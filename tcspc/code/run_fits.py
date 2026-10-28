"""
Batch runner for the TCSPC reanalysis: every global fit the report needs.

Primary model, established by validation (see out/validation_B_vs_published.csv):
    k = 2 exponentials + zero-lifetime scatter term + per-acquisition constant baseline,
    lifetimes linked within each fluorophore series, amplitudes free per acquisition.

The baseline is NOT in the handover's written specification but is demonstrably part of
the original estimator: without it the k=2 fit is catastrophic (D/dof 3.74, tau2 driven to
10.9 ns) and with it the published values reproduce to ~1%. The data contain a ~50 ns
instrumental tail carrying ~1.1% of counts, which is what the baseline absorbs.

Writes out/fit_runs.json. Re-run with --only <name> to redo one run.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import das_io
import fit_tcspc as ft

TCSPC_DIR = os.environ.get("TCSPC_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NS = ft.NS_PER_CHANNEL


def build():
    recs = das_io.load_folder(TCSPC_DIR)
    acqs, vend = [], {}
    for r in recs:
        vend[(r["dye"], r["stem_bp"])] = (int(r["bi_fit"]["lo_ch"]),
                                          int(r["bi_fit"]["hi_reduced_ch"]) + 1)
        for i in range(r["n_acq"]):
            acqs.append(ft.Acquisition(r["dye"], r["stem_bp"], i + 1,
                                       r["decays"][i].copy(), r["prompts"][i].copy()))
    return acqs, vend


def set_window(acqs, vend, kind, d_lo=0, d_hi=0, frac=0.01, margin=10, n_sd=3.0, smooth=15):
    """kind: 'das6' | 'rule'. d_lo/d_hi shift the rule-based edges for the sensitivity sweep."""
    for a in acqs:
        if kind == "das6":
            a.lo, a.hi = vend[(a.dye, a.stem_bp)]
        else:
            a.lo = ft.window_start(a.prompt, frac=frac, margin=margin) + d_lo
            a.hi = ft.window_end(a.decay, n_sd=n_sd, smooth=smooth) + d_hi
        a.lo = max(int(a.lo), 0)
        a.hi = min(int(a.hi), a.decay.size)
    return acqs


def summarise(g, acqs):
    """Everything the report needs from one global fit, JSON-serialisable."""
    per = []
    for a, f in zip(acqs, g.per_acq):
        m, r = ft.weighted_residuals(a, g.taus_ns, f["B"], f["s"], f["shift"], f["bg"])
        ta, ti = ft.mean_lifetimes(g.taus_ns, f["B"])
        per.append(dict(
            dye=a.dye, stem_bp=a.stem_bp, acq=a.acq, lo=int(a.lo), hi=int(a.hi),
            lo_ns=a.lo * NS, hi_ns=a.hi * NS,
            B=[float(v) for v in f["B"]],
            a_pct=[float(v) for v in ft.amplitude_fractions_pct(f["B"])],
            s=float(f["s"]), shift_ch=float(f["shift"]), bg=float(f["bg"]),
            scatter_pct=float(ft.scatter_fraction_pct(a, g.taus_ns, f["B"], f["s"],
                                                      f["shift"], f["bg"])),
            tau_amp_ns=ta, tau_int_ns=ti,
            deviance=float(f["deviance"]), n_channels=int(f["n_channels"]),
            dev_per_dof_acq=float(f["deviance"] / (f["n_channels"] - (len(f["B"]) + 3))),
            DW=float(ft.durbin_watson(a.decay[a.slc], m[a.slc])),
        ))
    return dict(taus_ns=[float(v) for v in g.taus_ns], deviance=g.deviance,
                n_params=g.n_params, n_channels=g.n_channels, dof=g.dof,
                dev_per_dof=g.dev_per_dof, scatter=bool(g.scatter),
                background=bool(g.background), meta=g.meta, per_acq=per)


def main(only=None):
    acqs, vend = build()
    SAL = [a for a in acqs if a.dye == "SALAD1"]
    TO1 = [a for a in acqs if a.dye == "TO1-B"]
    series = {"TO1-B": TO1, "SALAD1": SAL}
    init = {2: [2.1, 5.0], 3: [1.0, 3.5, 10.0]}

    runs = []
    # --- primary and old-window comparison, both model orders ---
    for wkind in ("das6", "rule"):
        for k in (2, 3):
            for sname in ("TO1-B", "SALAD1"):
                runs.append(dict(name=f"{sname}_k{k}_{wkind}", series=sname, k=k,
                                 window=dict(kind=wkind), scatter=True, background=True))
    # --- section 2: does the scatter term survive the new window? ---
    for sname in ("TO1-B", "SALAD1"):
        runs.append(dict(name=f"{sname}_k2_rule_noscatter", series=sname, k=2,
                         window=dict(kind="rule"), scatter=False, background=True))
        runs.append(dict(name=f"{sname}_k2_rule_nobaseline", series=sname, k=2,
                         window=dict(kind="rule"), scatter=True, background=False))
    # --- section 1: sensitivity to both window edges ---
    for sname in ("TO1-B", "SALAD1"):
        for d_lo in (-20, -10, +10, +20):
            runs.append(dict(name=f"{sname}_k2_rule_lo{d_lo:+d}", series=sname, k=2,
                             window=dict(kind="rule", d_lo=d_lo), scatter=True, background=True))
        for d_hi in (-200, -100, +100, +200):
            runs.append(dict(name=f"{sname}_k2_rule_hi{d_hi:+d}", series=sname, k=2,
                             window=dict(kind="rule", d_hi=d_hi), scatter=True, background=True))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fit_runs.json")
    store = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for cfg in runs:
        if only and cfg["name"] != only:
            continue
        if cfg["name"] in store and not only:
            continue
        acq = series[cfg["series"]]
        set_window(acq, vend, **cfg["window"])
        g = ft.fit_global(acq, k=cfg["k"], scatter=cfg["scatter"],
                          background=cfg["background"], taus_ns_init=init[cfg["k"]])
        store[cfg["name"]] = dict(config=cfg, **summarise(g, acq))
        json.dump(store, open(path, "w", encoding="utf-8"), indent=1)
        print(f"{cfg['name']:32s} taus={np.round(g.taus_ns,4)} D/dof={g.dev_per_dof:.4f}", flush=True)
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    main(only=ap.parse_args().only)
