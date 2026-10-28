"""
Builds every table the reanalysis report needs from out/fit_runs.json, and exports the
fitted curves and weighted residuals for the decay figures.

Tables written to out/:
    window_comparison.csv     component and mean lifetimes, DAS6 window vs the new rule,
                              with the published values as a third reference column
    window_sensitivity.csv    tau_amp under both window edges moved, vs the 0.54%
                              spread across independent TO1-B preparations
    model_order.csv           k=2 vs k=3 under the new window: deviance/DoF, Durbin-Watson,
                              third-component lifetime and amplitude, replicate reproducibility
    scatter_term.csv          with/without the s*IRF term, and without the baseline
    colour_shift.csv          fitted shift under both windows, and the peak's position
                              relative to the window start
    primary_report.csv        the section 4 deliverable, per construct
    primary_per_acquisition.csv   per-acquisition amplitudes feeding the above
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import das_io
import fit_tcspc as ft

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NS = ft.NS_PER_CHANNEL
PUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "tcspc_global_fit_results.csv")


def load_runs():
    return json.load(open(os.path.join(OUT, "fit_runs.json"), encoding="utf-8"))


def _per(run):
    return pd.DataFrame(run["per_acq"])


def construct_summary(run):
    """Collapse a global fit to one row per construct.

    Uncertainties on the mean lifetimes are propagated from the REPLICATE SPREAD in the
    amplitudes, not from asymptotic errors on the fitted parameters - the shared
    lifetimes are common to every acquisition in the series, so all of the
    construct-to-construct variation lives in the amplitudes.
    """
    taus = np.array(run["taus_ns"])
    p = _per(run)
    rows = []
    for (dye, bp), g in p.groupby(["dye", "stem_bp"]):
        a_pct = np.vstack(g.a_pct.values)
        rows.append(dict(
            dye=dye, stem_bp=int(bp), n_acq=len(g),
            **{f"tau{i+1}_ns": taus[i] for i in range(len(taus))},
            **{f"a{i+1}_pct": a_pct[:, i].mean() for i in range(a_pct.shape[1])},
            **{f"a{i+1}_pct_sd": a_pct[:, i].std(ddof=1) for i in range(a_pct.shape[1])},
            tau_amp_ns=g.tau_amp_ns.mean(), tau_amp_sd=g.tau_amp_ns.std(ddof=1),
            tau_int_ns=g.tau_int_ns.mean(), tau_int_sd=g.tau_int_ns.std(ddof=1),
            scatter_pct=g.scatter_pct.mean(), scatter_pct_sd=g.scatter_pct.std(ddof=1),
            shift_ch=g.shift_ch.mean(), shift_ch_sd=g.shift_ch.std(ddof=1),
            baseline_cts=g.bg.mean(),
            lo_ch=g.lo.mean(), hi_ch=g.hi.mean(),
            lo_ns=g.lo_ns.mean(), hi_ns=g.hi_ns.mean(),
            dev_per_dof=g.dev_per_dof_acq.mean(), DW=g.DW.mean(),
        ))
    return pd.DataFrame(rows).sort_values(["dye", "stem_bp"])


def window_comparison(runs):
    pub = pd.read_csv(PUB)
    pub = pub[pub.model == "k2_scatter"][["dye", "stem_bp", "tau1_ns", "tau2_ns",
                                          "a1_pct", "tau_amp_ns", "tau_int_ns"]]
    pub.columns = ["dye", "stem_bp"] + [f"{c}_published" for c in
                                        ["tau1_ns", "tau2_ns", "a1_pct", "tau_amp_ns", "tau_int_ns"]]
    frames = []
    for wkind, label in (("das6", "das6"), ("rule", "new")):
        parts = [construct_summary(runs[f"{s}_k2_{wkind}"]) for s in ("TO1-B", "SALAD1")
                 if f"{s}_k2_{wkind}" in runs]
        if not parts:
            continue
        d = pd.concat(parts)
        keep = ["dye", "stem_bp", "tau1_ns", "tau2_ns", "a1_pct", "tau_amp_ns", "tau_amp_sd",
                "tau_int_ns", "tau_int_sd", "scatter_pct", "shift_ch", "baseline_cts",
                "lo_ch", "hi_ch", "dev_per_dof", "DW"]
        d = d[keep].rename(columns={c: f"{c}_{label}" for c in keep[2:]})
        frames.append(d.set_index(["dye", "stem_bp"]))
    out = pd.concat(frames, axis=1).reset_index().merge(pub, on=["dye", "stem_bp"], how="left")
    for c in ("tau1_ns", "tau2_ns", "tau_amp_ns", "tau_int_ns"):
        if f"{c}_das6" in out and f"{c}_new" in out:
            out[f"d_{c}_new_vs_das6_pct"] = 100 * (out[f"{c}_new"] - out[f"{c}_das6"]) / out[f"{c}_das6"]
        if f"{c}_new" in out and f"{c}_published" in out:
            out[f"d_{c}_new_vs_published_pct"] = (
                100 * (out[f"{c}_new"] - out[f"{c}_published"]) / out[f"{c}_published"])
    return out


def window_sensitivity(runs):
    """tau_amp under each perturbed window edge, referenced to the unperturbed rule."""
    rows = []
    for s in ("TO1-B", "SALAD1"):
        base = f"{s}_k2_rule"
        if base not in runs:
            continue
        b = construct_summary(runs[base])
        for tag, dch in [("lo-20", -20), ("lo-10", -10), ("lo+10", +10), ("lo+20", +20),
                         ("hi-200", -200), ("hi-100", -100), ("hi+100", +100), ("hi+200", +200)]:
            name = f"{s}_k2_rule_{'lo' if tag.startswith('lo') else 'hi'}{dch:+d}"
            if name not in runs:
                continue
            d = construct_summary(runs[name])
            m = b[["dye", "stem_bp", "tau_amp_ns", "tau_int_ns"]].merge(
                d[["dye", "stem_bp", "tau_amp_ns", "tau_int_ns"]], on=["dye", "stem_bp"],
                suffixes=("_base", "_pert"))
            rows.append(dict(series=s, edge=tag, delta_ch=dch,
                             taus_ns=np.round(runs[name]["taus_ns"], 4).tolist(),
                             d_tau_amp_pct_mean=100 * ((m.tau_amp_ns_pert - m.tau_amp_ns_base)
                                                       / m.tau_amp_ns_base).mean(),
                             d_tau_amp_pct_max=100 * ((m.tau_amp_ns_pert - m.tau_amp_ns_base)
                                                      / m.tau_amp_ns_base).abs().max(),
                             d_tau_int_pct_mean=100 * ((m.tau_int_ns_pert - m.tau_int_ns_base)
                                                       / m.tau_int_ns_base).mean(),
                             dev_per_dof=runs[name]["dev_per_dof"]))
    return pd.DataFrame(rows)


def model_order(runs):
    """k=2 vs k=3 under both windows, with the third component's reproducibility."""
    rows = []
    for s in ("TO1-B", "SALAD1"):
        for wkind in ("das6", "rule"):
            n2, n3 = f"{s}_k2_{wkind}", f"{s}_k3_{wkind}"
            if n2 not in runs or n3 not in runs:
                continue
            r2, r3 = runs[n2], runs[n3]
            p3 = _per(r3)
            a3 = np.vstack(p3.a_pct.values)[:, 2]
            rows.append(dict(series=s, window=wkind,
                             taus_k2=np.round(r2["taus_ns"], 4).tolist(),
                             taus_k3=np.round(r3["taus_ns"], 4).tolist(),
                             dev_per_dof_k2=r2["dev_per_dof"], dev_per_dof_k3=r3["dev_per_dof"],
                             DW_k2=_per(r2).DW.mean(), DW_k3=p3.DW.mean(),
                             tau3_ns=r3["taus_ns"][2],
                             a3_pct_mean=a3.mean(), a3_pct_sd=a3.std(ddof=1),
                             a3_pct_min=a3.min(), a3_pct_max=a3.max(),
                             a3_cv_pct=100 * a3.std(ddof=1) / a3.mean()))
    return pd.DataFrame(rows)


def scatter_and_baseline(runs):
    rows = []
    for s in ("TO1-B", "SALAD1"):
        for tag, name in (("scatter + baseline", f"{s}_k2_rule"),
                          ("baseline only (no scatter)", f"{s}_k2_rule_noscatter"),
                          ("scatter only (no baseline)", f"{s}_k2_rule_nobaseline")):
            if name not in runs:
                continue
            r = runs[name]
            p = _per(r)
            rows.append(dict(series=s, model=tag, taus_ns=np.round(r["taus_ns"], 4).tolist(),
                             dev_per_dof=r["dev_per_dof"], DW=p.DW.mean(),
                             scatter_pct=p.scatter_pct.mean(), baseline_cts=p.bg.mean(),
                             tau_amp_ns=p.tau_amp_ns.mean(), n_params=r["n_params"]))
    return pd.DataFrame(rows)


def colour_shift(runs):
    rows = []
    for s in ("TO1-B", "SALAD1"):
        for wkind in ("das6", "rule"):
            name = f"{s}_k2_{wkind}"
            if name not in runs:
                continue
            p = _per(runs[name])
            rows.append(dict(series=s, window=wkind, shift_mean_ch=p.shift_ch.mean(),
                             shift_sd_ch=p.shift_ch.std(ddof=1),
                             shift_min_ch=p.shift_ch.min(), shift_max_ch=p.shift_ch.max(),
                             shift_mean_ps=1000 * NS * p.shift_ch.mean(),
                             lo_mean_ch=p.lo.mean()))
    return pd.DataFrame(rows)


def export_curves(runs, primary="rule", constructs=((8,), (12,))):
    """Fitted curve and weighted residuals on the data channel grid, for the decay figures.

    Exports 8 and 12 bp with each fluorogen, which is what the chapter's decay figure needs.
    """
    recs = das_io.load_folder(os.environ.get("TCSPC_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")))
    by = {(r["dye"], r["stem_bp"]): r for r in recs}
    want = [(d, bp) for d in ("TO1-B", "SALAD1") for bp in (8, 12)]
    frames = []
    for dye, bp in want:
        name = f"{dye}_k2_{primary}"
        if name not in runs or (dye, bp) not in by:
            continue
        run = runs[name]
        taus = np.array(run["taus_ns"])
        for rec in [x for x in run["per_acq"] if x["dye"] == dye and x["stem_bp"] == bp]:
            src = by[(dye, bp)]
            i = rec["acq"] - 1
            a = ft.Acquisition(dye, bp, rec["acq"], src["decays"][i], src["prompts"][i],
                               lo=rec["lo"], hi=rec["hi"])
            m, r = ft.weighted_residuals(a, taus, np.array(rec["B"]), rec["s"],
                                         rec["shift_ch"], rec["bg"])
            ch = np.arange(a.decay.size)
            inw = (ch >= a.lo) & (ch < a.hi)
            frames.append(pd.DataFrame(dict(
                dye=dye, stem_bp=bp, acq=rec["acq"], channel=ch, time_ns=ch * NS,
                prompt=a.prompt, decay=a.decay, fit=m, weighted_residual=r,
                in_window=inw)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    runs = load_runs()
    built = {}
    for name, fn in [("window_comparison", window_comparison),
                     ("window_sensitivity", window_sensitivity),
                     ("model_order", model_order),
                     ("scatter_term", scatter_and_baseline),
                     ("colour_shift", colour_shift)]:
        try:
            t = fn(runs)
            if len(t):
                t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
                built[name] = t
        except Exception as e:
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
    # section 4 primary deliverable
    parts = [construct_summary(runs[f"{s}_k2_rule"]) for s in ("TO1-B", "SALAD1")
             if f"{s}_k2_rule" in runs]
    if parts:
        prim = pd.concat(parts)
        prim.to_csv(os.path.join(OUT, "primary_report.csv"), index=False)
        built["primary_report"] = prim
        peracq = pd.concat([_per(runs[f"{s}_k2_rule"]) for s in ("TO1-B", "SALAD1")
                            if f"{s}_k2_rule" in runs])
        peracq.to_csv(os.path.join(OUT, "primary_per_acquisition.csv"), index=False)
    cur = export_curves(runs)
    if len(cur):
        cur.to_csv(os.path.join(OUT, "decay_curves_8_12bp.csv"), index=False)
        built["decay_curves"] = cur
    print("built:", ", ".join(built))
    return built


if __name__ == "__main__":
    main()
