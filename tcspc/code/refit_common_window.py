#!/usr/bin/env python
"""Refit every Mango II TCSPC decay over ONE common fit window.

Why. The pipeline's end rule (first channel after the peak where the 15-point smoothed
decay falls below background + 3 SD) was applied to each file separately, so replicates
of one construct were cut at ends up to 15 ns apart. This script keeps the rule but
applies it consistently: every acquisition is fitted from the common start (IRF at 1 %
of its peak, minus 10 channels) to the EARLIEST per-file end in the dataset, so every
decay still carries signal above background across the whole window.

Why not the pooled end. Applying the end rule to summed decays moves the end out to
~119 ns, because summing raises the counts. That drags the fit into the ~54 ns tail
that a constant baseline cannot absorb: the short lifetime shifts by ~7 % and both fit
statistics get worse. It also makes the window depend on how many files are pooled.

Everything else is unchanged: fit_tcspc.fit_global with k = 2, scatter term, fitted
baseline, lifetimes linked within each dye series, Poisson maximum likelihood.

Usage:

    python refit_common_window.py      # decay files in ../data, stored workbook beside this script
    python refit_common_window.py --workbook TCSPC_reanalysis_tables.xlsx "path/to/*_batch.das"

    --per-dye          one window per dye (each dye's earliest end) instead of one for all
    --out DIR          output folder, one CSV file per table (default out/common_window
                       beside this script)
    --from-workbook    test mode: fit the 8 and 12 bp decays stored in the workbook itself

File patterns are expanded by the script itself, so they work in every shell.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_tcspc as F                                                   # noqa: E402

K, SCATTER, BACKGROUND = 2, True, True          # the production model, unchanged
REPLACED = ("primary_per_acquisition", "primary_report", "fit_windows", "decay_curves_8_12bp")


# ----------------------------------------------------------------------------------
# loading and labelling
# ----------------------------------------------------------------------------------
def load_from_das(files, inventory):
    """Read .das files with the pipeline's own reader and label each decay by matching its
    total counts to the acquisition inventory, which reproduces the existing labels exactly."""
    import das_global as G
    loaded = [x for x in G.load_decays(files) if G._bp(x[0]) != 9999]      # blanks have no bp prefix
    fingerprint = {int(r.total_counts): (r.dye, int(r.stem_bp), int(r.acq)) for r in inventory.itertuples()}
    acqs, unmatched = [], []
    for sample, dec, irf, src, idx in loaded:
        key = fingerprint.get(int(round(dec.sum())))
        if key is None:
            unmatched.append(f"{src} decay {idx} (sample {sample}, total {int(dec.sum())})")
            continue
        acqs.append(F.Acquisition(dye=key[0], stem_bp=key[1], acq=key[2], decay=dec, prompt=irf))
    if unmatched:
        sys.exit("Stopped: these decays match no row of the acquisition inventory:\n  " + "\n  ".join(unmatched))
    return acqs


def load_from_workbook(path):
    d = pd.read_excel(path, sheet_name="decay_curves_8_12bp")
    return [F.Acquisition(dye=dye, stem_bp=int(bp), acq=int(acq),
                          decay=g.sort_values("channel").decay.values.astype(float),
                          prompt=g.sort_values("channel").prompt.values.astype(float))
            for (dye, bp, acq), g in d.groupby(["dye", "stem_bp", "acq"])]


def check_complete(acqs, inventory, test_mode):
    labels = {(a.dye, a.stem_bp, a.acq) for a in acqs}
    if len(labels) != len(acqs):
        sys.exit("Stopped: two decays received the same label; check for duplicate files.")
    expected = {(r.dye, int(r.stem_bp), int(r.acq)) for r in inventory.itertuples()}
    missing = sorted(expected - labels)
    if missing and not test_mode:
        sys.exit(f"Stopped: {len(missing)} acquisitions in the inventory were not found: {missing}")
    return missing


# ----------------------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------------------
def set_common_windows(acqs, per_dye=False):
    """Common start from the IRF; end at the earliest per-file end (overall, or per dye)."""
    per_file = {}
    for a in acqs:
        F.apply_window(a)
        per_file[(a.dye, a.stem_bp, a.acq)] = a.hi
    groups = {}
    for a in acqs:
        groups.setdefault(a.dye if per_dye else "all", []).append(a)
    for g in groups.values():
        lo, hi = min(a.lo for a in g), min(a.hi for a in g)
        for a in g:
            a.lo, a.hi = lo, hi
    return per_file


# ----------------------------------------------------------------------------------
# fitting and tables in the workbook's own format
# ----------------------------------------------------------------------------------
def fit_all(acqs):
    fits = {}
    for dye in sorted({a.dye for a in acqs}):
        grp = [a for a in acqs if a.dye == dye]
        print(f"fitting {dye}: {len(grp)} acquisitions, window {grp[0].lo}-{grp[0].hi} "
              f"({grp[0].lo * F.NS_PER_CHANNEL:.1f}-{grp[0].hi * F.NS_PER_CHANNEL:.1f} ns) ...", flush=True)
        gf = F.fit_global(grp, k=K, background=BACKGROUND, scatter=SCATTER)
        print(f"  linked lifetimes {np.round(gf.taus_ns, 4)} ns")
        fits[dye] = (grp, gf)
    return fits


def per_acquisition_table(fits):
    rows = []
    for dye, (grp, gf) in fits.items():
        for a, f in zip(grp, gf.per_acq):
            B = np.asarray(f["B"], float)
            tau_amp, tau_int = F.mean_lifetimes(gf.taus_ns, B)
            m, _ = F.weighted_residuals(a, gf.taus_ns, B, f["s"], f["shift"], f["bg"])
            rows.append(dict(dye=a.dye, stem_bp=a.stem_bp, acq=a.acq, lo=a.lo, hi=a.hi,
                             lo_ns=a.lo * F.NS_PER_CHANNEL, hi_ns=a.hi * F.NS_PER_CHANNEL,
                             B=str(list(map(float, B))), a_pct=str(list(map(float, F.amplitude_fractions_pct(B)))),
                             s=float(f["s"]), shift_ch=float(f["shift"]), bg=float(f["bg"]),
                             scatter_pct=F.scatter_fraction_pct(a, gf.taus_ns, B, f["s"], f["shift"], f["bg"]),
                             tau_amp_ns=tau_amp, tau_int_ns=tau_int, deviance=float(f["deviance"]),
                             n_channels=int(f["n_channels"]),
                             dev_per_dof_acq=float(f["deviance"]) / (int(f["n_channels"]) - 5),
                             DW=F.durbin_watson(a.decay[a.slc], m[a.slc])))
    return pd.DataFrame(rows).sort_values(["dye", "stem_bp", "acq"], ascending=[False, True, True])


def report_table(per, fits):
    out = []
    for (dye, bp), g in per.groupby(["dye", "stem_bp"], sort=False):
        taus = fits[dye][1].taus_ns
        a1 = g.a_pct.map(lambda s: eval(s)[0]); a2 = g.a_pct.map(lambda s: eval(s)[1])
        out.append(dict(dye=dye, stem_bp=bp, n_acq=len(g), tau1_ns=taus[0], tau2_ns=taus[1],
                        a1_pct=a1.mean(), a2_pct=a2.mean(), a1_pct_sd=a1.std(), a2_pct_sd=a2.std(),
                        tau_amp_ns=g.tau_amp_ns.mean(), tau_amp_sd=g.tau_amp_ns.std(),
                        tau_int_ns=g.tau_int_ns.mean(), tau_int_sd=g.tau_int_ns.std(),
                        scatter_pct=g.scatter_pct.mean(), scatter_pct_sd=g.scatter_pct.std(),
                        shift_ch=g.shift_ch.mean(), shift_ch_sd=g.shift_ch.std(), baseline_cts=g.bg.mean(),
                        lo_ch=g.lo.mean(), hi_ch=g.hi.mean(), lo_ns=g.lo_ns.mean(), hi_ns=g.hi_ns.mean(),
                        dev_per_dof=g.dev_per_dof_acq.mean(), DW=g.DW.mean()))
    return pd.DataFrame(out)


def windows_table(old, acqs, per_file):
    w = old.copy() if old is not None else pd.DataFrame([dict(dye=a.dye, stem_bp=a.stem_bp, acq=a.acq) for a in acqs])
    by = {(a.dye, a.stem_bp, a.acq): a for a in acqs}
    w["per_file_hi"] = [per_file.get((r.dye, int(r.stem_bp), int(r.acq)), np.nan) for r in w.itertuples()]
    for col, fn in (("new_lo", lambda a: a.lo), ("new_hi", lambda a: a.hi),
                    ("new_lo_ns", lambda a: a.lo * F.NS_PER_CHANNEL), ("new_hi_ns", lambda a: a.hi * F.NS_PER_CHANNEL),
                    ("n_ch_new", lambda a: a.hi - a.lo)):
        w[col] = [fn(by[(r.dye, int(r.stem_bp), int(r.acq))]) if (r.dye, int(r.stem_bp), int(r.acq)) in by else np.nan
                  for r in w.itertuples()]
    return w


def curves_table(old, fits):
    """Refresh the fit, residual and window columns of the stored 8 and 12 bp decays."""
    if old is None:
        return None
    c = old.copy()
    for dye, (grp, gf) in fits.items():
        for a, f in zip(grp, gf.per_acq):
            sel = (c.dye == a.dye) & (c.stem_bp == a.stem_bp) & (c.acq == a.acq)
            if not sel.any():
                continue
            m, r = F.weighted_residuals(a, gf.taus_ns, np.asarray(f["B"]), f["s"], f["shift"], f["bg"])
            ch = c.loc[sel, "channel"].values.astype(int)
            c.loc[sel, "fit"] = m[ch]; c.loc[sel, "weighted_residual"] = r[ch]
            c.loc[sel, "in_window"] = (ch >= a.lo) & (ch < a.hi)
    return c


def changes_table(old_rep, new_rep):
    m = new_rep.merge(old_rep, on=["dye", "stem_bp"], suffixes=("", "_old"))
    cols = ["dye", "stem_bp", "tau1_ns_old", "tau1_ns", "tau2_ns_old", "tau2_ns", "a2_pct_old", "a2_pct",
            "tau_amp_ns_old", "tau_amp_ns", "tau_amp_sd_old", "tau_amp_sd", "hi_ns_old", "hi_ns",
            "dev_per_dof_old", "dev_per_dof", "DW_old", "DW"]
    m = m[cols].copy()
    m["tau_amp_change_pct"] = 100 * (m.tau_amp_ns / m.tau_amp_ns_old - 1)
    return m


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("das", nargs="*", default=[os.path.join(HERE, os.pardir, "data", "*_batch.das")])
    ap.add_argument("--workbook", default=os.path.join(HERE, "TCSPC_reanalysis_tables.xlsx"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "common_window"))
    ap.add_argument("--per-dye", action="store_true")
    ap.add_argument("--from-workbook", action="store_true")
    a = ap.parse_args()

    book = pd.read_excel(a.workbook, sheet_name=None)
    inventory = book["acquisition_inventory"]
    if a.from_workbook:
        acqs = load_from_workbook(a.workbook)
    else:
        files = sorted({f for p in a.das for f in glob.glob(p)})
        if not files:
            sys.exit("No .das files given. Pass the complex files, for example:  data/*bp*.das")
        acqs = load_from_das(files, inventory)
    missing = check_complete(acqs, inventory, a.from_workbook)
    print(f"{len(acqs)} acquisitions loaded and labelled" + (f" (test mode; {len(missing)} not in workbook)" if missing else ""))

    per_file = set_common_windows(acqs, per_dye=a.per_dye)
    ends = sorted({x.hi for x in acqs})
    print(f"common window: start {acqs[0].lo}, end {ends} (per-file ends ranged "
          f"{min(per_file.values())}-{max(per_file.values())})")

    fits = fit_all(acqs)
    per = per_acquisition_table(fits)
    rep = report_table(per, fits)

    new_book = dict(book)
    new_book["primary_per_acquisition"] = per
    new_book["primary_report"] = rep
    new_book["fit_windows"] = windows_table(book.get("fit_windows"), acqs, per_file)
    if book.get("decay_curves_8_12bp") is not None:
        new_book["decay_curves_8_12bp"] = curves_table(book["decay_curves_8_12bp"], fits)
    chg = changes_table(book["primary_report"], rep)
    new_book["common_window_changes"] = chg
    note = pd.DataFrame([dict(sheet="common_window_changes",
                              description=("Refit over one common window (refit_common_window.py). "
                                           f"Replaced: {', '.join(REPLACED)}. Other sheets are from the "
                                           "per-file run and were not recomputed."))])
    if "INDEX" in new_book:
        new_book["INDEX"] = pd.concat([book["INDEX"], note], ignore_index=True)

    os.makedirs(a.out, exist_ok=True)
    for name, df in new_book.items():
        if df is not None:
            df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)

    show = chg[["dye", "stem_bp", "tau_amp_ns_old", "tau_amp_ns", "tau_amp_change_pct", "dev_per_dof_old", "dev_per_dof"]]
    print("\n" + show.round(4).to_string(index=False))
    t12 = rep[rep.stem_bp == 12].set_index("dye").tau_amp_ns
    o12 = book["primary_report"][book["primary_report"].stem_bp == 12].set_index("dye").tau_amp_ns
    if {"SALAD1", "TO1-B"} <= set(t12.index):
        print(f"\n12 bp ratio SALAD1 / TO1-B: {o12['SALAD1'] / o12['TO1-B']:.4f} before, "
              f"{t12['SALAD1'] / t12['TO1-B']:.4f} after")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
