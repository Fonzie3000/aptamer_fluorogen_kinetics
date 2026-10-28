#!/usr/bin/env python
"""Bulk-emission, lifetime and emitting-fraction numbers of Chapter 4, written to one CSV file.

Sections 4.3.1 and 4.3.2 of the thesis quote these numbers (pages 138 to 145). Their
calculations sat in the Chapter 4 figure code, which this repository leaves out; they are
copied here without the plotting. The arithmetic is unchanged. Some numbers had no code:
the integrated ratio, the A22U ratios, the split of the lifetime difference and the two
spreads of the lifetime across constructs. They are plain ratios and differences of the
same inputs, written out below.

Inputs
    ../data/screen_spectra_triplicates.xlsx          bulk screen: four aptamer-fluorogen
                                                     combinations, three replicates each
    out/common_window/primary_per_acquisition.csv    written by refit_common_window.py
    out/common_window/primary_report.csv             written by refit_common_window.py
    ../../smtirf/kinetics/tables/brightness_ratio_by_condition.csv
                                                     written by smtirf/kinetics/analyse_smtirf.py;
                                                     optional, used only for the single-molecule
                                                     rows, which are left empty without it

Output
    out/ratio_numbers.csv    one row per number: quantity, value, unit, thesis page, input

Run (after refit_common_window.py):   python ratio_numbers.py
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.join(HERE, os.pardir, "data", "screen_spectra_triplicates.xlsx")
COMMON = os.path.join(HERE, "out", "common_window")
SM_TABLE = os.path.join(HERE, os.pardir, os.pardir, "smtirf", "kinetics", "tables",
                        "brightness_ratio_by_condition.csv")
OUT = os.path.join(HERE, "out", "ratio_numbers.csv")

TRAP = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz

# Imported constants, as set in the Chapter 4 figure code
EPS_RATIO = 47284 / 52142                # extinction ratio of the bound complexes, Yang et al. 2025
KR_PERMITTED = (0.72, 1.17)              # radiative-rate ratio the Strickler-Berg relation allows,
                                         # imported: propagated over the band-shape assumptions in the
                                         # earlier analysis, and not recomputed from these spectra
FB_BAND = (0.745, 1.343)                 # multiplicative interval on the emitting-fraction requirement,
                                         # propagated over the extinction ratio, the radiative term and
                                         # the lifetime ratio; imported with KR_PERMITTED
CONC_SHOWN = (1.0, 2.0, 4.0, 8.0)        # the reported series; 0.5 and 16 nM are excluded
POWER_MW = 1.5                           # laser power of the concentration series


# ------------------------------------------------------------------ bulk screen spectra
def _screen_replicates(path=SCREEN):
    """Blank-subtracted triplicate spectra of the four-combination screen.

    The workbook has sheets A to D, each holding replicate columns with the water blank
    subtracted once, named like 'A1-H2O'. Returns (wavelength, {sheet: replicate array},
    {sheet: combination label written in that sheet}).
    """
    xl = pd.ExcelFile(path)
    if not {"A", "B", "C", "D"} <= set(xl.sheet_names):
        sys.exit(f"{path}: expected sheets A to D")
    out, labels, wl = {}, {}, None
    for sh in "ABCD":
        d = xl.parse(sh)
        cols = [c for c in d.columns if re.fullmatch(rf"{sh}\d+-H2O", str(c).strip())]
        if len(cols) < 2:
            break
        wl = pd.to_numeric(d.iloc[:, 0], errors="coerce").values
        out[sh] = d[cols].apply(pd.to_numeric, errors="coerce").values
        labels[sh] = next((str(c) for c in d.columns if re.search(r"(wt|mt|a22u).*\+", str(c), re.I)), sh)
    if len(out) != 4:
        sys.exit(f"{path}: replicate columns like 'A1-H2O' not found in every sheet")
    ok = np.isfinite(wl)
    return wl[ok], {k: v[ok] for k, v in out.items()}, labels


def _spectra():
    """Wavelength axis and one replicate group per combination, water subtracted once.

    The combination of each sheet is read from the label written in it: 'wt' or 'mt' for
    the aptamer, and TO1-B or 365a-061, SALAD1's compound code, for the fluorogen.
    """
    wl, reps, labels = _screen_replicates()
    groups = {}
    for sh, arr in reps.items():
        lab = labels[sh].lower()
        apt = "A22U" if ("mt" in lab or "a22u" in lab) else "MII wt"
        dye = "TO1-B" if "to1" in lab else "SALAD1"
        groups[f"{apt} + {dye}"] = pd.DataFrame(arr, columns=[f"{sh}{i + 1}" for i in range(arr.shape[1])])
    return pd.Series(wl), groups


def _band(wl, y):
    """Band statistics by the rules the earlier analysis used.

    The spectra begin above the blue half-maximum, so the band's blue edge is the first
    recorded wavelength and its red edge the half-maximum crossing; the width in
    wavenumbers is that span converted at its midpoint. The emission factor is the
    inverse mean of the cubed inverse wavenumber, with the wavelength Jacobian.
    """
    wl = np.asarray(wl, float); y = np.asarray(y, float)
    i = int(np.nanargmax(y)); hm = y[i] / 2
    r = next(j for j in range(i, len(y)) if y[j] <= hm)
    wr = float(np.interp(hm, [y[r], y[r - 1]], [wl[r], wl[r - 1]]))
    bl = [j for j in range(i, -1, -1) if y[j] <= hm]
    wlo = float(np.interp(hm, [y[bl[0]], y[bl[0] + 1]], [wl[bl[0]], wl[bl[0] + 1]])) if bl else float(wl[0])
    nu = 1e7 / wl; F = y * wl ** 2; o = np.argsort(nu)
    return dict(peak_nm=float(wl[i]), centroid_nm=float(TRAP(y * wl, wl) / TRAP(y, wl)),
                fwhm_nm=wr - wlo, fwhm_cm1=(wr - wlo) * 1e7 / ((wr + wlo) / 2) ** 2,
                emission_factor=float(TRAP(F[o], nu[o]) / TRAP(F[o] * nu[o] ** -3, nu[o])))


def _screen_constants(wl, g):
    """Bulk peak ratio and Strickler-Berg term, SALAD1 to TO1-B, from the wild-type spectra."""
    T, S = g["MII wt + TO1-B"].mean(axis=1).values, g["MII wt + SALAD1"].mean(axis=1).values
    bT, bS = _band(wl.values, T), _band(wl.values, S)
    ratio = float(np.nanmax(S) / np.nanmax(T))
    sb = EPS_RATIO * (bS["emission_factor"] / bT["emission_factor"]) * (bS["fwhm_cm1"] / bT["fwhm_cm1"])
    return ratio, float(sb), bT, bS


# ------------------------------------------------------------------ lifetimes
def _tc_acq(folder=COMMON):
    """TCSPC amplitude-weighted and intensity-weighted lifetimes, mean and SD over acquisitions."""
    a = pd.read_csv(os.path.join(folder, "primary_per_acquisition.csv"))
    return (a.groupby(["dye", "stem_bp"])
             .agg(tau_amp_ns=("tau_amp_ns", "mean"), tau_amp_sd=("tau_amp_ns", "std"),
                  tau_int_ns=("tau_int_ns", "mean"), tau_int_sd=("tau_int_ns", "std"),
                  n=("acq", "count")).reset_index())


def _tau_amp_ratio(folder=COMMON):
    """SALAD1 over TO1-B amplitude-weighted lifetime at 12 bp."""
    a = _tc_acq(folder)
    a = a[a.stem_bp == 12].set_index("dye").tau_amp_ns
    return float(a["SALAD1"] / a["TO1-B"])


def _lifetime_split(folder=COMMON):
    """Split of the 12 bp mean-lifetime difference into a population part and a lifetime part.

    tau_amp = (1 - a2) tau1 + a2 tau2. The population part changes a2 from the TO1-B to
    the SALAD1 value with the lifetimes held; the lifetime part changes the lifetimes with
    a2 held. The two orders give different splits, so the two are averaged.
    """
    r = pd.read_csv(os.path.join(folder, "primary_report.csv"))
    r = r[r.stem_bp == 12].set_index("dye")
    t1A, t2A, a2A = (float(r.loc["TO1-B", c]) for c in ("tau1_ns", "tau2_ns", "a2_pct"))
    t1B, t2B, a2B = (float(r.loc["SALAD1", c]) for c in ("tau1_ns", "tau2_ns", "a2_pct"))
    a2A, a2B = a2A / 100, a2B / 100
    tau = lambda t1, t2, a2: (1 - a2) * t1 + a2 * t2                       # noqa: E731
    TA, TB = tau(t1A, t2A, a2A), tau(t1B, t2B, a2B)
    pop = ((tau(t1A, t2A, a2B) - TA) + (TB - tau(t1B, t2B, a2A))) / 2
    return TB - TA, pop, (TB - TA) - pop


def _construct_spread(folder=COMMON):
    """Spread of the amplitude-weighted lifetime across constructs, as quoted in the thesis.

    TO1-B, seven constructs: standard deviation over mean. SALAD1, two constructs:
    difference over mean.
    """
    r = pd.read_csv(os.path.join(folder, "primary_report.csv"))
    t = r[r.dye == "TO1-B"].tau_amp_ns.values
    s = r[r.dye == "SALAD1"].tau_amp_ns.values
    return (100 * t.std(ddof=1) / t.mean(), len(t),
            100 * (s.max() - s.min()) / s.mean(), len(s))


# ------------------------------------------------------------------ the requirement
def fb_required(uniform_pred, sb_ratio, tau_amp_ratio, borrow_eps=True, borrow_sb=True):
    """Emitting-fraction ratio required by the bulk and lifetime measurements."""
    d = (EPS_RATIO if borrow_eps else 1.0) * (sb_ratio if borrow_sb else 1.0)
    return uniform_pred / (d * tau_amp_ratio)


def _single_molecule_ratio(path):
    """Per-complex brightness ratio across the reported concentrations at 1.5 mW."""
    b = pd.read_csv(path)
    if "power_mW" in b.columns:
        b = b[b.power_mW == POWER_MW]
    keep = b[b.conc_nM.isin(CONC_SHOWN)]
    b = keep if len(keep) else b
    return float(b.ratio_median.median()), float(b.ratio_median.min()), float(b.ratio_median.max())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common-window", default=COMMON, help="folder written by refit_common_window.py")
    ap.add_argument("--sm-table", default=SM_TABLE, help="brightness_ratio_by_condition.csv from smtirf/kinetics")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.common_window, "primary_per_acquisition.csv")):
        sys.exit(f"{a.common_window} has no primary_per_acquisition.csv; run refit_common_window.py first")

    wl, g = _spectra()
    uniform_pred, sb_ratio, bT, bS = _screen_constants(wl, g)
    peak = {k: float(np.nanmax(v.mean(axis=1).values)) for k, v in g.items()}
    area = {k: float(TRAP(v.mean(axis=1).values, wl.values)) for k, v in g.items()}
    tau_ratio = _tau_amp_ratio(a.common_window)
    diff, pop, life = _lifetime_split(a.common_window)
    cv_t, n_t, rd_s, n_s = _construct_spread(a.common_window)
    fb_measured = fb_required(uniform_pred, sb_ratio, tau_ratio, borrow_eps=False, borrow_sb=False)
    fb_eps = fb_required(uniform_pred, sb_ratio, tau_ratio, borrow_eps=True, borrow_sb=False)
    fb_full = fb_required(uniform_pred, sb_ratio, tau_ratio)

    SCR, CW, IMP = "screen_spectra_triplicates.xlsx", "out/common_window", "imported constant"
    rows = [
        ("bulk ratio SALAD1/TO1-B on wild type, band maximum", uniform_pred, "", 138, SCR),
        ("bulk ratio SALAD1/TO1-B on wild type, integrated band", area["MII wt + SALAD1"] / area["MII wt + TO1-B"], "", 138, SCR),
        ("A22U over wild type with TO1-B, band maximum", peak["A22U + TO1-B"] / peak["MII wt + TO1-B"], "", 138, SCR),
        ("A22U over wild type with SALAD1, band maximum", peak["A22U + SALAD1"] / peak["MII wt + SALAD1"], "", 138, SCR),
        ("best A22U combination over wild type with SALAD1, band maximum",
         max(peak["A22U + TO1-B"], peak["A22U + SALAD1"]) / peak["MII wt + SALAD1"], "", 138, SCR),
        ("emission maximum, SALAD1 on wild type", bS["peak_nm"], "nm", 138, SCR),
        ("emission maximum, TO1-B on wild type", bT["peak_nm"], "nm", 138, SCR),
        ("lifetime ratio SALAD1/TO1-B, amplitude-weighted, 12 bp", tau_ratio, "", 141, CW),
        ("mean-lifetime difference SALAD1 - TO1-B, 12 bp", diff, "ns", 141, CW),
        ("part of the difference from the population shift, order-averaged", pop, "ns", 141, CW),
        ("part of the difference from the longer long-lived state, order-averaged", life, "ns", 141, CW),
        (f"TO1-B lifetime across {n_t} constructs, standard deviation over mean", cv_t, "%", 141, CW),
        (f"SALAD1 lifetime between {n_s} constructs, difference over mean", rd_s, "%", 141, CW),
        ("bulk ratio over lifetime ratio", fb_measured, "", 141, f"{SCR}; {CW}"),
        ("extinction ratio SALAD1/TO1-B (Yang et al. 2025)", EPS_RATIO, "", 143, IMP),
        ("radiative-rate ratio permitted by Strickler-Berg, lower", KR_PERMITTED[0], "", 143, IMP),
        ("radiative-rate ratio permitted by Strickler-Berg, upper", KR_PERMITTED[1], "", 143, IMP),
        ("radiative-rate ratio needed with no difference in emitting fraction", fb_eps, "", 143, f"{SCR}; {CW}; {IMP}"),
        ("Strickler-Berg radiative term", sb_ratio, "", 143, f"{SCR}; {IMP}"),
        ("emitting-fraction ratio required, with the imported terms", fb_full, "", 143, f"{SCR}; {CW}; {IMP}"),
        ("emitting-fraction ratio required, interval lower", fb_full * FB_BAND[0], "", 143, IMP),
        ("emitting-fraction ratio required, interval upper", fb_full * FB_BAND[1], "", 143, IMP),
        ("upper bound on the TO1-B emitting fraction, with the imported terms", 1 / fb_full, "", 143, ""),
        ("upper bound on the TO1-B emitting fraction, measured ratios alone", 1 / fb_measured, "", 143, ""),
    ]
    if os.path.exists(a.sm_table):
        sm, sm_lo, sm_hi = _single_molecule_ratio(a.sm_table)
        src = "smtirf/kinetics/tables/brightness_ratio_by_condition.csv"
        note = src
    else:
        sm = sm_lo = sm_hi = np.nan
        src = "not found; run smtirf/kinetics first, then this script again"
        note = src
        print(f"note: {a.sm_table} not found; the single-molecule rows are left empty")
    rows += [
        ("single-molecule brightness ratio SALAD1/TO1-B, median over 1 to 8 nM at 1.5 mW", sm, "", 145, src),
        ("single-molecule brightness ratio, lowest condition", sm_lo, "", 145, src),
        ("single-molecule brightness ratio, highest condition", sm_hi, "", 145, src),
        ("single-molecule ratio over lifetime ratio", sm / tau_ratio, "", 141, f"{note}; {CW}"),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value", "unit", "thesis_page", "input"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.max_colwidth", 80, "display.width", 140):
        print(out[["quantity", "value", "unit", "thesis_page"]].round(4).to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
