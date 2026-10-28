#!/usr/bin/env python
"""Single- against double-exponential fits for every dwell-time distribution in
the concentration series.

The reported fits use the two-component model. This module is the comparison
behind that choice: both models fitted to every distribution, with the evidence
for each written to one table.

Dwell sets are the ADOPTED ones, read from dwell_times_windowed.csv.gz:
  bound    repeated-binding sites only, floor MIN_DUR frames, the per-dye shared
           upper limit entered as right-censoring (apply_upper), boundary-truncated
           events right-censored
  unbound  interior gaps only (terminal dark intervals excluded, see
           FITTING_REPORT.md), floor max(MIN_DUR, MAX_GAP + 1) frames

Both models are fitted by analyse_smtirf.fit_dwell_set -- the pipeline's verified
estimator -- never by a local reimplementation.

Overlay. The model curve on a density histogram is the left-truncated PMF,
    P(D = d) = [sum_i a_i p_i q_i^(d-1)] / [sum_i a_i q_i^(a-1)],  q_i = exp(-1/tau_i)
divided by the frame time. Bins are integer numbers of frames; the empirical
density is count / (N * bin width), with N the number of observations at or
above the floor, so model and data share one normalisation.

Writes  tables/fit_compare_single_double.csv

Run after the earlier steps (run_kinetics.py runs every step in order):   python fit_compare.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                        # noqa: E402
import dwell_filters as DF                                        # noqa: E402

TAB, FT = A.TAB, A.FRAME_TIME_S
REF_PW = A.DEFAULT_POWER_MW
DMIN_ON = A.MIN_DUR
DMIN_OFF = max(A.MIN_DUR, A.MAX_GAP + 1)
XMAX_Q = 0.995          # x-range per panel: this quantile of the plotted dwells


def artefact_ceiling(kind):
    """Largest per-dwell dBIC the detection chain produced from a genuinely
    SINGLE-exponential truth (simulation E1). Read from the table, never typed:
    a real preference for two components must exceed it. For unbound dwells E1
    never produced a positive preference, so the ceiling there is at most zero."""
    p = os.path.join(TAB, "sim_e1_recovery.csv")
    if not os.path.exists(p):
        return np.nan
    e1 = pd.read_csv(p)
    col, n = (("on_dBIC", "n_on_dwells") if kind == "on" else ("off_dBIC", "n_off_dwells"))
    return float((e1[col] / e1[n]).max())


def pmf(d, floor, tau=None, af=None, t1=None, t2=None):
    """Left-truncated frame-discretised PMF, exactly as nll_geom defines it."""
    d = np.asarray(d, float)
    if tau is not None:
        q = np.exp(-1 / tau)
        return (1 - q) * q ** (d - floor)
    q1, q2 = np.exp(-1 / t1), np.exp(-1 / t2)
    num = af * (1 - q1) * q1 ** (d - 1) + (1 - af) * (1 - q2) * q2 ** (d - 1)
    den = af * q1 ** (floor - 1) + (1 - af) * q2 ** (floor - 1)
    return num / den


def frame_bins(lo, hi, n=36):
    """Integer-frame bin edges, one frame wide at short times and widening
    geometrically, every width a whole number of frames."""
    e = np.unique(np.round(np.geomspace(lo, hi + 1, n)).astype(int))
    e = np.unique(np.r_[np.arange(lo, min(lo + 8, e[-1])), e])
    return e - 0.5


def dwell_sets():
    D = pd.read_csv(os.path.join(TAB, "dwell_times_windowed.csv.gz"), low_memory=False)
    D = D[D.power_mW == REF_PW]
    lim = pd.read_csv(os.path.join(TAB, "filter_shared_upper_limit.csv"))
    lim_on = lim[lim.dwell == "on"].groupby("fluorogen").shared_limit_frames.first().to_dict()
    on = D[(D.type == "on") & D.passes_repeat.astype(bool) & (D.frames >= DMIN_ON)]
    off = D[(D.type == "off") & (~D.boundary_gap.astype(bool)) & (D.frames >= DMIN_OFF)]
    return on, off, lim_on


def fit_both(g, floor, cap=None):
    cm, cx = g[~g.censored.astype(bool)], g[g.censored.astype(bool)]
    d, k = cm.frames.values.astype(float), cm.key.values
    dc, kc = cx.frames.values.astype(float), cx.key.values
    if cap is not None:
        d, k, dc, kc = A.apply_upper(d, k, dc, kc, cap)
    r = A.fit_dwell_set(d, k, dc, kc, floor, n_boot=0, double=True)
    n = r["n_dwells"]
    return dict(n_dwells=int(n), n_complete=int(r["n_complete"]),
                n_censored=int(r["n_censored"]), n_molecules=int(r["n_molecules"]),
                tau_single_fr=r["tau_single"], a_fast=r["a_fast"],
                tau_fast_fr=r["tau_fast"], tau_slow_fr=r["tau_slow"],
                tau_amp_fr=r["tau_amp_weighted"],
                bic_single=r["bic_single"], bic_double=r["bic_double"],
                dBIC=r["dBIC"], dBIC_per_dwell=r["dBIC"] / n,
                LR_stat=r["LR_stat"], LR_p=r["LR_p"]), d


def build():
    on, off, lim_on = dwell_sets()
    rows, plotdata = [], {}
    for kind, src, floor in (("on", on, DMIN_ON), ("off", off, DMIN_OFF)):
        for (fl, c), g in src.groupby(["fluorogen", "conc_nM"]):
            cap = lim_on.get(fl) if kind == "on" else None
            res, d_complete = fit_both(g, floor, cap)
            res.update(fluorogen=fl, conc_nM=c, power_mW=REF_PW, dwell=kind,
                       floor_frames=floor, upper_limit_frames=cap)
            # goodness of each overlay against the binned data: the median and
            # worst absolute log ratio of model to observed density over bins
            # holding at least 20 observations
            v = g.frames.values.astype(float)
            hi = np.quantile(v, XMAX_Q)
            e = frame_bins(floor, hi)
            cnt, _ = np.histogram(v[~g.censored.astype(bool).values], bins=e)
            w = np.diff(e)
            emp = cnt / (len(v) * w)
            mids = [np.arange(int(np.ceil(a)), int(np.floor(b)) + 1) for a, b in zip(e[:-1], e[1:])]
            ms = np.array([pmf(m, floor, tau=res["tau_single_fr"]).mean() for m in mids])
            md = np.array([pmf(m, floor, af=res["a_fast"], t1=res["tau_fast_fr"],
                               t2=res["tau_slow_fr"]).mean() for m in mids])
            ok = cnt >= 20
            lr_s = np.abs(np.log(ms[ok] / emp[ok])); lr_d = np.abs(np.log(md[ok] / emp[ok]))
            res.update(median_abs_logratio_single=float(np.median(lr_s)),
                       median_abs_logratio_double=float(np.median(lr_d)),
                       max_abs_logratio_single=float(lr_s.max()),
                       max_abs_logratio_double=float(lr_d.max()),
                       n_bins_scored=int(ok.sum()))
            rows.append(res)
            plotdata[(kind, fl, c)] = (v, g.censored.astype(bool).values, e, floor, res)
    R = pd.DataFrame(rows)
    for k_ in ("tau_single", "tau_fast", "tau_slow", "tau_amp"):
        R[k_ + "_s"] = R[k_ + "_fr"] * FT
    R["preferred_by_BIC"] = np.where(R.dBIC > 0, "double", "single")
    R = R.sort_values(["dwell", "fluorogen", "conc_nM"]).reset_index(drop=True)
    R.to_csv(os.path.join(TAB, "fit_compare_single_double.csv"), index=False)
    A.log(f"fit_compare: {len(R)} distributions, both models each")
    return R, plotdata


def main():
    R, _ = build()
    return R


if __name__ == "__main__":
    main()
