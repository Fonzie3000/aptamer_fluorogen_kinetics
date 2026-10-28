#!/usr/bin/env python3
"""Measure where the dwell-time fits should start and stop, then apply it.

Stage 1  detection efficiency against true episode duration, per fluorogen, by
         simulation at each condition's own fitted rates, brightness and noise.
         Sets the lower truncation (MIN_DUR).
Stage 2  a closed-loop check: simulate with KNOWN rates, fit at the candidate
         truncation, and confirm the estimator recovers the input before the
         value is applied to real data.
Stage 3  refit every real condition under the current and the measured lower
         limit, and with and without an upper limit, and tabulate the difference.

Must be run as a script (not piped to stdin): the simulation uses
multiprocessing, and on macOS the spawned children re-import __main__.

  python run_truncation_study.py [--n-mol 600] [--stage 1,2,3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_smtirf as A          # noqa: E402
import simulate_validate as S      # noqa: E402

TAB = os.path.join(HERE, "tables")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage1(n_mol):
    log("stage 1: detection efficiency vs true episode duration")
    df, pool = S.exp6_detection_efficiency(n_mol, TAB, TAB)
    return df, pool


def stage2(n_mol, candidates):
    """Closed loop: does the estimator recover a known input at this truncation?

    Simulated with a single-exponential bound dwell at the measured brightness
    and noise, so any departure of the recovered lifetime from the input is
    attributable to the truncation choice and the detection chain, not to the
    sample.
    """
    log(f"stage 2: estimator recovery at candidate truncations {candidates}")
    rows = []
    for fl, bright, noise, bg in (("SALAD1", 321.0, 50.3, 5.4),
                                  ("TO1-B", 272.0, 49.7, 11.7)):
        for k_off in (0.35, 0.20, 0.80):
            dur, rec = None, None
            for dmin in candidates:
                old = A.MIN_DUR
                A.MIN_DUR = dmin
                try:
                    summ, _ = S.run_condition(0.021, k_off, n_mol=n_mol,
                                              seed=S.SEED + 300 + dmin,
                                              bright=bright, noise=noise, bg=bg)
                finally:
                    A.MIN_DUR = old
                rows.append(dict(fluorogen=fl, brightness=bright, noise_sd=noise,
                                 min_dur=dmin, k_off_true=k_off,
                                 tau_on_true_s=1 / k_off,
                                 tau_on_fit_s=summ.get("tau_on_fit_s"),
                                 tau_on_bias=summ.get("tau_on_bias"),
                                 tau_off_true_s=summ.get("tau_off_true_s"),
                                 tau_off_fit_s=summ.get("tau_off_fit_s"),
                                 tau_off_bias=summ.get("tau_off_bias"),
                                 detection_ratio=summ.get("detection_ratio"),
                                 n_on_dwells=summ.get("n_on_dwells")))
                log(f"  {fl:7s} k_off={k_off:.2f} dmin={dmin}: "
                    f"tau_on bias {rows[-1]['tau_on_bias']:.3f}, "
                    f"tau_off bias {rows[-1]['tau_off_bias']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "sim_e7_truncation_recovery.csv"), index=False)
    return df


def stage3(min_dur_measured, max_dur_variant, min_dur_previous=2):
    """Refit the real data under each combination of limits."""
    log("stage 3: refitting real conditions under each pair of limits")
    import pickle
    import glob
    ev = pd.concat([pickle.load(open(f, "rb"))["ev"]
                    for f in sorted(glob.glob(os.path.join(HERE, "cache", "*.pkl")))],
                   ignore_index=True)
    log(f"  {len(ev):,} events from cache")
    ON, OFF, BND = A.extract_dwells(ev, A.MAX_GAP, edge_correction=A.EDGE_CORRECTION)

    rows = []
    # Labelled by the VALUES, not by "current"/"measured": once the measured
    # limit is adopted as MIN_DUR those labels collapse onto each other.
    lims = sorted({min_dur_previous, min_dur_measured, A.MIN_DUR})
    variants = ([(f"min_dur={d}", d, None) for d in lims]
                + [(f"min_dur={d} + max_dur={max_dur_variant}", d, max_dur_variant)
                   for d in lims])
    for label, dmin, dmax in variants:
        dmin_off = max(dmin, A.MAX_GAP + 1)
        for (fl, c, pw), g in ON.groupby(["fluorogen", "conc_nM", "power_mW"]):
            oi = OFF[(OFF.fluorogen == fl) & (OFF.conc_nM == c) & (OFF.power_mW == pw)]
            ob = BND[(BND.fluorogen == fl) & (BND.conc_nM == c) & (BND.power_mW == pw)]
            cm = g[(~g.censored) & (g.frames >= dmin)]
            cx = g[g.censored & (g.frames >= dmin)]
            d, k, dc, kc = A.apply_upper(cm.frames.values.astype(float), cm.key.values,
                                         cx.frames.values.astype(float), cx.key.values,
                                         dmax)
            r_on = A.fit_dwell_set(d, k, dc, kc, dmin, n_boot=0)
            do, ko, dco, kco, amin = A.off_arrays(oi, ob, dmin_off)
            do, ko, dco, kco = A.apply_upper(do, ko, dco, kco, dmax)
            r_off = A.fit_dwell_set(do, ko, dco, kco, dmin_off, dmin_c=amin, n_boot=0)
            on_med = float(cm.frames.median()) if len(cm) else np.nan
            off_med = float(oi.loc[oi.frames >= dmin_off, "frames"].median())
            rows.append(dict(
                variant=label, min_dur=dmin, max_dur=dmax if dmax else 0,
                fluorogen=fl, conc_nM=c, power_mW=pw,
                n_on_complete=len(cm), n_on_censored=len(cx),
                n_on_recensored=int((cm.frames >= dmax).sum()) if dmax else 0,
                tau_on_s=r_on["tau_single"] * A.FRAME_TIME_S,
                tau_off_s=r_off["tau_single"] * A.FRAME_TIME_S,
                on_median_s=on_med * A.FRAME_TIME_S,
                off_median_s=off_med * A.FRAME_TIME_S,
                on_dBIC=r_on["dBIC"], off_dBIC=r_off["dBIC"],
                k_off_per_s=1 / (r_on["tau_single"] * A.FRAME_TIME_S),
                inv_tau_off_per_s=1 / (r_off["tau_single"] * A.FRAME_TIME_S)))
        log(f"  {label}: done")
    df = pd.DataFrame(rows)
    df["on_tau_over_median"] = df.tau_on_s / df.on_median_s
    df["off_tau_over_median"] = df.tau_off_s / df.off_median_s
    df.to_csv(os.path.join(TAB, "truncation_comparison.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=600)
    ap.add_argument("--stage", default="1,2,3")
    ap.add_argument("--min-dur-measured", type=int, default=None)
    ap.add_argument("--min-dur-previous", type=int, default=2,
                    help="the limit to compare the measured one against (default 2, "
                         "the value assumed before it was measured)")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stage.split(",")}
    t0 = time.time()

    mdm = args.min_dur_measured
    if "1" in stages:
        _, pool = stage1(args.n_mol)
        v = pool.min_dur_adopted.dropna()
        mdm = int(v.iloc[0]) if len(v) else None
        log(f"  measured lower truncation = {mdm} frames")
    if mdm is None:
        mdm = A.MIN_DUR
        log(f"  using MIN_DUR = {mdm} (stage 1 not run or inconclusive)")
    if "2" in stages:
        cands = sorted({A.MIN_DUR, mdm, mdm + 1, max(1, mdm - 1)})
        stage2(args.n_mol, cands)
    if "3" in stages:
        stage3(mdm, A.MAX_DUR_VARIANT, min_dur_previous=args.min_dur_previous)
    log(f"done in {(time.time()-t0)/60:.1f} min")
