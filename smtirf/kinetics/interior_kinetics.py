#!/usr/bin/env python
"""Kinetics under the interior-only unbound treatment, using the pipeline's fitter.

Why this module exists
----------------------
The terminal (boundary) dark interval after a molecule's last event had been
entered into the unbound likelihood as a right-censored observation. That is
correct only while the molecule is still cycling; when it has stopped emitting,
the interval is the photostability endpoint and not a waiting time. Ground-truth
simulation puts the error at a factor of two (estimator_recovery_with_termination
.csv). The unbound fits here therefore use INTERIOR gaps only.

A first version of these tables was produced with a locally written
Nelder-Mead fit rather than analyse_smtirf.fit_dwell_set, and disagreed with the
pipeline by 0.6-2.5%. Everything below calls fit_dwell_set, so one verified
estimator produces every number.

fit_dwell_set bootstraps only the single-exponential lifetime, so the interval on
the amplitude-weighted mean is obtained here by resampling MOLECULES and
re-calling that same function -- never by substituting a different fit.

Writes: kinetics_interior_only.csv, unbound_interior_only_fit.csv,
        kon_interior_only.csv, unbound_component_bootstrap.csv,
        unbound_component_concentration_response.csv,
        unbound_slope_by_component.csv

Run after run_all.sh:   python interior_kinetics.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                        # noqa: E402
import dwell_filters as DF                                        # noqa: E402

TAB, FT = A.TAB, A.FRAME_TIME_S
REF_PW = A.DEFAULT_POWER_MW
N_BOOT = 120
SEED = 0
DMIN_OFF = max(A.MIN_DUR, A.MAX_GAP + 1)      # bridging makes shorter gaps impossible
K_BULK = {"SALAD1": 0.69, "TO1-B": 0.85}


def _fit(d, k, dc, kc, dmin):
    return A.fit_dwell_set(d, k, dc, kc, dmin, n_boot=0, double=True)


def boot_fit(d, k, dc, kc, dmin, n_boot=N_BOOT, seed=SEED):
    """Point estimate plus a cluster bootstrap over molecules, both from
    fit_dwell_set. Returns the amplitude-weighted mean, the slow component and
    their intervals; a resample whose fit fails is skipped and counted."""
    r = _fit(d, k, dc, kc, dmin)
    keys = np.concatenate([k, kc]) if kc.size else k
    mols = np.unique(keys)
    idx_c = {m: np.flatnonzero(k == m) for m in mols}
    idx_x = {m: np.flatnonzero(kc == m) for m in mols} if kc.size else {}
    rg = np.random.default_rng(seed)
    bm, bs, nfail = [], [], 0
    for _ in range(n_boot):
        pick = rg.choice(mols, mols.size, replace=True)
        ic = np.concatenate([idx_c[m] for m in pick]) if mols.size else np.array([], int)
        ix = (np.concatenate([idx_x[m] for m in pick]) if idx_x else np.array([], int))
        if ic.size < 50:
            nfail += 1
            continue
        try:
            q = _fit(d[ic], k[ic], dc[ix] if ix.size else np.array([]),
                     kc[ix] if ix.size else np.array([]), dmin)
            bm.append(q["tau_amp_weighted"]); bs.append(q["tau_slow"])
        except Exception:
            nfail += 1
    pc = lambda v, p: float(np.percentile(v, p)) * FT if len(v) else np.nan
    return dict(mean_s=r["tau_amp_weighted"] * FT,
                mean_lo=pc(bm, 2.5), mean_hi=pc(bm, 97.5),
                slow_s=r["tau_slow"] * FT, slow_lo=pc(bs, 2.5), slow_hi=pc(bs, 97.5),
                fast_s=r["tau_fast"] * FT, a_fast=r["a_fast"],
                single_s=r["tau_single"] * FT, dBIC=r["dBIC"],
                n_complete=int(r["n_complete"]), n_censored=int(r["n_censored"]),
                n_molecules=int(r["n_molecules"]), n_boot_failed=nfail)


def line_fit(x, y, ylo, yhi, seed=1):
    """Weighted straight line with BOTH interval types: measurement-error only,
    and lack-of-fit (inflated by chi2/dof, t on n-2). With few points the second
    dominates and is the one to quote."""
    sd = np.maximum((yhi - ylo) / (2 * 1.96), 1e-12)
    w = 1 / sd ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    Aw = X * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
    res = y - X @ coef
    dof = max(x.size - 2, 1)
    chi2 = float((w * res ** 2).sum() / dof)
    cov = np.linalg.inv(X.T @ (X * w[:, None])) * max(chi2, 1.0)
    t = st.t.ppf(0.975, dof)
    rg = np.random.default_rng(seed)
    dr = np.array([np.linalg.lstsq(Aw, (y + rg.normal(0, sd)) * np.sqrt(w),
                                   rcond=None)[0] for _ in range(2000)])
    return dict(slope=coef[0], intercept=coef[1], chi2_per_dof=chi2, dof=dof,
                R2=1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                slope_mc_lo=np.percentile(dr[:, 0], 2.5),
                slope_mc_hi=np.percentile(dr[:, 0], 97.5),
                slope_lo=coef[0] - t * np.sqrt(cov[0, 0]),
                slope_hi=coef[0] + t * np.sqrt(cov[0, 0]),
                icpt_lo=coef[1] - t * np.sqrt(cov[1, 1]),
                icpt_hi=coef[1] + t * np.sqrt(cov[1, 1]))


def logfit(x, y):
    """Power-law exponent with a t interval. A diffusion-limited rebinding wait
    requires -1; a process independent of free dye requires 0."""
    lx, ly = np.log(x), np.log(y)
    p, b = np.polyfit(lx, ly, 1)
    res = ly - (p * lx + b)
    dof = max(x.size - 2, 1)
    se = np.sqrt((res ** 2).sum() / dof / ((lx - lx.mean()) ** 2).sum())
    t = st.t.ppf(0.975, dof)
    return p, p - t * se, p + t * se, 1 - (res ** 2).sum() / ((ly - ly.mean()) ** 2).sum()


def build():
    t0 = time.time()
    ON0, OFFI0, OFFB0 = DF.windowed_dwells()
    ONr, _, _, _, _ = DF.repeat_filter(ON0, OFFI0, OFFB0)
    lim_on, _ = DF.shared_upper_limit(ONr)
    A.log(f"interior kinetics: bound upper limits (frames) {lim_on}, "
          f"bound floor {A.MIN_DUR}, unbound floor {DMIN_OFF}")

    rows, comp = [], []
    for (fl, c, pw), g in ONr[ONr.power_mW == REF_PW].groupby(
            ["fluorogen", "conc_nM", "power_mW"]):
        cm, cx = g[~g.censored], g[g.censored]
        cm, cx = cm[cm.frames >= A.MIN_DUR], cx[cx.frames >= A.MIN_DUR]
        d, k, dc, kc = A.apply_upper(cm.frames.values.astype(float), cm.key.values,
                                     cx.frames.values.astype(float), cx.key.values,
                                     lim_on.get(fl))
        b = boot_fit(d, k, dc, kc, A.MIN_DUR)

        o = OFFI0[(OFFI0.fluorogen == fl) & (OFFI0.conc_nM == c) & (OFFI0.power_mW == pw)]
        o = o[o.frames >= DMIN_OFF]
        u = boot_fit(o.frames.values.astype(float), o.key.values,
                     np.array([]), np.array([]), DMIN_OFF)

        ob = OFFB0[(OFFB0.fluorogen == fl) & (OFFB0.conc_nM == c) & (OFFB0.power_mW == pw)]
        oi_all = OFFI0[(OFFI0.fluorogen == fl) & (OFFI0.conc_nM == c)
                       & (OFFI0.power_mW == pw)]
        rows.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw, n_molecules=b["n_molecules"],
            n_on=b["n_complete"], n_interior=u["n_complete"],
            tau_on_s=b["mean_s"], tau_on_lo=b["mean_lo"], tau_on_hi=b["mean_hi"],
            on_a_fast=b["a_fast"], on_tau_fast_s=b["fast_s"], on_tau_slow_s=b["slow_s"],
            k_off_per_s=1 / b["mean_s"], k_off_lo_per_s=1 / b["mean_hi"],
            k_off_hi_per_s=1 / b["mean_lo"],
            tau_off_interior_s=u["mean_s"], tau_off_lo=u["mean_lo"],
            tau_off_hi=u["mean_hi"], inv_tau_off_per_s=1 / u["mean_s"],
            off_a_fast=u["a_fast"], off_tau_fast_s=u["fast_s"],
            off_tau_slow_s=u["slow_s"], off_slow_lo=u["slow_lo"],
            off_slow_hi=u["slow_hi"],
            K_D_per_condition_nM=u["mean_s"] * c / b["mean_s"],
            boot_failed=b["n_boot_failed"] + u["n_boot_failed"]))
        comp.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw,
            n_interior=len(oi_all), n_censored=len(ob),
            frac_censored=len(ob) / max(len(oi_all) + len(ob), 1),
            med_interior_s=float(oi_all.frames.median()) * FT if len(oi_all) else np.nan,
            med_censored_s=float(ob.frames.median()) * FT if len(ob) else np.nan,
            a_fast_interior=u["a_fast"], tau_fast_interior_s=u["fast_s"],
            tau_slow_interior_s=u["slow_s"], mean_interior_s=u["mean_s"]))
        A.log(f"  {fl} {c:g} nM: tau_on {b['mean_s']:.3f} s, "
              f"tau_off {u['mean_s']:.2f} s ({time.time()-t0:.0f} s)")

    K = pd.DataFrame(rows).sort_values(["fluorogen", "conc_nM"]).reset_index(drop=True)
    for p_, l_, h_ in [("tau_on_s", "tau_on_lo", "tau_on_hi"),
                       ("tau_off_interior_s", "tau_off_lo", "tau_off_hi"),
                       ("k_off_per_s", "k_off_lo_per_s", "k_off_hi_per_s")]:
        assert ((K[p_] >= K[l_] - 1e-9) & (K[p_] <= K[h_] + 1e-9)).all(), p_
    K.to_csv(os.path.join(TAB, "kinetics_interior_only.csv"), index=False)

    # the with-terminal comparison, for the inflation factors
    C = pd.DataFrame(comp).sort_values(["fluorogen", "conc_nM"]).reset_index(drop=True)
    wt = []
    for r in C.itertuples():
        o = OFFI0[(OFFI0.fluorogen == r.fluorogen) & (OFFI0.conc_nM == r.conc_nM)
                  & (OFFI0.power_mW == r.power_mW)]
        ob = OFFB0[(OFFB0.fluorogen == r.fluorogen) & (OFFB0.conc_nM == r.conc_nM)
                   & (OFFB0.power_mW == r.power_mW)]
        o = o[o.frames >= DMIN_OFF]
        q = _fit(o.frames.values.astype(float), o.key.values,
                 ob.frames.values.astype(float), ob.key.values, DMIN_OFF)
        wt.append(dict(a_fast_withcens=q["a_fast"],
                       tau_fast_withcens_s=q["tau_fast"] * FT,
                       tau_slow_withcens_s=q["tau_slow"] * FT,
                       mean_withcens_s=q["tau_amp_weighted"] * FT))
    C = pd.concat([C, pd.DataFrame(wt)], axis=1)
    C["slow_inflation"] = C.tau_slow_withcens_s / C.tau_slow_interior_s
    C["mean_inflation"] = C.mean_withcens_s / C.mean_interior_s
    C.to_csv(os.path.join(TAB, "unbound_interior_only_fit.csv"), index=False)

    BS = K[["fluorogen", "conc_nM", "n_molecules", "tau_off_interior_s", "tau_off_lo",
            "tau_off_hi", "off_tau_slow_s", "off_slow_lo", "off_slow_hi"]].rename(
        columns={"tau_off_interior_s": "mean_s", "tau_off_lo": "mean_lo",
                 "tau_off_hi": "mean_hi", "off_tau_slow_s": "slow_s",
                 "off_slow_lo": "slow_lo", "off_slow_hi": "slow_hi"})
    BS.to_csv(os.path.join(TAB, "unbound_component_bootstrap.csv"), index=False)

    # component-wise concentration response
    cr = []
    for fl, g in C.groupby("fluorogen"):
        g = g.sort_values("conc_nM")
        for lbl, col in [("fast component τ", "tau_fast_interior_s"),
                         ("slow component τ", "tau_slow_interior_s"),
                         ("amplitude-weighted mean", "mean_interior_s"),
                         ("fast amplitude", "a_fast_interior")]:
            p, lo, hi, r2 = logfit(g.conc_nM.values, g[col].values)
            cr.append(dict(fluorogen=fl, quantity=lbl, exponent=p, lo=lo, hi=hi, R2=r2,
                           consistent_with_rebinding=(np.nan if "amplitude" == lbl[-9:]
                                                      else bool(lo <= -1 <= hi)),
                           consistent_with_flat=bool(lo <= 0 <= hi)))
    pd.DataFrame(cr).to_csv(
        os.path.join(TAB, "unbound_component_concentration_response.csv"), index=False)

    # slope on the mean vs on the slow component, both with measured uncertainty
    sl = []
    for fl, g in BS.groupby("fluorogen"):
        g = g.sort_values("conc_nM")
        for lbl, (p_, l_, h_) in [("mean (as reported)", ("mean_s", "mean_lo", "mean_hi")),
                                  ("slow component only", ("slow_s", "slow_lo", "slow_hi"))]:
            L = line_fit(g.conc_nM.values, 1 / g[p_].values,
                         1 / g[h_].values, 1 / g[l_].values)
            sl.append(dict(fluorogen=fl, source=lbl, k_on=L["slope"],
                           lo=L["slope_lo"], hi=L["slope_hi"],
                           mc_lo=L["slope_mc_lo"], mc_hi=L["slope_mc_hi"],
                           resolved=bool(L["slope_lo"] > 0), intercept=L["intercept"],
                           chi2_per_dof=L["chi2_per_dof"], R2=L["R2"],
                           uncertainty=f"molecule bootstrap, {N_BOOT} resamples",
                           interval=f"lack-of-fit (t, {L['dof']} dof), inflated by chi2/dof"))
    pd.DataFrame(sl).to_csv(
        os.path.join(TAB, "unbound_slope_by_component.csv"), index=False)

    # the association-rate line, from the mean
    old = os.path.join(TAB, "kon_intercept_fits.csv")
    prev = pd.read_csv(old).set_index("fluorogen") if os.path.exists(old) else None
    kon = []
    for fl, g in K.groupby("fluorogen"):
        g = g.sort_values("conc_nM")
        L = line_fit(g.conc_nM.values, 1 / g.tau_off_interior_s.values,
                     1 / g.tau_off_hi.values, 1 / g.tau_off_lo.values)
        res = L["slope_lo"] > 0
        koff = float(g.k_off_per_s.mean())
        row = dict(fluorogen=fl, k_on_per_nM_per_s=L["slope"], k_on_lo=L["slope_lo"],
                   k_on_hi=L["slope_hi"], k_on_mc_lo=L["slope_mc_lo"],
                   k_on_mc_hi=L["slope_mc_hi"], slope_resolved=res,
                   intercept_per_s=L["intercept"], icpt_lo=L["icpt_lo"],
                   icpt_hi=L["icpt_hi"],
                   intercept_excludes_zero=bool(L["icpt_lo"] > 0),
                   chi2_per_dof=L["chi2_per_dof"], R2=L["R2"],
                   k_off_mean_per_s=koff,
                   K_D_from_slope_nM=(koff / L["slope"] if res else np.nan),
                   K_D_bulk_nM=K_BULK.get(fl, np.nan))
        if prev is not None and fl in prev.index:
            row["k_on_previous"] = float(prev.loc[fl, "k_on_per_nM_per_s"])
            row["intercept_previous"] = float(prev.loc[fl, "intercept_per_s"])
            row["k_on_change"] = row["k_on_per_nM_per_s"] / row["k_on_previous"]
            row["intercept_change"] = row["intercept_per_s"] / row["intercept_previous"]
        kon.append(row)
    pd.DataFrame(kon).to_csv(os.path.join(TAB, "kon_interior_only.csv"), index=False)

    A.log(f"interior_kinetics done in {(time.time()-t0)/60:.1f} min; "
          f"{int(K.boot_failed.sum())} bootstrap resamples skipped")
    return K, C


if __name__ == "__main__":
    build()
