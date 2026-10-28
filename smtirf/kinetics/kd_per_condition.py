#!/usr/bin/env python
"""The dissociation constant from each single measurement, by every route open to one.

A K_D from ONE concentration is an extrapolation, not a measurement, because a
single point cannot distinguish a binding curve from any other curve through
that point. This file computes it anyway, by every route the data permit, for
one purpose: a constant of the interaction must come out the same from every
condition, so a route whose value drifts with concentration is thereby shown to
be invalid. That is a stronger statement than any single number.

Routes, and what each one assumes.

  R1  dwell route            K_D = k_off * tau_off * c
      Needs nothing but the two dwell times. Assumes both are the rates of a
      single two-state process, which the window validation already showed is
      false for tau_off -- it is not recovered under any analysis window.

  R2  occupancy, naive       K_D = c * (1 - theta) / theta
      Assumes the occupancy ceiling is unity. It is not: occupancy saturates far
      below full, so this route overestimates K_D by a factor that grows as
      theta approaches its true ceiling. Reported because it is what a reader
      computes by hand, and because the size of its error is worth showing.

  R3  occupancy, with ceiling  K_D = c * (theta_max - theta) / theta
      Uses the ceiling fitted across the whole series, so it is NOT purely a
      single-condition estimate -- it borrows one parameter from the series. It
      is the only occupancy route that can be constant across conditions, and
      it is undefined where the measured occupancy exceeds the fitted ceiling,
      which is itself a signal that the ceiling fit is unusable.

  R4  averaged occupancy     as R2, but with occupancy averaged over the
      acquisition rather than extrapolated to its first frame. Occupancy relaxes
      5- to 45-fold during every acquisition, so this averages over a
      relaxation and is reported only to show how far that displaces the answer.

Every value is quoted with an interval, and every route with a validity flag.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                      # noqa: E402

TAB = A.TAB
BULK = {"SALAD1": 0.69, "TO1-B": 0.85}


def _decay(t, amp, k, plateau):
    return amp * np.exp(-k * t) + plateau


def initial_occupancy(all_powers=True):
    """Occupancy extrapolated to the first frame, with a standard error.

    The standard error comes from the covariance of the decay fit, propagated to
    amplitude + plateau. Without it these K_D values would be numbers without
    uncertainties, which are not results.
    """
    z = np.load(os.path.join(TAB, "prebleach_occupancy_curves.npz"))
    rows = []
    for key in z.files:
        fl, c, pw = key.split("|")
        occ = z[key]
        t = np.arange(len(occ)) * A.FRAME_TIME_S
        try:
            p, cov = curve_fit(_decay, t, occ,
                               p0=[occ[:10].mean(), 0.02, occ[-500:].mean()],
                               maxfev=20000)
            th0 = p[0] + p[2]
            # var(amp + plateau) = var(amp) + var(plateau) + 2 cov(amp, plateau)
            se = float(np.sqrt(max(cov[0, 0] + cov[2, 2] + 2 * cov[0, 2], 0)))
            r2 = 1 - ((occ - _decay(t, *p)) ** 2).sum() / ((occ - occ.mean()) ** 2).sum()
        except Exception:
            th0, se, r2 = np.nan, np.nan, np.nan
        rows.append(dict(fluorogen=fl, conc_nM=float(c), power_mW=float(pw),
                         theta0=th0, theta0_se=se, decay_R2=r2))
    d = pd.DataFrame(rows)
    return d if all_powers else d[d.power_mW == A.DEFAULT_POWER_MW]


def two_point_langmuir(io, n_mc=4000, seed=0):
    """K_D from every PAIR of concentrations, with the occupancy ceiling cancelled.

    This is the route that gives TO1-B a number. For a single-site isotherm
    theta(c) = theta_max * c / (K + c), the ratio of occupancies at two
    concentrations does not contain theta_max at all:

        r = theta(c1)/theta(c2) = c1 (K + c2) / (c2 (K + c1))
        =>  K = c1 c2 (1 - r) / (r c2 - c1)

    So no ceiling has to be fitted, no series has to be pooled, and the estimate
    survives the failure that disqualified route R3 for TO1-B. What it cannot
    survive is occupancy that does not rise with concentration: r >= c1/c2 * ...
    yields a negative or infinite K, which is reported as unphysical rather than
    clipped, because those pairs are the evidence that the isotherm is not being
    sampled in its responsive range.

    Uncertainty is Monte Carlo over the two occupancy standard errors.
    """
    rg = np.random.default_rng(seed)
    rows = []
    for (fl, pw), g in io.groupby(["fluorogen", "power_mW"]):
        g = g.sort_values("conc_nM").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a_, b_ = g.loc[i], g.loc[j]
                c1, c2 = a_.conc_nM, b_.conc_nM
                d1 = rg.normal(a_.theta0, max(a_.theta0_se, 1e-12), n_mc)
                d2 = rg.normal(b_.theta0, max(b_.theta0_se, 1e-12), n_mc)
                r = d1 / d2
                k = c1 * c2 * (1 - r) / (r * c2 - c1)
                phys = np.isfinite(k) & (k > 0)
                r0 = a_.theta0 / b_.theta0
                k0 = c1 * c2 * (1 - r0) / (r0 * c2 - c1)
                rows.append(dict(
                    fluorogen=fl, power_mW=pw, c_low_nM=c1, c_high_nM=c2,
                    theta_low=a_.theta0, theta_high=b_.theta0,
                    occupancy_ratio=r0,
                    K_D_nM=k0 if np.isfinite(k0) and k0 > 0 else np.nan,
                    K_D_lo=float(np.percentile(k[phys], 2.5)) if phys.sum() > 50 else np.nan,
                    K_D_hi=float(np.percentile(k[phys], 97.5)) if phys.sum() > 50 else np.nan,
                    frac_draws_physical=float(phys.mean()),
                    physical=bool(np.isfinite(k0) and k0 > 0),
                    note=("" if np.isfinite(k0) and k0 > 0 else
                          "unphysical: occupancy does not rise between these two "
                          "concentrations")))
    return pd.DataFrame(rows)


def extrapolate_dwell_to_zero(K):
    """The dwell-route K_D extrapolated to zero dye concentration.

    The dwell route drifts monotonically with concentration in both dyes, which
    disqualifies any single value. It does not disqualify the intercept: the
    same argument the chapter already applies to laser power -- where the
    zero-power extrapolation is the quantity reported because the artefact
    scales with the variable -- applies to a concentration-scaling artefact.

    Two intervals are reported, as everywhere else in this project: one
    propagating the per-condition measurement error, one letting the scatter of
    the points about the line set the scale (t on n-2 degrees of freedom). The
    second is the one to quote.
    """
    from scipy import stats as st
    rows = []
    for (fl, pw), g in K.groupby(["fluorogen", "power_mW"]):
        g = g.sort_values("conc_nM")
        if len(g) < 3:
            continue
        x = g.conc_nM.values.astype(float)
        y = g.R1_dwell_nM.values.astype(float)
        sd = np.maximum((g.R1_hi.values - g.R1_lo.values) / 3.92, 1e-9)
        w = 1 / sd ** 2
        X = np.vstack([x, np.ones_like(x)]).T
        XtW = X.T * w
        coef = np.linalg.solve(XtW @ X, XtW @ y)
        pred = X @ coef
        dof = len(x) - 2
        ss_w = float((w * (y - pred) ** 2).sum())
        s2 = ss_w / dof
        cov_meas = np.linalg.inv(XtW @ X)
        cov_lof = cov_meas * s2
        t = st.t.ppf(0.975, dof)
        se_m, se_l = np.sqrt(cov_meas[1, 1]), np.sqrt(cov_lof[1, 1])
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(dict(
            fluorogen=fl, power_mW=pw, n_conc=len(x),
            slope_nM_per_nM=coef[0], intercept_nM=coef[1],
            intercept_meas_lo=coef[1] - 1.96 * se_m,
            intercept_meas_hi=coef[1] + 1.96 * se_m,
            intercept_lof_lo=coef[1] - t * se_l,
            intercept_lof_hi=coef[1] + t * se_l,
            intercept_resolved=bool(coef[1] - t * se_l > 0),
            chi2_per_dof=s2, R2=r2, K_D_bulk_nM=BULK.get(fl, np.nan)))
    return pd.DataFrame(rows)


def anchored_estimate(K, tp, n_mc=20000, seed=1):
    """A K_D for the dye whose own isotherm cannot be fitted, by anchoring the
    dwell-route RATIO to the other dye's directly measured value.

    The logic is the rule this project already uses to choose analysis
    conventions: a multiplicative bias that is common to both dyes cancels in a
    ratio, even when it destroys each absolute value. The dwell route is the
    clearest case. Its absolute K_D is invalid -- it drifts 14- to 23-fold with
    concentration -- yet the SALAD1-to-TO1-B ratio taken at MATCHED
    concentration is stable, and lands close to the ratio of the two published
    bulk values. So the ratio is usable where neither absolute is.

        K_D(TO1-B) = [K_D_dwell(TO1-B) / K_D_dwell(SALAD1)]  x  K_D(SALAD1)

    with K_D(SALAD1) taken from the two-point occupancy route, which needs no
    fitted ceiling and is physical at every pair for that dye.

    This is an ANCHORED estimate, not an independent measurement: it inherits
    every error in the SALAD1 anchor, and it assumes the dwell-route bias is
    common to both dyes. Both assumptions are stated in the output so the number
    cannot be quoted as though TO1-B had been measured directly.
    """
    rg = np.random.default_rng(seed)
    k = K[K.power_mW == A.DEFAULT_POWER_MW]
    piv = k.pivot(index="conc_nM", columns="fluorogen", values="R1_dwell_nM").dropna()
    if piv.shape[1] < 2 or piv.empty:
        return pd.DataFrame()
    ratio = (piv["TO1-B"] / piv["SALAD1"]).values
    anchor = tp[(tp.power_mW == A.DEFAULT_POWER_MW) & (tp.fluorogen == "SALAD1")
                & tp.physical].K_D_nM.dropna().values
    if not len(anchor):
        return pd.DataFrame()
    draws = rg.choice(ratio, n_mc) * rg.choice(anchor, n_mc)
    return pd.DataFrame([dict(
        target="TO1-B", anchor="SALAD1",
        ratio_median=float(np.median(ratio)), ratio_min=float(ratio.min()),
        ratio_max=float(ratio.max()), n_conc_in_ratio=len(ratio),
        anchor_K_D_nM=float(np.median(anchor)), n_anchor_pairs=len(anchor),
        K_D_nM=float(np.median(draws)),
        K_D_lo=float(np.percentile(draws, 2.5)),
        K_D_hi=float(np.percentile(draws, 97.5)),
        K_D_bulk_nM=BULK["TO1-B"],
        bulk_ratio=BULK["TO1-B"] / BULK["SALAD1"],
        assumptions=("anchored, not independent: inherits the SALAD1 anchor's error "
                     "and assumes the dwell-route concentration bias is common to "
                     "both dyes"))])


def build():
    kw = pd.read_csv(os.path.join(TAB, "kinetics_windowed.csv"))
    io = initial_occupancy()
    # Stored input: the isotherm fitted interactively on 14 September to the initial
    # fraction bound; it supplies the ceiling of route R3 only (see README).
    lang = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stored",
                                    "prebleach_initial_occupancy_langmuir.csv"))
    lang = lang[lang.source.str.startswith("initial")].set_index("fluorogen")
    bf = pd.read_csv(os.path.join(TAB, "bound_fraction_vs_concentration.csv"))
    d = kw.merge(io, on=["fluorogen", "conc_nM", "power_mW"], how="left")
    d = d.merge(bf[["fluorogen", "conc_nM", "theta_all", "theta_binders"]],
                on=["fluorogen", "conc_nM"], how="left")

    rows = []
    for r in d.itertuples():
        c = r.conc_nM
        tmax = float(lang.theta_max.get(r.fluorogen, np.nan))
        tmax_se = float(lang.theta_max_se.get(r.fluorogen, np.nan))
        tmax_r2 = float(lang.R2.get(r.fluorogen, np.nan))

        # R1: dwell route, interval from the two bootstrap intervals
        kd1 = r.k_off_per_s * r.off_tau_single_s * c
        kd1_lo = (1 / r.on_tau_hi_s) * r.off_tau_lo_s * c
        kd1_hi = (1 / r.on_tau_lo_s) * r.off_tau_hi_s * c

        # R2: naive occupancy, ceiling assumed unity
        th, se = r.theta0, r.theta0_se
        kd2 = c * (1 - th) / th if np.isfinite(th) and th > 0 else np.nan
        # dK/dtheta = -c / theta^2
        kd2_se = abs(c / th ** 2) * se if np.isfinite(th) and th > 0 else np.nan

        # R3: occupancy with the series-fitted ceiling
        if np.isfinite(th) and th > 0 and np.isfinite(tmax):
            kd3 = c * (tmax - th) / th
            # propagate both theta and theta_max
            kd3_se = c * np.sqrt((tmax_se / th) ** 2 + (tmax * se / th ** 2) ** 2)
            defined = th < tmax
        else:
            kd3, kd3_se, defined = np.nan, np.nan, False

        # R4: acquisition-averaged occupancy, naive ceiling
        tha = r.theta_all if np.isfinite(getattr(r, "theta_all", np.nan)) else np.nan
        kd4 = c * (1 - tha) / tha if np.isfinite(tha) and tha > 0 else np.nan

        rows.append(dict(
            fluorogen=r.fluorogen, conc_nM=c, power_mW=r.power_mW,
            n_molecules=r.n_molecules, on_dwells=r.on_n_complete,
            tau_on_s=r.on_tau_single_s, tau_off_s=r.off_tau_single_s,
            k_off_per_s=r.k_off_per_s,
            theta0=th, theta0_se=se, theta_avg=tha,
            R1_dwell_nM=kd1, R1_lo=kd1_lo, R1_hi=kd1_hi,
            R2_naive_nM=kd2, R2_se=kd2_se,
            R3_ceiling_nM=kd3 if defined else np.nan, R3_se=kd3_se if defined else np.nan,
            R3_defined=defined,
            R3_note=("" if defined else
                     "undefined: measured occupancy exceeds the fitted ceiling"),
            R4_averaged_nM=kd4,
            theta_max_used=tmax, theta_max_fit_R2=tmax_r2,
            K_D_bulk_nM=BULK.get(r.fluorogen, np.nan)))
    K = pd.DataFrame(rows).sort_values(["fluorogen", "power_mW", "conc_nM"])

    # A constant of the interaction must not drift with concentration. Scoring
    # each route on that, per fluorogen, at the reference power.
    ver = []
    ref = K[K.power_mW == A.DEFAULT_POWER_MW]
    for fl, g in ref.groupby("fluorogen"):
        for route, col in [("R1 dwell", "R1_dwell_nM"), ("R2 naive occupancy", "R2_naive_nM"),
                           ("R3 occupancy with ceiling", "R3_ceiling_nM"),
                           ("R4 averaged occupancy", "R4_averaged_nM")]:
            v = g.sort_values("conc_nM")[col].dropna()
            if len(v) < 2:
                ver.append(dict(fluorogen=fl, route=route, n_defined=len(v),
                                spread_fold=np.nan, geo_mean_nM=np.nan,
                                monotone_with_conc=None, ceiling_fit_R2=np.nan,
                                verdict="too few conditions defined"))
                continue
            spread = v.max() / v.min() if v.min() > 0 else np.inf
            mono = bool(np.all(np.diff(v.values) > 0) or np.all(np.diff(v.values) < 0))
            # A route that borrows the occupancy ceiling is only as good as the
            # isotherm that supplied it. TO1-B's ceiling comes from a fit with
            # R2 below zero -- worse than a horizontal line -- so its R3 values
            # are arithmetic on an unusable parameter, whatever their spread.
            ceil_r2 = float(g.theta_max_fit_R2.iloc[0])
            borrows = route.startswith("R3")
            if borrows and not (ceil_r2 > 0.5):
                verdict = (f"not usable: the ceiling it borrows comes from an "
                           f"isotherm with R2 = {ceil_r2:.3f}")
            elif mono and spread > 3:
                verdict = "invalid: drifts monotonically with concentration"
            elif spread > 10:
                verdict = "invalid: spread exceeds an order of magnitude"
            elif spread <= 5:
                verdict = "approximately constant"
            else:
                verdict = "borderline: spread 5-10 fold"
            ver.append(dict(
                fluorogen=fl, route=route, n_defined=len(v),
                spread_fold=spread, geo_mean_nM=float(np.exp(np.log(v).mean())),
                monotone_with_conc=mono, ceiling_fit_R2=ceil_r2 if borrows else np.nan,
                verdict=verdict))
    V = pd.DataFrame(ver)
    # --- the two routes that give BOTH dyes a number -----------------------
    tp = two_point_langmuir(io)
    tp.to_csv(os.path.join(TAB, "kd_two_point.csv"), index=False)
    ex = extrapolate_dwell_to_zero(K)
    ex.to_csv(os.path.join(TAB, "kd_dwell_extrapolated.csv"), index=False)

    # pooled two-point summary, reference power only
    sm = []
    for fl, g in tp[tp.power_mW == A.DEFAULT_POWER_MW].groupby("fluorogen"):
        ok = g[g.physical]
        v = ok.K_D_nM.dropna()
        sm.append(dict(
            fluorogen=fl, n_pairs=len(g), n_physical=len(ok),
            frac_physical=len(ok) / len(g) if len(g) else np.nan,
            K_D_median_nM=float(v.median()) if len(v) else np.nan,
            K_D_q25=float(v.quantile(.25)) if len(v) else np.nan,
            K_D_q75=float(v.quantile(.75)) if len(v) else np.nan,
            K_D_min=float(v.min()) if len(v) else np.nan,
            K_D_max=float(v.max()) if len(v) else np.nan,
            K_D_bulk_nM=BULK.get(fl, np.nan)))
    S2 = pd.DataFrame(sm)
    S2.to_csv(os.path.join(TAB, "kd_two_point_summary.csv"), index=False)

    an = anchored_estimate(K, tp)
    if len(an):
        an.to_csv(os.path.join(TAB, "kd_anchored_TO1B.csv"), index=False)

    K.to_csv(os.path.join(TAB, "kd_per_condition.csv"), index=False)
    V.to_csv(os.path.join(TAB, "kd_route_verdicts.csv"), index=False)
    A.log(f"wrote kd_per_condition.csv ({len(K)} conditions) and kd_route_verdicts.csv")
    return K, V, tp, ex, S2, an


if __name__ == "__main__":
    build()
