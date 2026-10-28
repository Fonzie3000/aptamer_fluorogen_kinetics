#!/usr/bin/env python
"""Methods absorbed from a reference single-molecule kinetics app, applied here.

Four things in that code are worth having and were NOT in this pipeline. They
are implemented below, faithfully, and run on the same windowed dwell set that
produces the headline numbers so the comparison is of estimators and not of
data.

1. OBSERVATION-WINDOW DIAGNOSTICS. The ratio of a fitted lifetime to the length
   of the recording, with an advisory threshold at 0.1, plus counts of events
   truncated at each boundary. A lifetime that is not small against the
   recording cannot be estimated from that recording whatever the estimator, and
   no amount of censoring machinery repairs it. This is the single most useful
   idea in the reference code: it is a cheap, model-free check that arrives at
   the same verdict this project reached only through simulation.

2. FIVE ALTERNATIVE SINGLE-EXPONENTIAL ESTIMATORS, each with R^2: the
   arithmetic mean, a count-weighted log-linear fit to the binned density, a
   nonlinear A*exp(-kt) fit to the same density, a left-truncated exponential
   PDF fitted to the histogram, a nonlinear fit to the empirical survival
   function, and a linear regression of log-survival above a tail cutoff. Used
   here NOT as competitors to the maximum-likelihood fit but as a diagnostic:
   for a true single exponential all of them agree, so the SPREAD between them
   measures how badly a one-component description fails. That is a second,
   independent read on the same question BIC answers.

3. MEAN-DWELL K_D, K_D = mean(T_off) * [L] / mean(T_on), equation 8 of
   Lee, Kim & Lee, Mol. Cells 2024, 47(10):100112. A literature-anchored route
   that uses no fitted parameter at all.

4. BETWEEN-FILE UNCERTAINTY: mean and sample standard deviation of each
   lifetime across the fields of one condition, as an alternative to the
   bootstrap over molecules. The bootstrap answers "how well is this field
   sampled", the between-field SD answers "how reproducible is this condition";
   the second is the larger and the more honest of the two when they disagree.

Deliberately NOT absorbed, with reasons:
  * dropping boundary-truncated events. This project right-censors them, which
    uses the information they carry instead of discarding it.
  * a maximum-duration filter that deletes long dwells. This project censors at
    the cutoff instead, because deleting biases the lifetime down.
  * single-exponential-only reporting. Both distributions here are decisively
    two-component, so a single tau is reported only alongside the two-component
    fit.
  * the reference histogram estimator as a headline. Its bins are not aligned
    to whole frames, which biases it low on frame-quantised data; it is
    computed here only as one of the six spread diagnostics.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                      # noqa: E402
import window_select as WS                                      # noqa: E402

TAB = A.TAB
FT = A.FRAME_TIME_S
WINDOW_RATIO_WARN = 0.10        # reference app's advisory threshold
SURV_CUTOFF = 0.05              # tail cutoff for the log-survival regression


# ----------------------------------------------------------------- estimators
def _r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _empirical_survival(t):
    t = np.sort(np.asarray(t, float))
    x = np.unique(t)
    s = np.array([(t >= v).mean() for v in x])
    return x, s


def est_mean(t):
    return dict(tau=float(np.mean(t)), r2=np.nan)


def est_density(t, nonlinear=False):
    """Count-weighted fit to the binned density; the reference app's default."""
    t = np.asarray(t, float)
    if t.size < 5 or t.min() >= t.max():
        return dict(tau=np.nan, r2=np.nan)
    nb = int(np.clip(np.sqrt(t.size), 15, 40))
    edges = np.linspace(t.min(), t.max(), nb + 1)
    cnt, _ = np.histogram(t, bins=edges)
    dens, _ = np.histogram(t, bins=edges, density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m = (cnt > 0) & (dens > 0)
    if m.sum() < 2:
        return dict(tau=np.nan, r2=np.nan)
    x, y, w = ctr[m], dens[m], np.sqrt(cnt[m].astype(float))
    c = np.polyfit(x, np.log(y), 1, w=w)
    k = -c[0]
    if not k > 0:
        return dict(tau=np.nan, r2=np.nan)
    if not nonlinear:
        return dict(tau=1 / k, r2=_r2(np.log(y), np.polyval(c, x)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(lambda tt, Aa, kk: Aa * np.exp(-kk * tt), x, y,
                             p0=[float(np.exp(c[1])), k], sigma=1 / w,
                             bounds=([0, 1e-12], [np.inf, np.inf]), maxfev=5000)
        if p[1] > 0:
            return dict(tau=1 / p[1],
                        r2=_r2(y, p[0] * np.exp(-p[1] * x)))
    except Exception:
        pass
    return dict(tau=1 / k, r2=_r2(np.log(y), np.polyval(c, x)))


def est_histogram(t):
    """Left-truncated exponential PDF fitted to the histogram.

    Bins are NOT frame-aligned here, deliberately: this reproduces the
    reference implementation so its bias on frame-quantised data is visible
    rather than hidden.
    """
    t = np.asarray(t, float)
    if t.size < 5:
        return dict(tau=np.nan, r2=np.nan)
    hi = np.percentile(t, 99)
    d = t[t <= hi]
    if d.size < 5 or d.min() >= d.max():
        return dict(tau=np.nan, r2=np.nan)
    nb = int(np.clip(np.sqrt(d.size), 15, 40))
    edges = np.linspace(d.min(), d.max(), nb + 1)
    cnt, _ = np.histogram(d, bins=edges)
    dens, _ = np.histogram(d, bins=edges, density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m = (cnt > 0) & (dens > 0)
    if m.sum() < 2:
        return dict(tau=np.nan, r2=np.nan)
    tmin = float(edges[0])
    x, y, w = ctr[m], dens[m], np.sqrt(cnt[m].astype(float))
    ms = float(np.mean(d)) - tmin
    k0 = 1 / ms if ms > 0 else 1 / float(np.mean(d))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(lambda tt, kk: kk * np.exp(-kk * (tt - tmin)), x, y,
                             p0=[k0], sigma=1 / w, bounds=([1e-10], [1000 * k0]),
                             maxfev=5000)
        k = float(p[0])
        if k > 0:
            return dict(tau=1 / k,
                        r2=_r2(y, k * np.exp(-k * (x - tmin))))
    except Exception:
        pass
    return dict(tau=np.nan, r2=np.nan)


def est_survival(t):
    x, s = _empirical_survival(t)
    if x.size < 3:
        return dict(tau=np.nan, r2=np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(lambda tt, Aa, kk: Aa * np.exp(-kk * tt), x, s,
                             p0=[1.0, 1 / max(np.mean(t), 1e-9)], maxfev=5000)
        if p[1] > 0:
            return dict(tau=1 / p[1], r2=_r2(s, p[0] * np.exp(-p[1] * x)))
    except Exception:
        pass
    return dict(tau=np.nan, r2=np.nan)


def est_slope(t, cutoff=SURV_CUTOFF):
    x, s = _empirical_survival(t)
    m = s > cutoff
    if m.sum() < 3:
        return dict(tau=np.nan, r2=np.nan)
    c = np.polyfit(x[m], np.log(s[m]), 1)
    k = -c[0]
    if not k > 0:
        return dict(tau=np.nan, r2=np.nan)
    return dict(tau=1 / k, r2=_r2(np.log(s[m]), np.polyval(c, x[m])))


ESTIMATORS = {"mean": lambda t: est_mean(t),
              "density_loglin": lambda t: est_density(t, False),
              "density_nonlin": lambda t: est_density(t, True),
              "histogram_trunc": est_histogram,
              "survival_nonlin": est_survival,
              "logsurv_slope": est_slope}


# ------------------------------------------------------------------- data
def windowed_dwells():
    """Rebuild the exact dwell set behind kinetics_windowed.csv."""
    ev = WS.load_events()
    wdef = pd.read_csv(os.path.join(TAB, "window_definitions.csv"))
    # The table holds every candidate rule; only the ADOPTED one is in force,
    # and passing all of them silently duplicates every key.
    if "rule" in wdef.columns:
        wdef = wdef[wdef.rule == WS.ADOPTED].copy()
    assert wdef.key.is_unique, "window definitions must be one row per condition"
    ev_w, lengths = WS.apply_window(ev, wdef)
    ON, OFF_I, OFF_B = A.extract_dwells(ev_w, A.MAX_GAP)
    return ON, OFF_I, OFF_B, wdef, lengths


def build():
    ON, OFF_I, OFF_B, wdef, lengths = windowed_dwells()
    kw = pd.read_csv(os.path.join(TAB, "kinetics_windowed.csv"))
    dmin_on, dmin_off = A.MIN_DUR, max(A.MIN_DUR, A.MAX_GAP + 1)
    keys = ["fluorogen", "conc_nM", "power_mW"]

    # window length per condition, in seconds, from the adopted windows
    wlen = (wdef.assign(fluorogen=wdef.key.str.split("|").str[0],
                        conc_nM=wdef.key.str.split("|").str[1].astype(float),
                        power_mW=wdef.key.str.split("|").str[2].astype(float))
            .groupby(keys).win_frames.median() * FT).rename("window_s")

    rows, diag = [], []
    for (fl, c, pw), g in ON.groupby(keys):
        kwr = kw[(kw.fluorogen == fl) & (kw.conc_nM == c) & (kw.power_mW == pw)]
        if not len(kwr):
            continue
        kwr = kwr.iloc[0]
        W = float(wlen.get((fl, c, pw), np.nan))
        off = OFF_I[(OFF_I.fluorogen == fl) & (OFF_I.conc_nM == c)
                    & (OFF_I.power_mW == pw)]
        offb = OFF_B[(OFF_B.fluorogen == fl) & (OFF_B.conc_nM == c)
                     & (OFF_B.power_mW == pw)]

        for kind, src, dmin, mle in (("on", g[~g.censored], dmin_on,
                                      kwr.on_tau_single_s),
                                     ("off", off, dmin_off,
                                      kwr.off_tau_single_s)):
            t = src.frames.values.astype(float)
            t = t[t >= dmin] * FT                       # seconds, as the app uses
            if t.size < 20:
                continue
            for name, fn in ESTIMATORS.items():
                r = fn(t)
                rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, dwell=kind,
                                 estimator=name, tau_s=r["tau"], R2=r["r2"],
                                 n_dwells=int(t.size),
                                 tau_MLE_s=float(mle),
                                 ratio_to_MLE=r["tau"] / float(mle)
                                 if np.isfinite(r["tau"]) and mle else np.nan))
            rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, dwell=kind,
                             estimator="MLE_single_censored", tau_s=float(mle),
                             R2=np.nan, n_dwells=int(t.size),
                             tau_MLE_s=float(mle), ratio_to_MLE=1.0))

        # ---- observation-window diagnostics, the reference app's idea
        rb_on = int(g.censored.sum())
        diag.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw, window_s=W,
            acquisition_s=A.N_FRAMES_EXPECTED * FT,
            tau_on_s=float(kwr.on_tau_single_s), tau_off_s=float(kwr.off_tau_single_s),
            bound_window_ratio=float(kwr.on_tau_single_s) / W if W else np.nan,
            unbound_window_ratio=float(kwr.off_tau_single_s) / W if W else np.nan,
            right_boundary_on_events=rb_on,
            boundary_off_gaps=int(len(offb)),
            frac_on_right_truncated=rb_on / max(len(g), 1),
            frac_off_gaps_at_boundary=len(offb) / max(len(off) + len(offb), 1)))

    E = pd.DataFrame(rows)
    D = pd.DataFrame(diag)
    D["warnings"] = [
        ";".join([w for w, ok in
                  (("bound_lifetime_not_small_vs_window", r.bound_window_ratio >= WINDOW_RATIO_WARN),
                   ("unbound_lifetime_not_small_vs_window", r.unbound_window_ratio >= WINDOW_RATIO_WARN))
                  if ok]) or "none" for r in D.itertuples()]

    # ---- spread across estimators: a model-adequacy read independent of BIC
    sp = (E[E.estimator != "MLE_single_censored"]
          .groupby(keys + ["dwell"])
          .agg(n_estimators=("tau_s", lambda s: int(s.notna().sum())),
               tau_min_s=("tau_s", "min"), tau_max_s=("tau_s", "max"),
               tau_median_s=("tau_s", "median"),
               R2_median=("R2", "median"))
          .reset_index())
    sp["spread_fold"] = sp.tau_max_s / sp.tau_min_s
    sp = sp.merge(kw[keys + ["on_dBIC", "off_dBIC"]], on=keys, how="left")
    sp["dBIC_same_dwell"] = np.where(sp.dwell == "on", sp.on_dBIC, sp.off_dBIC)
    sp = sp.drop(columns=["on_dBIC", "off_dBIC"])
    # Do the two model-adequacy diagnostics actually co-locate? Asserting that
    # they do was wrong: on the RAW dBIC they are uncorrelated. dBIC scales with
    # the number of dwells, so it is not comparable across conditions of very
    # different size; per-dwell it is. Both correlations are computed and
    # written out so the claim in the report is whatever the numbers say.
    from scipy.stats import spearmanr
    sp["dBIC_per_dwell"] = sp.dBIC_same_dwell / sp.n_dwells.replace(0, np.nan) \
        if "n_dwells" in sp.columns else np.nan
    if "n_dwells" not in sp.columns:
        nn = (E[E.estimator == "MLE_single_censored"]
              .groupby(keys + ["dwell"]).n_dwells.first().reset_index())
        sp = sp.merge(nn, on=keys + ["dwell"], how="left")
        sp["dBIC_per_dwell"] = sp.dBIC_same_dwell / sp.n_dwells.replace(0, np.nan)
    cr = []
    for lab, sub in (("all", sp), ("on", sp[sp.dwell == "on"]),
                     ("off", sp[sp.dwell == "off"])):
        d = sub.dropna(subset=["spread_fold", "dBIC_same_dwell"])
        if len(d) < 4:
            continue
        r1 = spearmanr(d.spread_fold, d.dBIC_same_dwell)
        r2 = spearmanr(d.spread_fold, d.dBIC_per_dwell)
        cr.append(dict(subset=lab, n=len(d),
                       rho_spread_vs_dBIC=r1.statistic, p_raw=r1.pvalue,
                       rho_spread_vs_dBIC_per_dwell=r2.statistic, p_per_dwell=r2.pvalue))
    pd.DataFrame(cr).to_csv(
        os.path.join(TAB, "cross_estimator_bic_correlation.csv"), index=False)

    # ---- mean-dwell K_D, Lee et al. 2024 eq 8
    mk = []
    for (fl, c, pw), g in E[E.estimator == "mean"].groupby(keys):
        on = g[g.dwell == "on"].tau_s
        off = g[g.dwell == "off"].tau_s
        if not (len(on) and len(off)):
            continue
        ton, toff = float(on.iloc[0]), float(off.iloc[0])
        mk.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                       mean_ton_s=ton, mean_toff_s=toff,
                       K_D_mean_dwell_nM=toff * c / ton,
                       k_off_from_mean_per_s=1 / ton,
                       k_on_from_mean_per_nM_per_s=1 / (toff * c),
                       K_D_bulk_nM=0.69 if fl == "SALAD1" else 0.85))
    MK = pd.DataFrame(mk)

    # ---- between-field mean +/- sample SD, the alternative uncertainty
    bf = []
    pf = os.path.join(TAB, "kinetics_per_field.csv")
    if os.path.exists(pf):
        F = pd.read_csv(pf)
        for col in ("on_tau_single_s", "off_tau_single_s"):
            if col not in F.columns:
                continue
            gg = (F.groupby([k for k in keys if k in F.columns])[col]
                  .agg(n_fields="size", mean="mean",
                       sd=lambda s: float(np.std(s, ddof=1)) if s.size > 1 else 0.0)
                  .reset_index())
            gg["quantity"] = col
            gg["cv"] = gg.sd / gg["mean"]
            bf.append(gg)
    BF = pd.concat(bf, ignore_index=True) if bf else pd.DataFrame()

    E.to_csv(os.path.join(TAB, "cross_estimator_tau.csv"), index=False)
    sp.to_csv(os.path.join(TAB, "cross_estimator_spread.csv"), index=False)
    D.to_csv(os.path.join(TAB, "observation_window_diagnostics.csv"), index=False)
    MK.to_csv(os.path.join(TAB, "kd_mean_dwell_lee2024.csv"), index=False)
    if len(BF):
        BF.to_csv(os.path.join(TAB, "between_field_uncertainty.csv"), index=False)
    A.log(f"cross_estimators: {len(E)} estimator fits, {len(D)} conditions diagnosed")
    return E, sp, D, MK, BF


if __name__ == "__main__":
    build()
