"""Censoring-aware mixture analysis of per-molecule brightness.

Why this file exists
--------------------
The brightness distribution is hard-censored: a molecule enters it only if its
bound level exceeds its own background by 3 background standard deviations, so
the observed density is zero below that per-molecule floor and the true density
below it is unobserved. A plain Gaussian mixture fitted to such a sample will
place a component against the censoring boundary whether or not a dim
subpopulation exists, because that is the cheapest way to reproduce the sharp
left edge. A two-component preference obtained that way is therefore not
evidence for a dim population.

Two estimators are provided.

1. `fit_truncated_mixture` -- a K-component Gaussian mixture in log-brightness
   with the floor entered as a KNOWN per-observation left-truncation point. The
   likelihood of an observed molecule is

       f(y_i) = sum_k w_k phi((y_i-mu_k)/s_k)/s_k
                / sum_k w_k * Phi((mu_k-a_i)/s_k)

   where a_i = log(3 sigma_i) is that molecule's own floor. The denominator is
   the probability that a molecule of the mixture is detectable at all, so the
   fitted weights and locations refer to the UNDERLYING population rather than
   to the detected sample.

2. `fit_margin_mixture` -- a plain mixture fitted only to molecules a stated
   margin above their own floor. This discards the censoring boundary instead of
   modelling it, and is the more conservative of the two.

A dim subpopulation is claimed only if the extra component survives BOTH, and
sits clear of the floor in both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from scipy import stats

FLOOR_SIGMA = 3.0          # the idealiser's acceptance threshold
MARGIN = 1.5               # "clear of the floor" = brightness > MARGIN * floor


def _unpack(theta, K):
    if K == 1:
        return np.array([0.0]), theta[0:1], np.exp(theta[1:2])
    logits = np.r_[0.0, theta[:K - 1]]
    logw = logits - logsumexp(logits)
    mu = theta[K - 1:2 * K - 1]
    s = np.exp(theta[2 * K - 1:3 * K - 1])
    return logw, mu, s


def _nll_trunc(theta, y, a, K):
    logw, mu, s = _unpack(theta, K)
    if np.any(s < 1e-4) or np.any(s > 20):
        return 1e12
    z = (y[:, None] - mu) / s
    logpdf = logw - np.log(s) - 0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
    # detectability of each component at each molecule's own floor
    logS = logw + log_ndtr((mu - a[:, None]) / s)
    ll = np.sum(logsumexp(logpdf, axis=1) - logsumexp(logS, axis=1))
    return -ll if np.isfinite(ll) else 1e12


def fit_truncated_mixture(brightness, floor, K, n_init=5, seed=0):
    """MLE of a K-component lognormal mixture with per-molecule left truncation."""
    y, a = np.log(np.asarray(brightness, float)), np.log(np.asarray(floor, float))
    ok = np.isfinite(y) & np.isfinite(a)
    y, a = y[ok], a[ok]
    rg = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        if K == 1:
            x0 = np.r_[np.median(y) + rg.normal(0, .3), np.log(y.std())]
        else:
            x0 = np.r_[rg.normal(0, 1, K - 1),
                       np.sort(rg.choice(y, K, replace=False)),
                       np.log(np.full(K, y.std() * rg.uniform(.4, 1.0)))]
        r = minimize(_nll_trunc, x0, args=(y, a, K), method="L-BFGS-B",
                     options=dict(maxiter=2000, maxfun=20000, ftol=1e-12, gtol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    logw, mu, s = _unpack(best.x, K)
    npar = 3 * K - 1
    order = np.argsort(mu)
    return dict(K=K, n=y.size, nll=float(best.fun),
                bic=npar * np.log(y.size) + 2 * best.fun,
                weights=np.exp(logw)[order], medians=np.exp(mu)[order],
                sigmas=s[order], converged=bool(best.success))


def fit_margin_mixture(brightness, floor, K, margin=MARGIN, n_init=10, seed=0):
    """Plain (untruncated) mixture fitted only above margin * floor."""
    b, f = np.asarray(brightness, float), np.asarray(floor, float)
    keep = np.isfinite(b) & np.isfinite(f) & (b > margin * f)
    y = np.log(b[keep])
    if y.size < 200:
        return dict(K=K, n=int(y.size), nll=np.nan, bic=np.nan,
                    weights=np.array([np.nan]), medians=np.array([np.nan]),
                    sigmas=np.array([np.nan]), converged=False)
    rg = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        mu = rg.choice(y, K, replace=False)
        s = np.full(K, y.std())
        w = np.full(K, 1 / K)
        prev = -np.inf
        for _ in range(500):
            p = w * stats.norm.pdf(y[:, None], mu, s)
            tot = p.sum(1)
            ll = np.log(np.maximum(tot, 1e-300)).sum()
            g = p / tot[:, None]
            nk = g.sum(0) + 1e-12
            w = nk / y.size
            mu = (g * y[:, None]).sum(0) / nk
            s = np.sqrt((g * (y[:, None] - mu) ** 2).sum(0) / nk).clip(1e-4)
            if abs(ll - prev) < 1e-9 * abs(ll):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, w.copy(), mu.copy(), s.copy())
    ll, w, mu, s = best
    npar = 3 * K - 1
    order = np.argsort(mu)
    return dict(K=K, n=int(y.size), nll=float(-ll),
                bic=npar * np.log(y.size) - 2 * ll,
                weights=w[order], medians=np.exp(mu)[order], sigmas=s[order],
                converged=True)


def _nll_pinned(params, y, a, mu_dim, w_dim):
    """Two-component truncated mixture with the DIM component pinned at a given
    location and population weight; only the bright component is free."""
    mu2, ls2, ls1 = params
    s2, s1 = np.exp(ls2), np.exp(ls1)
    if not (1e-3 < s1 < 20 and 1e-3 < s2 < 20):
        return 1e12
    logw = np.log(np.array([w_dim, 1 - w_dim]).clip(1e-12))
    mu, s = np.array([mu_dim, mu2]), np.array([s1, s2])
    z = (y[:, None] - mu) / s
    logpdf = logw - np.log(s) - 0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
    logS = logw + log_ndtr((mu - a[:, None]) / s)
    ll = np.sum(logsumexp(logpdf, axis=1) - logsumexp(logS, axis=1))
    return -ll if np.isfinite(ll) else 1e12


def profile_dark_fraction(B, group_cols=("fluorogen", "conc_nM", "power_mW"),
                          floor_fracs=(0.2, 0.4, 0.6, 0.8, 1.0),
                          weights=(0.05, 0.2, 0.4, 0.6, 0.8),
                          min_n=400, seed=0):
    """Profile likelihood over a pinned dim population.

    For each condition, a second component is PINNED at a stated fraction of the
    median detection floor and at a stated share of the population, and the
    remaining parameters are re-optimised. Reporting dNLL against the
    single-component fit shows directly how much the data constrain the
    undetected population: where dNLL is ~0 across a wide range of weights, the
    dark fraction is not identifiable from the brightness distribution and no
    claim about it can be made from these data.
    """
    rows = []
    for keys, g in B.groupby(list(group_cols)):
        b, f = np.asarray(g.brightness, float), np.asarray(g.floor, float)
        ok = np.isfinite(b) & np.isfinite(f) & (b > 0) & (f > 0)
        if ok.sum() < min_n:
            continue
        y, a = np.log(b[ok]), np.log(f[ok])
        med_floor = float(np.median(f[ok]))
        base = fit_truncated_mixture(b[ok], f[ok], 1, n_init=4, seed=seed)
        for ff in floor_fracs:
            mu_dim = np.log(ff * med_floor)
            for w in weights:
                r = minimize(_nll_pinned,
                             [np.median(y), np.log(y.std()), np.log(y.std() * .5)],
                             args=(y, a, mu_dim, w), method="L-BFGS-B",
                             options=dict(maxiter=800, ftol=1e-12))
                rows.append(dict(zip(group_cols, keys)) | dict(
                    n=int(y.size), median_floor=med_floor,
                    dim_median_over_floor=ff, dim_median_counts=ff * med_floor,
                    dim_population_weight=w, nll=float(r.fun),
                    nll_1component=base["nll"],
                    dNLL_vs_1component=float(r.fun) - base["nll"]))
    return pd.DataFrame(rows)


def mixture_table(B, group_cols=("fluorogen", "conc_nM", "power_mW"),
                  Ks=(1, 2, 3), margin=MARGIN, min_n=400, max_n=5000, seed=0,
                  n_init=3):
    """Both estimators, every K, every condition, with the floor-proximity
    diagnostics needed to judge whether an extra component is real.

    Conditions larger than max_n are subsampled to that size (the same draw for
    every K, so the BIC comparison within a condition stays valid); a
    three-component fit is amply determined by several thousand molecules.
    """
    rg0 = np.random.default_rng(seed)
    rows = []
    for keys, g in B.groupby(list(group_cols)):
        fl = np.asarray(g.floor, float)
        b = np.asarray(g.brightness, float)
        ok = np.isfinite(fl) & np.isfinite(b)
        if ok.sum() < min_n:
            continue
        b, fl = b[ok], fl[ok]
        if b.size > max_n:
            pick = rg0.choice(b.size, max_n, replace=False)
            b, fl = b[pick], fl[pick]
        med_floor = float(np.median(fl))
        for K in Ks:
            for est, fn in (("truncation-aware", fit_truncated_mixture),
                            ("above-margin", fit_margin_mixture)):
                r = (fn(b, fl, K, n_init=n_init, seed=seed) if est == "truncation-aware"
                     else fn(b, fl, K, margin=margin, n_init=n_init, seed=seed))
                dim_med = float(r["medians"][0])
                rows.append(dict(zip(group_cols, keys)) | dict(
                    estimator=est, K=K, n=r["n"], bic=r["bic"], nll=r["nll"],
                    median_floor=med_floor,
                    weights=np.round(r["weights"], 4).tolist(),
                    component_medians=np.round(r["medians"], 1).tolist(),
                    component_sigmas=np.round(r["sigmas"], 3).tolist(),
                    dimmest_component_median=dim_med,
                    dimmest_over_floor=dim_med / med_floor,
                    dimmest_clear_of_floor=bool(dim_med > margin * med_floor),
                    converged=r["converged"]))
    tab = pd.DataFrame(rows)
    # per (condition, estimator): which K wins, and is the dim component real?
    verdict = []
    for (est, *keys), g in tab.groupby(["estimator"] + list(group_cols)):
        g = g.sort_values("bic")
        bestK = int(g.K.iloc[0])
        b1 = float(g.loc[g.K == 1, "bic"].iloc[0])
        bk = float(g.bic.iloc[0])
        row = g.iloc[0]
        verdict.append(dict(estimator=est, **dict(zip(group_cols, keys)),
                            best_K=bestK, bic_1=b1, bic_best=bk, dBIC_vs_1=b1 - bk,
                            dimmest_over_floor=row.dimmest_over_floor,
                            dim_component_clear_of_floor=row.dimmest_clear_of_floor,
                            supports_dim_subpopulation=bool(
                                bestK > 1 and row.dimmest_clear_of_floor)))
    return tab, pd.DataFrame(verdict)
