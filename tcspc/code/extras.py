"""
The three open questions of section 5 of the reanalysis brief.

Q1  Is the short component genuinely shared between the two fluorogens?
    The primary fit links lifetimes WITHIN a fluorophore series but not across the two.
    Here both series are fitted jointly with tau1 forced to a single common value and
    tau2 free per dye, and the deviance is compared against the sum of the two separate
    fits. The comparison is a nested likelihood-ratio test with one degree of freedom
    (two free tau1 values collapse to one).

    Caution on interpretation: with ~600 000 counts per decay, a deviance difference is
    statistically significant at absurdly small effect sizes - the same reason the
    handover forbids deviance F-tests for construct comparisons. The number that matters
    is therefore how much the CONSTRAINT MOVES the lifetimes and mean lifetimes, not
    whether the likelihood-ratio p-value clears 0.05.

Q2  How much of the mean-lifetime difference between the fluorogens is amplitude and
    how much is lifetime? Done by substituting one factor at a time, on 8 bp AND 12 bp,
    since those are the two constructs measured with both fluorogens.

Q3  Is the long component present in a blank? Answered in the main analysis: the ~54 ns
    tail is present in the Ludox scattering standard at the same lifetime. See
    out/slow_tail_characterisation.csv. The dye-without-aptamer blank is NOT among these
    files, so that half of the question stays open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import fit_tcspc as ft

NS = ft.NS_PER_CHANNEL


def fit_fixed_slow(acqs, tau_slow_ns=54.4, scatter=True, background=True,
                   taus_ns_init=(2.1, 5.0)):
    """k = 2 free lifetimes PLUS a third component FIXED at the measured instrumental tail.

    Motivation for section 3: the decays contain a ~54 ns component that is also present
    in the Ludox scattering standard, and a flat baseline cannot represent it. If a third
    FREE exponential is merely absorbing that mismatch, then pinning the third lifetime to
    the independently measured 54 ns should recover most of k=3's improvement while adding
    no free lifetime - and the fitted third lifetime of a free k=3 fit should not be
    interpreted as photophysical.
    """
    init = np.asarray(taus_ns_init, float)

    def total(logp):
        p = np.exp(logp)
        if p[1] <= p[0] or p[1] >= tau_slow_ns:
            return 1e18
        taus_ch = np.array([p[0], p[1], tau_slow_ns]) / NS
        tot, fits = 0.0, []
        for a in acqs:
            f = ft._fit_one_acquisition(a, taus_ch, x0=None,
                                        background=background, scatter=scatter)
            tot += f["deviance"]; fits.append(f)
        total.fits = fits
        total.taus = np.array([p[0], p[1], tau_slow_ns])
        return tot

    res = minimize(total, np.log(init), method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-4, maxiter=2000, maxfev=4000))
    total(res.x)
    n_ch = sum(f["n_channels"] for f in total.fits)
    n_par = 2 + (3 + 1 + (1 if scatter else 0) + (1 if background else 0)) * len(acqs)
    dev = float(res.fun)
    a3 = np.array([100 * f["B"][2] / f["B"].sum() for f in total.fits])
    return dict(taus_ns=total.taus, deviance=dev, dof=n_ch - n_par,
                dev_per_dof=dev / (n_ch - n_par),
                DW=float(np.mean([ft.durbin_watson(
                    a.decay[a.slc],
                    ft.weighted_residuals(a, total.taus, f["B"], f["s"], f["shift"], f["bg"])[0][a.slc])
                    for a, f in zip(acqs, total.fits)])),
                a3_pct_mean=float(a3.mean()), a3_pct_sd=float(a3.std(ddof=1)),
                tau_amp_ns=float(np.mean([ft.mean_lifetimes(total.taus, f["B"])[0]
                                          for f in total.fits])),
                tau_amp_2comp_ns=float(np.mean([ft.mean_lifetimes(total.taus[:2], f["B"][:2])[0]
                                                for f in total.fits])),
                fits=total.fits)


def fit_shared_short(acqs_by_dye: dict, scatter=True, background=True,
                     tau1_init=2.2, tau2_init=None):
    """tau1 common to both fluorogens, tau2 free per fluorogen.

    Outer parameters: [tau1_shared, tau2_dyeA, tau2_dyeB]. Each acquisition is fitted with
    (tau1_shared, tau2 of its own dye), amplitudes and shift free per acquisition.
    """
    dyes = list(acqs_by_dye)
    if tau2_init is None:
        tau2_init = {d: 5.0 for d in dyes}

    def total(logp):
        p = np.exp(logp)
        tau1 = p[0]
        tot = 0.0
        fits = {}
        for i, d in enumerate(dyes):
            tau2 = p[1 + i]
            if tau2 <= tau1:
                return 1e18
            taus_ch = np.array([tau1, tau2]) / NS
            fl = []
            for a in acqs_by_dye[d]:
                f = ft._fit_one_acquisition(a, taus_ch, x0=None,
                                            background=background, scatter=scatter)
                tot += f["deviance"]; fl.append(f)
            fits[d] = (np.array([tau1, tau2]), fl)
        total.fits = fits
        return tot

    x0 = np.log([tau1_init] + [tau2_init[d] for d in dyes])
    res = minimize(total, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-4, maxiter=2000, maxfev=4000))
    total(res.x)
    p = np.exp(res.x)
    return dict(tau1_shared_ns=float(p[0]),
                tau2_ns={d: float(p[1 + i]) for i, d in enumerate(dyes)},
                deviance=float(res.fun), fits=total.fits)


def shared_short_test(runs, acqs_by_dye):
    """Constrained vs unconstrained: deviance, and how far the lifetimes actually move."""
    unc = {d: runs[f"{d}_k2_rule"] for d in acqs_by_dye}
    D_unc = sum(r["deviance"] for r in unc.values())
    t1 = {d: r["taus_ns"][0] for d, r in unc.items()}
    t2 = {d: r["taus_ns"][1] for d, r in unc.items()}
    con = fit_shared_short(acqs_by_dye, tau1_init=float(np.mean(list(t1.values()))),
                           tau2_init=t2)
    dD = con["deviance"] - D_unc
    rows = []
    for d in acqs_by_dye:
        taus_c = np.array([con["tau1_shared_ns"], con["tau2_ns"][d]])
        fl = con["fits"][d][1]
        ta = np.mean([ft.mean_lifetimes(taus_c, f["B"])[0] for f in fl])
        ti = np.mean([ft.mean_lifetimes(taus_c, f["B"])[1] for f in fl])
        pu = pd.DataFrame(unc[d]["per_acq"])
        rows.append(dict(dye=d,
                         tau1_unconstrained_ns=t1[d], tau1_shared_ns=con["tau1_shared_ns"],
                         d_tau1_pct=100 * (con["tau1_shared_ns"] - t1[d]) / t1[d],
                         tau2_unconstrained_ns=t2[d], tau2_constrained_ns=taus_c[1],
                         d_tau2_pct=100 * (taus_c[1] - t2[d]) / t2[d],
                         tau_amp_unconstrained_ns=pu.tau_amp_ns.mean(), tau_amp_constrained_ns=ta,
                         d_tau_amp_pct=100 * (ta - pu.tau_amp_ns.mean()) / pu.tau_amp_ns.mean(),
                         tau_int_unconstrained_ns=pu.tau_int_ns.mean(), tau_int_constrained_ns=ti))
    summary = dict(deviance_unconstrained=D_unc, deviance_constrained=con["deviance"],
                   delta_deviance=dD, df=1, p_value=float(chi2.sf(max(dD, 0), 1)),
                   tau1_shared_ns=con["tau1_shared_ns"],
                   note="delta-deviance p-values are unusable at ~600k counts; judge by how "
                        "far the constraint moves tau_amp, not by p")
    return pd.DataFrame(rows), summary


def decompose_mean_lifetime(runs, window="rule", constructs=(8, 12)):
    """Substitute amplitudes and lifetimes one at a time, for each shared construct.

    tau_amp = sum_j a_j tau_j with a normalised, so with amplitudes a and lifetimes t:
        observed_TO1  = tau_amp(a_TO1, t_TO1)
        observed_SAL  = tau_amp(a_SAL, t_SAL)
        lifetime-only = tau_amp(a_TO1, t_SAL)   amplitudes held at TO1-B
        amplitude-only= tau_amp(a_SAL, t_TO1)   lifetimes held at TO1-B
    The two partial effects need not sum to the total; the residual is the interaction.
    """
    out = []
    tT = np.array(runs[f"TO1-B_k2_{window}"]["taus_ns"])
    tS = np.array(runs[f"SALAD1_k2_{window}"]["taus_ns"])
    pT = pd.DataFrame(runs[f"TO1-B_k2_{window}"]["per_acq"])
    pS = pd.DataFrame(runs[f"SALAD1_k2_{window}"]["per_acq"])
    for bp in constructs:
        aT = np.vstack(pT[pT.stem_bp == bp].a_pct.values).mean(0) / 100
        aS = np.vstack(pS[pS.stem_bp == bp].a_pct.values).mean(0) / 100
        obsT = float((aT * tT).sum()); obsS = float((aS * tS).sum())
        life = float((aT * tS).sum()); amp = float((aS * tT).sum())
        total = obsS - obsT
        out.append(dict(stem_bp=bp,
                        tau_amp_TO1B_ns=obsT, tau_amp_SALAD1_ns=obsS, total_diff_ns=total,
                        lifetime_only_ns=life, lifetime_effect_ns=life - obsT,
                        amplitude_only_ns=amp, amplitude_effect_ns=amp - obsT,
                        interaction_ns=total - (life - obsT) - (amp - obsT),
                        lifetime_share_pct=100 * (life - obsT) / total,
                        amplitude_share_pct=100 * (amp - obsT) / total,
                        a1_TO1B_pct=100 * aT[0], a1_SALAD1_pct=100 * aS[0]))
    return pd.DataFrame(out)
