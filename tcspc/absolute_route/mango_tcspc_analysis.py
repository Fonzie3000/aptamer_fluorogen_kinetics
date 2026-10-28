#!/usr/bin/env python3
"""
Mango II fluorescence lifetime and brightness analysis - complete, self-contained.

Writes its tables as CSV files. Chapter 4 of the thesis uses one of them,
strickland_berg_absolute.csv (Supplementary Figure S4.6 and section 4.3.2).

Usage
-----
    python mango_tcspc_analysis.py            # writes results/ (CSV tables)
    python mango_tcspc_analysis.py --quiet    # suppress the console summary

Requires: numpy>=1.20, pandas, scipy, openpyxl.
Expects the six input files in data/ beside this script:
    tcspc_global_fit_results.csv     global fit output, k=2+scatter and k=3+scatter
    bulk_brightness_MII_TO1B.csv     stem-series summary (means + population SDs)
    stem_Brightness.xlsx             stem-series RAW replicate readings (n=3)
    FI_4.xlsx                        emission spectra, MII wt / A22U x TO1-B / SALAD1
    literature_photophysics.csv      Yang 2025 Table 1
    literature_lifetimes.csv         Trachman 2018 lifetimes

Naming: SALAD1 appears as "SALAD" in raw .das filenames and as "365a-061" in some
spreadsheet headers; TO1-Biotin appears as "MII". All three refer to the same compounds.

CONVENTIONS, which matter in this dataset
-----------------------------------------
* Every lifetime is named with its model (k2_scatter / k3_scatter) and weighting
  (amplitude- or intensity-weighted). tau_amp and tau_int differ by ~17% here.
* Phi = k_r * tau_amp is exact for a multi-exponential decay whose sub-populations share
  a radiative rate, so tau_amp - NOT tau_int - is the lifetime that enters any
  quantum-yield or k_r algebra. tau_int is the more model-robust quantity but does not
  appear in that relation.
* Uncertainties are SD unless labelled SE or as a Monte Carlo percentile interval.
* Asymptotic standard errors from the fit are never used; replicate/preparation scatter is.

Author's note on one trap: in F ~ eps*c*Phi with Phi = k_r*f_B*tau_amp, the extinction
coefficient enters TWICE and both are correct - once as the absorption cross-section at
the excitation wavelength, and again inside k_r via the Strickland-Berg integrated
absorption. It is the same number, so its uncertainty is perfectly correlated and enters
squared. Omitting the second factor understates the bright-fraction ratio (1.85 vs 2.04).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
TRAP = getattr(np, "trapezoid", None) or np.trapz     # numpy 2.x renamed trapz

# Physical / instrumental constants
N_BUFFER = 1.334        # refractive index of aqueous buffer; cancels in all ratios here
IRF_FWHM_NS = 0.20      # assumed DeltaFlex response width, used only to state a threshold
GAUSS_AREA = np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))   # 1.0645, Gaussian band area factor

SPEC_COLS = ["MIIwt_TO1B", "MIIwt_SALAD1", "A22U_TO1B", "A22U_SALAD1"]


# ==================================================================================
# 1. Loading
# ==================================================================================
def read_commented_csv(path, header_startswith):
    """The brightness file has QUOTED '#' preamble lines, so pandas' comment= will not
    strip them (each line begins with '"'). Locate the real header row instead."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith(header_startswith))
    preamble = [l.strip('"').lstrip("# ").strip() for l in lines[:hdr]]
    return pd.read_csv(path, skiprows=hdr), preamble


def load_all(data_dir=DATA):
    d = {}
    fit = pd.read_csv(f"{data_dir}/tcspc_global_fit_results.csv")
    d["fit"] = fit
    d["T2"] = fit[(fit.model == "k2_scatter") & (fit.dye == "TO1-B")].sort_values("stem_bp").reset_index(drop=True)
    d["S2"] = fit[(fit.model == "k2_scatter") & (fit.dye == "SALAD1")].sort_values("stem_bp").reset_index(drop=True)
    d["T3"] = fit[(fit.model == "k3_scatter") & (fit.dye == "TO1-B")].sort_values("stem_bp").reset_index(drop=True)
    d["S3"] = fit[(fit.model == "k3_scatter") & (fit.dye == "SALAD1")].sort_values("stem_bp").reset_index(drop=True)

    d["bright"], d["bright_preamble"] = read_commented_csv(
        f"{data_dir}/bulk_brightness_MII_TO1B.csv", "stem_bp")

    lit = pd.read_csv(f"{data_dir}/literature_photophysics.csv")
    d["litT"] = lit[lit.dye == "TO1-biotin"].iloc[0]
    d["litS"] = lit[lit.dye == "SALAD1"].iloc[0]
    d["lit_tau"] = pd.read_csv(f"{data_dir}/literature_lifetimes.csv")

    sp = pd.read_excel(f"{data_dir}/FI_4.xlsx")
    sp.columns = ["wl"] + SPEC_COLS
    d["spec"] = sp.dropna().astype(float).sort_values("wl").reset_index(drop=True)

    st = pd.read_excel(f"{data_dir}/stem_Brightness.xlsx", header=None)
    stem = pd.DataFrame({"label": st.iloc[1].values, "I": st.iloc[2].values.astype(float)})
    stem["stem_bp"] = stem.label.str.extract(r"(\d+)bp").astype(int)
    stem["rep"] = stem.label.str.extract(r"_(\d)").astype(int)
    d["stem_raw"] = stem
    return d


# ==================================================================================
# 2. Descriptive statistics and the correlation that does NOT establish decoupling
# ==================================================================================
def descriptive_stats(D):
    """Reproduces (and corrects) the summary numbers quoted in the project brief."""
    I = D["bright"].intensity_au.values
    sd0 = D["bright"].sd_au.values                 # these are POPULATION SDs, see below
    T2 = D["T2"]
    ta, ti = T2.tau_amp_ns.values, T2.tau_int_ns.values
    i12 = int(np.where(D["bright"].stem_bp.values == 12)[0][0])
    i14 = int(np.where(D["bright"].stem_bp.values == 14)[0][0])
    out = {
        "brightness_range_pct_of_mean": 100 * (I.max() - I.min()) / I.mean(),
        "brightness_mean_supplied_sd_au": sd0.mean(),
        "brightness_mean_supplied_sd_pct": 100 * sd0.mean() / I.mean(),
        "var_between_constructs": I.var(ddof=1),
        "var_within_mean_of_variances": (sd0**2).mean(),
        "ratio_pooled_variance": I.var(ddof=1) / (sd0**2).mean(),
        "ratio_squared_mean_sd_as_in_brief": I.var(ddof=1) / sd0.mean()**2,
        "step_12_14_au": I[i12] - I[i14],
        "TO1B_tau_amp_range_pct": 100 * (ta.max() - ta.min()) / ta.mean(),
        "TO1B_tau_int_range_pct": 100 * (ti.max() - ti.min()) / ti.mean(),
        "TO1B_tau_amp_sd_across_preps_ns": ta.std(ddof=1),
        "TO1B_tau_amp_sd_across_preps_pct": 100 * ta.std(ddof=1) / ta.mean(),
        "TO1B_a2_range_pct_of_mean": 100 * (T2.a2_pct.max() - T2.a2_pct.min()) / T2.a2_pct.mean(),
    }
    out["brightness_over_tau_variation_factor"] = (
        out["brightness_range_pct_of_mean"] / out["TO1B_tau_amp_range_pct"])
    return out


def pearson_with_ci(x, y, alpha=0.05):
    """Pearson r with a Fisher-z CI. At n=7 the CI is the honest output, not the p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((n - 2) / max(1 - r**2, 1e-300))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return dict(n=n, r=r, p=p,
                ci_lo=float(np.tanh(np.arctanh(r) - zc * se)),
                ci_hi=float(np.tanh(np.arctanh(r) + zc * se)))


def brightness_lifetime_correlations(D):
    b = D["bright"].set_index("stem_bp").intensity_au
    t2, t3 = D["T2"].set_index("stem_bp"), D["T3"].set_index("stem_bp")
    bp = sorted(set(b.index) & set(t2.index))
    series = [("tau_amp_k2", t2.tau_amp_ns), ("tau_int_k2", t2.tau_int_ns),
              ("a2_pct_k2", t2.a2_pct), ("a1_pct_k2", t2.a1_pct),
              ("tau_amp_k3", t3.tau_amp_ns), ("tau_int_k3", t3.tau_int_ns)]
    rows = []
    for name, col in series:
        rows.append(dict(predictor=name, **pearson_with_ci(b.loc[bp].values, col.loc[bp].values)))
    return pd.DataFrame(rows)


def variance_share(D):
    """How much of the brightness variation could tau_amp carry, on a log scale?

    Between-construct variances are corrected for the within-construct noise floor, so
    this is an upper bound on the share attributable to lifetime.
    """
    F = D["bright"].intensity_au.values
    sF = D["bright"].sd_au.values
    ta = D["T2"].tau_amp_ns.values
    sta = D["T2"].tau_amp_sd.values
    sd_lnF_b, sd_lnF_w = np.std(np.log(F), ddof=1), np.mean(sF / F)
    sd_lnt_b, sd_lnt_w = np.std(np.log(ta), ddof=1), np.mean(sta / ta)
    var_F = max(sd_lnF_b**2 - sd_lnF_w**2, 0.0)
    var_t = max(sd_lnt_b**2 - sd_lnt_w**2, 0.0)
    return dict(sd_lnF_between_pct=100 * sd_lnF_b, sd_lnF_within_pct=100 * sd_lnF_w,
                sd_lnF_corrected_pct=100 * np.sqrt(var_F),
                sd_lntau_between_pct=100 * sd_lnt_b, sd_lntau_within_pct=100 * sd_lnt_w,
                sd_lntau_corrected_pct=100 * np.sqrt(var_t),
                max_variance_share_pct=100 * var_t / var_F if var_F > 0 else np.nan)


def stem_series_stats(D):
    """Proper statistics on the RAW replicates.

    Two corrections to the supplied summary file: n = 3 (the column was empty), and the
    supplied SDs are population SDs (ddof=0), a factor sqrt(3/2)=1.225 too small. The
    brief's '4.0 sigma' for the 12->14 bp step divided a difference of MEANS by a
    per-observation SD; the correct denominator is the SE of the difference.
    """
    stem = D["stem_raw"]
    bps = sorted(stem.stem_bp.unique())
    groups = [stem[stem.stem_bp == b].I.values for b in bps]
    F, p = stats.f_oneway(*groups)
    n = len(groups[0])
    ms_between = np.var([g.mean() for g in groups], ddof=1) * n
    ms_within = np.mean([g.var(ddof=1) for g in groups])
    g12 = stem[stem.stem_bp == 12].I.values
    g14 = stem[stem.stem_bp == 14].I.values
    w = stats.ttest_ind(g12, g14, equal_var=False)
    se = float(np.sqrt(g12.var(ddof=1) / len(g12) + g14.var(ddof=1) / len(g14)))
    per_construct = pd.DataFrame({
        "stem_bp": bps,
        "mean_au": [g.mean() for g in groups],
        "sd_sample_ddof1_au": [g.std(ddof=1) for g in groups],
        "sd_population_ddof0_au": [g.std(ddof=0) for g in groups],
        "sem_au": [g.std(ddof=1) / np.sqrt(len(g)) for g in groups],
        "n": [len(g) for g in groups],
    })
    summary = dict(anova_F=float(F), anova_df1=len(groups) - 1,
                   anova_df2=sum(len(g) for g in groups) - len(groups), anova_p=float(p),
                   ms_between_over_ms_within=float(ms_between / ms_within),
                   pooled_within_sd_au=float(np.sqrt(ms_within)),
                   pooled_within_sd_pct=float(100 * np.sqrt(ms_within) / stem.I.mean()),
                   step_12_14_au=float(g12.mean() - g14.mean()), step_se_au=se,
                   step_in_SE_units=float((g12.mean() - g14.mean()) / se),
                   welch_t=float(w.statistic), welch_p=float(w.pvalue))
    return summary, per_construct


# ==================================================================================
# 3. Apparent radiative and non-radiative rates
# ==================================================================================
def shape_lack_of_fit(D):
    """Is the non-monotonic brightness shape real, or would a monotonic trend do?

    Classical lack-of-fit F-test: fit intensity ~ stem_bp (linear), then split the
    residual sum of squares into pure error (from the n=3 replicates) and lack of fit
    (systematic departure from the straight line). A significant lack of fit means the
    construct-to-construct pattern is NOT captured by a monotonic trend.
    """
    stem = D["stem_raw"]
    x, y = stem.stem_bp.values.astype(float), stem.I.values
    slope, intercept, r, p_lin, se = stats.linregress(x, y)
    ss_res = float(np.sum((y - (intercept + slope * x)) ** 2))
    groups = [stem[stem.stem_bp == b].I.values for b in sorted(stem.stem_bp.unique())]
    ss_pe = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups]))
    df_pe = sum(len(g) - 1 for g in groups)
    ss_lof, df_lof = ss_res - ss_pe, len(groups) - 2
    F = (ss_lof / df_lof) / (ss_pe / df_pe)
    return dict(linear_slope_au_per_bp=float(slope), linear_slope_se=float(se),
                linear_p=float(p_lin), linear_r2=float(r**2),
                ss_lack_of_fit=ss_lof, df_lack_of_fit=df_lof,
                ss_pure_error=ss_pe, df_pure_error=df_pe,
                F_lack_of_fit=float(F),
                p_lack_of_fit=float(stats.f.sf(F, df_lof, df_pe)))


def detectable_effects(D, n=3, power=0.80, alpha=0.05):
    """Smallest two-construct difference resolvable at the given power, by measurement.

    Two-sample t-test approximation: delta = (t_{1-a/2,df} + t_{power,df}) * SD * sqrt(2/n).
    Uses the pooled within-construct SD for brightness (from the raw replicates) and the
    mean within-construct replicate SD for tau_amp.
    """
    df = 2 * (n - 1)
    k = (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * np.sqrt(2 / n)
    stem = D["stem_raw"]
    groups = [stem[stem.stem_bp == b].I.values for b in sorted(stem.stem_bp.unique())]
    sd_F = float(np.sqrt(np.mean([g.var(ddof=1) for g in groups])))
    meanF = float(stem.I.mean())
    sd_t = float(D["T2"].tau_amp_sd.mean())
    meant = float(D["T2"].tau_amp_ns.mean())
    return dict(n_per_construct=n, power=power, multiplier=float(k),
                brightness_pooled_within_sd_au=sd_F,
                brightness_min_detectable_au=k * sd_F,
                brightness_min_detectable_pct=100 * k * sd_F / meanF,
                tau_amp_mean_within_sd_ns=sd_t,
                tau_amp_min_detectable_ns=k * sd_t,
                tau_amp_min_detectable_pct=100 * k * sd_t / meant,
                observed_step_12_14_pct=100 * (
                    stem[stem.stem_bp == 12].I.mean() - stem[stem.stem_bp == 14].I.mean()) / meanF,
                observed_SALAD1_vs_TO1B_tau_amp_pct=100 * (
                    D["S2"].tau_amp_ns.mean() / D["T2"].tau_amp_ns.mean() - 1))


def amp_weighted_rate(row, k):
    """<k>_amp = sum_i a_i / tau_i, the amplitude-weighted mean DECAY rate.

    This is not 1/tau_amp, because <1/tau> != 1/<tau>. For a heterogeneous sample the
    amplitude-weighted rate is the defensible population average, so k_nr is taken as
    <k>_amp - k_r_app. The two conventions differ by 14-18% in this dataset. Both are
    tabulated; the naive one should not be used.
    """
    a = np.array([row[f"a{i}_pct"] for i in range(1, k + 1)], float) / 100.0
    tau = np.array([row[f"tau{i}_ns"] for i in range(1, k + 1)], float)
    a = a / a.sum()
    return float((a / tau).sum())


def rate_table(D):
    """Apparent rates per construct. Phi is IMPORTED (Yang 2025) and is used here only to
    show why the imported value is untenable; the final model in section 5 avoids it."""
    rows = []
    for label, df, k in [("TO1-B", D["T2"], 2), ("SALAD1", D["S2"], 2),
                         ("TO1-B", D["T3"], 3), ("SALAD1", D["S3"], 3)]:
        phi = float(D["litT"].QY) if label == "TO1-B" else float(D["litS"].QY)
        for _, r in df.iterrows():
            kr = phi / r.tau_amp_ns                     # ns^-1
            kamp = amp_weighted_rate(r, k)
            rows.append(dict(dye=label, model=f"k{k}_scatter", stem_bp=int(r.stem_bp),
                             tau_amp_ns=r.tau_amp_ns, tau_int_ns=r.tau_int_ns, Phi_lit=phi,
                             kr_app_per_ns=kr, kr_app_per_s=kr * 1e9,
                             k_amp_per_ns=kamp, inv_tau_amp_per_ns=1.0 / r.tau_amp_ns,
                             knr_app_per_ns=kamp - kr,
                             knr_naive_per_ns=1.0 / r.tau_amp_ns - kr))
    return pd.DataFrame(rows)


def tau_amp_ratio(D):
    """tau_amp ratio SALAD1/TO1-B with an honest uncertainty.

    Statistical part: SE of the mean over INDEPENDENT preparations (7 TO1-B constructs,
    2 SALAD1 constructs - these are separate tubes; the 3 aliquots within each are
    technical replicates and are not treated as independent). Systematic part: the
    k=2 vs k=3 model spread. The 8 and 12 bp constructs exist for both dyes, giving a
    matched-construct cross-check.
    """
    tT, tS = D["T2"].tau_amp_ns.values, D["S2"].tau_amp_ns.values
    r2 = tS.mean() / tT.mean()
    r3 = D["S3"].tau_amp_ns.values.mean() / D["T3"].tau_amp_ns.values.mean()
    rel_stat = float(np.hypot(tT.std(ddof=1) / np.sqrt(len(tT)) / tT.mean(),
                              tS.std(ddof=1) / np.sqrt(len(tS)) / tS.mean()))
    rel_model = float(abs(r3 - r2) / r2)
    matched = {int(bp): float(D["S2"].loc[D["S2"].stem_bp == bp, "tau_amp_ns"].iloc[0] /
                              D["T2"].loc[D["T2"].stem_bp == bp, "tau_amp_ns"].iloc[0])
               for bp in (8, 12)}
    return dict(ratio_k2=float(r2), ratio_k3=float(r3), rel_stat=rel_stat,
                rel_model=rel_model, rel_total=float(np.hypot(rel_stat, rel_model)),
                matched_8bp=matched[8], matched_12bp=matched[12],
                tau_T_mean=float(tT.mean()), tau_T_sd=float(tT.std(ddof=1)),
                tau_S_mean=float(tS.mean()), tau_S_sd=float(tS.std(ddof=1)),
                tau_int_ratio_k2=float(D["S2"].tau_int_ns.mean() / D["T2"].tau_int_ns.mean()),
                tau_int_ratio_k3=float(D["S3"].tau_int_ns.mean() / D["T3"].tau_int_ns.mean()))


# ==================================================================================
# 4. Strickland-Berg
# ==================================================================================
def sb_prefactor(eps_max, lam_abs_nm, lam_em_nm, n=N_BUFFER):
    """k_r = prefactor * (absorption FWHM in cm^-1), Gaussian-band approximation of

        k_r = 2.88e-9 n^2 <nu_f^-3>^-1 Int eps(nu) dnu/nu

    with Int eps dnu/nu ~= 1.0645 eps_max FWHM / nu_abs and <nu_f^-3>^-1 ~= nu_em^3.
    Returns s^-1 per cm^-1 of band width. Used for the ABSOLUTE route only.
    """
    nu_abs, nu_em = 1e7 / lam_abs_nm, 1e7 / lam_em_nm
    return 2.88e-9 * n**2 * nu_em**3 * (eps_max / nu_abs) * GAUSS_AREA


def nu_minus3(y, wl):
    """<nu_f^-3> for the Strickland-Berg emission factor, computed exactly from a spectrum.

    <nu^-3> = Int I(nu) nu^-3 dnu / Int I(nu) dnu. Substituting nu = 1e7/lambda and
    I(nu) = I(lambda) lambda^2 / 1e7 makes the Jacobian cancel, leaving
    Int I_lambda (lambda/1e7)^3 dlambda / Int I_lambda dlambda.
    """
    return float(TRAP(y * (wl / 1e7) ** 3, wl) / TRAP(y, wl))


def band_stats(y, wl):
    """Peak, intensity centroid and FWHM (nm and cm^-1) of an emission band."""
    i = int(np.argmax(y))
    half = y.max() / 2
    lo = float(np.interp(half, y[: i + 1], wl[: i + 1]))     # rising edge
    hi = float(np.interp(-half, -y[i:], wl[i:]))             # falling edge
    return dict(peak_nm=float(wl[i]),
                centroid_nm=float(TRAP(y * wl, wl) / TRAP(y, wl)),
                fwhm_nm=hi - lo,
                fwhm_cm1=1e7 * (hi - lo) / ((lo + hi) / 2) ** 2)


def sb_ratio_measured(D, use_mirror_image=True):
    """Defensible k_r,SALAD1/k_r,TO1-B, using the MEASURED emission spectra.

    k_r ratio = (eps ratio) x (<nu^-3>_T/<nu^-3>_S) x (absorption band-width ratio).
    The emission factor is measured exactly. Absorbance data is unavailable for this
    project, so the absorption band-width ratio is taken from the EMISSION band widths
    via the mirror-image rule - an approximation, and the weakest link in the chain.
    Set use_mirror_image=False to drop it (band-width ratio = 1).
    """
    sp = D["spec"]
    eps_r = float(D["litS"].eps_M1cm1) / float(D["litT"].eps_M1cm1)
    em = nu_minus3(sp.MIIwt_TO1B.values, sp.wl.values) / nu_minus3(sp.MIIwt_SALAD1.values, sp.wl.values)
    bw = 1.0
    if use_mirror_image:
        bT = band_stats(sp.MIIwt_TO1B.values, sp.wl.values)["fwhm_cm1"]
        bS = band_stats(sp.MIIwt_SALAD1.values, sp.wl.values)["fwhm_cm1"]
        bw = bS / bT
    return eps_r * em * bw, dict(eps_ratio=eps_r, emission_factor=em, bandwidth_ratio=bw)


def sb_absolute_table(D, widths=(1500, 1700, 1900, 2100, 2409, 2500)):
    """Absolute route: f_B = k_r_app / k_r_Strickland-Berg, sweeping the unknown band width.

    f_B(SALAD1) > 1 is unphysical and is what flags Yang's Phi(SALAD1)=0.722 as too high.
    """
    pT = sb_prefactor(float(D["litT"].eps_M1cm1), float(D["litT"].lambda_ex), float(D["litT"].lambda_em))
    pS = sb_prefactor(float(D["litS"].eps_M1cm1), float(D["litS"].lambda_ex), float(D["litS"].lambda_em))
    krT = float(D["litT"].QY) / D["T2"].tau_amp_ns.mean() * 1e9
    krS = float(D["litS"].QY) / D["S2"].tau_amp_ns.mean() * 1e9
    rows = [dict(band_fwhm_cm1=w, kr_SB_TO1B_per_s=pT * w, kr_SB_SALAD1_per_s=pS * w,
                 fB_TO1B=krT / (pT * w), fB_SALAD1=krS / (pS * w),
                 Phi_SALAD1_ceiling=pS * w * D["S2"].tau_amp_ns.mean() / 1e9) for w in widths]
    meta = dict(prefactor_TO1B=pT, prefactor_SALAD1=pS, kr_app_TO1B_per_s=krT,
                kr_app_SALAD1_per_s=krS, width_for_fB_SALAD1_equals_1=krS / pS)
    return pd.DataFrame(rows), meta


# ==================================================================================
# 5. The dark-fraction model
# ==================================================================================
def fbright_ratio(R_F, eps_ratio, kr_ratio, tau_ratio, c_ratio=1.0):
    """f_B(SALAD1)/f_B(TO1-B) from F ~ eps * c * k_r * f_B * tau_amp.

    CAREFUL: eps enters twice and both are correct. `eps_ratio` is the absorption
    cross-section at the excitation wavelength; `kr_ratio` from sb_ratio_measured ALREADY
    contains a second factor of eps because k_r scales with the integrated absorption.
    Pass eps_ratio=1 and kr_ratio=1 together for the assumption-free floor.
    """
    return R_F / (eps_ratio * c_ratio * kr_ratio * tau_ratio)


def fbright_ratio_mc(R_F, eps_ratio, em_factor, bw_ratio, tau_ratio, tau_rel,
                     rel_RF=0.05, rel_eps=0.05, rel_em=0.01, rel_bw=0.10,
                     n=300_000, seed=20240517):
    """Monte Carlo interval on the bright-fraction ratio.

    rel_RF and rel_eps are STATED ASSUMPTIONS, not fitted errors: the emission spectra are
    single acquisitions (n=1) and Yang publish no uncertainty on eps. eps is drawn ONCE
    and used squared, because it is literally the same number in both places.
    """
    rng = np.random.default_rng(seed)
    ln = lambda c, s: c * np.exp(rng.normal(0, s, n))
    d_eps = ln(eps_ratio, rel_eps)
    fb = ln(R_F, rel_RF) / (d_eps**2 * ln(em_factor, rel_em) *
                            ln(bw_ratio, rel_bw) * ln(tau_ratio, tau_rel))
    return fb


def dark_pool_photon_fraction(f_dark, tau_dark_ns, tau_bright_ns):
    """Share of DETECTED photons contributed by an unresolved dark pool.

    A term A*exp(-t/tau_D) convolved with the IRF tends to A*tau_D*IRF(t) as tau_D -> 0,
    so the fitted s*IRF coefficient scales with the pool's PHOTON COUNT, not its
    population. A large dark pool is photon-poor, which is why it hides under scatter.
    """
    f_bright = 1.0 - f_dark
    return (f_dark * tau_dark_ns) / (f_dark * tau_dark_ns + f_bright * tau_bright_ns)


def max_tau_dark(f_dark, tau_bright_ns, photon_budget):
    """Largest tau_dark whose photon contribution stays inside a given budget."""
    b = photon_budget
    return b * (1.0 - f_dark) * tau_bright_ns / (f_dark * (1 - b))


def dark_pool_visibility(D, f_bright_TO1B,
                         taus_ps=(1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200),
                         budgets=(0.001, 0.005, 0.01, 0.0217, 0.05)):
    tauB = float(D["T2"].tau_amp_ns.mean())
    f_dark = 1.0 - f_bright_TO1B
    vis = pd.DataFrame([dict(tau_dark_ps=t,
                             photon_pct=100 * dark_pool_photon_fraction(f_dark, t / 1000, tauB))
                        for t in taus_ps])
    bnd = pd.DataFrame([dict(photon_budget_pct=100 * b,
                             max_tau_dark_ps=1000 * max_tau_dark(f_dark, tauB, b))
                        for b in budgets])
    return vis, bnd, dict(f_dark=f_dark, tau_bright_ns=tauB,
                          scatter_pct_TO1B=float(D["T2"].scatter_pct.iloc[0]),
                          scatter_pct_SALAD1=float(D["S2"].scatter_pct.iloc[0]))


def stem_required_fbright(D):
    """For one dye at fixed eps, F ~ c * f_B * tau_amp, so f_B*c ~ F/tau_amp.

    The result is f_B*c normalised to its mean. It CANNOT be split into a dark fraction
    and a concentration/occupancy difference by these data - the two enter identically.
    """
    t = D["T2"].set_index("stem_bp")
    b = D["bright"].set_index("stem_bp")
    bp = sorted(set(t.index) & set(b.index))
    F, sF = b.loc[bp, "intensity_au"].values, b.loc[bp, "sd_au"].values
    ta, sta = t.loc[bp, "tau_amp_ns"].values, t.loc[bp, "tau_amp_sd"].values
    q = F / ta
    qn = q / q.mean()
    return pd.DataFrame(dict(stem_bp=bp, F_au=F, F_sd_au=sF, tau_amp_ns=ta,
                             tau_amp_sd_ns=sta, fB_times_c_rel=qn,
                             fB_times_c_rel_sd=qn * np.hypot(sF / F, sta / ta)))


def bound_fraction(c_rna_nM, c_dye_nM, kd_nM):
    """1:1 binding, exact quadratic. Returns the fraction of DYE bound."""
    b = c_rna_nM + c_dye_nM + kd_nM
    ab = 0.5 * (b - np.sqrt(np.clip(b * b - 4 * c_rna_nM * c_dye_nM, 0, None)))
    return ab / c_dye_nM


def occupancy_scan(D, concs_nM=(0.1, 0.5, 1, 2, 5, 10, 100, 1000)):
    """Can K_D differences explain the 2x (nM) -> 4x (uM) change in the intensity ratio?

    Equimolar RNA and dye, the least favourable realistic case. The ceiling on the
    occupancy ratio as c -> 0 is K_D(TO1-B)/K_D(SALAD1).
    """
    kdT, kdS = float(D["litT"].KD_nM), float(D["litS"].KD_nM)
    rows = [dict(conc_nM=c, frac_bound_TO1B=bound_fraction(c, c, kdT),
                 frac_bound_SALAD1=bound_fraction(c, c, kdS),
                 occupancy_ratio_S_over_T=bound_fraction(c, c, kdS) / bound_fraction(c, c, kdT))
            for c in concs_nM]
    return pd.DataFrame(rows), dict(KD_ratio_ceiling=kdT / kdS)


# ==================================================================================
# 6. Emission spectra: brightness measures and the A22U test
# ==================================================================================
def window_integral(y, wl, centre, width=10.0):
    m = (wl >= centre - width / 2) & (wl <= centre + width / 2)
    return float(TRAP(y[m], wl[m]))


def brightness_measures(D):
    """Peak height, the 535/10 nm window used in the supplied summary CSV, and the full
    band integral. All three agree on the wt ratio to within 5%, so the choice of measure
    is not load-bearing."""
    sp = D["spec"]
    rows = {}
    for c in SPEC_COLS:
        y, wl = sp[c].values, sp.wl.values
        rows[c] = dict(peak=float(y.max()), win535=window_integral(y, wl, 535.0),
                       full=float(TRAP(y, wl)))
    return pd.DataFrame(rows).T


def spectral_band_table(D):
    sp = D["spec"]
    return pd.DataFrame({c: band_stats(sp[c].values, sp.wl.values) for c in SPEC_COLS}).T


def a22u_test(B, measure="peak"):
    """Same-dye comparison, in which eps, k_r and Phi cancel EXACTLY:

        F(A22U)/F(wt) = (c ratio) x (f_B ratio) x (tau_amp ratio)

    Only concentration, bright fraction and lifetime remain. This is the cleanest test
    available to the project and needs no imported photophysics - but it requires
    tau_amp for the A22U complexes, which has not been measured.
    """
    return {"TO1-B": float(B.loc["A22U_TO1B", measure] / B.loc["MIIwt_TO1B", measure]),
            "SALAD1": float(B.loc["A22U_SALAD1", measure] / B.loc["MIIwt_SALAD1", measure])}


def trachman_reconciliation(D):
    """Which mean type gives Trachman's quoted 1.51 ns, and what k_r does it imply?"""
    row = D["lit_tau"].iloc[0]
    t1, t2, tm = float(row.tau1_ns), float(row.tau2_ns), float(row.tau_mean_ns)
    a1 = (t2 - tm) / (t2 - t1)                       # amplitude fraction giving tau_amp = 1.51
    tint = (a1 * t1**2 + (1 - a1) * t2**2) / (a1 * t1 + (1 - a1) * t2)
    phi = float(row.QY)
    kr_trach = phi / tm * 1e9
    pT = sb_prefactor(float(D["litT"].eps_M1cm1), float(D["litT"].lambda_ex), float(D["litT"].lambda_em))
    return dict(tau1_ns=t1, tau2_ns=t2, quoted_mean_ns=tm, QY=phi,
                a1_for_amp_weighted=a1, implied_tau_int_ns=tint,
                this_work_tau_amp_ns=float(D["T2"].tau_amp_ns.mean()),
                this_work_tau_int_ns=float(D["T2"].tau_int_ns.mean()),
                kr_app_trachman_per_s=kr_trach,
                kr_app_this_work_per_s=float(D["litT"].QY) / D["T2"].tau_amp_ns.mean() * 1e9,
                kr_discrepancy_factor=kr_trach / (float(D["litT"].QY) / D["T2"].tau_amp_ns.mean() * 1e9),
                fB_TO1B_on_trachman_numbers_at_1900cm1=kr_trach / (pT * 1900))


# ==================================================================================
# 7. Main
# ==================================================================================
def main(quiet=False):
    os.makedirs(OUT, exist_ok=True)
    D = load_all()
    say = (lambda *a: None) if quiet else print

    desc = descriptive_stats(D)
    corr = brightness_lifetime_correlations(D)
    vshare = variance_share(D)
    stem_sum, stem_per = stem_series_stats(D)
    rates = rate_table(D)
    tr = tau_amp_ratio(D)
    kr_def, sbmeta = sb_ratio_measured(D)
    sb_tab, sb_meta = sb_absolute_table(D)
    B = brightness_measures(D)
    bands = spectral_band_table(D)
    occ, occ_meta = occupancy_scan(D)
    trach = trachman_reconciliation(D)

    eps_r = sbmeta["eps_ratio"]
    R_F_peak = float(B.loc["MIIwt_SALAD1", "peak"] / B.loc["MIIwt_TO1B", "peak"])
    R_F_full = float(B.loc["MIIwt_SALAD1", "full"] / B.loc["MIIwt_TO1B", "full"])
    fb_floor = fbright_ratio(R_F_peak, 1.0, 1.0, tr["ratio_k2"])
    fb_eps = fbright_ratio(R_F_peak, eps_r, 1.0, tr["ratio_k2"])
    fb_full_pt = fbright_ratio(R_F_peak, eps_r, kr_def, tr["ratio_k2"])
    fb_mc = fbright_ratio_mc(R_F_peak, eps_r, sbmeta["emission_factor"],
                             sbmeta["bandwidth_ratio"], tr["ratio_k2"], tr["rel_total"])
    q = np.percentile(fb_mc, [2.5, 50, 97.5])
    vis, bnd, vismeta = dark_pool_visibility(D, 1.0 / q[1])
    stemreq = stem_required_fbright(D)
    a22 = a22u_test(B)
    phi_S_implied = float(D["litT"].QY) * R_F_peak / eps_r

    headline = pd.DataFrame([
        ("R_F peak, SALAD1/TO1-B, MII wt", R_F_peak),
        ("R_F 535/10nm window", float(B.loc["MIIwt_SALAD1","win535"]/B.loc["MIIwt_TO1B","win535"])),
        ("R_F full band integral", R_F_full),
        ("tau_amp ratio, k2_scatter", tr["ratio_k2"]),
        ("tau_amp ratio, k3_scatter", tr["ratio_k3"]),
        ("tau_amp ratio rel. uncertainty", tr["rel_total"]),
        ("tau_int ratio, k2_scatter", tr["tau_int_ratio_k2"]),
        ("eps ratio (Yang 2025)", eps_r),
        ("SB emission factor (measured)", sbmeta["emission_factor"]),
        ("emission FWHM ratio (measured)", sbmeta["bandwidth_ratio"]),
        ("defensible k_r ratio", kr_def),
        ("f_B ratio, no imported numbers", fb_floor),
        ("f_B ratio, eps only", fb_eps),
        ("f_B ratio, full (point estimate)", fb_full_pt),
        ("f_B ratio, full (MC median)", q[1]),
        ("f_B ratio, full (MC 2.5%)", q[0]),
        ("f_B ratio, full (MC 97.5%)", q[2]),
        ("max f_B(TO1-B)", 1.0 / q[1]),
        ("TO1-B dark fraction, min", 1.0 - 1.0 / q[1]),
        ("Phi(SALAD1) implied by your data", phi_S_implied),
        ("Phi(SALAD1) reported by Yang", float(D["litS"].QY)),
        ("A22U/wt brightness, TO1-B", a22["TO1-B"]),
        ("A22U/wt brightness, SALAD1", a22["SALAD1"]),
        ("SALAD1/TO1-B on A22U", float(B.loc["A22U_SALAD1","peak"]/B.loc["A22U_TO1B","peak"])),
        ("kr_app TO1-B (s^-1, Yang Phi)", sb_meta["kr_app_TO1B_per_s"]),
        ("kr_app SALAD1 (s^-1, Yang Phi)", sb_meta["kr_app_SALAD1_per_s"]),
        ("band FWHM for f_B(SALAD1)=1 (cm^-1)", sb_meta["width_for_fB_SALAD1_equals_1"]),
        ("occupancy ratio ceiling (KD_T/KD_S)", occ_meta["KD_ratio_ceiling"]),
    ], columns=["quantity", "value"])

    tables = {
        "headline_results": headline,
        "descriptive_stats": pd.DataFrame(desc.items(), columns=["quantity", "value"]),
        "brightness_lifetime_correlations": corr,
        "variance_share": pd.DataFrame(vshare.items(), columns=["quantity", "value"]),
        "stem_series_stats": pd.DataFrame(stem_sum.items(), columns=["quantity", "value"]),
        "stem_series_per_construct": stem_per,
        "stem_series_required_fbright": stemreq,
        "rate_table": rates,
        "tau_amp_ratio": pd.DataFrame(tr.items(), columns=["quantity", "value"]),
        "strickland_berg_absolute": sb_tab,
        "spectral_integrals": B.rename_axis("complex").reset_index(),
        "spectral_band_stats": bands.rename_axis("complex").reset_index(),
        "dark_pool_visibility": vis,
        "dark_pool_tau_bounds": bnd,
        "shape_lack_of_fit": pd.DataFrame(shape_lack_of_fit(D).items(), columns=["quantity", "value"]),
        "detectable_effects": pd.DataFrame(detectable_effects(D).items(), columns=["quantity", "value"]),
        "occupancy_scan": occ,
        "trachman_reconciliation": pd.DataFrame(trach.items(), columns=["quantity", "value"]),
    }
    for name, t in tables.items():
        t.to_csv(f"{OUT}/{name}.csv", index=False)

    say("\n=== headline results ===")
    say(headline.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    say(f"\nstem series: ANOVA F({stem_sum['anova_df1']},{stem_sum['anova_df2']}) = "
        f"{stem_sum['anova_F']:.2f}, p = {stem_sum['anova_p']:.2e}; "
        f"12 vs 14 bp = {stem_sum['step_in_SE_units']:.1f} SE (Welch p = {stem_sum['welch_p']:.4f})")
    say(f"brightness variance tau_amp could explain: <= {vshare['max_variance_share_pct']:.2f}% "
        f"(variation ratio {desc['brightness_over_tau_variation_factor']:.1f}x)")
    say(f"\nwrote {len(tables)} tables to {OUT}")
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    main(quiet=ap.parse_args().quiet)
