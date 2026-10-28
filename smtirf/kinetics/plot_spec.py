#!/usr/bin/env python
"""Filter cascade and plotting specification for the four dwell-time plots.

The four plots are the on-time and off-time DISTRIBUTIONS (counts per bin, or
equivalently the survival function) and the on-time and off-time DENSITIES
(the same histograms normalised to unit area). This file produces, per
condition, the numbers needed to draw them and the exact filter that was
applied to get there.

THE DISTINCTION THIS FILE ENFORCES
----------------------------------
A dwell may be removed for one reason only: that it is not measured correctly.
It may never be removed because it makes the plot untidy. Long dwells are the
untidy ones and they are also real, so the tail is handled by choosing what the
AXIS shows, not by choosing what the data contains:

  FILTER      changes the data entering the fit and the histogram. Every one
              below is a named parameter with a measured justification and a
              count in dwell_filter_cascade.csv.
  DISPLAY     changes only the axis limits. Nothing is deleted; the fraction of
              dwells beyond the plotted range is reported so the reader knows
              what is off-screen.

Reporting "n dwells" for a plot always means the count AFTER filtering and
BEFORE display clipping, because the clipped ones are still in the fit.

BIN ALIGNMENT
-------------
Durations are integers in frames. A bin width that is not a whole number of
frames puts a different number of possible integer values in different bins,
which stamps a sawtooth on the histogram that is an artefact of binning and
nothing else. Every bin edge here is at a half-integer frame, so each bin
contains exactly `bin_frames` possible durations.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                      # noqa: E402
import window_select as WS                                      # noqa: E402

TAB = A.TAB
FT = A.FRAME_TIME_S

# ---- display choices (these move the AXIS, never the data) -----------------
# The x-range is set in units of the fitted lifetime, NOT by a percentile of
# the data. On a log-y axis an exponential is a straight line, so the useful
# range is however many lifetimes still carry counts; a percentile range on a
# heavy tail stretches the axis over a region with almost no data and forces
# bins so wide that the short-time behaviour — where all the information is —
# collapses into two or three bars.
N_TAU_DISPLAY = 6            # exp(-6) = 0.25% of the distribution remains
MIN_COUNT_PER_BIN = 5        # bins below this are noise-dominated on a log axis
MAX_BINS = 60                # cap, so a long range does not give unreadable bars

# ---- the one genuinely differential filter --------------------------------
# Detection of short episodes is not equally complete for the two dyes. Pooled
# over concentrations the simulated recovery at 3 frames is 1.000 [0.998,1.000]
# for SALAD1 but 0.961 [0.932,0.977] for TO1-B, and at 4 frames 1.000 and
# 0.985 [0.966,0.994]. MIN_DUR = 3 is therefore complete for one dye and ~4%
# short for the other IN THE FIRST BIN, which is the bin the eye weights most.
# For a figure that compares the two dyes the shared floor is the safer choice.
COMPARISON_FLOOR = 4         # frames; use for two-dye figures
def floors():
    return {"fit (headline, MIN_DUR)": A.MIN_DUR,
            "two-dye comparison figure": COMPARISON_FLOOR}


def windowed():
    ev = WS.load_events()
    wdef = pd.read_csv(os.path.join(TAB, "window_definitions.csv"))
    wdef = wdef[wdef.rule == WS.ADOPTED]
    ev_w, _ = WS.apply_window(ev, wdef)
    return ev, ev_w


def cascade():
    """Count what each named filter removes, per condition and dwell type."""
    ev_raw, ev_w = windowed()
    ON0, OFFI0, OFFB0 = A.extract_dwells(ev_w, 0)          # no gap bridging
    ON, OFFI, OFFB = A.extract_dwells(ev_w, A.MAX_GAP)     # headline bridging
    keys = ["fluorogen", "conc_nM", "power_mW"]
    dmin_off = max(A.MIN_DUR, A.MAX_GAP + 1)
    rows = []
    for (fl, c, pw), g in ON.groupby(keys):
        g0 = ON0[(ON0.fluorogen == fl) & (ON0.conc_nM == c) & (ON0.power_mW == pw)]
        oi = OFFI[(OFFI.fluorogen == fl) & (OFFI.conc_nM == c) & (OFFI.power_mW == pw)]
        ob = OFFB[(OFFB.fluorogen == fl) & (OFFB.conc_nM == c) & (OFFB.power_mW == pw)]
        raw = ev_raw[(ev_raw.fluorogen == fl) & (ev_raw.conc_nM == c)
                     & (ev_raw.power_mW == pw)]
        comp = g[~g.censored]
        rows.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw, dwell="on",
            step0_events_in_acquisition=len(raw),
            step1_inside_window=len(g0),
            step2_after_gap_bridging=len(g),
            step3_complete_uncensored=len(comp),
            step3_censored_kept_in_fit=int(g.censored.sum()),
            step4_ge_MIN_DUR=int((comp.frames >= A.MIN_DUR).sum()),
            step4_ge_comparison_floor=int((comp.frames >= COMPARISON_FLOOR).sum()),
            n_for_plot=int((comp.frames >= A.MIN_DUR).sum())))
        rows.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw, dwell="off",
            step0_events_in_acquisition=len(raw),
            step1_inside_window=len(OFFI0[(OFFI0.fluorogen == fl)
                                          & (OFFI0.conc_nM == c)
                                          & (OFFI0.power_mW == pw)]),
            step2_after_gap_bridging=len(oi),
            step3_complete_uncensored=len(oi),
            step3_censored_kept_in_fit=len(ob),
            step4_ge_MIN_DUR=int((oi.frames >= dmin_off).sum()),
            step4_ge_comparison_floor=int((oi.frames >= dmin_off).sum()),
            n_for_plot=int((oi.frames >= dmin_off).sum())))
    Cd = pd.DataFrame(rows)
    for a, b in (("step1_inside_window", "step0_events_in_acquisition"),
                 ("step2_after_gap_bridging", "step1_inside_window"),
                 ("n_for_plot", "step2_after_gap_bridging")):
        Cd[f"frac_{a}"] = Cd[a] / Cd[b].replace(0, np.nan)
    Cd.to_csv(os.path.join(TAB, "dwell_filter_cascade.csv"), index=False)
    return Cd, ON, OFFI, OFFB


def spec(ON, OFFI, OFFB=None, floor_on=None, floor_off=None):
    """Per condition: bin width, display range, and the curve to overlay.

    The overlay is renormalised to the plotted floor. A single exponential
    fitted with left truncation at t0 has density k*exp(-k(t-t0)) for t >= t0;
    if the figure starts at a different floor the curve must be re-based there
    or it will sit above or below the bars for reasons that have nothing to do
    with the fit quality.
    """
    kw = pd.read_csv(os.path.join(TAB, "kinetics_windowed.csv"))
    keys = ["fluorogen", "conc_nM", "power_mW"]
    f_on = A.MIN_DUR if floor_on is None else floor_on
    f_off = max(A.MIN_DUR, A.MAX_GAP + 1) if floor_off is None else floor_off
    rows = []
    def n_censored(kind, fl, c, pw, src):
        if kind == "on":
            g = src[(src.fluorogen == fl) & (src.conc_nM == c) & (src.power_mW == pw)]
            return int(g.censored.sum())
        if OFFB is None:
            return 0
        g = OFFB[(OFFB.fluorogen == fl) & (OFFB.conc_nM == c) & (OFFB.power_mW == pw)]
        return len(g)

    for kind, src, floor in (("on", ON, f_on), ("off", OFFI, f_off)):
        for (fl, c, pw), g in src.groupby(keys):
            v = g[~g.censored].frames.values if kind == "on" else g.frames.values
            v = np.asarray(v, float)
            v = v[v >= floor]
            if v.size < 30:
                continue
            k = kw[(kw.fluorogen == fl) & (kw.conc_nM == c) & (kw.power_mW == pw)]
            k = k.iloc[0] if len(k) else None
            tau_col = "on_tau_single_s" if kind == "on" else "off_tau_single_s"
            tau_s = float(k[tau_col]) if k is not None else np.nan
            tau_fr = tau_s / FT if np.isfinite(tau_s) else np.nanmean(v)
            hi_disp = float(min(floor + N_TAU_DISPLAY * tau_fr, v.max()))
            # Finest WHOLE-frame bin that still keeps the last displayed bin
            # above the noise floor. Whole frames so every bin holds the same
            # number of possible integer durations.
            bw = 1
            while bw < 200:
                edges = np.arange(floor - 0.5, hi_disp + bw + 0.5, bw)
                cnt, _ = np.histogram(v, bins=edges)
                if (len(edges) - 1) <= MAX_BINS and cnt.size and \
                        cnt[-1] >= MIN_COUNT_PER_BIN:
                    break
                bw += 1
            edges = np.arange(floor - 0.5, hi_disp + bw + 0.5, bw)
            cnt, _ = np.histogram(v, bins=edges)
            last_good = np.flatnonzero(cnt >= MIN_COUNT_PER_BIN)
            x_max_usable = (edges[last_good[-1] + 1] if last_good.size
                            else edges[-1])
            rows.append(dict(
                fluorogen=fl, conc_nM=c, power_mW=pw, dwell=kind,
                n_plotted=int(v.size),
                # A histogram shows only the uncensored observations. For
                # off-times the censored ones (intervals still running at a
                # window edge) are numerous and are the LONGEST, so a density
                # drawn from the plotted subset understates the tail even
                # though the FIT, which right-censors them, does not. Where
                # this fraction is large, show the Kaplan-Meier survival curve
                # instead of, or beside, the density.
                n_censored_not_plotted=n_censored(kind, fl, c, pw, src),
                frac_censored=(n_censored(kind, fl, c, pw, src) /
                               max(1, v.size + n_censored(kind, fl, c, pw, src))),
                floor_frames=int(floor),
                floor_s=floor * FT,
                bin_frames=int(bw), bin_s=bw * FT, n_bins=int(len(edges) - 1),
                x_min_s=(floor - 0.5) * FT, x_max_display_s=hi_disp * FT,
                x_max_usable_s=float(x_max_usable) * FT,
                recommended_x_max_s=float(x_max_usable) * FT,
                # How much of the exponential decay the axis can actually show.
                # For off-times this is limited by the recording, not by the
                # choice of range: six lifetimes is longer than the acquisition,
                # so the plot shows only the first part of the curve and the
                # shape is correspondingly weakly constrained. Below about 2
                # the distribution should not be called exponential on the
                # strength of the figure.
                tau_multiples_shown=((hi_disp - floor) * FT / tau_s
                                     if tau_s and np.isfinite(tau_s) else np.nan),
                display_range_requested_in_tau=N_TAU_DISPLAY,
                range_limited_by=("data/recording" if hi_disp >= v.max()
                                  else "requested tau multiple"),
                frac_beyond_display=float((v > hi_disp).mean()),
                n_beyond_display=int((v > hi_disp).sum()),
                longest_dwell_s=float(v.max()) * FT,
                y_scale="log", hist_norm_for_density="density=True (unit area)",
                tau_single_s=tau_s,
                overlay_rate_per_s=1 / tau_s if tau_s and np.isfinite(tau_s) else np.nan,
                overlay_form=("density: (1/tau)*exp(-(t-t0)/tau) with t0 = "
                              "floor_s;  counts: multiply by n_plotted*bin_s"),
                tau_fast_s=float(k.get(f"{kind}_tau_fast_s", np.nan))
                if k is not None else np.nan,
                tau_slow_s=float(k.get(f"{kind}_tau_slow_s", np.nan))
                if k is not None else np.nan,
                a_fast=float(k.get(f"{kind}_a_fast", np.nan))
                if k is not None else np.nan,
                dBIC_double_minus_single=float(k.get(f"{kind}_dBIC", np.nan))
                if k is not None else np.nan))
    S = pd.DataFrame(rows)
    # For a figure that overlays the two dyes the bins and range must be the
    # SAME for both, or the bars are not comparable. Take the coarser bin and
    # the longer range of the pair, so neither dye is shown beyond where its
    # own counts support it.
    shared = (S.groupby(["conc_nM", "power_mW", "dwell"])
              .agg(shared_floor_frames=("floor_frames", "max"),
                   shared_bin_frames=("bin_frames", "max"),
                   shared_x_max_s=("x_max_display_s", "max"),
                   n_dyes=("fluorogen", "nunique")).reset_index())
    shared["shared_n_bins"] = (
        (shared.shared_x_max_s / FT - shared.shared_floor_frames)
        / shared.shared_bin_frames).round().astype(int)
    S = S.merge(shared, on=["conc_nM", "power_mW", "dwell"], how="left")
    S.to_csv(os.path.join(TAB, "plot_specification.csv"), index=False)
    return S


def build():
    Cd, ON, OFFI, OFFB = cascade()
    S = spec(ON, OFFI, OFFB)
    A.log(f"wrote dwell_filter_cascade.csv ({len(Cd)} rows) and "
          f"plot_specification.csv ({len(S)} rows)")
    return Cd, S


if __name__ == "__main__":
    build()
