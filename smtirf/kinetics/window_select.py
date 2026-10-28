#!/usr/bin/env python
"""Step 3 of the consolidated workflow: choose the analysis window.

WHY THIS STEP EXISTS
--------------------
Two things vary between acquisitions for reasons unrelated to the quantity being
measured. Each field was illuminated before recording to bring the density of
emitting molecules down to where single molecules resolve, and that exposure was
longer where more molecules were occupied, so it varies alongside concentration.
And occupancy is not stationary within an acquisition: it falls 5- to 45-fold
over 250 s in every file. These are one problem seen from two sides -- the
occupied population is decaying, and acquisitions did not all start at the same
point on that decay.

The fix is to analyse one window inside each acquisition, chosen by a rule that
puts every acquisition at a comparable point on its own relaxation, and to
extract dwells from that window only.

THE THREE CANDIDATE RULES
-------------------------
A  matched occupancy   window starts where occupancy first falls to OCC_FRAC of
                       that acquisition's own initial occupancy. Aligns by how
                       far the relaxation has run, which is the thing that
                       differs between acquisitions.
B  matched clock time  window starts at a fixed frame for every acquisition.
                       Simpler, weaker: acquisitions entered the decay at
                       different points, so a fixed offset does not align them.
C  early window        a short window at the very start. Closest to the
                       pre-illumination state and so most comparable with bulk,
                       but the fewest events and the worst statistics.

A and B are given the SAME window length so they differ only in alignment; C is
necessarily shorter. The literal specification for B -- several times the
longest fitted unbound lifetime -- is not achievable: the longest tau_off is
about 107 s, so three times it is 3208 frames against a 2500-frame acquisition.
B is therefore implemented at the largest offset that leaves A's window length
intact, which is the latest start any acquisition receives under A. That makes
A and B the same total observation time at the same worst-case offset, differing
only in whether the start is aligned to the relaxation or to the clock.

Events are clipped to the window and re-based to window coordinates. An event
straddling the window start has an unobserved beginning, so it is dropped from
the on-time set rather than entered with a false duration; an event straddling
the window end is right-censored, exactly as at the acquisition boundary.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A  # noqa: E402

TAB = A.TAB

# ---- parameters of step 3 ----------------------------------------------------
# The window parameters live in analyse_smtirf.py with every other analysis
# choice, so that one parameter block states what the pipeline in force does.
OCC_FRAC = A.WINDOW_OCC_FRAC
EARLY_LEN = A.WINDOW_EARLY_LEN
MIN_WINDOW = 200       # refuse to analyse a window shorter than this. Rule C is
                       # deliberately short (EARLY_LEN), so this guards only against
                       # a DEGENERATE window -- the zero-length case that arises when
                       # an acquisition never reaches the occupancy threshold.
MIN_EVENTS_TRUST = 3   # complete dwells a trace must contribute to be graded
                       # informative about the lifetime
RULES = ("A_matched_occupancy", "B_matched_clock", "C_early")
ADOPTED = A.WINDOW_RULE  # chosen on simulated recovery: the most
                       # CONSISTENT across conditions (spread 1.32x against
                       # 1.59-1.85x), which is what matters because a constant
                       # multiplicative bias cancels in a ratio or a trend.
                       # It runs about 10% low in absolute terms.


def initial_occupancy(occ, n=10):
    """Occupancy at the start of an acquisition, averaged over n frames."""
    return float(np.mean(occ[:n]))


def build_windows(curves, occ_frac=OCC_FRAC, early_len=EARLY_LEN,
                  n_frames=A.N_FRAMES_EXPECTED, write=False):
    """Window [w0, w1) per acquisition under each rule.

    The common window length is set by the acquisition that takes longest to
    reach the occupancy threshold, so that every acquisition gets the same
    observation time and the fits are not weighted by exposure.
    """
    starts, never = {}, []
    for key, occ in curves.items():
        init = initial_occupancy(occ)
        thr = occ_frac * init
        below = np.flatnonzero(occ <= thr)
        if below.size:
            starts[key] = int(below[0])
        else:
            # The acquisition never relaxes to the threshold. That is not a
            # late start -- it means there is no relaxation to skip, so the
            # whole acquisition is already at a comparable point and the window
            # opens at frame 0. Treating it as a start of n_frames would give a
            # zero-length window and silently poison every other acquisition,
            # because the common length is set by the latest start.
            starts[key] = 0
            never.append(key)
    win_len = int(n_frames - max(starts.values()))
    clock_start = int(max(starts.values()))
    if win_len < MIN_WINDOW:
        raise ValueError(
            f"common window is {win_len} frames, below MIN_WINDOW={MIN_WINDOW}: "
            f"the latest threshold crossing is frame {clock_start}. Raise "
            f"OCC_FRAC (currently {occ_frac}) so windows open earlier, or "
            f"analyse the slowest acquisitions separately.")
    if never:
        A.log(f"build_windows: {len(never)} acquisition(s) never reach "
              f"{occ_frac:g}x initial occupancy; window opens at frame 0 for them")
    rows = []
    for key, occ in curves.items():
        fl, c, pw = key.split("|")
        init = initial_occupancy(occ)
        for rule in RULES:
            if rule == "A_matched_occupancy":
                w0 = starts[key]
                w1 = w0 + win_len
            elif rule == "B_matched_clock":
                w0, w1 = clock_start, clock_start + win_len
            else:
                w0, w1 = 0, early_len
            w1 = min(w1, n_frames)
            seg = occ[w0:w1]
            rows.append(dict(
                rule=rule, key=key, fluorogen=fl, conc_nM=float(c),
                power_mW=float(pw), w0=w0, w1=w1, win_frames=w1 - w0,
                win_s=(w1 - w0) * A.FRAME_TIME_S,
                occ_initial=init, occ_at_w0=float(occ[w0]),
                occ_window_mean=float(seg.mean()) if len(seg) else np.nan,
                occ_ratio_to_initial=float(occ[w0]) / init if init > 0 else np.nan,
                # residual non-stationarity inside the window: fold change from
                # the first tenth of the window to the last tenth
                residual_fold=float(seg[:max(len(seg)//10, 1)].mean() /
                                    seg[-max(len(seg)//10, 1):].mean())
                if len(seg) >= 20 and seg[-max(len(seg)//10, 1):].mean() > 0 else np.nan))
    W = pd.DataFrame(rows)
    # Writing the canonical table is OPT-IN, and only stage_apply does it.
    # Inferring it from the parameters was not enough: the validation stage
    # builds windows from SIMULATED curves at the default threshold, which
    # silently replaced the real acquisitions in window_definitions.csv and
    # crashed the figure on a key that no longer existed. A scan over thresholds
    # had done the same thing one level up. The table now changes only when the
    # analysis that uses it says so.
    if write:
        W.to_csv(os.path.join(TAB, "window_definitions.csv"), index=False)
    return W


def apply_window(ev, windows, n_frames=A.N_FRAMES_EXPECTED):
    """Clip an event table to each acquisition's window and re-base to window
    coordinates.

    Returns (clipped events, per-file window length). An event straddling w0 is
    DROPPED: its start was not observed, so its duration is unknown and entering
    it would understate the dwell. An event straddling w1 is kept and flagged
    end-truncated, which the likelihood already handles as right-censoring.
    """
    key = ev.fluorogen.astype(str) + "|" + ev.conc_nM.astype(str) + "|" + \
        ev.power_mW.astype(str)
    w = windows.set_index("key")[["w0", "w1"]]
    # tolerate float/str key formatting differences
    idx = pd.Index(key).map(lambda k: k if k in w.index else _match_key(k, w.index))
    w0 = w.w0.reindex(idx).values
    w1 = w.w1.reindex(idx).values
    e = ev.copy()
    e["_w0"], e["_w1"] = w0, w1
    keep = (e.end > e._w0) & (e.start < e._w1) & np.isfinite(e._w0)
    e = e[keep].copy()
    straddles_start = e.start < e._w0
    e = e[~straddles_start].copy()
    e["end_trunc"] = e.end_trunc.astype(bool) | (e.end > e._w1)
    e["end"] = np.minimum(e.end, e._w1)
    e["start"] = e.start - e._w0
    e["end"] = e["end"] - e._w0
    e["duration"] = e.end - e.start
    e = e[e.duration >= 1].drop(columns=["_w0", "_w1"])
    lengths = dict(zip(windows.key, windows.win_frames))
    return e, lengths


def load_events():
    """Every event, from the per-file caches written by the main pipeline."""
    import glob
    import pickle
    evs = []
    for p in sorted(glob.glob(os.path.join(A.HERE, "cache", "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        if d.get("ev") is not None and len(d["ev"]):
            evs.append(d["ev"])
    return pd.concat(evs, ignore_index=True)


def load_curves():
    z = np.load(os.path.join(TAB, "prebleach_occupancy_curves.npz"))
    return {k: z[k] for k in z.files}


def fit_window(ev_win, win_frames, n_boot=0, seed=0):
    """Steps 4 and 5 applied to a windowed event table. The dwell extraction,
    truncation limits, censoring rules and likelihood are UNCHANGED -- only the
    frames the events come from differ."""
    assert win_frames >= MIN_WINDOW, (
        f"window of {win_frames} frames is below MIN_WINDOW={MIN_WINDOW}; a "
        "degenerate window produces negative boundary gaps and a likelihood "
        "that does not converge")
    ON, OFF, BND = A.extract_dwells(ev_win, A.MAX_GAP, n_frames=win_frames,
                                    edge_correction=A.EDGE_CORRECTION)
    assert (BND.frames >= 0).all(), "negative boundary gap: events outside the window"
    dmin_off = max(A.MIN_DUR, A.MAX_GAP + 1)
    rows = []
    for (fl, c, pw), g in ON.groupby(["fluorogen", "conc_nM", "power_mW"]):
        oi = OFF[(OFF.fluorogen == fl) & (OFF.conc_nM == c) & (OFF.power_mW == pw)]
        ob = BND[(BND.fluorogen == fl) & (BND.conc_nM == c) & (BND.power_mW == pw)]
        cm = g[(~g.censored) & (g.frames >= A.MIN_DUR)]
        cx = g[g.censored & (g.frames >= A.MIN_DUR)]
        d_on, k_on_, dc_on, kc_on = A.apply_upper(
            cm.frames.values.astype(float), cm.key.values,
            cx.frames.values.astype(float), cx.key.values, A.MAX_DUR)
        if d_on.size < 30:
            continue
        son = A.fit_dwell_set(d_on, k_on_, dc_on, kc_on, A.MIN_DUR,
                              n_boot=n_boot, seed=seed)
        do, ko, dco, kco, dmin_used = A.off_arrays(oi, ob, dmin_off)
        soff = (A.fit_dwell_set(do, ko, dco, kco, dmin_used, dmin_c=1,
                                n_boot=n_boot, seed=seed)
                if do.size >= 30 else None)
        r = dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                 n_molecules=int(pd.unique(g.key).size),
                 on_n_complete=int(d_on.size), on_n_censored=int(dc_on.size),
                 on_tau_single_s=son["tau_single"] * A.FRAME_TIME_S,
                 on_tau_lo_s=son.get("tau_single_lo", np.nan) * A.FRAME_TIME_S,
                 on_tau_hi_s=son.get("tau_single_hi", np.nan) * A.FRAME_TIME_S,
                 on_tau_fast_s=son["tau_fast"] * A.FRAME_TIME_S,
                 on_tau_slow_s=son["tau_slow"] * A.FRAME_TIME_S,
                 on_a_fast=son["a_fast"], on_dBIC=son["dBIC"],
                 k_off_per_s=1.0 / (son["tau_single"] * A.FRAME_TIME_S))
        if soff is not None:
            r.update(off_n=int(do.size),
                     off_tau_single_s=soff["tau_single"] * A.FRAME_TIME_S,
                     off_tau_lo_s=soff.get("tau_single_lo", np.nan) * A.FRAME_TIME_S,
                     off_tau_hi_s=soff.get("tau_single_hi", np.nan) * A.FRAME_TIME_S,
                     off_dBIC=soff["dBIC"],
                     inv_tau_off_per_s=1.0 / (soff["tau_single"] * A.FRAME_TIME_S))
        rows.append(r)
    return pd.DataFrame(rows), ON, OFF, BND


def occupancy_curve_from_events(ev, n_mol, n_frames=A.N_FRAMES_EXPECTED):
    """Occupancy per frame, from event boundaries. Same reconstruction used on
    the real data, so the window rules see the same kind of input."""
    delta = np.zeros(n_frames + 2)
    np.add.at(delta, np.clip(ev.start.values.astype(int), 0, n_frames), 1)
    np.add.at(delta, np.clip(ev.end.values.astype(int), 0, n_frames), -1)
    return np.cumsum(delta)[:n_frames] / max(n_mol, 1)


def simulate_set(conditions, n_mol, seed):
    """Simulate one set of acquisitions and return (events, occupancy curves).

    Factored out so a parameter scan can reuse a single set of traces: the
    question in a scan is what the RULE does to fixed data, so re-simulating per
    parameter value would add sampling noise to the very comparison being made.
    """
    import simulate_validate as S
    evs, curves_sim = {}, {}
    for cond in conditions:
        name = cond["name"]
        _, ev = S.run_condition(
            cond["k_on_eff"], cond["k_off"], n_mol=n_mol, seed=seed,
            k_bleach=cond.get("k_bleach", 0.0),
            k_site_loss=cond.get("k_site_loss", 0.0),
            site_persist=cond.get("site_persist", 0.0),
            t_offset=cond.get("t_offset", 0.0))
        if ev is None or not len(ev):
            continue
        ev = ev.copy()
        ev["fluorogen"], ev["conc_nM"], ev["power_mW"] = name, 1.0, 1.5
        evs[name] = ev
        curves_sim[f"{name}|1.0|1.5"] = occupancy_curve_from_events(ev, n_mol)
    if not evs:
        return None, None
    return pd.concat(evs.values(), ignore_index=True), curves_sim


SIM_N_MOL_DEFAULT = 4000
OCC_FRAC_SCAN = (0.10, 0.15, 0.25, 0.40, 0.60)


def scan_occ_frac_sim(conditions=None, n_mol=SIM_N_MOL_DEFAULT, seeds=(11, 23, 37),
                      fracs=OCC_FRAC_SCAN):
    """Recovery of known rates as a function of the one free parameter in step 3.

    OCC_FRAC is the only number in the window rule that was not forced by the
    data, so it is scanned rather than asserted. The traces are simulated ONCE
    per seed and every threshold is applied to the same traces, so differences
    between thresholds are the rule's doing and not sampling.
    """
    conditions = SIM_CONDITIONS if conditions is None else conditions
    truth = {c["name"]: c for c in conditions}
    rows = []
    for seed in seeds:
        ev_all_sim, curves_sim = simulate_set(conditions, n_mol, seed)
        if ev_all_sim is None:
            continue
        A.log(f"  scan seed {seed} simulated")
        for frac in fracs:
            try:
                Wd = build_windows(curves_sim, occ_frac=frac)
            except ValueError as e:              # degenerate window at this frac
                rows.append(dict(seed=seed, occ_frac=frac, condition=None,
                                 failed=str(e)[:80]))
                continue
            w = Wd[Wd.rule == "A_matched_occupancy"]
            ev_w, _ = apply_window(ev_all_sim, w)
            k, *_ = fit_window(ev_w, int(w.win_frames.iloc[0]))
            for _, r in k.iterrows():
                t = truth[r.fluorogen]
                rows.append(dict(
                    seed=seed, occ_frac=frac, condition=r.fluorogen, failed=None,
                    win_frames=int(w.win_frames.iloc[0]),
                    k_off_recovery=r.k_off_per_s / t["k_off"],
                    tau_off_recovery=r.get("off_tau_single_s", np.nan) * t["k_on_eff"],
                    n_complete=r.on_n_complete))
    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(TAB, "window_occfrac_scan_sim.csv"), index=False)
    return V


def scan_occ_frac_real(fracs=OCC_FRAC_SCAN):
    """The same scan on the real data: do the reported conclusions depend on it?"""
    ev_all, curves = load_events(), load_curves()
    rows = []
    for frac in fracs:
        try:
            Wd = build_windows(curves, occ_frac=frac)
        except ValueError as e:
            A.log(f"  occ_frac={frac}: {e}")
            continue
        w = Wd[Wd.rule == "A_matched_occupancy"]
        ev_w, _ = apply_window(ev_all, w)
        k, *_ = fit_window(ev_w, int(w.win_frames.iloc[0]))
        k["occ_frac"] = frac
        k["win_frames"] = int(w.win_frames.iloc[0])
        k = k.merge(w[["fluorogen", "conc_nM", "power_mW", "w0", "residual_fold"]],
                    on=["fluorogen", "conc_nM", "power_mW"], how="left")
        rows.append(k)
        A.log(f"  occ_frac={frac}: window {int(w.win_frames.iloc[0])} frames")
    K = pd.concat(rows, ignore_index=True)
    K.to_csv(os.path.join(TAB, "window_occfrac_scan_real.csv"), index=False)

    # do the CONCLUSIONS move, not just the numbers?
    summ = []
    for frac, g in K[K.power_mW == A.DEFAULT_POWER_MW].groupby("occ_frac"):
        pv = g.pivot_table(index="conc_nM", columns="fluorogen", values="k_off_per_s")
        sal = pv["SALAD1"].dropna()
        to = pv["TO1-B"].dropna()
        summ.append(dict(
            occ_frac=frac, win_frames=int(g.win_frames.iloc[0]),
            residual_fold_median=float(g.residual_fold.median()),
            k_off_SALAD_lo=sal.min(), k_off_SALAD_hi=sal.max(),
            SALAD_fold_range=sal.max() / sal.min(),
            TO_fold_range=to.max() / to.min(),
            ratio_at_lowest_c=(sal.iloc[0] / to.iloc[0]) if len(sal) and len(to) else np.nan,
            SALAD_rises=bool(sal.iloc[-1] > sal.iloc[0])))
    S_ = pd.DataFrame(summ)
    S_.to_csv(os.path.join(TAB, "window_occfrac_conclusions.csv"), index=False)
    return K, S_


def validate_rules(conditions, n_mol=1200, seed=11, out_tab=None):
    """Simulate acquisitions with KNOWN rates, a decaying occupied population and
    a variable start offset, then apply each candidate rule and ask whether the
    input rates come back.

    A rule that does not recover known rates is not a fix however reasonable it
    sounds, so this runs before any rule is adopted. The truth is the per-site
    k_off and k_on used to generate the traces; recovery is fitted / true.
    """
    import simulate_validate as S
    rows, curves_sim, evs = [], {}, {}
    for cond in conditions:
        name = cond["name"]
        A.log(f"  simulating {name}")
        summ, ev = S.run_condition(
            cond["k_on_eff"], cond["k_off"], n_mol=n_mol, seed=seed,
            k_bleach=cond.get("k_bleach", 0.0),
            k_site_loss=cond.get("k_site_loss", 0.0),
            site_persist=cond.get("site_persist", 0.0),
            t_offset=cond.get("t_offset", 0.0))
        if ev is None or not len(ev):
            continue
        ev = ev.copy()
        ev["fluorogen"], ev["conc_nM"], ev["power_mW"] = name, 1.0, 1.5
        evs[name] = ev
        curves_sim[f"{name}|1.0|1.5"] = occupancy_curve_from_events(ev, n_mol)
    if not evs:
        return pd.DataFrame()

    ev_all_sim = pd.concat(evs.values(), ignore_index=True)
    Wsim = build_windows(curves_sim)
    truth = {c["name"]: c for c in conditions}

    for rule in RULES + ("whole_acquisition",):
        if rule == "whole_acquisition":
            ev_w, win = ev_all_sim, A.N_FRAMES_EXPECTED
        else:
            w = Wsim[Wsim.rule == rule]
            ev_w, _ = apply_window(ev_all_sim, w)
            win = int(w.win_frames.iloc[0])
        A.log(f"  fitting rule {rule}")
        k, *_ = fit_window(ev_w, win)
        for _, r in k.iterrows():
            t = truth[r.fluorogen]
            rows.append(dict(
                rule=rule, condition=r.fluorogen,
                k_off_true=t["k_off"], k_off_fit=r.k_off_per_s,
                k_off_recovery=r.k_off_per_s / t["k_off"],
                tau_on_true_s=1.0 / t["k_off"], tau_on_fit_s=r.on_tau_single_s,
                tau_off_true_s=1.0 / t["k_on_eff"],
                tau_off_fit_s=r.get("off_tau_single_s", np.nan),
                tau_off_recovery=(r.get("off_tau_single_s", np.nan) *
                                  t["k_on_eff"]),
                k_site_loss=t.get("k_site_loss", 0.0),
                t_offset=t.get("t_offset", 0.0),
                n_complete=r.on_n_complete, n_molecules=r.n_molecules))
    V = pd.DataFrame(rows)
    if out_tab:
        V.to_csv(out_tab, index=False)
        Wsim.to_csv(out_tab.replace(".csv", "_windows.csv"), index=False)
    return V


def per_trace_diagnostics(ev_win, windows, win_frames, mol=None):
    """Is the adopted window defensible for EACH trace, and how far can each be
    trusted?

    A window rule chosen on the population average can still be wrong for an
    individual molecule, so every trace is checked on four things, each of which
    is a reason a trace could fail to support the fit:

      stationary   the molecule's own event rate does not differ between the
                   first and second half of the window. Tested per molecule with
                   a two-sided binomial test against an even split. This has
                   little power at small event counts, which is itself reported
                   rather than hidden: a molecule with 3 events cannot be shown
                   to be non-stationary, so it is graded 'untested', not 'pass'.
      informative  contributes at least MIN_EVENTS_TRUST complete dwells to the
                   window, so it carries information about the lifetime.
      resolved     bound-to-background contrast is at least CONTRAST_MIN, so its
                   events are real transitions rather than noise excursions.
      uncensored   less than half its dwells are censored at a window edge, so
                   its contribution is not dominated by bounds.

    The grade is deliberately conservative: 'untested' never counts as a pass.
    """
    from scipy import stats as st
    half = win_frames / 2.0
    g = ev_win.assign(key=ev_win.file.astype(str) + "|" + ev_win.mol_idx.astype(str))
    rows = []
    for key, e in g.groupby("key", sort=False):
        n = len(e)
        n_first = int((e.start < half).sum())
        p = (float(st.binomtest(n_first, n, 0.5).pvalue) if n >= 1 else np.nan)
        n_complete = int((~e.end_trunc.astype(bool) & (e.duration >= A.MIN_DUR)).sum())
        n_cens = int(e.end_trunc.astype(bool).sum())
        rows.append(dict(key=key, file=e.file.iloc[0], mol_idx=e.mol_idx.iloc[0],
                         fluorogen=e.fluorogen.iloc[0], conc_nM=e.conc_nM.iloc[0],
                         power_mW=e.power_mW.iloc[0], n_events_window=n,
                         n_first_half=n_first, n_complete=n_complete,
                         n_censored=n_cens, split_p=p,
                         mean_dur_window=float(e.duration.mean())))
    D = pd.DataFrame(rows)
    if mol is not None and len(mol):
        m = mol.assign(key=mol.file.astype(str) + "|" + mol.mol_idx.astype(str))
        D = D.merge(m[["key", "contrast", "brightness", "background_sd"]],
                    on="key", how="left")
    # power to detect a half-window imbalance: the smallest count at which a
    # 2:1 split would reach p<0.05
    D["testable"] = D.n_events_window >= 8
    D["stationary"] = np.where(~D.testable, "untested",
                               np.where(D.split_p >= 0.05, "pass", "fail"))
    D["informative"] = D.n_complete >= MIN_EVENTS_TRUST
    D["resolved"] = (D.contrast >= A.CONTRAST_VARIANT
                     if "contrast" in D.columns else True)
    D["uncensored"] = D.n_censored < 0.5 * D.n_events_window
    D["trust"] = np.select(
        [(D.stationary == "fail"),
         (~D.resolved) | (~D.uncensored),
         (D.stationary == "pass") & D.informative,
         (D.stationary == "untested") & D.informative],
        ["low (non-stationary)", "low (unresolved or censored)",
         "high", "moderate (too few events to test)"],
        default="moderate (too few events to test)")
    return D


def _match_key(k, index):
    """Key formatting is not guaranteed identical between the curve archive and
    the event table (1.0 vs 1); match on the numeric triple instead."""
    try:
        fl, c, pw = k.split("|")
        for cand in index:
            f2, c2, p2 = cand.split("|")
            if f2 == fl and float(c2) == float(c) and float(p2) == float(pw):
                return cand
    except Exception:
        pass
    return np.nan


# ============================== driver ======================================
# The conditions below span the decay rates and start offsets the DATA show
# (relaxation time constants of 11-56 s, a persistent fraction of 4-17%, and
# untreated fields entering the decay at t=0 against pre-bleached ones entering
# later). They are the ground truth the candidate rules are scored against.
SIM_CONDITIONS = [
    dict(name="clean_slow", k_on_eff=0.021, k_off=0.35, k_site_loss=0.018,
         site_persist=0.17, t_offset=0),
    dict(name="clean_fast", k_on_eff=0.034, k_off=0.35, k_site_loss=0.060,
         site_persist=0.10, t_offset=0),
    dict(name="treat_mid", k_on_eff=0.052, k_off=0.35, k_site_loss=0.040,
         site_persist=0.06, t_offset=30),
    dict(name="treat_fast", k_on_eff=0.065, k_off=0.35, k_site_loss=0.090,
         site_persist=0.04, t_offset=45),
    dict(name="treat_bleach", k_on_eff=0.052, k_off=0.35, k_site_loss=0.040,
         site_persist=0.06, t_offset=30, k_bleach=0.40),
    dict(name="no_decay", k_on_eff=0.034, k_off=0.35, k_site_loss=0.0,
         site_persist=1.0, t_offset=0),
]
SIM_SEEDS = (11, 23, 37)
SIM_N_MOL = SIM_N_MOL_DEFAULT


def stage_validate(n_mol=SIM_N_MOL, seeds=SIM_SEEDS):
    """Score the candidate rules against known rates. Several seeds at high n:
    this project has twice mistaken simulation noise on a marginal criterion for
    a real difference, so a rule is not adopted until the between-condition
    spread is clearly larger than the seed-to-seed spread."""
    outs = []
    for seed in seeds:
        v = validate_rules(SIM_CONDITIONS, n_mol=n_mol, seed=seed)
        v["seed"] = seed
        outs.append(v)
        A.log(f"  seed {seed} done")
    V = pd.concat(outs, ignore_index=True)
    V.to_csv(os.path.join(TAB, "window_rule_validation.csv"), index=False)
    m = V.groupby(["rule", "condition"]).k_off_recovery.mean().unstack(1)
    sd = V.groupby(["rule", "condition"]).k_off_recovery.std().groupby("rule").mean()
    S = pd.DataFrame(dict(mean_recovery=m.mean(axis=1),
                          across_condition_sd=m.std(axis=1),
                          across_condition_range=m.max(axis=1) / m.min(axis=1),
                          worst_condition=m.min(axis=1), best_condition=m.max(axis=1),
                          seed_sd=sd))
    S["bias_vs_noise"] = S.across_condition_sd / S.seed_sd
    S.index.name = "rule"
    S.to_csv(os.path.join(TAB, "window_rule_scores.csv"))
    return V, S


def stage_apply(n_boot=None):
    """Apply the adopted rule to the real data and refit. Steps 4 and 5 are
    unchanged; only the frames the dwells come from differ."""
    ev_all, curves = load_events(), load_curves()
    Wd = build_windows(curves, write=True)      # the window set actually in force
    w = Wd[Wd.rule == ADOPTED]
    ev_w, _ = apply_window(ev_all, w)
    win = int(w.win_frames.iloc[0])
    A.log(f"{ADOPTED}: {len(ev_w):,} of {len(ev_all):,} events, window {win} frames")
    K, ON, OFF, BND = fit_window(ev_w, win,
                                 n_boot=A.N_BOOT if n_boot is None else n_boot,
                                 seed=A.SEED)
    K["rule"], K["win_frames"] = ADOPTED, win
    K = K.merge(w[["fluorogen", "conc_nM", "power_mW", "w0", "w1", "occ_at_w0",
                   "occ_initial", "occ_window_mean", "residual_fold"]],
                on=["fluorogen", "conc_nM", "power_mW"], how="left")
    K.to_csv(os.path.join(TAB, "kinetics_windowed.csv"), index=False)

    # all three rules, for the candidate comparison table
    cand = []
    for rule in RULES:
        ww = Wd[Wd.rule == rule]
        ew, _ = apply_window(ev_all, ww)
        kk, *_ = fit_window(ew, int(ww.win_frames.iloc[0]))
        cand.append(kk.assign(rule=rule))
    pd.concat(cand, ignore_index=True).to_csv(
        os.path.join(TAB, "window_candidate_kinetics.csv"), index=False)

    # what moved, against the whole-acquisition analysis
    whole = pd.read_csv(os.path.join(TAB, "kinetics_per_condition.csv"))[
        ["fluorogen", "conc_nM", "power_mW", "on_tau_single_s", "off_tau_single_s",
         "k_off_per_s"]]
    mv = K[K.power_mW == A.DEFAULT_POWER_MW].merge(
        whole, on=["fluorogen", "conc_nM", "power_mW"], suffixes=("_win", "_whole"))
    mv["tau_on_ratio"] = mv.on_tau_single_s_win / mv.on_tau_single_s_whole
    mv["k_off_ratio"] = mv.k_off_per_s_win / mv.k_off_per_s_whole
    mv["tau_off_ratio"] = mv.off_tau_single_s_win / mv.off_tau_single_s_whole
    mv.to_csv(os.path.join(TAB, "window_what_moved.csv"), index=False)

    # per-trace justification, and whether the weak traces drive the answer
    mol = pd.read_csv(os.path.join(TAB, "brightness_per_molecule.csv"))
    D = per_trace_diagnostics(ev_w, w, win, mol=mol)
    D.to_csv(os.path.join(TAB, "window_per_trace_trust.csv"), index=False)
    ev_w = ev_w.assign(key=ev_w.file.astype(str) + "|" + ev_w.mol_idx.astype(str))
    subsets = {"all traces (headline)": D.key,
               "high trust only": D.loc[D.trust == "high", "key"],
               "high + moderate (drop failures)":
                   D.loc[~D.trust.str.startswith("low"), "key"],
               "drop non-stationary only":
                   D.loc[D.trust != "low (non-stationary)", "key"]}
    rows = []
    for name, keys in subsets.items():
        kk, *_ = fit_window(ev_w[ev_w.key.isin(set(keys))], win)
        rows.append(kk[kk.power_mW == A.DEFAULT_POWER_MW].assign(subset=name))
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(TAB, "window_trust_sensitivity.csv"), index=False)
    return K, D


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="validate,apply,scan",
                    help="comma-separated: validate, apply, scan")
    ap.add_argument("--n-mol", type=int, default=SIM_N_MOL)
    args = ap.parse_args()
    stages = {s.strip() for s in args.stage.split(",")}
    t0 = time.time()
    if "validate" in stages:
        A.log("window: scoring candidate rules against known rates")
        stage_validate(n_mol=args.n_mol)
    if "apply" in stages:
        A.log("window: applying the adopted rule to the real data")
        stage_apply()
    if "scan" in stages:
        A.log("window: scanning the occupancy threshold on the real data")
        scan_occ_frac_real()
        A.log("window: scanning the occupancy threshold in simulation")
        scan_occ_frac_sim(n_mol=args.n_mol)
    A.log(f"window_select done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    # MUST be guarded: simulate_validate uses multiprocessing with the spawn
    # start method, so every worker re-imports this module. Without the guard
    # each worker relaunches the whole simulation and the run deadlocks with no
    # output at all.
    main()
