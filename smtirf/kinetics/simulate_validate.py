"""Ground-truth validation of the smTIRF dwell-time estimators.

What this does
--------------
Generates synthetic intensity traces from a KNOWN two-state binding process at
the experimental frame time, brightness and noise level, optionally with
fluorophore blinking and photobleaching injected; idealises them with a replica
of the change-point idealisation (ruptures KernelCPD, linear kernel, PELT,
minimum segment size 1, penalty 10, length-weighted Otsu two-level split,
3-sigma acceptance); and runs the same estimators used on the real data. The
recovered rates are then compared with the values put in.

This converts the estimators from asserted to demonstrated, and separates
properties of the sample from artefacts of the detection chain. In particular it
answers three questions the real data cannot answer on their own:

  Q1  Does a genuinely single-exponential dwell distribution get reported as
      two-component after passing through detection at these efficiencies?
  Q2  Does blinking at the measured rate generate a non-zero intercept in
      1/tau_off against concentration, and how much of the measured intercept
      can it account for?
  Q3  Does the apparent k_off drift with concentration when the true k_off is
      held fixed and only background and noise scale as measured?

Run:  python simulate_validate.py            (all experiments)
      python simulate_validate.py --quick    (fewer molecules, for a smoke test)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import ruptures as rpt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_smtirf as A            # frame-discretised MLE, dwell extraction

TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)

# ------------------------------- parameters --------------------------------
DT = A.FRAME_TIME_S
N_FRAMES = A.N_FRAMES_EXPECTED
PENALTY = 10.0                 # CPD penalty used by the trace export
MIN_SIZE = 1                   # minimum segment size, frames
CONTRAST_SIGMA = 3.0           # binder acceptance threshold

N_MOL = 900                    # molecules per simulated condition
SEED = 7

# reference condition, taken from the measured tables (SALAD1, 0.5 nM, 1.5 mW)
REF_BRIGHT = 321.0             # bound-state brightness, counts
REF_NOISE = 50.3               # background standard deviation, counts
REF_BG = 5.4                   # residual background offset, counts

# blinking: a bound dye goes dark and returns. The return rate sets the gap
# LENGTH (mean dark time ~ 1/k_return); the darkening rate sets how OFTEN.
BLINK_RETURN = 5.0             # s^-1  -> mean dark interval 0.2 s = 2 frames
BLINK_DARK_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)   # s^-1


def weighted_otsu(values, weights):
    """Length-weighted Otsu threshold on segment means."""
    order = np.argsort(values)
    v, w = values[order], weights[order].astype(float)
    wsum = w.sum()
    if wsum <= 0 or v.size < 2:
        return np.inf
    cw = np.cumsum(w)[:-1]
    cm = np.cumsum(w * v)[:-1]
    total = (w * v).sum()
    w0, w1 = cw, wsum - cw
    m0 = cm / np.maximum(w0, 1e-12)
    m1 = (total - cm) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.argmax(between))
    return 0.5 * (v[k] + v[k + 1])


def simulate_molecule(rng, k_on_eff, k_off, bright=REF_BRIGHT, noise=REF_NOISE,
                      bg=REF_BG, k_blink_dark=0.0, k_blink_return=BLINK_RETURN,
                      k_bleach=0.0, n_frames=N_FRAMES, dt=DT,
                      k_site_loss=0.0, site_persist=0.0, t_offset=0.0):
    """One molecule. Returns (trace, true occupancy intervals in seconds).

    Occupancy is a continuous-time two-state Markov chain, so transitions fall
    mid-frame and produce the partial-intensity edge frames seen in real data.
    Fluorophore blinking and bleaching act WITHIN an occupancy episode; a
    bleached dye stays dark until the complex dissociates, after which a fresh
    dye from the free pool can bind. That is the exchange behaviour of the tag.
    """
    T = n_frames * dt
    # Site inactivation: the observed occupancy relaxation is a loss of the
    # binding-competent population, not a change in the per-site rates. A
    # fraction `site_persist` of sites never inactivate; the rest inactivate at
    # `k_site_loss`. `t_offset` is how far into that decay the acquisition
    # starts, which is what the pre-acquisition illumination sets and what
    # differs between fields. Inactivation ENDS any episode in progress, which
    # is what truncates bound dwells.
    if k_site_loss > 0 and rng.random() >= site_persist:
        t_death = rng.exponential(1.0 / k_site_loss) - t_offset
        if t_death <= 0:
            return bg + rng.normal(0, noise, n_frames), []
        T = min(T, t_death)
    # Start AT EQUILIBRIUM. The sample sits in dye solution in the dark before
    # recording, so a fraction k_on/(k_on+k_off) of sites is already occupied at
    # frame 0 and the acquisition opens on a decaying transient. Starting every
    # molecule unbound would build in a RISE to steady state instead, which is
    # the opposite of what the data show and would make the window rules face
    # the wrong problem.
    theta_eq = k_on_eff / (k_on_eff + k_off)
    bound = bool(rng.random() < theta_eq)
    occ, t = [], 0.0
    start = 0.0
    while t < T:
        if not bound:
            t += rng.exponential(1.0 / k_on_eff)
            if t >= T:
                break
            start, bound = t, True
        else:
            t += rng.exponential(1.0 / k_off)
            occ.append((start, min(t, T)))
            bound = False

    emit = []
    for a, b in occ:
        tt, fluor = a, True
        while tt < b:
            if fluor:
                r = k_blink_dark + k_bleach
                if r <= 0:
                    emit.append((tt, b))
                    break
                nxt = tt + rng.exponential(1.0 / r)
                emit.append((tt, min(nxt, b)))
                if nxt >= b:
                    break
                if rng.random() < k_bleach / r:
                    break                       # bleached: dark until dissociation
                fluor, tt = False, nxt
            else:
                if k_blink_return <= 0:
                    break
                nxt = tt + rng.exponential(1.0 / k_blink_return)
                if nxt >= b:
                    break
                fluor, tt = True, nxt

    # exact per-frame emitting fraction
    frac = np.zeros(n_frames)
    for a, b in emit:
        i0, i1 = int(a / dt), min(int(np.ceil(b / dt)), n_frames)
        for i in range(i0, i1):
            lo, hi = i * dt, (i + 1) * dt
            frac[i] += max(0.0, min(b, hi) - max(a, lo)) / dt
    frac = np.clip(frac, 0, 1)
    trace = bg + bright * frac + rng.normal(0, noise, n_frames)
    return trace, occ


def idealise(trace, penalty=PENALTY, min_size=MIN_SIZE, contrast_sigma=CONTRAST_SIGMA):
    """Replica of the CPD idealiser of the trace export. Returns per-frame 0/1 states."""
    med = np.median(trace)
    sd = 1.4826 * np.median(np.abs(trace - med))
    if sd <= 0:
        return np.zeros(trace.size, int)
    z = (trace - med) / sd
    bkps = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(
        z.reshape(-1, 1)).predict(pen=penalty)
    edges = np.r_[0, np.asarray(bkps, int)]
    means = np.array([trace[edges[i]:edges[i + 1]].mean() for i in range(edges.size - 1)])
    lens = np.diff(edges).astype(float)
    if means.size < 2:
        return np.zeros(trace.size, int)
    thr = weighted_otsu(means, lens)
    lab = (means > thr).astype(int)
    states = np.zeros(trace.size, int)
    for i in range(edges.size - 1):
        states[edges[i]:edges[i + 1]] = lab[i]
    b0, b1 = trace[states == 0], trace[states == 1]
    if b1.size == 0 or b0.size < 2:
        return np.zeros(trace.size, int)
    if (b1.mean() - b0.mean()) < contrast_sigma * b0.std(ddof=1):
        return np.zeros(trace.size, int)       # rejected: no two-level structure
    return states


def states_to_events(states, mol_idx, n_frames=N_FRAMES):
    """Export conventions: End exclusive, StartTrunc/EndTrunc at the boundaries."""
    out = []
    if states.sum() == 0:
        return out
    d = np.diff(np.r_[0, states, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for s, e in zip(starts, ends):
        out.append(dict(file="sim", mol_idx=mol_idx, start=int(s), end=int(e),
                        duration=int(e - s), snr=np.nan,
                        start_trunc=bool(s == 0), end_trunc=bool(e == n_frames),
                        fluorogen="sim", conc_nM=np.nan, power_mW=np.nan))
    return out


def _one(args):
    (i, seed, kw) = args
    rng = np.random.default_rng(seed)
    trace, occ = simulate_molecule(rng, **kw)
    states = idealise(trace)
    ev = states_to_events(states, i)
    true_on = np.array([b - a for a, b in occ])
    # a true episode is "resolvable" if it covers at least one whole frame
    return ev, true_on, int(states.sum()), int(len(occ))


def _one_efficiency(args):
    """Per-episode detection outcome, for measuring where the lower truncation
    of the dwell-time fits should sit.

    Recovery is scored against the TRUE occupancy episodes of a trace that looks
    like a real one -- generated from that condition's own fitted rates,
    brightness and noise -- rather than against isolated events injected into a
    blank trace. That matters because the idealiser's two-level split and its
    3-sigma acceptance test are properties of the whole trace: a trace
    containing only 1-frame events would be accepted or rejected on different
    grounds than a real one, and would give the wrong efficiency.

    An episode counts as recovered if the idealiser called at least one bound
    frame overlapping the frames the episode covers.
    """
    (i, seed, kw) = args
    rng = np.random.default_rng(seed)
    trace, occ = simulate_molecule(rng, **kw)
    states = idealise(trace)
    dt = kw.get("dt", DT)
    nf = kw.get("n_frames", N_FRAMES)
    dur_s, rec = [], []
    for a, b in occ:
        f0 = int(np.floor(a / dt))
        f1 = int(np.ceil(b / dt))
        f0, f1 = max(f0, 0), min(max(f1, f0 + 1), nf)
        dur_s.append(b - a)
        rec.append(bool(states[f0:f1].any()))
    return np.array(dur_s), np.array(rec, dtype=bool)


def efficiency_run(k_on_eff, k_off, n_mol=N_MOL, seed=SEED, workers=None, **kw):
    """Detection efficiency as a function of TRUE episode duration."""
    kw = dict(k_on_eff=k_on_eff, k_off=k_off, **kw)
    tasks = [(i, seed * 1000003 + i, kw) for i in range(n_mol)]
    workers = workers or max(1, min(cpu_count() - 1, 7))
    with Pool(workers) as pool:
        res = pool.map(_one_efficiency, tasks, chunksize=8)
    dur = np.concatenate([r[0] for r in res]) if res else np.array([])
    rec = np.concatenate([r[1] for r in res]) if res else np.array([], bool)
    return dur, rec


def _wilson(k, n, z=1.96):
    """Wilson interval; the normal approximation is unusable near p = 1."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_condition(k_on_eff, k_off, n_mol=N_MOL, seed=SEED, workers=None, **kw):
    """Simulate, idealise and fit one condition. Returns (summary dict, events df)."""
    kw = dict(k_on_eff=k_on_eff, k_off=k_off, **kw)
    tasks = [(i, seed * 1000003 + i, kw) for i in range(n_mol)]
    workers = workers or max(1, min(cpu_count() - 1, 7))
    with Pool(workers) as pool:
        res = pool.map(_one, tasks, chunksize=8)
    events, true_on, bound_frames, n_true = [], [], 0, 0
    n_with_events = 0
    for ev, ton, bf, nt in res:
        events.extend(ev)
        true_on.append(ton)
        bound_frames += bf
        n_true += nt
        n_with_events += int(len(ev) > 0)
    ev = pd.DataFrame(events)
    true_on = np.concatenate(true_on) if true_on else np.array([])
    summ = dict(k_on_eff_true=k_on_eff, k_off_true=k_off,
                tau_on_true_s=1.0 / k_off, tau_off_true_s=1.0 / k_on_eff,
                n_molecules=n_mol, n_mol_with_events=n_with_events,
                n_true_episodes=n_true, n_detected_events=len(ev),
                detection_ratio=len(ev) / max(n_true, 1),
                bound_fraction_true=float(np.sum(true_on) / (n_mol * N_FRAMES * DT)),
                bound_fraction_observed=bound_frames / (n_mol * N_FRAMES))
    summ.update({k: v for k, v in kw.items() if k not in ("k_on_eff", "k_off")})
    if ev.empty:
        return summ, ev
    ON, OFF, BND = A.extract_dwells(ev, A.MAX_GAP, n_frames=N_FRAMES)
    dmin_off = max(A.MIN_DUR, A.MAX_GAP + 1)
    cm = ON[(~ON.censored) & (ON.frames >= A.MIN_DUR)]
    cx = ON[ON.censored & (ON.frames >= A.MIN_DUR)]
    r_on = A.fit_dwell_set(cm.frames.values.astype(float), cm.key.values,
                           cx.frames.values.astype(float), cx.key.values,
                           A.MIN_DUR, n_boot=100, seed=seed)
    oi = OFF[OFF.frames >= dmin_off]
    ob = BND[BND.frames >= 1]
    r_off = A.fit_dwell_set(oi.frames.values.astype(float), oi.key.values,
                            ob.frames.values.astype(float), ob.key.values,
                            dmin_off, dmin_c=1, n_boot=100, seed=seed + 1)
    summ.update(
        tau_on_fit_s=r_on["tau_single"] * DT,
        tau_on_lo_s=r_on.get("tau_single_lo", np.nan) * DT,
        tau_on_hi_s=r_on.get("tau_single_hi", np.nan) * DT,
        tau_off_fit_s=r_off["tau_single"] * DT,
        tau_off_lo_s=r_off.get("tau_single_lo", np.nan) * DT,
        tau_off_hi_s=r_off.get("tau_single_hi", np.nan) * DT,
        k_off_fit=1.0 / (r_on["tau_single"] * DT),
        inv_tau_off_per_s_fit=1.0 / (r_off["tau_single"] * DT),
        on_dBIC=r_on.get("dBIC"), on_LR_p=r_on.get("LR_p"),
        on_a_fast=r_on.get("a_fast"),
        on_tau_fast_s=(r_on.get("tau_fast") or np.nan) * DT,
        on_tau_slow_s=(r_on.get("tau_slow") or np.nan) * DT,
        off_dBIC=r_off.get("dBIC"),
        n_on_dwells=int(len(cm) + len(cx)), n_off_dwells=int(len(oi)),
        frac_1frame_events=float((ev.duration == 1).mean()),
        events_per_binder=len(ev) / max(n_with_events, 1))
    summ["tau_on_bias"] = summ["tau_on_fit_s"] / summ["tau_on_true_s"]
    summ["tau_off_bias"] = summ["tau_off_fit_s"] / summ["tau_off_true_s"]
    summ["on_model_called"] = "double" if (r_on.get("dBIC") or 0) > 10 else "single"
    summ["off_model_called"] = "double" if (r_off.get("dBIC") or 0) > 10 else "single"
    # gap statistics, for comparison with the measured dark-gap excess
    raw = []
    for _, g in ev.sort_values(["mol_idx", "start"]).groupby("mol_idx"):
        s, e = g.start.values, g.end.values
        if s.size > 1:
            raw.extend((s[1:] - e[:-1]).tolist())
    raw = np.array(raw, float)
    summ["n_raw_gaps"] = int(raw.size)
    summ["frac_gaps_1to4"] = float(np.mean((raw >= 1) & (raw <= 4))) if raw.size else np.nan
    return summ, ev


# ============================ experiments ==================================
def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def exp1_recovery(n_mol, out):
    """Q1. Known single-exponential on- and off-times, no blinking, no bleaching.
    Tests estimator bias and, critically, how often a genuinely one-component
    distribution is reported as two-component."""
    rows = []
    grid = [(0.021, 0.35), (0.021, 0.20), (0.021, 1.00),
            (0.010, 0.35), (0.042, 0.35), (0.084, 0.35)]
    for i, (kon, koff) in enumerate(grid):
        s, _ = run_condition(kon, koff, n_mol=n_mol, seed=SEED + i)
        rows.append(s)
        log(f"  E1 k_off={koff:.2f} k_on_eff={kon:.3f} -> tau_on {s.get('tau_on_fit_s', np.nan):.3f}s "
            f"(true {1/koff:.3f}, bias {s.get('tau_on_bias', np.nan):.3f}) "
            f"tau_off {s.get('tau_off_fit_s', np.nan):.2f}s (true {1/kon:.2f}, "
            f"bias {s.get('tau_off_bias', np.nan):.3f}) on-model={s.get('on_model_called')}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sim_e1_recovery.csv"), index=False)
    return df


def exp2_blinking(n_mol, out):
    """Q2. Blinking injected at a range of rates, at a fixed true k_on and k_off.
    A concentration-independent dark process should appear as an intercept in
    1/tau_off against concentration; here it is measured directly as the inflation
    of 1/tau_off at fixed concentration."""
    rows = []
    for i, kbd in enumerate(BLINK_DARK_GRID):
        s, _ = run_condition(0.021, 0.35, n_mol=n_mol, seed=SEED + 20 + i,
                             k_blink_dark=kbd, k_blink_return=BLINK_RETURN)
        s["k_blink_dark"] = kbd
        rows.append(s)
        log(f"  E2 blink_dark={kbd:.2f}/s -> 1/tau_off {s.get('inv_tau_off_per_s_fit', np.nan):.4f} "
            f"(true k_on_eff 0.0210) frac_gaps_1to4 {s.get('frac_gaps_1to4', np.nan):.3f} "
            f"tau_on {s.get('tau_on_fit_s', np.nan):.3f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sim_e2_blinking.csv"), index=False)
    return df


def exp2b_long_blinks(n_mol, out):
    """Blinks whose dark interval EXCEEDS the gap-bridging window, so they are
    not repaired by merging. These fragment one occupancy episode into several
    apparent events and are the mechanism that could manufacture an intercept."""
    rows = []
    for i, kret in enumerate((5.0, 2.0, 1.0, 0.5)):
        s, _ = run_condition(0.021, 0.35, n_mol=n_mol, seed=SEED + 90 + i,
                             k_blink_dark=0.30, k_blink_return=kret)
        s.update(k_blink_dark=0.30, k_blink_return=kret,
                 mean_dark_frames=1.0 / kret / DT)
        rows.append(s)
        log(f"  E2b mean dark {1/kret/DT:.0f} frames -> 1/tau_off "
            f"{s.get('inv_tau_off_per_s_fit', np.nan):.4f} (true 0.0210), "
            f"tau_on {s.get('tau_on_fit_s', np.nan):.3f}s (true 2.857), "
            f"events/binder {s.get('events_per_binder', np.nan):.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sim_e2b_long_blinks.csv"), index=False)
    return df


def exp5_two_component(n_mol, out):
    """Positive control for the model-selection step: the truth really is a
    mixture of two dissociation rates. Confirms the two-component test has power
    and recovers the components, so that a 'single' verdict elsewhere is
    informative rather than a lack of sensitivity."""
    rows = []
    grid = [(0.7, 0.20, 2.0), (0.5, 0.15, 1.5), (0.8, 0.30, 3.0)]
    for i, (frac_fast, k_slow, k_fast) in enumerate(grid):
        # a molecule is assigned one of two dissociation rates for the whole trace
        rg = np.random.default_rng(SEED + 110 + i)
        tasks = []
        for m in range(n_mol):
            koff = k_fast if rg.random() < frac_fast else k_slow
            tasks.append((m, SEED * 7919 + m + 1000 * i,
                          dict(k_on_eff=0.021, k_off=koff)))
        with Pool(max(1, min(cpu_count() - 1, 7))) as pool:
            res = pool.map(_one, tasks, chunksize=8)
        events, nwe = [], 0
        for ev, _, _, _ in res:
            events.extend(ev)
            nwe += int(len(ev) > 0)
        ev = pd.DataFrame(events)
        ON, _, _ = A.extract_dwells(ev, A.MAX_GAP, n_frames=N_FRAMES)
        cm = ON[(~ON.censored) & (ON.frames >= A.MIN_DUR)]
        cx = ON[ON.censored & (ON.frames >= A.MIN_DUR)]
        r = A.fit_dwell_set(cm.frames.values.astype(float), cm.key.values,
                            cx.frames.values.astype(float), cx.key.values,
                            A.MIN_DUR, n_boot=0, seed=SEED)
        rows.append(dict(frac_fast_true=frac_fast, tau_fast_true_s=1 / k_fast,
                         tau_slow_true_s=1 / k_slow, n_molecules=n_mol,
                         n_on_dwells=len(cm) + len(cx),
                         a_fast_fit=r.get("a_fast"),
                         tau_fast_fit_s=(r.get("tau_fast") or np.nan) * DT,
                         tau_slow_fit_s=(r.get("tau_slow") or np.nan) * DT,
                         dBIC=r.get("dBIC"),
                         model_called="double" if (r.get("dBIC") or 0) > 10 else "single"))
        log(f"  E5 true {frac_fast:.0%} at {1/k_fast:.2f}s + {1-frac_fast:.0%} at "
            f"{1/k_slow:.2f}s -> fit {r.get('a_fast', float('nan')):.2f} at "
            f"{(r.get('tau_fast') or float('nan'))*DT:.2f}s + "
            f"{(r.get('tau_slow') or float('nan'))*DT:.2f}s, "
            f"dBIC {r.get('dBIC', float('nan')):.0f} -> {rows[-1]['model_called']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sim_e5_two_component.csv"), index=False)
    return df


def exp6_detection_efficiency(n_mol, out, tab, nominals=(1, 2, 3, 4, 5, 7, 10),
                              target=None):
    """Where should the lower truncation of the dwell-time fits sit?

    For each condition, simulate at that condition's own fitted rates and its
    MEASURED brightness, background and noise, then score what fraction of true
    occupancy episodes the idealiser recovers as a function of the episode's true
    duration. The lower truncation must sit at or above the duration where
    recovery reaches `target`, otherwise the likelihood is fitted to a sample
    that is missing short dwells in a duration-dependent way.

    Efficiency depends on brightness, and the two fluorogens differ in
    brightness, so this is measured separately for each -- but the truncation
    finally adopted is the SAME for both, taken from whichever needs the larger
    value, so the comparison between them cannot be biased by the choice.
    """
    target = A.DETECTION_TARGET if target is None else target
    kin = pd.read_csv(os.path.join(tab, "kinetics_per_condition.csv"))
    bg = pd.read_csv(os.path.join(tab, "background_vs_concentration.csv"))
    ref = kin[kin.power_mW == A.DEFAULT_POWER_MW].merge(
        bg[["fluorogen", "conc_nM", "power_mW", "background", "noise_sd", "brightness"]],
        on=["fluorogen", "conc_nM", "power_mW"], how="inner")
    rows = []
    for r in ref.sort_values(["fluorogen", "conc_nM"]).itertuples():
        k_off = 1.0 / r.on_tau_single_s
        k_on_eff = 1.0 / r.off_tau_single_s
        dur, rec = efficiency_run(k_on_eff, k_off, n_mol=n_mol,
                                  seed=SEED + 200 + int(r.conc_nM * 10),
                                  bright=float(r.brightness), noise=float(r.noise_sd),
                                  bg=float(r.background))
        nom = np.rint(dur / DT).astype(int)
        for d in nominals:
            m = nom == d
            k, n = int(rec[m].sum()), int(m.sum())
            lo, hi = _wilson(k, n)
            rows.append(dict(fluorogen=r.fluorogen, conc_nM=r.conc_nM,
                             power_mW=r.power_mW, brightness=r.brightness,
                             noise_sd=r.noise_sd, nominal_frames=d,
                             n_episodes=n, n_recovered=k,
                             recovery=k / n if n else np.nan,
                             ci_lo=lo, ci_hi=hi))
        m = nom > max(nominals)
        k, n = int(rec[m].sum()), int(m.sum())
        lo, hi = _wilson(k, n)
        rows.append(dict(fluorogen=r.fluorogen, conc_nM=r.conc_nM, power_mW=r.power_mW,
                         brightness=r.brightness, noise_sd=r.noise_sd,
                         nominal_frames=-1, n_episodes=n, n_recovered=k,
                         recovery=k / n if n else np.nan, ci_lo=lo, ci_hi=hi))
        log(f"  E6 {r.fluorogen:7s} {r.conc_nM:>4g} nM  "
            + "  ".join(f"{d}f:{rows[-len(nominals)-1+i]['recovery']:.3f}"
                        for i, d in enumerate(nominals)))
    df = pd.DataFrame(rows)
    # Per CONDITION, the shortest duration clearing the target. Pooling across
    # concentrations lets the well-behaved conditions mask a bad one, so the
    # per-condition requirement is reported alongside the pooled decision.
    need = []
    for (fl, c, pw), g in df[df.nominal_frames > 0].groupby(
            ["fluorogen", "conc_nM", "power_mW"]):
        ok = g.loc[g.ci_lo >= target, "nominal_frames"]
        need.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                         min_dur_required=int(ok.min()) if len(ok) else None,
                         recovery_at_2=float(g.loc[g.nominal_frames == 2, "recovery"].iloc[0]),
                         recovery_at_3=float(g.loc[g.nominal_frames == 3, "recovery"].iloc[0]),
                         recovery_at_4=float(g.loc[g.nominal_frames == 4, "recovery"].iloc[0])))
    need = pd.DataFrame(need)
    need.to_csv(os.path.join(out, "sim_e6_required_per_condition.csv"), index=False)
    df = df.merge(need[["fluorogen", "conc_nM", "power_mW", "min_dur_required"]],
                  on=["fluorogen", "conc_nM", "power_mW"], how="left")
    df.to_csv(os.path.join(out, "sim_e6_detection_efficiency.csv"), index=False)

    # pooled over concentrations, per fluorogen: the value that decides MIN_DUR
    pool = (df[df.nominal_frames > 0].groupby(["fluorogen", "nominal_frames"])
            .agg(n_episodes=("n_episodes", "sum"), n_recovered=("n_recovered", "sum"))
            .reset_index())
    pool["recovery"] = pool.n_recovered / pool.n_episodes
    pool[["ci_lo", "ci_hi"]] = pool.apply(
        lambda x: pd.Series(_wilson(x.n_recovered, x.n_episodes)), axis=1)
    chosen = {}
    for fl, g in pool.groupby("fluorogen"):
        ok = g[g.ci_lo >= target].nominal_frames
        chosen[fl] = int(ok.min()) if len(ok) else None
    vals = [v for v in chosen.values() if v is not None]
    pool["meets_target_lower_ci"] = pool.ci_lo >= target
    pool["target"] = target
    pool["min_dur_for_this_fluorogen"] = pool.fluorogen.map(chosen)
    pool["min_dur_adopted"] = max(vals) if vals else None
    pool.to_csv(os.path.join(out, "sim_e6_efficiency_pooled.csv"), index=False)
    log(f"  E6 lower truncation: per-fluorogen {chosen}, "
        f"adopted {max(vals) if vals else None} frames "
        f"(recovery lower CI >= {target:.2f})")
    return df, pool


def exp3_concentration(n_mol, out, bg_table=None):
    """Q3. True k_off and k_on held FIXED; only concentration (through k_on_eff)
    and the measured background/noise scaling vary. Any drift in the recovered
    k_off, or curvature in 1/tau_off against concentration, is an artefact of the
    detection chain rather than a property of the sample."""
    K_ON = 0.0424          # per nM per s, so k_on_eff = K_ON * c
    K_OFF = 0.35
    concs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    # measured background and noise at 1.5 mW (SALAD1); falls back to flat values
    meas = {}
    if bg_table is not None and os.path.exists(bg_table):
        bg = pd.read_csv(bg_table)
        bg = bg[(bg.fluorogen == "SALAD1") & (bg.power_mW == 1.5)]
        meas = {float(r.conc_nM): (float(r.background), float(r.noise_sd),
                                   float(r.brightness)) for r in bg.itertuples()}
    rows = []
    for i, c in enumerate(concs):
        b, nz, br = meas.get(c, (REF_BG, REF_NOISE, REF_BRIGHT))
        s, _ = run_condition(K_ON * c, K_OFF, n_mol=n_mol, seed=SEED + 40 + i,
                             bright=br, noise=nz, bg=b)
        s.update(conc_nM=c, k_on_true_per_nM=K_ON, noise_used=nz, bright_used=br)
        rows.append(s)
        log(f"  E3 c={c:>5g} nM (noise {nz:.1f}, bright {br:.0f}) -> "
            f"k_off_fit {s.get('k_off_fit', np.nan):.3f} (true {K_OFF:.3f}) "
            f"1/tau_off {s.get('inv_tau_off_per_s_fit', np.nan):.4f} "
            f"(true {K_ON*c:.4f})")
    df = pd.DataFrame(rows)
    # the same slope-and-intercept fit applied to the real series
    ok = df.dropna(subset=["inv_tau_off_per_s_fit"])
    x, y = ok.conc_nM.values, ok.inv_tau_off_per_s_fit.values
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    df["fitted_k_on_per_nM"] = slope
    df["fitted_intercept_per_s"] = icpt
    df["fitted_R2"] = r2
    df["k_on_recovery"] = slope / K_ON
    df.to_csv(os.path.join(out, "sim_e3_concentration.csv"), index=False)
    log(f"  E3 slope fit: k_on = {slope:.5f} /nM/s (true {K_ON:.5f}, "
        f"recovery {slope/K_ON:.2f}x), intercept = {icpt:.5f} /s, R2 = {r2:.3f}")
    return df


def exp3b_matched_occupancy(n_mol, out, k_bleach=0.40):
    """The E3 sweep is informative only in the occupancy regime the experiment
    actually occupies. Plain E3 reaches 49-65% occupancy at the top of the range,
    where events merge; the real data sit at 5-13% throughout. Re-running with
    photobleaching of the bound dye included both matches the measured tau_on and
    tests whether bleaching alone reproduces the observed occupancy ceiling --
    the interpretation the Langmuir amplitude was given."""
    K_ON, K_OFF = 0.0424, 0.35
    rows = []
    for i, c in enumerate([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]):
        s, _ = run_condition(K_ON * c, K_OFF, n_mol=n_mol, seed=SEED + 130 + i,
                             k_bleach=k_bleach)
        s.update(conc_nM=c, k_on_true_per_nM=K_ON, k_bleach=k_bleach)
        rows.append(s)
        log(f"  E3b c={c:>5g} nM -> k_off_fit {s.get('k_off_fit', np.nan):.3f} "
            f"(true {K_OFF:.3f}), tau_on {s.get('tau_on_fit_s', np.nan):.3f}s, "
            f"occupancy obs {s.get('bound_fraction_observed', np.nan):.4f} "
            f"true {s.get('bound_fraction_true', np.nan):.4f}")
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["inv_tau_off_per_s_fit"])
    slope, icpt = np.polyfit(ok.conc_nM.values, ok.inv_tau_off_per_s_fit.values, 1)
    pred = slope * ok.conc_nM.values + icpt
    r2 = 1 - ((ok.inv_tau_off_per_s_fit.values - pred) ** 2).sum() / \
        ((ok.inv_tau_off_per_s_fit.values - ok.inv_tau_off_per_s_fit.mean()) ** 2).sum()
    df["fitted_k_on_per_nM"] = slope
    df["fitted_intercept_per_s"] = icpt
    df["fitted_R2"] = r2
    df["k_off_fold_range"] = df.k_off_fit.max() / df.k_off_fit.min()
    df.to_csv(os.path.join(out, "sim_e3b_matched_occupancy.csv"), index=False)
    log(f"  E3b slope fit: k_on = {slope:.5f} /nM/s (true {K_ON:.5f}), "
        f"intercept = {icpt:.5f} /s, R2 = {r2:.3f}, "
        f"apparent k_off spans {df.k_off_fit.max()/df.k_off_fit.min():.2f}x")
    return df


def exp4_bleaching(n_mol, out):
    """Bound dwell against bleaching rate, at fixed true k_off. Quantifies how
    much of the measured power dependence of tau_on bleaching can account for."""
    rows = []
    for i, kb in enumerate((0.0, 0.1, 0.2, 0.4, 0.8)):
        s, _ = run_condition(0.021, 0.35, n_mol=n_mol, seed=SEED + 60 + i, k_bleach=kb)
        s["k_bleach"] = kb
        rows.append(s)
        log(f"  E4 k_bleach={kb:.2f}/s -> tau_on {s.get('tau_on_fit_s', np.nan):.3f}s "
            f"(true occupancy tau {1/0.35:.3f}s) "
            f"bound_frac obs {s.get('bound_fraction_observed', np.nan):.4f} "
            f"true {s.get('bound_fraction_true', np.nan):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sim_e4_bleaching.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer molecules")
    ap.add_argument("--n-mol", type=int, default=None)
    args = ap.parse_args()
    n_mol = args.n_mol or (150 if args.quick else N_MOL)
    t0 = time.time()
    log(f"validation with {n_mol} molecules per condition, {N_FRAMES} frames, dt={DT} s")
    log("E1: recovery of known single-exponential dwells")
    e1 = exp1_recovery(n_mol, TAB)
    log("E2: blinking injected at fixed true rates")
    e2 = exp2_blinking(n_mol, TAB)
    log("E2b: blinks longer than the gap-bridging window")
    e2b = exp2b_long_blinks(n_mol, TAB)
    log("E5: positive control, truth is genuinely two-component")
    e5 = exp5_two_component(n_mol, TAB)
    log("E3: concentration sweep at fixed true k_off, measured noise scaling")
    e3 = exp3_concentration(n_mol, TAB,
                            bg_table=os.path.join(TAB, "background_vs_concentration.csv"))
    log("E3b: concentration sweep at the occupancy the experiment actually reaches")
    e3b = exp3b_matched_occupancy(n_mol, TAB)
    log("E4: photobleaching of the bound dye")
    e4 = exp4_bleaching(n_mol, TAB)
    log(f"done in {(time.time()-t0)/60:.1f} min; tables written")


if __name__ == "__main__":
    main()
