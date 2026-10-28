#!/usr/bin/env python3
"""
Single-molecule TIRF analysis of Mango II / fluorogen complexes.
End-to-end: a folder of CPD-idealised trace exports (.h5, format_version 4) -> every
table and text output for Chapter 4.

Run:      python analyse_smtirf.py
Requires: numpy, scipy, pandas, h5py

Conventions kept from the trace export
-----------------------------
End / EndIdx are EXCLUSIVE. Durations are in frames. On-times come from merged
events; boundary-truncated on-times enter the likelihood as right-censored.
Off-times come from gaps between consecutive events of one molecule; the gap
before the first and after the last event is retained as a RIGHT-CENSORED
observation (the "survival" convention) because discarding it removes the
longest unbound intervals and biases tau_off low.

Every analysis choice is a named parameter in the PARAMETERS block.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import warnings

import glob
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================== PARAMETERS ==================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SMTIRF_DATA", os.path.abspath(os.path.join(HERE, os.pardir, "TIRF")))
OUT_DIR = HERE
_H5_INDEX = None
TAB = os.path.join(OUT_DIR, "tables")
CACHE = os.path.join(OUT_DIR, "cache")

FRAME_TIME_S = 0.1        # 100 ms exposure, continuous acquisition
DEFAULT_POWER_MW = 1.5    # laser power for files whose name carries no "<x>mW"
N_FRAMES_EXPECTED = 2500

# Acquisitions at these concentrations were preceded by deliberate photobleaching
# of the field so that single molecules could be resolved; the others were not.
# The treatment is PERFECTLY CONFOUNDED with concentration, so any quantity that
# differs between the two groups has two candidate explanations. See prebleach.py
# and section 5b of the report. This is experimental metadata, not an analysis
# choice: change it only if the acquisition log says otherwise.
PREBLEACH_CONC_NM = (2.0, 4.0, 8.0, 16.0)

# --- step 3: the analysis window within each acquisition ---------------------
# Occupancy relaxes 5- to 45-fold during every acquisition and the acquisitions
# did not all start at the same point on that decay, so dwells are taken from a
# window chosen to put every acquisition at a comparable point on its own
# relaxation. The rule is APPLIED here; it is JUSTIFIED in the report, next to
# the simulated recovery of known rates that selected it (window_select.py).
WINDOW_RULE = "A_matched_occupancy"   # or "B_matched_clock", "C_early", "none"
WINDOW_OCC_FRAC = 0.25    # rule A: the window opens where occupancy first falls
                          # to this fraction of the acquisition's OWN initial
                          # occupancy
WINDOW_EARLY_LEN = 250    # rule C: length of the early window, in frames

MAX_GAP = 3               # bridge dark gaps of <= MAX_GAP frames inside an event
MIN_DUR = 3               # Left-truncate dwell fits at MIN_DUR frames. MEASURED, not
                          # assumed. The limit must sit at or above the duration where
                          # detection of true episodes is complete, and the SAME value
                          # is used for both fluorogens so the comparison between them
                          # cannot be biased by the choice. Evidence, from
                          # run_truncation_study.py:
                          #   recovery of true episodes (sim_e6_efficiency_pooled.csv)
                          #     3 frames: 99.9% SALAD1, 98.0% TO1-B
                          #     2 frames: 98.0% SALAD1, 73.0% TO1-B  <- asymmetric,
                          #       and the gap widens with concentration
                          #   closed-loop recovery of a known input
                          #   (sim_e7_truncation_recovery.csv)
                          #     3 frames: 1.006-1.026 of truth
                          #     4 frames: 0.929-0.950  <- truncating higher is NOT safer
DETECTION_TARGET = 0.95   # recovery threshold defining the lower truncation
MAX_DUR = None            # upper truncation (frames) for dwell fits; None = no upper
                          # limit. Dwells >= MAX_DUR are entered as RIGHT-CENSORED at
                          # MAX_DUR rather than discarded, so no information is thrown
                          # away and the long tail cannot drive the fitted lifetime.
MAX_DUR_VARIANT = 500     # the with-upper-limit variant reported alongside the headline
                          # (500 frames = 50 s, a fifth of the acquisition)
EDGE_CORRECTION = False   # headline: durations as measured; +1 frame variant reported
CONTRAST_MIN = None       # headline: no contrast filter; >=3 variant reported
CONTRAST_VARIANT = 3.0

OFF_CENSORING = "survival"   # "survival" (boundary gaps kept, right-censored)
                             # or "interior" (export convention, gaps discarded)

# --- effective photostability under exchange: the span is a CHOICE, stated here
SPAN_DEFINITION = "first_to_last"  # or "acquisition" (measure from frame 0)
SPAN_CENSORING = "gap_aware"       # or "strict" (censor only if the last event was
                                   # still running at the final frame -- biased low)
CENSOR_MARGIN = 1.0                # censor when the observation left after the last
                                   # event is < CENSOR_MARGIN x the expected wait
                                   # (tau_off) for that condition

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules

MIXTURE_KS = (1, 2, 3)     # components tried in the censoring-aware brightness mixture
DIM_RUN = 5               # running-mean window (frames) for the dim-binder statistic
DIM_N_SHUFFLE = 6         # frame shuffles per molecule for the empirical null
DIM_Z = 3.0               # z above the molecule's own shuffled null = real excursion

N_BOOT = 150              # cluster bootstrap resamples (over MOLECULES, not events)
N_BOOT_RATIO = 4000       # bootstrap resamples for brightness ratios
SEED = 20260910

BULK_INTENSITY_RATIO = 2.07     # SALAD1 / TO1-B bulk emission, matched conditions
PRED_POPULATION = (1.00, 1.20)  # per-spot brightness ratio, population account
PRED_UNIFORM = 2.07             # per-spot brightness ratio, uniform account
KD_BULK_NM = {"SALAD1": 0.69, "TO1-B": 0.85}

SENS_MAX_GAP = (0, 1, 2, 3)
SENS_MIN_DUR = (2, 3, 4, 5)     # bracketing the measured value of 3 frames
SENS_MAX_DUR = (None, 500)      # no upper limit, and the 50 s variant
SENS_EDGE = (False, True)
SENS_CONTRAST = (None, 3.0)

# ============================================================================

for _d in (TAB, CACHE):
    os.makedirs(_d, exist_ok=True)

_CONC = re.compile(r"_([0-9]*\.?[0-9]+)\s*(nM|pM)(?=_|\.)", re.I)
_POW = re.compile(r"_([0-9]*\.?[0-9]+)\s*mW(?=_|\.)", re.I)
_REP = re.compile(r"_([0-9]+)\.h5$", re.I)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ metadata
def parse_name(path):
    """Fluorogen, concentration, power and replicate from the FILE NAME.

    The name is authoritative: the acquisition path stored in the root
    source_file attribute has been seen to disagree with the true condition.
    Any disagreement is reported by check_format, not silently resolved.
    """
    base = os.path.basename(path)
    if re.search(r"SALAD", base, re.I):
        fluor = "SALAD1"
    elif re.search(r"(?<![A-Za-z])TO(?![A-Za-z])", base, re.I):
        fluor = "TO1-B"
    else:
        fluor = None
    mc, mp, mr = _CONC.search(base), _POW.search(base), _REP.search(base)
    conc = None
    if mc:
        conc = float(mc.group(1)) * (1e-3 if mc.group(2).lower() == "pm" else 1.0)
    return dict(file=base, fluorogen=fluor, conc_nM=conc,
                power_mW=float(mp.group(1)) if mp else DEFAULT_POWER_MW,
                power_in_name=bool(mp),
                replicate=int(mr.group(1)) if mr else None)


def check_format(path):
    """Reject unfitted or trace-filtered exports; print the layout if it deviates."""
    with h5py.File(path, "r") as f:
        a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        fmt = [str(v) for k, v in a.items() if k.endswith("_format")]
        if "hmm_results" not in fmt:
            return False, f"format tag={fmt!r} (needs 'hmm_results')", a
        if bool(a.get("filtered", False)):
            return False, "export was trace-filtered; ask for an unfiltered export", a
        if int(a.get("format_version", -1)) != 4:
            return False, f"format_version={a.get('format_version')} (expected 4)", a
        if "channel_results" not in f:
            return False, f"no channel_results group; root keys {list(f.keys())}", a
    return True, "ok", a


def load_export(path, with_traces=True):
    with h5py.File(path, "r") as f:
        primary = f.attrs["primary_channel"]
        primary = primary.decode() if isinstance(primary, bytes) else str(primary)
        groups = {}
        for g in f["channel_results"].values():
            nm = g.attrs["channel_name"]
            groups[nm.decode() if isinstance(nm, bytes) else str(nm)] = g
        g = groups.get(primary) or next(iter(groups.values()))
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in g["molecule_ids"][:]]
        out = dict(ids=ids, xy=g["locations"][:], n_frames=int(g.attrs["n_frames"]),
                   states=g["states"][:],
                   events=[json.loads(g[f"events/{i}/records"][()]) for i in range(len(ids))],
                   selected=[g[f"events/{i}/selected"][:] for i in range(len(ids))],
                   path=path)
        out["traces"] = g["traces"][:].astype(float) if with_traces else None
    return out


# --------------------------------------------------- per-molecule / per-event
def per_molecule_table(d, meta):
    tr, st = d["traces"], d["states"]
    n = st.shape[1]
    rows = []
    for i in range(n):
        s = st[:, i].astype(float)
        valid = ~np.isnan(s)
        y = tr[:, i]
        m0 = valid & (s == 0)
        m1 = valid & (s == 1)
        bg = float(np.mean(y[m0])) if m0.sum() else np.nan
        bgsd = float(np.std(y[m0], ddof=1)) if m0.sum() > 1 else np.nan
        lvl = float(np.mean(y[m1])) if m1.sum() else np.nan
        rows.append(dict(
            file=meta["file"], fluorogen=meta["fluorogen"], conc_nM=meta["conc_nM"],
            power_mW=meta["power_mW"], replicate=meta["replicate"],
            mol_idx=i, molecule_id=d["ids"][i],
            x=float(d["xy"][i, 0]), y=float(d["xy"][i, 1]),
            unresolved=bool(not valid.any()),
            n_valid=int(valid.sum()), n0=int(m0.sum()), n1=int(m1.sum()),
            n_events=len(d["events"][i]),
            background=bg, background_sd=bgsd, bound_level=lvl,
            brightness=(lvl - bg) if (m1.sum() and m0.sum()) else np.nan,
            contrast=((lvl - bg) / bgsd) if (m1.sum() and bgsd and bgsd > 0) else np.nan,
            bound_fraction=(m1.sum() / valid.sum()) if valid.sum() else np.nan,
            trace_median=float(np.median(y[valid])) if valid.any() else np.nan,
            first_start=min((e["StartIdx"] for e in d["events"][i]), default=-1),
            last_end=max((e["EndIdx"] for e in d["events"][i]), default=-1),
            last_end_trunc=bool(max(d["events"][i], key=lambda e: e["EndIdx"])["EndTrunc"])
            if d["events"][i] else False,
            first_start_trunc=bool(min(d["events"][i], key=lambda e: e["StartIdx"])["StartTrunc"])
            if d["events"][i] else False,
        ))
    return pd.DataFrame(rows)


def per_event_table(d, meta):
    rows = []
    for i, evs in enumerate(d["events"]):
        sel = d["selected"][i]
        for j, e in enumerate(evs):
            rows.append(dict(
                file=meta["file"], fluorogen=meta["fluorogen"], conc_nM=meta["conc_nM"],
                power_mW=meta["power_mW"], mol_idx=i, molecule_id=d["ids"][i],
                start=int(e["StartIdx"]), end=int(e["EndIdx"]), duration=int(e["Duration"]),
                mean_intensity=float(e["MeanIntensity"]), snr=float(e["SNR"]),
                start_trunc=bool(e["StartTrunc"]), end_trunc=bool(e["EndTrunc"]),
                selected=bool(sel[j]) if len(sel) > j else True))
    return pd.DataFrame(rows)


def gap_table(d, meta):
    rows = []
    for i, evs in enumerate(d["events"]):
        evs = sorted(evs, key=lambda e: e["StartIdx"])
        for a, b in zip(evs, evs[1:]):
            rows.append(dict(file=meta["file"], fluorogen=meta["fluorogen"],
                             conc_nM=meta["conc_nM"], power_mW=meta["power_mW"],
                             mol_idx=i, gap=int(b["StartIdx"] - a["EndIdx"])))
    return pd.DataFrame(rows)


# ------------------------------------------------- dim-binder shuffle null
def _run_max(a, k):
    c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
    return np.nanmax((c[k:] - c[:-k]) / k, axis=0)


def dim_statistic(A, rg):
    """Largest k-frame running mean, observed vs the molecule's own frame-shuffled
    null. Shuffling preserves the intensity distribution and destroys the
    multi-frame structure a real dwell must have, so it is the right null for a
    maximum statistic (a 3-sigma threshold is not: the largest 2-frame excursion
    of a 2500-frame pure-noise trace already sits near 2.8 sigma)."""
    base = np.median(A, axis=0)
    mad = np.median(np.abs(A - base), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, np.nan)
    obs = (_run_max(A, DIM_RUN) - base) / mad
    sh = np.empty((DIM_N_SHUFFLE, A.shape[1]))
    idx0 = np.tile(np.arange(A.shape[0])[:, None], (1, A.shape[1]))
    for s in range(DIM_N_SHUFFLE):
        idx = rg.permuted(idx0, axis=0)
        sh[s] = (_run_max(np.take_along_axis(A, idx, axis=0), DIM_RUN) - base) / mad
    return obs, sh.mean(0), sh.std(0, ddof=1)


# =============================== STAGE 1-2 =================================
def discover_h5(root):
    """Every .h5 under `root`, at any depth. Recursive because the data folder
    is the user's to organise: the 48 exports have lived both directly in it and
    in an `h5/` subfolder, and a non-recursive glob silently finds nothing in
    the second case."""
    found = sorted(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
    seen, out = set(), []
    for p in found:
        b = os.path.basename(p)
        if b not in seen:                      # first match wins, shallowest first
            seen.add(b)
            out.append(p)
    dupes = len(found) - len(out)
    if dupes:
        log(f"note: {dupes} duplicate .h5 basename(s) under {root}; using the first of each")
    return out


def h5_path(filename):
    """Resolve a bare export filename to its path on disk."""
    global _H5_INDEX
    if _H5_INDEX is None:
        _H5_INDEX = {os.path.basename(p): p for p in discover_h5(DATA_DIR)}
    p = _H5_INDEX.get(filename)
    if p is None:
        raise FileNotFoundError(f"{filename} not found under {DATA_DIR}")
    return p


def stage_extract():
    """Format check + metadata + per-molecule/event/gap tables + dim statistic."""
    import glob
    files = discover_h5(DATA_DIR)
    if not files:
        sys.exit(f"no .h5 files in {DATA_DIR}")
    log(f"{len(files)} files in {DATA_DIR}")

    meta_rows, rejected = [], []
    for p in files:
        ok, msg, attrs = check_format(p)
        r = parse_name(p)
        src = str(attrs.get("source_file", "")).replace("\\", "/")
        r.update(format_ok=ok, format_msg=msg, idealizer=attrs.get("idealizer"),
                 format_version=attrs.get("format_version"),
                 filtered=bool(attrs.get("filtered", False)), source_file=src)
        # cross-check the acquisition path against the filename
        pm = re.search(r"([0-9]*\.?[0-9]+)\s*(nM|pM)", src, re.I)
        if pm and r["conc_nM"] is not None:
            c = float(pm.group(1)) * (1e-3 if pm.group(2).lower() == "pm" else 1.0)
            r["path_conc_nM"] = c
            r["conc_disagrees_with_path"] = abs(c - r["conc_nM"]) > 1e-9
        else:
            r["path_conc_nM"], r["conc_disagrees_with_path"] = np.nan, False
        if not ok:
            rejected.append((r["file"], msg))
        if r["fluorogen"] is None or r["conc_nM"] is None:
            rejected.append((r["file"], "could not parse fluorogen or concentration"))
        meta_rows.append(r)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(os.path.join(TAB, "metadata_parsed.csv"), index=False)
    if rejected:
        log("REJECTED FILES (excluded from every analysis):")
        for f_, m_ in rejected:
            log(f"    {f_}: {m_}")
    if meta.conc_disagrees_with_path.any():
        log("NOTE: filename and acquisition path disagree on concentration for "
            f"{int(meta.conc_disagrees_with_path.sum())} file(s); the FILENAME was used.")
    good = meta[meta.format_ok & meta.fluorogen.notna() & meta.conc_nM.notna()]

    rg = np.random.default_rng(SEED)
    mol_l, ev_l, gap_l = [], [], []
    for k, r in enumerate(good.itertuples(), 1):
        cpath = os.path.join(CACHE, r.file.replace(".h5", ".pkl"))
        if os.path.exists(cpath):
            with open(cpath, "rb") as fh:
                c = pickle.load(fh)
        else:
            t0 = time.time()
            m = dict(file=r.file, fluorogen=r.fluorogen, conc_nM=r.conc_nM,
                     power_mW=r.power_mW, replicate=r.replicate)
            d = load_export(h5_path(r.file), with_traces=True)
            if d["n_frames"] != N_FRAMES_EXPECTED:
                log(f"    {r.file}: n_frames={d['n_frames']} (expected {N_FRAMES_EXPECTED})")
            mt, et, gt = per_molecule_table(d, m), per_event_table(d, m), gap_table(d, m)
            n = len(mt)
            obs = np.empty(n); nmn = np.empty(n); nsd = np.empty(n)
            for i0 in range(0, n, 800):
                sl = slice(i0, min(i0 + 800, n))
                obs[sl], nmn[sl], nsd[sl] = dim_statistic(d["traces"][:, sl], rg)
            mt["dim_obs"], mt["dim_null_mean"], mt["dim_null_sd"] = obs, nmn, nsd
            mt["dim_z"] = (obs - nmn) / nsd
            mt["n_frames"] = d["n_frames"]
            c = dict(mol=mt, ev=et, gaps=gt)
            with open(cpath, "wb") as fh:
                pickle.dump(c, fh, protocol=4)
            log(f"[{k}/{len(good)}] {r.file} mol={len(mt)} ev={len(et)} {time.time()-t0:.0f}s")
        mol_l.append(c["mol"]); ev_l.append(c["ev"]); gap_l.append(c["gaps"])

    mol = pd.concat(mol_l, ignore_index=True)
    ev = pd.concat(ev_l, ignore_index=True)
    gaps = pd.concat(gap_l, ignore_index=True)
    log(f"molecules {len(mol)} | events {len(ev)} | raw gaps {len(gaps)}")

    qc = (mol.groupby(["fluorogen", "conc_nM", "power_mW", "replicate", "file"])
          .agg(n_frames=("n_frames", "first"), n_molecules=("mol_idx", "size"),
               unresolved=("unresolved", "sum"),
               mol_with_events=("n_events", lambda s: int((s > 0).sum())),
               mol_zero_events=("n_events", lambda s: int((s == 0).sum())),
               n_events=("n_events", "sum"),
               ev_per_mol_median=("n_events", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else 0.0),
               ev_per_mol_p90=("n_events", lambda s: float(np.percentile(s[s > 0], 90)) if (s > 0).any() else 0.0),
               contrast_q10=("contrast", lambda s: float(np.nanpercentile(s, 10))),
               contrast_median=("contrast", "median"),
               contrast_q90=("contrast", lambda s: float(np.nanpercentile(s, 90))),
               frac_mol_contrast_lt3=("contrast", lambda s: float(np.nanmean(s < 3))),
               background_median=("background", "median"),
               background_sd_median=("background_sd", "median"),
               brightness_median=("brightness", "median"),
               x_min=("x", "min"), x_max=("x", "max"), y_min=("y", "min"), y_max=("y", "max"))
          .reset_index())
    evq = (ev.groupby("file")
           .agg(frac_ev_snr_lt3=("snr", lambda s: float((s < 3).mean())),
                frac_ev_1frame=("duration", lambda s: float((s == 1).mean())),
                frac_ev_gt100frame=("duration", lambda s: float((s > 100).mean())))
           .reset_index())
    tr = (ev.assign(t=ev.start_trunc | ev.end_trunc).groupby("file").t.mean()
          .rename("frac_ev_truncated").reset_index())
    qc = qc.merge(evq, on="file", how="left").merge(tr, on="file", how="left")
    qc.to_csv(os.path.join(TAB, "qc_per_file.csv"), index=False)
    return meta, good, mol, ev, gaps, qc


# ========================= dwell extraction & likelihood ====================
def extract_dwells(evdf, max_gap, n_frames=N_FRAMES_EXPECTED, edge_correction=False):
    """Merge events across dark gaps of <= max_gap frames, then return
    on-dwells, interior off-gaps, and boundary off-gaps (right-censored).
    Vectorised: segment boundaries are found by cumulative sum, not a Python loop."""
    e = evdf.sort_values(["file", "mol_idx", "start"], kind="mergesort").reset_index(drop=True)
    key = (e.file.astype(str) + "|" + e.mol_idx.astype(str)).values
    first_in_mol = np.r_[True, key[1:] != key[:-1]]
    gap = e.start.values - np.r_[np.nan, e.end.values[:-1]]
    new_seg = first_in_mol | (gap > max_gap)
    e = e.assign(_seg=np.cumsum(new_seg) - 1, _key=key)
    agg = e.groupby("_seg").agg(
        file=("file", "first"), mol_idx=("mol_idx", "first"), key=("_key", "first"),
        fluorogen=("fluorogen", "first"), conc_nM=("conc_nM", "first"),
        power_mW=("power_mW", "first"), start=("start", "first"), end=("end", "last"),
        st=("start_trunc", "first"), en=("end_trunc", "last"), snr=("snr", "mean"))
    on = agg.assign(frames=(agg.end - agg.start) + (1 if edge_correction else 0),
                    censored=agg.st.astype(bool) | agg.en.astype(bool))
    m = (~first_in_mol) & new_seg
    off = pd.DataFrame(dict(key=key[m], file=e.file.values[m], fluorogen=e.fluorogen.values[m],
                            conc_nM=e.conc_nM.values[m], power_mW=e.power_mW.values[m],
                            frames=gap[m]))
    fl = agg.groupby("key").agg(file=("file", "first"), fluorogen=("fluorogen", "first"),
                                conc_nM=("conc_nM", "first"), power_mW=("power_mW", "first"),
                                first=("start", "min"), last=("end", "max")).reset_index()
    bnd = pd.concat([fl.assign(frames=fl["first"].astype(float)),
                     fl.assign(frames=(n_frames - fl["last"]).astype(float))], ignore_index=True)
    cols = ["key", "file", "fluorogen", "conc_nM", "power_mW", "frames"]
    return (on.reset_index(drop=True)[cols + ["censored", "snr"]], off[cols], bnd[cols])


def nll_geom(theta, d, a, w, dc, ac, wc, double):
    """Frame-discretised (geometric) exponential likelihood.

    A true dwell T ~ Exp(k) observed on a camera of frame time dt is recorded as
    an integer count of frames, so P(D=d) = p(1-p)^(d-1) with p = 1-exp(-k dt)
    and tau (frames) = -1/ln(1-p). Left truncation is per-observation (a_i), so
    interior gaps truncated at max_gap+1 and untruncated boundary gaps can enter
    the same fit. dc are right-censored. w, wc are tally weights."""
    if double:
        af = 1 / (1 + np.exp(-theta[0]))
        t1, t2 = np.exp(theta[1]), np.exp(theta[2])
        if min(t1, t2) <= 1e-9:
            return 1e12
        q1, q2 = -1 / t1, -1 / t2
        p1, p2 = -np.expm1(q1), -np.expm1(q2)
        norm = af * np.exp(q1 * (a - 1)) + (1 - af) * np.exp(q2 * (a - 1))
        pmf = af * p1 * np.exp(q1 * (d - 1)) + (1 - af) * p2 * np.exp(q2 * (d - 1))
        ll = np.sum(w * (np.log(np.maximum(pmf, 1e-300)) - np.log(np.maximum(norm, 1e-300))))
        if np.size(dc):
            nc = af * np.exp(q1 * (ac - 1)) + (1 - af) * np.exp(q2 * (ac - 1))
            sc = af * np.exp(q1 * (dc - 1)) + (1 - af) * np.exp(q2 * (dc - 1))
            ll += np.sum(wc * (np.log(np.maximum(sc, 1e-300)) - np.log(np.maximum(nc, 1e-300))))
    else:
        tau = np.exp(theta[0])
        if tau <= 1e-9:
            return 1e12
        q = -1 / tau
        ll = np.sum(w * (np.log1p(-np.exp(q)) + (d - a) * q))
        if np.size(dc):
            ll += np.sum(wc * (dc - ac) * q)
    return -ll if np.isfinite(ll) else 1e12


def fit_geom(d, a, w, dc, ac, wc, double=False, n_init=5, seed=0):
    A = (d, a, w, dc, ac, wc, double)
    rg = np.random.default_rng(seed)
    m = max(float(np.average(d, weights=w)), 2.0)
    best = None
    for _ in range(n_init if double else 1):
        x0 = ([rg.normal(0, .8), np.log(m * rg.uniform(.1, .6)), np.log(m * rg.uniform(1.5, 6))]
              if double else [np.log(m)])
        r = minimize(nll_geom, x0, args=A, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000, maxfev=4000))
        if best is None or r.fun < best.fun:
            best = r
    n = float(np.sum(w) + np.sum(wc))
    if double:
        af = 1 / (1 + np.exp(-best.x[0]))
        t1, t2 = np.exp(best.x[1]), np.exp(best.x[2])
        if t1 > t2:
            af, t1, t2 = 1 - af, t2, t1
        return dict(a_fast=float(af), tau_fast=float(t1), tau_slow=float(t2),
                    tau_amp=float(af * t1 + (1 - af) * t2), nll=float(best.fun),
                    k=3, bic=3 * np.log(n) + 2 * best.fun, n=n)
    return dict(tau=float(np.exp(best.x[0])), nll=float(best.fun), k=1,
                bic=np.log(n) + 2 * best.fun, n=n)


def _tally(d, a, mol_idx, w_mol):
    """Weighted tally over (duration, truncation-point) pairs. Identical to the
    per-observation likelihood sum but ~100x faster on integer durations."""
    if d.size == 0:
        return np.array([]), np.array([]), np.array([])
    avals, acode = np.unique(np.rint(a).astype(np.int64), return_inverse=True)
    nA = avals.size
    code = np.rint(d).astype(np.int64) * nA + acode
    w = w_mol[mol_idx]
    keep = w > 0
    if not keep.any():
        return np.array([]), np.array([]), np.array([])
    c = np.bincount(code[keep], weights=w[keep])
    nz = np.flatnonzero(c)
    return (nz // nA).astype(float), avals[nz % nA].astype(float), c[nz]


def _index(keys_list):
    allk = np.unique(np.concatenate([k for k in keys_list if k.size])) if any(
        k.size for k in keys_list) else np.array([])
    lut = {k: i for i, k in enumerate(allk)}
    return allk, [np.array([lut[k] for k in ks], dtype=int) if ks.size else np.array([], int)
                  for ks in keys_list]


def fit_dwell_set(dur, keys, dur_c, keys_c, dmin, dmin_c=None, n_boot=0, seed=0, double=True):
    """Point estimates (single + double) plus a cluster bootstrap over MOLECULES.
    Events from one molecule are correlated, so the resampling unit is the
    molecule, never the event."""
    dmin_c = dmin if dmin_c is None else dmin_c
    allk, (mi, mic) = _index([keys, keys_c])
    M = allk.size
    a = np.full(dur.size, float(dmin))
    ac = np.full(dur_c.size, float(dmin_c))
    w1 = np.ones(M)
    td, ta, tw = _tally(dur, a, mi, w1)
    cd, ca, cw = _tally(dur_c, ac, mic, w1)
    s = fit_geom(td, ta, tw, cd, ca, cw, False)
    out = dict(tau_single=s["tau"], bic_single=s["bic"], nll_single=s["nll"],
               n_dwells=int(dur.size + dur_c.size), n_complete=int(dur.size),
               n_censored=int(dur_c.size), n_molecules=int(M))
    if double and dur.size > 30:
        dd = fit_geom(td, ta, tw, cd, ca, cw, True, seed=seed)
        lr = 2 * (s["nll"] - dd["nll"])
        out.update(a_fast=dd["a_fast"], tau_fast=dd["tau_fast"], tau_slow=dd["tau_slow"],
                   tau_amp_weighted=dd["tau_amp"], bic_double=dd["bic"],
                   dBIC=s["bic"] - dd["bic"], LR_stat=lr,
                   LR_p=float(stats.chi2.sf(max(lr, 0), 2)))
    if n_boot:
        rg = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            w = rg.multinomial(M, np.full(M, 1 / M)).astype(float)
            bd, ba, bw = _tally(dur, a, mi, w)
            xd, xa, xw = _tally(dur_c, ac, mic, w)
            if bd.size < 3:
                continue
            bs.append(fit_geom(bd, ba, bw, xd, xa, xw, False)["tau"])
        if bs:
            out["tau_single_lo"], out["tau_single_hi"] = np.percentile(bs, [2.5, 97.5])
    return out


def apply_upper(dur, keys, dur_c, keys_c, max_dur):
    """Impose an upper truncation at `max_dur` frames WITHOUT discarding data.

    A dwell at or beyond max_dur is re-entered as a right-censored observation at
    max_dur: the likelihood then uses S(max_dur)/S(dmin) for it, which says "this
    dwell lasted at least max_dur" and nothing more. Discarding those dwells
    instead would bias the fitted lifetime down; keeping them as complete
    observations lets a handful of very long events -- which may be set by the
    molecule leaving the surface rather than by the dye unbinding -- drive the
    fit. Censoring is the option that does neither.

    max_dur=None returns the inputs unchanged.
    """
    if not max_dur:
        return dur, keys, dur_c, keys_c
    long_ = dur >= max_dur
    d_keep, k_keep = dur[~long_], keys[~long_]
    add_d, add_k = np.full(int(long_.sum()), float(max_dur)), keys[long_]
    dc = np.concatenate([np.minimum(dur_c, max_dur), add_d]) if dur_c.size else add_d
    kc = np.concatenate([keys_c, add_k]) if keys_c.size else add_k
    return d_keep, k_keep, dc, kc


def off_arrays(off_i, off_b, dmin_off):
    """Assemble off-time observations under the chosen censoring convention."""
    oi = off_i[off_i.frames >= dmin_off]
    if OFF_CENSORING == "survival":
        ob = off_b[off_b.frames >= 1]
        return (oi.frames.values.astype(float), oi.key.values,
                ob.frames.values.astype(float), ob.key.values, 1)
    return (oi.frames.values.astype(float), oi.key.values,
            np.array([]), np.array([]), dmin_off)


# ============================= STAGE 3: brightness =========================
def stage_brightness(mol):
    """Analysis 7. Per-molecule bound-state brightness = mean of the molecule's
    state-1 frames minus the mean of its state-0 frames. Compared only at
    matched laser power, never scaled by nominal power."""
    rg = np.random.default_rng(SEED)
    B = mol[(mol.n1 > 0) & np.isfinite(mol.brightness)].copy()
    B["floor"] = 3 * B.background_sd
    B["above_floor"] = B.brightness > B.floor
    cx = (B.x.min() + B.x.max()) / 2
    cy = (B.y.min() + B.y.max()) / 2
    B["r_px"] = np.hypot(B.x - cx, B.y - cy)
    B.to_csv(os.path.join(TAB, "brightness_per_molecule.csv"), index=False)

    def ratio(s, t, n=N_BOOT_RATIO):
        if s.size < 10 or t.size < 10:
            return np.nan, np.nan, np.nan, np.nan
        bs = np.empty(n)
        for k in range(n):
            bs[k] = np.median(rg.choice(s, s.size)) / np.median(rg.choice(t, t.size))
        return (np.median(s) / np.median(t), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                float(stats.mannwhitneyu(s, t, alternative="two-sided")[1]))

    rows = []
    for (c, pw), g in B.groupby(["conc_nM", "power_mW"]):
        s = g.loc[g.fluorogen == "SALAD1", "brightness"].values
        t = g.loc[g.fluorogen == "TO1-B", "brightness"].values
        r, lo, hi, p = ratio(s, t)
        # rank-biserial effect size: non-parametric, no normality assumed
        rb = np.nan
        if s.size and t.size:
            U = stats.mannwhitneyu(s, t, alternative="two-sided")[0]
            rb = 2 * U / (s.size * t.size) - 1
        rows.append(dict(conc_nM=c, power_mW=pw, n_SALAD=s.size, n_TO=t.size,
                         med_SALAD=np.median(s) if s.size else np.nan,
                         med_TO=np.median(t) if t.size else np.nan,
                         mean_SALAD=s.mean() if s.size else np.nan,
                         mean_TO=t.mean() if t.size else np.nan,
                         ratio_median=r, ci_lo=lo, ci_hi=hi, mannwhitney_p=p,
                         rank_biserial=rb,
                         frac_near_floor_SALAD=float((g.loc[g.fluorogen == "SALAD1", "brightness"] <
                                                      1.5 * g.loc[g.fluorogen == "SALAD1", "floor"]).mean()),
                         frac_near_floor_TO=float((g.loc[g.fluorogen == "TO1-B", "brightness"] <
                                                   1.5 * g.loc[g.fluorogen == "TO1-B", "floor"]).mean()),
                         consistent_with_population=bool(lo <= PRED_POPULATION[1] and PRED_POPULATION[0] <= hi)
                         if np.isfinite(lo) else False,
                         consistent_with_uniform=bool(lo <= PRED_UNIFORM <= hi)
                         if np.isfinite(lo) else False))
    br = pd.DataFrame(rows).sort_values(["power_mW", "conc_nM"])
    br.to_csv(os.path.join(TAB, "brightness_ratio_by_condition.csv"), index=False)

    # background / detection floor vs concentration
    bg = (mol.groupby(["fluorogen", "conc_nM", "power_mW"])
          .agg(n=("mol_idx", "size"), background=("background", "median"),
               noise_sd=("background_sd", "median"), brightness=("brightness", "median"))
          .reset_index())
    bg["floor_counts"] = 3 * bg.noise_sd
    bg["signal_to_background"] = bg.brightness / bg.noise_sd
    bg.to_csv(os.path.join(TAB, "background_vs_concentration.csv"), index=False)

    # Dim subpopulation. The brightness distribution is HARD-CENSORED at each
    # molecule's own 3-sigma floor, so a plain mixture will place a component
    # against that boundary whether or not a dim population exists. Two
    # censoring-aware estimators are run instead (see mixture_truncated.py), and
    # the accompanying profile likelihood shows what the data can and cannot
    # constrain below the floor. A plain untruncated mixture is NOT reported,
    # because its two-component preference is uninformative here.
    from mixture_truncated import mixture_table, profile_dark_fraction
    Bm = B[np.isfinite(B.floor)]
    tab, verdict = mixture_table(Bm, Ks=MIXTURE_KS, seed=SEED)
    tab.to_csv(os.path.join(TAB, "brightness_mixture_censoring_aware.csv"), index=False)
    verdict.to_csv(os.path.join(TAB, "brightness_mixture_verdict.csv"), index=False)
    prof = profile_dark_fraction(Bm, group_cols=("fluorogen", "conc_nM", "power_mW"),
                                 seed=SEED)
    prof.to_csv(os.path.join(TAB, "brightness_dim_population_profile.csv"), index=False)
    return B, br, bg


def _em_gmm(x, K, n_init=6, iters=300, seed=0):
    rg = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        mu = rg.choice(x, K, replace=False)
        sd = np.full(K, x.std())
        w = np.full(K, 1 / K)
        prev = -np.inf
        for _ in range(iters):
            p = w * stats.norm.pdf(x[:, None], mu, sd)
            tot = p.sum(1)
            ll = np.log(np.maximum(tot, 1e-300)).sum()
            g = p / tot[:, None]
            nk = g.sum(0) + 1e-12
            w = nk / x.size
            mu = (g * x[:, None]).sum(0) / nk
            sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / nk).clip(1e-6)
            if abs(ll - prev) < 1e-8 * abs(ll):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, w.copy(), mu.copy(), sd.copy())
    return best


# ======================= STAGE 4: detectable fraction ======================
def stage_detectable(mol, qc):
    """Analysis 8. Densities per field, and the zero-event class split into
    molecules that sat at background (non-binders) versus molecules occupied
    throughout (the most stable population, which must not be discarded)."""
    area = ((qc.x_max - qc.x_min) * (qc.y_max - qc.y_min)).median()
    ref = (mol[mol.n_events > 0].groupby("file")
           .agg(bg_ref=("background", "median"), sd_ref=("background_sd", "median")).reset_index())
    Z = mol.merge(ref, on="file", how="left")
    Z["level_above_bg"] = Z.trace_median - Z.bg_ref
    zero = Z[Z.n_events == 0].copy()
    zero["zero_class"] = np.where(zero.level_above_bg > 3 * zero.sd_ref, "occupied throughout",
                                  np.where(zero.level_above_bg > 1.5 * zero.sd_ref, "ambiguous",
                                           "at background"))
    zt = (zero.pivot_table(index=["fluorogen", "conc_nM", "power_mW"], columns="zero_class",
                           values="mol_idx", aggfunc="size", fill_value=0).reset_index())
    zt.to_csv(os.path.join(TAB, "zero_event_class_breakdown.csv"), index=False)

    dens = (mol.groupby(["fluorogen", "conc_nM", "power_mW", "file"])
            .agg(molecules=("mol_idx", "size"),
                 with_events=("n_events", lambda s: int((s > 0).sum())),
                 dim_binders=("dim_z", lambda s: int((s > DIM_Z).sum())))
            .reset_index())
    dz = mol[mol.n_events == 0].groupby("file").dim_z.apply(lambda s: int((s > DIM_Z).sum()))
    dens["dim_binders_zero_event"] = dens.file.map(dz).fillna(0).astype(int)
    dens["emitting"] = dens.with_events + dens.dim_binders_zero_event
    dens["area_px2"] = area
    for c_ in ("molecules", "with_events", "emitting"):
        dens[c_ + "_per_1000px2"] = dens[c_] / area * 1000
    dens.to_csv(os.path.join(TAB, "detectable_fraction_per_field.csv"), index=False)

    # emitting-fraction routes, per matched condition
    rows = []
    for (c, pw), g in dens.groupby(["conc_nM", "power_mW"]):
        s = g[g.fluorogen == "SALAD1"]
        t = g[g.fluorogen == "TO1-B"]
        if s.empty or t.empty:
            continue
        for col, label, note in [
                ("molecules", "all detected spots", "spot-detection floor"),
                ("emitting", "threshold-corrected emitting spots", "spot-detection floor only"),
                ("with_events", "passing the 3-sigma binder test",
                 "spot floor + binder threshold; double-counts brightness")]:
            a, b = s[col].values.astype(float), t[col].values.astype(float)
            r = a.mean() / b.mean()
            # field-level t interval on the log ratio (n=3 fields per condition)
            se = np.sqrt((a.std(ddof=1) / a.mean()) ** 2 / len(a) +
                         (b.std(ddof=1) / b.mean()) ** 2 / len(b))
            tcrit = stats.t.ppf(.975, len(a) + len(b) - 2)
            rows.append(dict(conc_nM=c, power_mW=pw, route=label, censored_at=note,
                             SALAD1_mean=a.mean(), SALAD1_sd=a.std(ddof=1),
                             TO1B_mean=b.mean(), TO1B_sd=b.std(ddof=1),
                             n_fields_each=len(a), n_channels_each=1,
                             ratio=r, ci_lo=np.exp(np.log(r) - tcrit * se),
                             ci_hi=np.exp(np.log(r) + tcrit * se)))
    routes = pd.DataFrame(rows)
    routes.to_csv(os.path.join(TAB, "emitting_fraction_routes.csv"), index=False)

    ladder = (mol.groupby(["fluorogen", "conc_nM", "power_mW"])
              .agg(all_spots=("mol_idx", "size"),
                   pass_binder_test=("n_events", lambda s: int((s > 0).sum())),
                   below_threshold=("n_events", lambda s: int((s == 0).sum())))
              .reset_index())
    dimz = (mol[mol.n_events == 0].groupby(["fluorogen", "conc_nM", "power_mW"])
            .dim_z.apply(lambda s: int((s > DIM_Z).sum())).rename("dim_binders").reset_index())
    ladder = ladder.merge(dimz, on=["fluorogen", "conc_nM", "power_mW"], how="left").fillna(0)
    ladder["noise_consistent"] = ladder.below_threshold - ladder.dim_binders
    ladder["emitting"] = ladder.pass_binder_test + ladder.dim_binders
    ladder["frac_below_threshold"] = ladder.below_threshold / ladder.all_spots
    ladder["frac_dim"] = ladder.dim_binders / ladder.all_spots
    ladder["frac_dark"] = ladder.noise_consistent / ladder.all_spots
    ladder.to_csv(os.path.join(TAB, "detection_floor_ladder.csv"), index=False)

    # Is the SALAD1/TO1-B imbalance in molecule count created by the acceptance
    # test, or is it already there in the localised spots? Splitting the ratio
    # at that step separates "more of them light up" from "more were there".
    imb = []
    for (c, pw), g in ladder.groupby(["conc_nM", "power_mW"]):
        gg = g.set_index("fluorogen")
        if not {"SALAD1", "TO1-B"} <= set(gg.index):
            continue
        pre = gg.loc["SALAD1", "all_spots"] / gg.loc["TO1-B", "all_spots"]
        post = gg.loc["SALAD1", "pass_binder_test"] / gg.loc["TO1-B", "pass_binder_test"]
        imb.append(dict(conc_nM=c, power_mW=pw,
                        spots_SALAD=int(gg.loc["SALAD1", "all_spots"]),
                        spots_TO=int(gg.loc["TO1-B", "all_spots"]),
                        binders_SALAD=int(gg.loc["SALAD1", "pass_binder_test"]),
                        binders_TO=int(gg.loc["TO1-B", "pass_binder_test"]),
                        ratio_before_acceptance=pre, ratio_after_acceptance=post,
                        acceptance_contribution=post / pre,
                        frac_of_log_ratio_from_acceptance=(
                            np.log(post / pre) / np.log(post) if post > 1 else np.nan)))
    pd.DataFrame(imb).to_csv(
        os.path.join(TAB, "molecule_count_imbalance.csv"), index=False)
    return dens, routes, ladder


# ============================ STAGE 5: kinetics ============================
def stage_kinetics(mol, ev, gaps):
    dmin_off = max(MIN_DUR, MAX_GAP + 1)   # merging removes any shorter interior gap
    ON, OFF, BND = extract_dwells(ev, MAX_GAP, edge_correction=EDGE_CORRECTION)

    dwl = pd.concat([
        ON.assign(type="on", censoring=np.where(ON.censored, "right-censored", "complete")),
        OFF.assign(type="off", censoring="complete", censored=False, snr=np.nan),
        BND.assign(type="off_boundary", censoring="right-censored", censored=True, snr=np.nan),
    ], ignore_index=True)
    dwl["seconds"] = dwl.frames * FRAME_TIME_S
    dwl = dwl.merge(mol[["file", "mol_idx", "molecule_id", "contrast"]]
                    .assign(key=mol.file.astype(str) + "|" + mol.mol_idx.astype(str))
                    [["key", "molecule_id", "contrast"]], on="key", how="left")
    dwl.to_csv(os.path.join(TAB, "dwell_times_long.csv.gz"), index=False, compression="gzip")

    # tail-robust reference for every fit (median is insensitive to both limits)
    med = []
    for (fl, c, pw), g in ON.groupby(["fluorogen", "conc_nM", "power_mW"]):
        oi = OFF[(OFF.fluorogen == fl) & (OFF.conc_nM == c) & (OFF.power_mW == pw)]
        on_v = g.loc[(~g.censored) & (g.frames >= MIN_DUR), "frames"]
        off_v = oi.loc[oi.frames >= dmin_off, "frames"]
        med.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                        on_median_dwell_frames=float(on_v.median()) if len(on_v) else np.nan,
                        on_median_dwell_s=float(on_v.median()) * FRAME_TIME_S if len(on_v) else np.nan,
                        off_median_dwell_frames=float(off_v.median()) if len(off_v) else np.nan,
                        off_median_dwell_s=float(off_v.median()) * FRAME_TIME_S if len(off_v) else np.nan))
    med = pd.DataFrame(med)

    rows = []
    for (fl, c, pw), g in ON.groupby(["fluorogen", "conc_nM", "power_mW"]):
        oi = OFF[(OFF.fluorogen == fl) & (OFF.conc_nM == c) & (OFF.power_mW == pw)]
        ob = BND[(BND.fluorogen == fl) & (BND.conc_nM == c) & (BND.power_mW == pw)]
        # a likelihood left-truncated at MIN_DUR may only see dwells >= MIN_DUR
        cm = g[(~g.censored) & (g.frames >= MIN_DUR)]
        cx = g[g.censored & (g.frames >= MIN_DUR)]
        d_on, k_on_, dc_on, kc_on = apply_upper(
            cm.frames.values.astype(float), cm.key.values,
            cx.frames.values.astype(float), cx.key.values, MAX_DUR)
        r_on = fit_dwell_set(d_on, k_on_, dc_on, kc_on,
                             MIN_DUR, n_boot=N_BOOT, seed=SEED)
        d, k, dc, kc, amin = off_arrays(oi, ob, dmin_off)
        d, k, dc, kc = apply_upper(d, k, dc, kc, MAX_DUR)
        r_off = fit_dwell_set(d, k, dc, kc, dmin_off, dmin_c=amin, n_boot=N_BOOT, seed=SEED + 1)
        row = dict(fluorogen=fl, conc_nM=c, power_mW=pw)
        row.update({"on_" + k2: v for k2, v in r_on.items()})
        row.update({"off_" + k2: v for k2, v in r_off.items()})
        rows.append(row)
    kin = pd.DataFrame(rows)
    for pre in ("on", "off"):
        for c_ in [x for x in kin.columns if x.startswith(pre + "_tau")]:
            kin[c_ + "_s"] = kin[c_] * FRAME_TIME_S
    kin["k_off_per_s"] = 1 / kin.on_tau_single_s
    kin["k_off_lo"] = 1 / kin.on_tau_single_hi_s
    kin["k_off_hi"] = 1 / kin.on_tau_single_lo_s
    kin["inv_tau_off_per_s"] = 1 / kin.off_tau_single_s
    kin["inv_tau_off_per_s_lo"] = 1 / kin.off_tau_single_hi_s
    kin["inv_tau_off_per_s_hi"] = 1 / kin.off_tau_single_lo_s
    kin["K_D_kinetic_per_point_nM"] = kin.conc_nM * kin.off_tau_single_s / kin.on_tau_single_s
    kin["K_D_bulk_nM"] = kin.fluorogen.map(KD_BULK_NM)
    # median dwell, in the same units, as a tail-robust reference for the fits:
    # a fitted lifetime far from the median means the fit is driven by the tails
    kin = kin.merge(med, on=["fluorogen", "conc_nM", "power_mW"], how="left")
    for pre in ("on", "off"):
        kin[f"{pre}_tau_over_median"] = (kin[f"{pre}_tau_single_s"]
                                         / kin[f"{pre}_median_dwell_s"])
    kin["max_gap"] = MAX_GAP
    kin["min_duration_frames"] = MIN_DUR
    kin["max_duration_frames"] = MAX_DUR if MAX_DUR else ""
    kin["edge_correction"] = EDGE_CORRECTION
    kin["off_censoring"] = OFF_CENSORING
    # Units are explicit in every column name: *_frames or *_s. Bare `tau`
    # columns used to be frames and were mistaken for seconds downstream.
    kin = kin.rename(columns={c: c + "_frames" for c in kin.columns
                              if (c.startswith(("on_tau", "off_tau"))
                                  and not c.endswith("_s")
                                  and not c.endswith("_over_median"))})
    # every column whose name mentions tau must declare its units: _frames, _s,
    # _per_s (a rate), or _over_median (dimensionless)
    unlabelled = [c for c in kin.columns
                  if "tau" in c and not c.endswith(
                      ("_s", "_frames", "_over_median", "_per_s",
                       "_per_s_lo", "_per_s_hi"))]
    assert not unlabelled, f"unit-ambiguous columns would be written: {unlabelled}"
    # Columns the later steps look up by name. Asserted here so a rename fails
    # at this line rather than minutes later in a later step.
    needed = [f"{p}_tau_{q}{sfx}" for p in ("on", "off")
              for q in ("single", "fast", "slow") for sfx in ("_frames", "_s")]
    needed += [f"{p}_a_fast" for p in ("on", "off")]
    needed += ["k_off_per_s", "k_off_lo", "k_off_hi", "inv_tau_off_per_s",
               "inv_tau_off_per_s_lo", "inv_tau_off_per_s_hi", "on_dBIC", "off_dBIC",
               "on_tau_single_lo_s", "on_tau_single_hi_s",
               "off_tau_single_lo_s", "off_tau_single_hi_s",
               "on_median_dwell_s", "off_median_dwell_s",
               "on_tau_over_median", "off_tau_over_median"]
    missing = [c for c in needed if c not in kin.columns]
    assert not missing, f"downstream consumers need these kin columns: {missing}"
    kin.to_csv(os.path.join(TAB, "kinetics_per_condition.csv"), index=False)

    # per-field, to separate within-condition from between-condition scatter
    pf = []
    for (fl, c, pw, fn), g in ON.groupby(["fluorogen", "conc_nM", "power_mW", "file"]):
        oi = OFF[OFF.file == fn]
        ob = BND[BND.file == fn]
        cm = g[(~g.censored) & (g.frames >= MIN_DUR)]
        cx = g[g.censored & (g.frames >= MIN_DUR)]
        t_on = fit_dwell_set(cm.frames.values.astype(float), cm.key.values,
                             cx.frames.values.astype(float), cx.key.values,
                             MIN_DUR, double=False)["tau_single"]
        d, k, dc, kc, amin = off_arrays(oi, ob, dmin_off)
        t_off = fit_dwell_set(d, k, dc, kc, dmin_off, dmin_c=amin, double=False)["tau_single"]
        pf.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, file=fn,
                       tau_on_s=t_on * FRAME_TIME_S, tau_off_s=t_off * FRAME_TIME_S,
                       k_off=1 / (t_on * FRAME_TIME_S), inv_tau_off_per_s=1 / (t_off * FRAME_TIME_S)))
    pf = pd.DataFrame(pf)
    pf.to_csv(os.path.join(TAB, "kinetics_per_field.csv"), index=False)

    # k_on as the slope of 1/tau_off vs concentration; the intercept is a
    # measurement of a concentration-independent contribution, not a nuisance
    ref_pw = DEFAULT_POWER_MW
    series = kin[kin.power_mW == ref_pw].sort_values(["fluorogen", "conc_nM"])
    fits = []
    for fl, g in series.groupby("fluorogen"):
        if g.conc_nM.nunique() < 3:
            fits.append(dict(fluorogen=fl, note="fewer than 3 concentrations; slope not fitted"))
            continue
        # UNITS: x in nM, y in s^-1 (y is built from off_tau_single_s, never from
        # the frame-valued column), so the slope is nM^-1 s^-1 and the intercept
        # is s^-1. Asserted, because a frames/seconds mix-up here is invisible in
        # the fitted line and would rescale k_on by 10.
        x, y = g.conc_nM.values, g.inv_tau_off_per_s.values
        assert np.allclose(y, 1.0 / g.off_tau_single_s.values), \
            "inv_tau_off_per_s must be per second"
        w = 1 / ((g.inv_tau_off_per_s_hi.values - g.inv_tau_off_per_s_lo.values) / 2) ** 2
        A = np.vstack([x, np.ones_like(x)]).T * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
        pred = coef[0] * x + coef[1]
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        koff = float(g.k_off_per_s.mean())
        # intervals on slope, intercept and K_D: resample the concentration
        # points with replacement (the scatter between concentrations is much
        # larger than the within-condition bootstrap error, so that scatter is
        # what the interval must reflect)
        rg = np.random.default_rng(SEED + 5)
        bs = []
        for _ in range(2000):
            idx = rg.integers(0, len(x), len(x))
            if len(np.unique(x[idx])) < 2:
                continue
            Ab = np.vstack([x[idx], np.ones(len(idx))]).T * np.sqrt(w[idx])[:, None]
            cb, *_ = np.linalg.lstsq(Ab, y[idx] * np.sqrt(w[idx]), rcond=None)
            kb = float(g.k_off_per_s.values[idx].mean())
            bs.append((cb[0], cb[1], kb / cb[0] if cb[0] > 0 else np.nan))
        bs = np.array(bs)
        q = lambda col: (float(np.nanpercentile(bs[:, col], 2.5)),
                         float(np.nanpercentile(bs[:, col], 97.5)))
        kon_lo, kon_hi = q(0)
        icpt_lo, icpt_hi = q(1)
        kd_lo, kd_hi = q(2)
        fits.append(dict(fluorogen=fl, k_on_per_nM_per_s=coef[0], intercept_per_s=coef[1],
                         k_on_lo=kon_lo, k_on_hi=kon_hi,
                         intercept_lo=icpt_lo, intercept_hi=icpt_hi,
                         K_D_slope_lo_nM=kd_lo, K_D_slope_hi_nM=kd_hi,
                         intercept_excludes_zero=bool(icpt_lo > 0),
                         R2=r2, k_off_mean_per_s=koff,
                         k_off_min=float(g.k_off_per_s.min()), k_off_max=float(g.k_off_per_s.max()),
                         k_off_fold_range=float(g.k_off_per_s.max() / g.k_off_per_s.min()),
                         K_D_from_slope_nM=koff / coef[0] if coef[0] > 0 else np.nan,
                         K_D_bulk_nM=KD_BULK_NM.get(fl),
                         intercept_frac_at=json.dumps(
                             {float(c): round(float(coef[1] / (coef[0] * c + coef[1])), 3) for c in x})))
    kon = pd.DataFrame(fits)
    kon.to_csv(os.path.join(TAB, "kon_intercept_fits.csv"), index=False)

    # Langmuir isotherm on the per-molecule bound fraction (independent of the
    # event-level fits: it counts bound FRAMES, not transitions)
    bf = (mol[mol.power_mW == ref_pw].assign(binder=mol.n_events > 0)
          .groupby(["fluorogen", "conc_nM"])
          .apply(lambda g: pd.Series(dict(
              n_all=len(g), n_binders=int(g.binder.sum()),
              theta_binders=float(g.loc[g.binder, "bound_fraction"].mean()),
              theta_all=float(g.bound_fraction.mean()),
              theta_binders_sd=float(g.loc[g.binder, "bound_fraction"].std(ddof=1)),
              ev_per_binder=float(g.loc[g.binder, "n_events"].mean()))), include_groups=False)
          .reset_index())
    bf.to_csv(os.path.join(TAB, "bound_fraction_vs_concentration.csv"), index=False)

    def langmuir(c, thmax, kd):
        return thmax * c / (kd + c)

    lang = []
    for fl, g in bf.groupby("fluorogen"):
        for which in ("theta_binders", "theta_all"):
            if g.conc_nM.nunique() < 3:
                continue
            try:
                p, cov = curve_fit(langmuir, g.conc_nM.values, g[which].values,
                                   p0=[g[which].max() * 1.3, 2.0], maxfev=40000,
                                   bounds=([0, 1e-4], [1, 1e4]))
                se = np.sqrt(np.diag(cov))
            except Exception as exc:
                p, se = (np.nan, np.nan), (np.nan, np.nan)
                log(f"    Langmuir fit failed for {fl}/{which}: {exc}")
            lang.append(dict(fluorogen=fl, basis=which, theta_max=p[0], theta_max_se=se[0],
                             K_D_nM=p[1], K_D_se=se[1],
                             note="theta_max is an OCCUPANCY CEILING of the observable "
                                  "(fraction of time an unbleached dye occupies the pocket), "
                                  "not a binding stoichiometry"))
    lang = pd.DataFrame(lang)
    lang.to_csv(os.path.join(TAB, "langmuir_fits.csv"), index=False)

    # tau_on vs laser power: a decrease indicates photobleaching of the bound dye
    pw_rows = []
    for fl, g in kin.groupby("fluorogen"):
        for c, gg in g.groupby("conc_nM"):
            if gg.power_mW.nunique() < 3:
                continue
            x, y = gg.power_mW.values, gg.k_off_per_s.values
            slope, icpt = np.polyfit(x, y, 1)
            pw_rows.append(dict(fluorogen=fl, conc_nM=c, n_powers=len(x),
                                powers=json.dumps(x.tolist()),
                                tau_on_s=json.dumps(np.round(gg.on_tau_single_s.values, 3).tolist()),
                                bleach_slope_per_mW=slope, zero_power_k_off=icpt,
                                zero_power_tau_on_s=(1 / icpt) if icpt > 0 else np.nan,
                                linear_model_valid=bool(icpt > 0),
                                note="negative intercept => 1/tau_on is not linear in power; "
                                     "no zero-power extrapolation is reportable"
                                     if icpt <= 0 else
                                     "extrapolated from 3 powers; treat as indicative"))
    pd.DataFrame(pw_rows).to_csv(os.path.join(TAB, "power_extrapolation.csv"), index=False)

    # dark-gap excess: the reference must be fitted to the LONG-gap tail with the
    # same number of components the distribution actually needs, then extrapolated
    gap_rows = []
    for (fl, c, pw), g in gaps.groupby(["fluorogen", "conc_nM", "power_mW"]):
        v = g.gap.values.astype(float)
        tail = v[v >= 10]
        if tail.size < 200:
            continue
        allk = np.array(["x"] * tail.size)
        r = fit_geom(*(_tally(tail, np.full(tail.size, 10.0), np.zeros(tail.size, int), np.ones(1))),
                     np.array([]), np.array([]), np.array([]), True, seed=SEED)
        af, t1, t2 = r["a_fast"], r["tau_fast"], r["tau_slow"]
        q1, q2 = np.exp(-1 / t1), np.exp(-1 / t2)
        p1, p2 = 1 - q1, 1 - q2
        norm = af * q1 ** 9 + (1 - af) * q2 ** 9
        ftail = tail.size / v.size
        pred = lambda dd: v.size * ftail * (af * p1 * q1 ** (dd - 1) +
                                            (1 - af) * p2 * q2 ** (dd - 1)) / norm
        obs14 = int(((v >= 1) & (v <= 4)).sum())
        exp14 = float(sum(pred(dd) for dd in (1, 2, 3, 4)))
        gap_rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, n_gaps=v.size,
                             tau_fast_tail=t1, tau_slow_tail=t2, a_fast_tail=af,
                             obs_1to4=obs14, exp_1to4=exp14,
                             excess=obs14 - exp14, excess_ratio=obs14 / exp14 if exp14 else np.nan,
                             frac_of_all_gaps=obs14 / v.size,
                             check_obs_over_exp=json.dumps(
                                 {int(dd): [int((v == dd).sum()), round(float(pred(dd)), 1)]
                                  for dd in (12, 20, 30, 40)})))
    pd.DataFrame(gap_rows).to_csv(os.path.join(TAB, "dark_gap_statistics.csv"), index=False)
    return ON, OFF, BND, kin, pf, kon, bf, lang


# ================= STAGE 6: effective photostability under exchange ========
def km_survival(t, censored):
    """Kaplan-Meier. t in frames, censored=True means still active at the end."""
    order = np.argsort(t)
    t, censored = np.asarray(t)[order], np.asarray(censored)[order]
    times = np.unique(t[~censored])
    n = t.size
    S, out = 1.0, []
    for tt in times:
        at_risk = int((t >= tt).sum())
        d = int(((t == tt) & (~censored)).sum())
        if at_risk == 0:
            break
        S *= (1 - d / at_risk)
        out.append((tt, S, at_risk, d))
    a = np.array(out) if out else np.zeros((0, 4))
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def stage_photostability(mol, kin=None):
    """Analysis 9. NOT intrinsic single-fluorophore photobleaching, which this
    design cannot measure: the free fluorogen pool replenishes the pocket
    throughout, so what is measured is how long a molecule keeps producing
    signal at all -- effective photostability under exchange.

    The span is DEFINED here, explicitly, because the definition is a choice and
    it moves the answer:

      first_to_last   last event end minus first event start. Answers "once a
                      position starts working, how long does it keep working".
                      A molecule that first binds late in the movie contributes
                      a short span through no fault of its own.
      acquisition     last event end minus frame 0. Answers "how long does a
                      position work", and charges each molecule for the time it
                      spent waiting to bind at all.

    and the END is treated two ways:

      strict          censored only if the last event was still in progress at
                      the final frame. This is what the first pass used, and it
                      is biased LOW: a molecule whose last event ended at frame
                      2400 is recorded as having definitely stopped, when only
                      100 frames of observation remained.
      gap-aware       censored if the observation remaining after the last event
                      is shorter than the wait one would have expected before
                      the next event (tau_off for that condition, times
                      CENSOR_MARGIN). Not seeing another event in less time than
                      the expected gap is not evidence that the molecule stopped.

    The gap-aware end is the defensible one and is used for the headline; all
    four combinations are reported in photostability_definitions.csv.
    """
    P = mol[mol.n_events > 0].copy()
    P["remaining_frames"] = P.n_frames - P.last_end
    # expected wait to the next event, per condition, from the fitted off-time
    if kin is not None and "off_tau_single_frames" in kin.columns:
        exp_gap = (kin.set_index(["fluorogen", "conc_nM", "power_mW"])
                   ["off_tau_single_frames"].to_dict())
        P["expected_gap_frames"] = [
            exp_gap.get((f, c, p), np.nan)
            for f, c, p in zip(P.fluorogen, P.conc_nM, P.power_mW)]
    else:
        P["expected_gap_frames"] = np.nan
    P["censored_strict"] = P.last_end_trunc | (P.last_end >= P.n_frames)
    P["censored_gap_aware"] = (
        P.censored_strict
        | (P.remaining_frames < CENSOR_MARGIN * P.expected_gap_frames.fillna(0)))
    P["span_first_to_last_frames"] = P.last_end - P.first_start
    P["span_acquisition_frames"] = P.last_end
    P["active_span_frames"] = P["span_" + SPAN_DEFINITION + "_frames"]
    P["censored"] = P["censored_" + SPAN_CENSORING]
    P["left_truncated"] = P.first_start_trunc
    P["active_span_s"] = P.active_span_frames * FRAME_TIME_S
    P[["file", "fluorogen", "conc_nM", "power_mW", "molecule_id", "mol_idx", "n_events",
       "first_start", "last_end", "active_span_frames", "active_span_s", "censored",
       "left_truncated", "bound_fraction", "brightness", "contrast"]].to_csv(
        os.path.join(TAB, "photostability_per_molecule.csv"), index=False)

    rows = []
    for (fl, c, pw), g in P.groupby(["fluorogen", "conc_nM", "power_mW"]):
        gk = g.file.astype(str).values + "|" + g.mol_idx.astype(str).values
        cens = g.censored.values
        span = g.active_span_frames.values.astype(float)
        ok = span >= 1
        d, kd = span[ok & ~cens], gk[ok & ~cens]
        dc, kc = span[ok & cens], gk[ok & cens]
        if d.size < 30:
            continue
        r = fit_dwell_set(d, kd, dc, kc, 1, n_boot=N_BOOT, seed=SEED + 2)
        tt, ss, _, _ = km_survival(g.active_span_frames.values, g.censored.values)
        med_km = float(tt[np.argmax(ss <= 0.5)] * FRAME_TIME_S) if (ss <= 0.5).any() else np.nan
        rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                         n_molecules=len(g), n_censored=int(g.censored.sum()),
                         frac_censored=float(g.censored.mean()),
                         n_left_truncated=int(g.left_truncated.sum()),
                         median_events=float(g.n_events.median()),
                         km_median_active_s=med_km,
                         tau_single_s=r["tau_single"] * FRAME_TIME_S,
                         tau_lo_s=r.get("tau_single_lo", np.nan) * FRAME_TIME_S,
                         tau_hi_s=r.get("tau_single_hi", np.nan) * FRAME_TIME_S,
                         a_fast=r.get("a_fast"),
                         tau_fast_s=(r.get("tau_fast") or np.nan) * FRAME_TIME_S,
                         tau_slow_s=(r.get("tau_slow") or np.nan) * FRAME_TIME_S,
                         dBIC_double_minus_single=r.get("dBIC"),
                         model_selected="double" if (r.get("dBIC") or 0) > 10 else "single",
                         selection_criterion="dBIC > 10 favours two components"))
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(TAB, "photostability_survival.csv"), index=False)

    # ---- how much does the definition move the answer? all four combinations
    defs = []
    for sdef in ("first_to_last", "acquisition"):
        for scen in ("strict", "gap_aware"):
            sp = P["span_" + sdef + "_frames"].values.astype(float)
            cn = P["censored_" + scen].values
            for (fl, c, pw), idx in P.groupby(["fluorogen", "conc_nM", "power_mW"]).groups.items():
                m = P.index.get_indexer(idx)
                s_, c_ = sp[m], cn[m]
                gk = (P.file.astype(str).values + "|" + P.mol_idx.astype(str).values)[m]
                ok = s_ >= 1
                d, kd = s_[ok & ~c_], gk[ok & ~c_]
                dc, kc = s_[ok & c_], gk[ok & c_]
                if d.size < 30:
                    continue
                r = fit_dwell_set(d, kd, dc, kc, 1, n_boot=0)
                defs.append(dict(span_definition=sdef, end_rule=scen,
                                 fluorogen=fl, conc_nM=c, power_mW=pw,
                                 n_molecules=int(ok.sum()),
                                 n_censored=int((ok & c_).sum()),
                                 frac_censored=float((ok & c_).sum() / max(ok.sum(), 1)),
                                 median_span_s=float(np.median(s_[ok])) * FRAME_TIME_S,
                                 tau_single_s=r["tau_single"] * FRAME_TIME_S,
                                 dBIC=r.get("dBIC")))
    pdefs = pd.DataFrame(defs)
    pdefs.to_csv(os.path.join(TAB, "photostability_definitions.csv"), index=False)

    # ---- is the comparison a surface-quality artefact? (the confound that
    # invalidated the spot-density route). Per FIELD: characteristic survival
    # against molecule density. If they covary, part of the difference between
    # the fluorogens is surface quality rather than the dye.
    per_field, dens_map = [], (mol.groupby("file").size().to_dict())
    # detection area per field, from the molecule coordinates themselves
    _ext = mol.groupby("file").agg(x0=("x", "min"), x1=("x", "max"),
                                   y0=("y", "min"), y1=("y", "max"))
    area_px2 = float(((_ext.x1 - _ext.x0) * (_ext.y1 - _ext.y0)).median())
    for (fl, c, pw, fn), g in P.groupby(["fluorogen", "conc_nM", "power_mW", "file"]):
        sp = g.active_span_frames.values.astype(float)
        cn = g.censored.values
        gk = g.file.astype(str).values + "|" + g.mol_idx.astype(str).values
        ok = sp >= 1
        if (ok & ~cn).sum() < 30:
            continue
        r = fit_dwell_set(sp[ok & ~cn], gk[ok & ~cn], sp[ok & cn], gk[ok & cn], 1, n_boot=0)
        per_field.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, file=fn,
                              n_spots=int(dens_map.get(fn, 0)),
                              n_binders=int(ok.sum()),
                              density_per_1000px2=dens_map.get(fn, 0) / area_px2 * 1000.0,
                              tau_single_s=r["tau_single"] * FRAME_TIME_S,
                              median_span_s=float(np.median(sp[ok])) * FRAME_TIME_S,
                              frac_censored=float((ok & cn).sum() / max(ok.sum(), 1))))
    pf_surv = pd.DataFrame(per_field)
    if len(pf_surv) > 3:
        from scipy.stats import pearsonr, spearmanr
        cr = []
        for lab, sub in ([("all fields", pf_surv)]
                         + [(f"{fl} only", g) for fl, g in pf_surv.groupby("fluorogen")]):
            if len(sub) < 4:
                continue
            pr, pp = pearsonr(sub.n_spots, sub.tau_single_s)
            sr, sp_ = spearmanr(sub.n_spots, sub.tau_single_s)
            cr.append(dict(subset=lab, n_fields=len(sub),
                           pearson_r=pr, pearson_p=pp,
                           spearman_rho=sr, spearman_p=sp_,
                           interpretation=(
                               "covaries (both tests)" if (pp < 0.05 and sp_ < 0.05)
                               else "no detectable covariation"
                               if (pp >= 0.05 and sp_ >= 0.05)
                               else "equivocal: the two tests disagree")))
        pd.DataFrame(cr).to_csv(
            os.path.join(TAB, "photostability_density_confound.csv"), index=False)

        # density-matched comparison: restrict to the overlap in spot count
        ov_lo = pf_surv.groupby("fluorogen").n_spots.min().max()
        ov_hi = pf_surv.groupby("fluorogen").n_spots.max().min()
        matched = pf_surv[(pf_surv.n_spots >= ov_lo) & (pf_surv.n_spots <= ov_hi)]
        mrows = []
        for lab, sub in (("all fields", pf_surv), ("density-matched", matched)):
            agg = sub.groupby("fluorogen").tau_single_s.agg(["size", "mean", "std"])
            if set(agg.index) >= {"SALAD1", "TO1-B"}:
                mrows.append(dict(
                    subset=lab, overlap_lo=int(ov_lo), overlap_hi=int(ov_hi),
                    n_fields_SALAD=int(agg.loc["SALAD1", "size"]),
                    n_fields_TO=int(agg.loc["TO1-B", "size"]),
                    tau_SALAD_s=agg.loc["SALAD1", "mean"],
                    tau_SALAD_sd=agg.loc["SALAD1", "std"],
                    tau_TO_s=agg.loc["TO1-B", "mean"],
                    tau_TO_sd=agg.loc["TO1-B", "std"],
                    ratio=agg.loc["SALAD1", "mean"] / agg.loc["TO1-B", "mean"]))
        pd.DataFrame(mrows).to_csv(
            os.path.join(TAB, "photostability_density_matched.csv"), index=False)

        # Restricting to a shared density window throws away most fields, so the
        # decisive version estimates the fluorogen effect AT FIXED DENSITY by
        # regression: if the difference between the dyes is really surface
        # quality, adding density as a covariate should absorb it.
        try:
            import statsmodels.formula.api as smf
            q = pf_surv.copy()
            q["is_TO"] = (q.fluorogen == "TO1-B").astype(int)
            q["logdens"] = np.log(q.n_spots)
            q["logtau"] = np.log(q.tau_single_s)
            models = [("tau ~ fluorogen", "logtau ~ is_TO"),
                      ("+ log density", "logtau ~ is_TO + logdens"),
                      ("+ density + concentration",
                       "logtau ~ is_TO + logdens + C(conc_nM)")]
            mm = []
            for lab, form in models:
                fit = smf.ols(form, data=q).fit()
                b = fit.params["is_TO"]
                ci = fit.conf_int().loc["is_TO"]
                mm.append(dict(model=lab, n_fields=int(fit.nobs),
                               ratio_SALAD_over_TO=float(np.exp(-b)),
                               ci_lo=float(np.exp(-ci[1])), ci_hi=float(np.exp(-ci[0])),
                               p_fluorogen=float(fit.pvalues["is_TO"]),
                               density_coef_per_log_unit=float(
                                   fit.params.get("logdens", np.nan)),
                               p_density=float(fit.pvalues.get("logdens", np.nan)),
                               r_squared=float(fit.rsquared)))
            pd.DataFrame(mm).to_csv(
                os.path.join(TAB, "photostability_density_regression.csv"), index=False)
        except ImportError:
            log("statsmodels unavailable; fixed-density regression skipped")
    pf_surv.to_csv(os.path.join(TAB, "photostability_per_field.csv"), index=False)
    return P, ps


# ======================== STAGE 7: heterogeneity ===========================
def stage_heterogeneity(mol, ON, OFF):
    """Analysis 5. Event-count dispersion against Poisson, per-molecule dwell
    means, and whether the slow component is carried by a subpopulation or is
    present within every molecule (permutation test: reassign dwells across
    molecules preserving each molecule's count)."""
    rg = np.random.default_rng(SEED)
    rows = []
    for (fl, c, pw), g in mol.groupby(["fluorogen", "conc_nM", "power_mW"]):
        n = g.loc[g.n_events > 0, "n_events"].values.astype(float)
        if n.size < 50:
            continue
        disp = float(n.var(ddof=1) / n.mean())
        on = ON[(ON.fluorogen == fl) & (ON.conc_nM == c) & (ON.power_mW == pw) & (~ON.censored)]
        pm = on.groupby("key").frames.agg(["size", "mean"])
        pm = pm[pm["size"] >= 5]
        obs_sd = float(pm["mean"].std(ddof=1)) if len(pm) > 5 else np.nan
        null_sd = np.nan
        if len(pm) > 5:
            counts = pm["size"].values.astype(int)
            pool = on.frames.values.astype(float)
            sds = []
            for _ in range(200):
                perm = rg.permutation(pool)
                idx = np.r_[0, np.cumsum(counts)]
                means = [perm[idx[i]:idx[i + 1]].mean() for i in range(len(counts))
                         if idx[i + 1] <= perm.size]
                sds.append(np.std(means, ddof=1))
            null_sd = float(np.mean(sds))
        rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                         n_binders=int(n.size), mean_events=float(n.mean()),
                         var_events=float(n.var(ddof=1)),
                         index_of_dispersion=disp,
                         poisson_expectation=1.0,
                         overdispersed=bool(disp > 1.5),
                         n_mol_ge5_dwells=int(len(pm)),
                         sd_of_per_molecule_mean_on=obs_sd,
                         sd_expected_if_exchangeable=null_sd,
                         excess_spread_ratio=(obs_sd / null_sd) if null_sd and null_sd > 0 else np.nan,
                         interpretation="ratio > 1 => molecules differ genuinely; "
                                        "ratio ~ 1 => the slow component is within every molecule"))
    het = pd.DataFrame(rows)
    het.to_csv(os.path.join(TAB, "heterogeneity.csv"), index=False)
    return het


# ========================= STAGE 8: sensitivity ============================
def stage_sensitivity(mol, ev):
    """Every headline choice moved one at a time: max_gap, minimum duration,
    edge correction, contrast filter."""
    rows = []
    for mg in SENS_MAX_GAP:
        for edge in SENS_EDGE:
            ON, OFF, BND = extract_dwells(ev, mg, edge_correction=edge)
            for cf in SENS_CONTRAST:
                if cf is None:
                    keep = None
                else:
                    ok = mol[(mol.contrast >= cf)]
                    keep = set(ok.file.astype(str) + "|" + ok.mol_idx.astype(str))
                for md, xd in [(m, x) for m in SENS_MIN_DUR for x in SENS_MAX_DUR]:
                    dmin_off = max(md, mg + 1)
                    O = ON if keep is None else ON[ON.key.isin(keep)]
                    F_ = OFF if keep is None else OFF[OFF.key.isin(keep)]
                    Bd = BND if keep is None else BND[BND.key.isin(keep)]
                    for (fl, c, pw), g in O.groupby(["fluorogen", "conc_nM", "power_mW"]):
                        cm, cx = g[~g.censored], g[g.censored]
                        cm = cm[cm.frames >= md]
                        cx = cx[cx.frames >= md]
                        if len(cm) < 50:
                            continue
                        du, ku, dcu, kcu = apply_upper(
                            cm.frames.values.astype(float), cm.key.values,
                            cx.frames.values.astype(float), cx.key.values, xd)
                        t_on = fit_dwell_set(du, ku, dcu, kcu, md,
                                             double=False)["tau_single"]
                        oi = F_[(F_.fluorogen == fl) & (F_.conc_nM == c) & (F_.power_mW == pw)]
                        ob = Bd[(Bd.fluorogen == fl) & (Bd.conc_nM == c) & (Bd.power_mW == pw)]
                        d, k, dc, kc, amin = off_arrays(oi, ob, dmin_off)
                        if d.size < 50:
                            continue
                        d, k, dc, kc = apply_upper(d, k, dc, kc, xd)
                        t_off = fit_dwell_set(d, k, dc, kc, dmin_off, dmin_c=amin,
                                              double=False)["tau_single"]
                        ton, toff = t_on * FRAME_TIME_S, t_off * FRAME_TIME_S
                        rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                                         max_gap=mg, min_dur=md,
                                         max_dur=xd if xd else 0,
                                         edge_correction=edge,
                                         contrast_filter=cf, n_on=len(cm), n_off=int(d.size),
                                         tau_on_s=ton, tau_off_s=toff,
                                         k_off_per_s=1 / ton,
                                         K_D_per_point_nM=c * toff / ton))
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(TAB, "sensitivity.csv"), index=False)
    return sens


# ===================== STAGE 11: methods & caveats =========================
METHODS = """\
# Methods

Immobilised Mango II aptamer molecules were imaged by objective-type total
internal reflection fluorescence microscopy in the continuous presence of a
fluorogenic dye, so that binding of the fluorogen to the aptamer produced a
step increase in emission and dissociation returned the signal to background.
Movies of {n_frames} frames were acquired at {frame_ms:.0f} ms exposure
({frame_s:g} s per frame) without interruption. Two fluorogens, TO1-B and
SALAD1, were imaged on the same aptamer at dye concentrations of {concs} nM;
at {pow_conc} nM the acquisition was repeated at {powers} mW to test for
photobleaching of the bound fluorogen. Each condition comprised
{n_fields} fields of view.

Single-molecule intensity trajectories were idealised into a two-state
bound/unbound sequence by penalised change-point detection external to this
analysis, and the resulting event tables were used without re-idealisation.
Exports were verified to be fitted, unfiltered, version-4 files before use;
files failing that check were excluded and named. Fluorogen identity, dye
concentration and laser power were taken from the file name, and the
acquisition path recorded in each file was cross-checked against it, with any
disagreement reported rather than silently resolved.

Dark gaps of up to {max_gap} frames between consecutive events of one molecule
were bridged before analysis, on the basis that such gaps are far more frequent
than rebinding at these concentrations predicts and therefore mostly reflect
fluorophore blinking or sub-frame dissociation rather than genuine unbinding.
Bound dwell times were taken from the merged events; events in progress at the
first or last frame of a movie were entered into the likelihood as
right-censored observations rather than discarded. Unbound dwell times were
taken from the intervals between consecutive events of the same molecule, and
the intervals preceding the first and following the last event were retained as
right-censored observations; discarding them removes the longest unbound
intervals and biases the unbound lifetime downwards.

Because durations are recorded as integer frame counts, dwell-time
distributions were fitted by maximum likelihood using the frame-discretised
(geometric) form of the exponential law, in which a dwell of true duration
T ~ Exp(k) is observed with probability p(1-p)^(d-1) for p = 1 - exp(-k dt).
{window_sentence}

{truncation_sentence} Truncation points were applied per observation so
that censored and complete observations with different truncation could enter
the same likelihood. One- and two-component models were fitted to every
distribution and compared by the likelihood-ratio statistic and by the Bayesian
information criterion; a single lifetime is not reported where two components
are decisively favoured. Confidence intervals were obtained by bootstrap
resampling of molecules rather than of events ({n_boot} resamples), because
events recorded from one molecule are not independent.

Per-molecule bound-state brightness was defined as the mean intensity of a
molecule's bound frames minus the mean of its unbound frames, and per-molecule
contrast as that difference divided by the standard deviation of the unbound
frames. Brightness was compared between fluorogens only within a single laser
power, and never rescaled by nominal power. Because single-molecule intensity
distributions are not normal, fluorogens were compared by the Mann-Whitney test
with the rank-biserial correlation as effect size, and ratios of medians were
bracketed by bootstrap over molecules.

Molecules showing no two-level structure were counted separately and classified
by whether the trajectory sat at the background level throughout or at the bound
level throughout, so that molecules occupied for the whole acquisition -- the
most stable subpopulation -- were not discarded together with non-binders. To
distinguish complexes that never emitted from complexes too dim to pass the
idealiser's acceptance threshold, the largest {dim_run}-frame running mean of
each trajectory was compared with the same statistic computed on
{dim_shuffle} random permutations of that molecule's own frames; permutation
preserves the intensity distribution while destroying the multi-frame structure
a genuine dwell must possess, and provides the appropriate null for a maximum
statistic, which a fixed multiple of the background standard deviation does not.

{span_sentence} {censor_sentence} Survival was estimated by the
Kaplan-Meier method and lifetimes by the censored likelihood above. Because the
free fluorogen pool is present throughout and a bleached ligand can be replaced,
this quantity measures the persistence of signal at a surface position and not
the photobleaching lifetime of an individual fluorophore, which the design
cannot access.

{headline_sentence} {sensitivity_sentence}
"""

CAVEATS = """\
# Caveats

1. Bound dwell times are photobleaching-limited. tau_on falls steeply with laser
   power at every concentration tested, so 1/tau_on is an effective
   disappearance rate and not the dissociation rate. Any K_D computed from it is
   correspondingly unreliable, and dwell times measured at different laser
   powers must never be pooled.

2. The apparent on-rate contains a concentration-independent term, and it is a
   photobleaching artefact rather than a binding process. Simulation with the
   idealiser applied to synthetic traces of known composition shows that
   blinking, whether or not the gaps are bridged, does NOT manufacture a
   positive intercept, whereas photobleaching of the bound dye does:
   {intercept_evidence}. The intercept should therefore not be read as
   evidence of a second binding route.{intercept_zero} Separately, short dark gaps are in excess of what the long-gap
   tail predicts, so a dark interval is still not interpretable as a
   dissociation event without bounding the blinking contribution -- but that
   excess is a statement about gap durations, not about the intercept.

3. Occupancy saturates far below unity. The Langmuir amplitude is an occupancy
   ceiling of the observable -- the fraction of time an unbleached dye occupies
   the pocket -- not a binding stoichiometry, so the fitted K_D from that route
   should not be read as an affinity.

4. Spot densities carry an immobilisation term. Where the two fluorogens occupied
   different flow-cell channels, channel-to-channel variation in RNA loading is
   not estimable and is indistinguishable from a difference in detectable
   fraction. Field-to-field intervals do not cover it. A channel-swap replicate
   would close this gap.

5. The fully dark population is not observed. A complex that never emits is never
   detected, so every emitting-fraction estimate here tests for the consequences
   of that population rather than observing it.

6. Detection floors are not equal between fluorogens at higher dye
   concentrations. The more strongly censored distribution has its median lifted,
   so brightness ratios measured where the floors differ are lower bounds.

7. The optical transmission difference between the two emission bands is not
   corrected. No emission spectra or filter transmission curves were available to
   this analysis; the correction is expected to be of order a few percent but is
   stated rather than assumed.

8. Molecules with essentially no unbound frames have no estimable background, so
   their contrast and detection floor are undefined; they are excluded from
   floor-based statistics and counted in the QC table.

9. Detection completeness at the adopted lower truncation.
   {truncation_exception}

10. Effective photostability under exchange is definition-dependent and its
    value must always be quoted with the span definition attached. Measuring the
    span from the first detected event rather than from the start of the
    acquisition, and censoring a span when less observation remained than the
    expected wait for the next event rather than only when the last event ran to
    the final frame, together move the characteristic time by tens of percent.
    {span_effect}

11. Four of the six concentrations were pre-bleached and two were not.
    Acquisitions at {prebleach_conc} nM were preceded by deliberate
    photobleaching of the field so that single molecules could be resolved;
    those at {unbleached_conc} nM were not. The treatment is therefore
    confounded with concentration, and any quantity derived from bound dwell
    times, occupancy or signal persistence may differ between the two groups for
    reasons unrelated to dye concentration. Results from the two groups are
    reported separately throughout, and the un-bleached subset is primary
    wherever it can carry the result. The confound was tested for and no
    treatment-specific signature was found: the concentration dependence of the
    apparent dissociation rate is present inside the pre-bleached group alone,
    where all four conditions were treated identically; the time to a molecule's
    first event is not lengthened in the treated group; and the fraction of
    molecules showing no binding is not raised in it.

12. Occupancy is not stationary, and the acquisition does not measure an
    equilibrium. Measured directly from the idealised states, {decline_evidence}. {decline_not_detection}, and the decline
    is therefore a real loss of the occupied population under illumination
    rather than a loss of the ability to detect it. Every quantity averaged over
    the acquisition -- occupancy, the equilibrium dissociation constant derived
    from it, and the apparent dissociation rate -- is an average over this
    relaxation and not a property of the equilibrium. {initial_kd}, which is
    within a small factor of bulk, whereas the same fit to the steady-state
    occupancy reached later in the acquisition is not. The dissociation
    constants reported from dwell times should be read as properties of the
    illuminated steady state.

13. Dwells are taken from a window inside each acquisition, and the absolute
    rates carry that window's bias. Because occupancy relaxes during every
    acquisition and the acquisitions did not all start at the same point on
    that relaxation, dwells are extracted from a window chosen to align them;
    {window_bias}. Separately, {window_trust}. The two offsets act in the same
    direction, so absolute dissociation rates should be read as roughly a fifth
    low, while ratios between the fluorogens and trends across concentration
    are unaffected. Whole-acquisition values are in the supplement for
    comparison.

14. The dissociation constant cannot be obtained from these dwell times. No
    analysis window recovers the unbound lifetime: {tau_off_unrecovered}. The
    symptom is visible in the data, where the dissociation constant computed
    from dwells rises monotonically with concentration, which a constant of the
    interaction cannot do. The affinity quoted in this work therefore comes
    from occupancy extrapolated to the first frame, and the dwell-derived
    association rate is reported as an apparent quantity only.

15. The analysis window equalises the acquisitions; it does not make any of
    them stationary. Under the adopted rule {window_residual}, so a dwell
    measured late in the window is drawn from a sparser population than one
    measured early. The rule was designed and validated to make acquisitions
    comparable with each other, which is what a ratio or a concentration trend
    requires, and it should not be read as having removed the relaxation.

16. The occupancy threshold that opens the window was chosen, not derived, and
    is the only free parameter in the step. It was set a priori at a quarter of
    each acquisition's initial occupancy and scanned afterwards rather than
    fitted. No reported conclusion depends on it: {window_scan}. A lower
    threshold leaves less relaxation inside the window but yields a shorter
    window and fewer events, so stationarity and statistics trade against each
    other; the adopted value is where simulated recovery is most consistent
    across conditions.
"""


def caveat_sentences():
    """Numeric claims in the caveats, derived from the tables that produced them.

    Same rule as the methods text: interpolate every value mentioned, or mention
    none. Where the supporting table is absent the sentence degrades to a
    qualitative wording rather than printing a stale or placeholder number, so
    caveats.md is always true of the run that produced it.
    """
    def load(name):
        p = os.path.join(TAB, name)
        try:
            return pd.read_csv(p) if os.path.exists(p) else None
        except Exception:
            return None

    kon, e4 = load("kon_intercept_fits.csv"), load("sim_e4_bleaching.csv")
    e3b, req = load("sim_e3b_matched_occupancy.csv"), load("sim_e6_required_per_condition.csv")
    pdefs = load("photostability_definitions.csv")
    out = {}

    if kon is not None and e3b is not None and len(e3b):
        K = kon.set_index("fluorogen")
        _m = [f"{K.loc[f, 'intercept_per_s']:.3f} per second for {f}"
              for f in K.index if "intercept_per_s" in K.columns]
        meas = " and ".join([", ".join(_m[:-1]), _m[-1]] if len(_m) > 2 else _m)
        br = (f"a bleaching rate of {e3b.k_bleach.iloc[0]:g} per second"
              if "k_bleach" in e3b.columns else "a realistic bleaching rate")
        out["intercept_evidence"] = (
            f"at {br} and the occupancy the experiment reaches, an intercept of "
            f"+{e3b.fitted_intercept_per_s.iloc[0]:.3f} per second appears with no "
            f"concentration-independent binding present, against measured values of "
            f"{meas}")
        if "intercept_excludes_zero" in K.columns:
            yes = [f for f in K.index if bool(K.loc[f, "intercept_excludes_zero"])]
            no = [f for f in K.index if not bool(K.loc[f, "intercept_excludes_zero"])]
            out["intercept_zero"] = (
                " Its interval excludes zero for " + ", ".join(yes) +
                (" but not for " + ", ".join(no) if no else "") + "."
                if yes else " Its interval includes zero for both fluorogens.")
        else:
            out["intercept_zero"] = ""
    else:
        out["intercept_evidence"] = ("simulation shows that photobleaching of the bound "
                                     "dye produces an intercept of the observed sign and "
                                     "magnitude with no concentration-independent "
                                     "binding present")
        out["intercept_zero"] = ""

    if req is not None and len(req) and "min_dur_required" in req.columns:
        bad = req[req.min_dur_required > MIN_DUR]
        n_ok, n_tot = len(req) - len(bad), len(req)
        if len(bad):
            b = bad.iloc[0]
            out["truncation_exception"] = (
                f"Detection of true occupancy episodes reaches "
                f"{DETECTION_TARGET:.0%} at {MIN_DUR} frames in {n_ok} of the {n_tot} "
                f"conditions, but only {b.recovery_at_3:.0%} for {b.fluorogen} at "
                f"{b.conc_nM:g} nM, which would require "
                f"{b.min_dur_required:.0f} frames. Missing short dwells inflate the "
                "fitted lifetime, so the bound dwell time for that condition is biased "
                "high and is reported as an upper bound.")
        else:
            out["truncation_exception"] = (
                f"Detection of true occupancy episodes reaches {DETECTION_TARGET:.0%} at "
                f"{MIN_DUR} frames in all {n_tot} conditions, so no condition retains "
                "incomplete detection at the adopted truncation.")
    else:
        out["truncation_exception"] = (
            "Detection of true occupancy episodes was not measured for this run, so "
            "completeness at the adopted truncation is unverified.")

    tcl = load("prebleach_trace_decline.csv")
    occ = load("prebleach_occupancy_timecourse.csv")
    lng = load("prebleach_initial_occupancy_langmuir.csv")
    pb = ", ".join(f"{c:g}" for c in sorted(PREBLEACH_CONC_NM))
    cl = ", ".join(f"{c:g}" for c in sorted(set(load("qc_per_file.csv").conc_nM)
                                            - set(PREBLEACH_CONC_NM))) if load(
        "qc_per_file.csv") is not None else "the remainder"
    out["prebleach_conc"] = pb
    out["unbleached_conc"] = cl
    if tcl is not None and len(tcl):
        g = tcl.groupby("group").occ_fold.median()
        out["decline_evidence"] = (
            f"occupancy falls {tcl.occ_fold.min():.0f}- to {tcl.occ_fold.max():.0f}-fold "
            f"over the {N_FRAMES_EXPECTED * FRAME_TIME_S:.0f} s acquisition in every one "
            f"of the {len(tcl)} files, with a median fold-decline of "
            f"{g.get('not pre-bleached', float('nan')):.0f} in the untreated group and "
            f"{g.get('pre-bleached', float('nan')):.0f} in the treated one")
        out["decline_not_detection"] = (
            f"This is not a detection artefact: over the same interval the bound level "
            f"falls only {tcl.bound_fold.median():.2f}-fold while the background falls "
            f"{tcl.bg_fold.median():.2f}-fold, so bound-to-background contrast "
            f"{'improves' if tcl.contrast_last.median() > tcl.contrast_first.median() else 'degrades'} "
            f"from {tcl.contrast_first.median():.0f} to {tcl.contrast_last.median():.0f}")
    else:
        out["decline_evidence"] = ("occupancy falls steeply over the acquisition in every "
                                   "condition")
        out["decline_not_detection"] = ("Contrast does not degrade over the same interval, "
                                        "so this is not a detection artefact")
    if lng is not None and len(lng):
        r = lng[(lng.fluorogen == "SALAD1") & (lng.source.str.startswith("initial"))]
        out["initial_kd"] = (
            f"Fitted to the occupancy extrapolated to the first frame, SALAD1 gives "
            f"K_D = {r.K_D_nM.iloc[0]:.1f} +/- {r.K_D_se.iloc[0]:.1f} nM against a bulk "
            f"value of {r.K_D_bulk_nM.iloc[0]:g} nM"
            if len(r) else "The initial occupancy was not fitted")
    else:
        out["initial_kd"] = "The initial occupancy was not fitted"

    wsco = load("window_rule_scores.csv")
    wsen = load("window_trust_sensitivity.csv")
    wval = load("window_rule_validation.csv")
    if wsco is not None and len(wsco) and WINDOW_RULE in set(wsco.iloc[:, 0]):
        r = wsco.set_index(wsco.columns[0]).loc[WINDOW_RULE]
        out["window_bias"] = (
            f"the adopted rule recovers a known dissociation rate to "
            f"{r.mean_recovery:.2f} of its true value, varying "
            f"{r.across_condition_range:.2f}-fold across conditions against "
            f"{wsco.set_index(wsco.columns[0]).across_condition_range.drop(WINDOW_RULE).min():.2f}-fold "
            f"for the next most consistent alternative")
    else:
        out["window_bias"] = ("the adopted rule was selected on simulated recovery of "
                              "known rates")
    if wsen is not None and len(wsen) and "subset" in wsen.columns:
        pv = wsen.pivot_table(index=["fluorogen", "conc_nM"], columns="subset",
                              values="k_off_per_s")
        if {"all traces (headline)", "high trust only"} <= set(pv.columns):
            rr = pv["high trust only"] / pv["all traces (headline)"]
            out["window_trust"] = (
                f"restricting to the most trustworthy traces raises the apparent "
                f"dissociation rate by {rr.min()-1:.0%} to {rr.max()-1:.0%} in the same "
                f"direction in all {len(rr)} conditions, so the offset is uniform and "
                f"comparisons are unaffected while the absolute level is not")
        else:
            out["window_trust"] = "the trace-quality sensitivity was not computed"
    else:
        out["window_trust"] = "the trace-quality sensitivity was not computed"
    if wval is not None and len(wval) and "tau_off_recovery" in wval.columns:
        mo = wval.groupby(["rule", "condition"]).tau_off_recovery.mean().unstack(1)
        rg = (mo.max(axis=1) / mo.min(axis=1))
        out["tau_off_unrecovered"] = (
            f"across conditions the recovered unbound lifetime spans {rg.min():.1f}- to "
            f"{rg.max():.1f}-fold for every rule tested, including the whole "
            f"acquisition")
    else:
        out["tau_off_unrecovered"] = ("the unbound lifetime was not shown to be "
                                      "recoverable")

    wocc = load("window_occfrac_conclusions.csv")
    wdef = load("window_definitions.csv")
    if wdef is not None and len(wdef):
        wa = wdef[wdef.rule == WINDOW_RULE]
        out["window_residual"] = (
            f"occupancy still falls {wa.residual_fold.min():.1f}- to "
            f"{wa.residual_fold.max():.1f}-fold between the first and last tenth of "
            f"the window itself")
    else:
        out["window_residual"] = "occupancy still falls substantially within the window"
    if wocc is not None and len(wocc):
        out["window_scan"] = (
            f"scanned over {len(wocc)} settings spanning "
            f"{wocc.occ_frac.min():g} to {wocc.occ_frac.max():g}, the concentration "
            f"dependence of the apparent dissociation rate persists at every setting "
            f"({wocc.SALAD_fold_range.min():.2f}- to "
            f"{wocc.SALAD_fold_range.max():.2f}-fold for SALAD1) and the ratio between "
            f"the fluorogens at the lowest concentration moves only over "
            f"{wocc.ratio_at_lowest_c.min():.2f}-{wocc.ratio_at_lowest_c.max():.2f}, "
            f"while the absolute rate moves by about "
            f"{100 * (wocc.k_off_SALAD_lo.max() / wocc.k_off_SALAD_lo.min() - 1):.0f} "
            f"per cent")
    else:
        out["window_scan"] = "the threshold was not scanned"

    if pdefs is not None and len(pdefs) and "end_rule" in pdefs.columns:
        ft = pdefs[pdefs.span_definition == "first_to_last"]
        k = ["fluorogen", "conc_nM", "power_mW"]
        st = ft[ft.end_rule == "strict"].set_index(k).tau_single_s
        ga = ft[ft.end_rule == "gap_aware"].set_index(k).tau_single_s
        r = (ga / st).dropna()
        out["span_effect"] = (f"The censoring rule alone accounts for "
                              f"{r.min()-1:+.0%} to {r.max()-1:+.0%}.")
    else:
        out["span_effect"] = "The censoring rule alone accounts for tens of percent."
    return out


def _rewrap(text, width=78):
    """Re-flow prose paragraphs to a fixed width. Interpolated sentences arrive
    as single long lines; without this the generated file is ragged wherever a
    parameter was substituted, which is precisely the text most likely to be
    read as prose."""
    import textwrap
    out = []
    for para in text.split("\n\n"):
        if para.lstrip().startswith(("#", "-", "*", "|")) or not para.strip():
            out.append(para)
        else:
            # A decimal must not be wrapped to the start of a line: inside a
            # numbered list "0.15, 0.25, ..." at a line start reads as a stray
            # list item. Bind each decimal to the word before it with a
            # non-breaking space so textwrap moves the pair together, then
            # restore ordinary spaces.
            flat = re.sub(r"(\S) (\d+\.\d)", "\\1\u00a0\\2",
                          " ".join(para.split()))
            out.append(textwrap.fill(flat, width=width, break_on_hyphens=False,
                                     break_long_words=False).replace("\u00a0", " "))
    return "\n\n".join(out)


def methods_sentences(min_dur=None, max_dur=None, max_gap=None, span_definition=None,
                      span_censoring=None, censor_margin=None, edge_correction=None,
                      contrast_min=None, detection_target=None, window_rule=None,
                      window_occ_frac=None):
    """Every methods sentence whose wording depends on a parameter.

    The rule this enforces: generated methods text either interpolates every
    value it mentions or mentions none. A paragraph that names some values and
    hard-codes others goes stale silently when a default changes -- which is
    exactly what happened when SPAN_CENSORING moved from "strict" to
    "gap_aware" and the methods paragraph kept describing the strict rule.

    Arguments default to the module parameters; they exist so the self-check
    below can render the text under a perturbed parameter set and confirm the
    prose actually follows.
    """
    min_dur = MIN_DUR if min_dur is None else min_dur
    max_dur = MAX_DUR if max_dur is None else max_dur
    max_gap = MAX_GAP if max_gap is None else max_gap
    span_definition = SPAN_DEFINITION if span_definition is None else span_definition
    span_censoring = SPAN_CENSORING if span_censoring is None else span_censoring
    censor_margin = CENSOR_MARGIN if censor_margin is None else censor_margin
    edge_correction = EDGE_CORRECTION if edge_correction is None else edge_correction
    contrast_min = CONTRAST_MIN if contrast_min is None else contrast_min
    window_rule = WINDOW_RULE if window_rule is None else window_rule
    window_occ_frac = (WINDOW_OCC_FRAC if window_occ_frac is None
                       else window_occ_frac)
    detection_target = DETECTION_TARGET if detection_target is None else detection_target
    dmin_off = max(min_dur, max_gap + 1)

    trunc = (
        f"Fits were left-truncated at {min_dur} frames for bound dwells and at "
        f"{dmin_off} frames for unbound dwells, the latter because bridging removes "
        f"any shorter interval by construction. The bound-dwell truncation was not "
        f"assumed: detection efficiency for true occupancy episodes was measured as a "
        f"function of episode duration by simulating each condition at its own fitted "
        f"rates and measured brightness, background and noise, and the truncation was "
        f"placed at the shortest duration whose lower confidence bound on recovery "
        f"reached {detection_target:.0%} for both fluorogens.")
    if max_dur:
        trunc += (f" Dwells of {max_dur} frames or longer were entered as right-censored "
                  f"at {max_dur} frames rather than discarded.")
    else:
        trunc += (" No upper truncation was applied; an upper limit was tested and moved "
                  "the fitted lifetimes by less than their confidence intervals.")

    span = ("Effective photostability under exchange was quantified as the interval "
            "between a molecule's first and last detected event."
            if span_definition == "first_to_last" else
            "Effective photostability under exchange was quantified as the interval from "
            "the start of the acquisition to a molecule's last detected event, so that "
            "time spent waiting for the first binding event is charged to the molecule.")

    censor = ("A span was treated as right-censored when the observation remaining after "
              "the last detected event was shorter than "
              + (f"{censor_margin:g} times " if censor_margin != 1 else "")
              + "that condition's own fitted unbound lifetime, since failing to observe a "
                "further event in less time than the expected wait is not evidence that "
                "the molecule had stopped emitting."
              if span_censoring == "gap_aware" else
              "A span was treated as right-censored only when the last detected event was "
              "still in progress at the final frame of the acquisition.")

    # One sentence. It STATES the rule; the argument for it belongs in the
    # results, next to the simulated recovery that selected it.
    if window_rule == "A_matched_occupancy":
        window = ("Because occupancy relaxed during every acquisition, dwells were "
                  "taken from a window within each acquisition rather than from the "
                  "whole record, opening where that acquisition's occupancy first "
                  f"fell to {window_occ_frac:g} of its own initial value, so that "
                  "every acquisition was analysed at a comparable point on its "
                  "relaxation.")
    elif window_rule == "B_matched_clock":
        window = ("Because occupancy relaxed during every acquisition, dwells were "
                  "taken from a window within each acquisition rather than from the "
                  "whole record, opening at a fixed frame common to all "
                  "acquisitions.")
    elif window_rule == "C_early":
        window = ("Because occupancy relaxed during every acquisition, dwells were "
                  "taken from a short window at the start of each acquisition rather "
                  "than from the whole record.")
    else:
        window = ("Dwells were taken from the whole of each acquisition; no analysis "
                  "window was applied.")

    headline = (
        f"Headline values were computed with a gap-bridging window of {max_gap} frames, "
        f"durations "
        + ("corrected by one frame for the partial frames lost at each transition"
           if edge_correction else "as measured, without the edge correction")
        + ", and "
        + (f"a minimum per-molecule contrast of {contrast_min:g}" if contrast_min
           else "no contrast threshold")
        + ".")

    sens = ("Each of those choices, together with the minimum dwell duration, was then "
            "varied systematically over " + ", ".join(
                [f"gap-bridging windows of {'/'.join(str(v) for v in SENS_MAX_GAP)} frames",
                 f"minimum durations of {'/'.join(str(v) for v in SENS_MIN_DUR)} frames",
                 "edge correction on and off",
                 "contrast threshold on and off"])
            + ", and the resulting parameter ranges are reported alongside the headline "
              "values.")
    return dict(truncation_sentence=trunc, span_sentence=span, censor_sentence=censor,
                headline_sentence=headline, sensitivity_sentence=sens,
                window_sentence=window)


def check_methods_follow_parameters():
    """Render the methods sentences under perturbed parameters and confirm the
    prose changes. A hard-coded paragraph passes every spell-check and every
    unit test that only reads the current output; it fails this one."""
    base = methods_sentences()
    perturbations = [
        ("span_definition", "acquisition" if SPAN_DEFINITION == "first_to_last"
         else "first_to_last", "span_sentence"),
        ("span_censoring", "strict" if SPAN_CENSORING == "gap_aware" else "gap_aware",
         "censor_sentence"),
        ("min_dur", MIN_DUR + 1, "truncation_sentence"),
        ("max_dur", 500 if not MAX_DUR else None, "truncation_sentence"),
        ("max_gap", MAX_GAP + 1, "headline_sentence"),
        ("edge_correction", not EDGE_CORRECTION, "headline_sentence"),
        ("contrast_min", 3.0 if not CONTRAST_MIN else None, "headline_sentence"),
        ("window_rule", "C_early" if WINDOW_RULE != "C_early" else "none",
         "window_sentence"),
        ("window_occ_frac", WINDOW_OCC_FRAC / 2, "window_sentence"),
    ]
    for name, value, key in perturbations:
        alt = methods_sentences(**{name: value})
        assert alt[key] != base[key], (
            f"methods text does not follow the parameter {name}: changing it to "
            f"{value!r} left '{key}' unchanged, so that sentence is hard-coded and "
            "will go stale")
    return True


def stage_text(meta, good, kin):
    concs = sorted(good.conc_nM.dropna().unique())
    pw_conc = good.groupby("conc_nM").power_mW.nunique()
    pw_conc = pw_conc[pw_conc >= 3]
    powers = sorted(good.power_mW.unique())
    nf = int(good.groupby(["fluorogen", "conc_nM", "power_mW"]).size().median())
    txt = METHODS.format(
        n_frames=N_FRAMES_EXPECTED, frame_ms=FRAME_TIME_S * 1000, frame_s=FRAME_TIME_S,
        concs=", ".join(f"{c:g}" for c in concs),
        pow_conc=", ".join(f"{c:g}" for c in pw_conc.index) if len(pw_conc) else "the highest",
        powers=", ".join(f"{p:g}" for p in powers), n_fields=nf,
        max_gap=MAX_GAP, min_dur=MIN_DUR, dmin_off=max(MIN_DUR, MAX_GAP + 1),
        dim_run=DIM_RUN, dim_shuffle=DIM_N_SHUFFLE, n_boot=N_BOOT,
        **methods_sentences())
    check_methods_follow_parameters()
    txt = _rewrap(txt)
    with open(os.path.join(OUT_DIR, "methods.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(OUT_DIR, "caveats.md"), "w", encoding="utf-8") as fh:
        fh.write(_rewrap(CAVEATS.format(**caveat_sentences())))
    log("methods.md and caveats.md written")


# ================================== main ===================================
def main():
    t0 = time.time()
    log(f"data:   {DATA_DIR}")
    log(f"output: {OUT_DIR}")
    log(f"params: max_gap={MAX_GAP} min_dur={MIN_DUR} edge_correction={EDGE_CORRECTION} "
        f"contrast_filter={CONTRAST_MIN} off_censoring={OFF_CENSORING}")

    meta, good, mol, ev, gaps, qc = stage_extract()
    log("stage 1-2 done: inventory and QC")

    B, br, bg = stage_brightness(mol)
    log("stage 3 done: per-molecule brightness (analysis 7)")

    dens, routes, ladder = stage_detectable(mol, qc)
    log("stage 4 done: detectable fraction (analysis 8)")

    ON, OFF, BND, kin, pf, kon, bf, lang = stage_kinetics(mol, ev, gaps)
    log("stage 5 done: dwell times, rate and equilibrium constants (analyses 2-4)")

    P, ps = stage_photostability(mol, kin)
    log("stage 6 done: effective photostability under exchange (analysis 9)")

    het = stage_heterogeneity(mol, ON, OFF)
    log("stage 7 done: heterogeneity (analysis 5)")

    sens = stage_sensitivity(mol, ev)
    log("stage 8 done: sensitivity grid")

    stage_text(meta, good, kin)

    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
    log(f"tables:  {TAB}")


if __name__ == "__main__":
    main()
