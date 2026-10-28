#!/usr/bin/env python
"""Sensitivity grid under the adopted analysis, for Figure 4.7B (panel G7).

stage_sensitivity() in analyse_smtirf.py was written before five decisions, so
tables/sensitivity.csv no longer describes the reported analysis:

  1. whole acquisitions        -> the adopted analysis window (window_definitions.csv)
  2. every molecule            -> repeated-binding sites, >= 2 bound dwells after bridging
  3. one exponential           -> two components; the amplitude-weighted mean is reported
  4. SENS_MAX_DUR (none / 500) -> the shared per-fluorogen upper limit, as censoring
  5. boundary dark intervals   -> not needed: Figure 4.7B shows bound lifetimes only

This script runs the same grid as stage_sensitivity (gap bridging x edge
correction x contrast threshold x minimum duration) with those decisions applied,
and writes tables/sensitivity_adopted.csv. It reads only what run_all.sh has
already written, and changes nothing else, sensitivity.csv included.

Run from the folder holding analyse_smtirf.py, after run_all.sh:

    python sensitivity_adopted.py
"""
from __future__ import annotations

import glob
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                      # noqa: E402
import window_select as WS                                      # noqa: E402

TAB, FT = A.TAB, A.FRAME_TIME_S
MIN_EVENTS_PER_SITE = 2        # the repeated-binding criterion, as in dwell_filters.py
MIN_COMPLETE_DWELLS = 50       # as in stage_sensitivity
KEYS = ["fluorogen", "conc_nM", "power_mW"]


def load_cache():
    """Molecule and event tables from the per-file caches, exactly as the pipeline uses them."""
    mols, evs = [], []
    for p in sorted(glob.glob(os.path.join(A.HERE, "cache", "*.pkl"))):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        if d.get("mol") is not None and len(d["mol"]):
            mols.append(d["mol"])
        if d.get("ev") is not None and len(d["ev"]):
            evs.append(d["ev"])
    if not evs:
        sys.exit("No cached events in cache/. Run run_all.sh first.")
    return pd.concat(mols, ignore_index=True), pd.concat(evs, ignore_index=True)


def upper_limits():
    """The shared per-fluorogen upper limit on bound dwells, in frames."""
    path = os.path.join(TAB, "filter_shared_upper_limit.csv")
    if not os.path.exists(path):
        sys.exit("tables/filter_shared_upper_limit.csv is missing. Run dwell_filters.py first.")
    L = pd.read_csv(path)
    if "dwell" in L.columns:
        L = L[L.dwell == "on"]
    return L.groupby("fluorogen").shared_limit_frames.first().to_dict()


def build():
    t0 = time.time()
    mol, ev = load_cache()
    wdef = pd.read_csv(os.path.join(TAB, "window_definitions.csv"))
    wdef = wdef[wdef.rule == WS.ADOPTED]
    assert wdef.key.is_unique, "window_definitions.csv holds more than one window per acquisition"
    ev_w, _ = WS.apply_window(ev, wdef)
    limits = upper_limits()
    A.log(f"sensitivity under the adopted analysis: window {WS.ADOPTED}, "
          f"upper limits (frames) {limits}")

    contrast_keys = {}
    for cf in A.SENS_CONTRAST:
        if cf is None:
            contrast_keys[cf] = None
        else:
            ok = mol[mol.contrast >= cf]
            contrast_keys[cf] = set(ok.file.astype(str) + "|" + ok.mol_idx.astype(str))

    rows, failed = [], 0
    for mg, edge in itertools.product(A.SENS_MAX_GAP, A.SENS_EDGE):
        ON, _off, _bnd = A.extract_dwells(ev_w, mg, edge_correction=edge)
        # repeated binding, counted after bridging so it counts episodes, not segments
        n_per = ON.groupby("key").size()
        ON = ON[ON.key.isin(set(n_per[n_per >= MIN_EVENTS_PER_SITE].index))]
        for cf in A.SENS_CONTRAST:
            O = ON if contrast_keys[cf] is None else ON[ON.key.isin(contrast_keys[cf])]
            for md in A.SENS_MIN_DUR:
                for (fl, c, pw), g in O.groupby(KEYS):
                    cm, cx = g[~g.censored], g[g.censored]
                    cm, cx = cm[cm.frames >= md], cx[cx.frames >= md]
                    if len(cm) < MIN_COMPLETE_DWELLS:
                        continue
                    d, k, dc, kc = A.apply_upper(
                        cm.frames.values.astype(float), cm.key.values,
                        cx.frames.values.astype(float), cx.key.values, limits.get(fl))
                    try:
                        r = A.fit_dwell_set(d, k, dc, kc, md, n_boot=0, double=True)
                    except Exception as exc:                       # one failed fit must not stop the grid
                        failed += 1
                        A.log(f"    fit failed: {fl} {c} nM {pw} mW, gap {mg}, floor {md}: {exc}")
                        continue
                    rows.append(dict(
                        fluorogen=fl, conc_nM=c, power_mW=pw,
                        max_gap=mg, min_dur=md, edge_correction=edge, contrast_filter=cf,
                        n_on=len(cm), n_sites=int(g.key.nunique()),
                        tau_on_amp_s=r.get("tau_amp_weighted", np.nan) * FT,
                        tau_on_fast_s=r.get("tau_fast", np.nan) * FT,
                        tau_on_slow_s=r.get("tau_slow", np.nan) * FT,
                        a_fast=r.get("a_fast", np.nan),
                        tau_on_single_s=r["tau_single"] * FT))
        A.log(f"  gap bridging {mg}, edge correction {edge}: {len(rows)} fits "
              f"({time.time() - t0:.0f} s)")
    S = pd.DataFrame(rows)
    out = os.path.join(TAB, "sensitivity_adopted.csv")
    S.to_csv(out, index=False)
    A.log(f"wrote {out}: {len(S)} fits, {failed} failed, {time.time() - t0:.0f} s")
    return S


if __name__ == "__main__":
    build()
