#!/usr/bin/env python
"""Fraction bound against time, one mean curve per condition, from the per-file extracts.

Two later steps read these curves: window_select.py sets the analysis window from them,
and kd_per_condition.py extrapolates them to the first frame for the two-point affinity.
The calculation is the curve part of occupancy_timecourse() in the earlier pre-bleach
analysis, copied unchanged. The output keeps its earlier name,
prebleach_occupancy_curves.npz, because both readers look for it by that name.

A molecule is bound at frame t if one of its events has start <= t < end. Each
acquisition gives the fraction of its molecules bound at every frame; the curve of a
condition is the mean over its acquisitions.

Reads   cache/*.pkl                              written by analyse_smtirf.py
Writes  tables/prebleach_occupancy_curves.npz   one array per condition, keyed
                                                 "fluorogen|conc_nM|power_mW"

Run after analyse_smtirf.py:   python occupancy_curves.py
"""
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_smtirf as A                                      # noqa: E402


def occupancy_curves(n_frames=A.N_FRAMES_EXPECTED):
    """Per-acquisition fraction bound at every frame, grouped by condition."""
    curves = {}
    for path in sorted(glob.glob(os.path.join(A.HERE, "cache", "*.pkl"))):
        try:
            d = pickle.load(open(path, "rb"))
        except Exception:
            continue
        ev, mol = d.get("ev"), d.get("mol")
        if ev is None or mol is None or not len(ev):
            continue
        fl = mol.fluorogen.iloc[0]
        c = float(mol.conc_nM.iloc[0])
        pw = float(mol.power_mW.iloc[0])
        n_mol = len(mol)
        delta = np.zeros(n_frames + 2)
        s = np.clip(ev.start.values.astype(int), 0, n_frames)
        e = np.clip(ev.end.values.astype(int), 0, n_frames)
        np.add.at(delta, s, 1)
        np.add.at(delta, e, -1)
        occ = np.cumsum(delta)[:n_frames] / max(n_mol, 1)
        key = (fl, c, pw)
        curves.setdefault(key, []).append(occ)
    return curves


def main():
    curves = occupancy_curves()
    if not curves:
        sys.exit("No cached events in cache/; run analyse_smtirf.py first.")
    out = os.path.join(A.TAB, "prebleach_occupancy_curves.npz")
    np.savez_compressed(out, **{f"{fl}|{c}|{pw}": np.mean(a, axis=0)
                                for (fl, c, pw), a in curves.items()})
    A.log(f"wrote {out}: {len(curves)} conditions, "
          f"{sum(len(a) for a in curves.values())} acquisitions")


if __name__ == "__main__":
    main()
