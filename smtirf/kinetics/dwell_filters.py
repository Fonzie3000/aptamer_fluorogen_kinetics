#!/usr/bin/env python
"""Two requested dwell filters, their cost, and the model comparison after them.

FILTER R -- repeated binding only. Molecules contributing exactly one bound
dwell are dropped, so every retained molecule shows re-binding.

FILTER U -- one shared upper limit per dye, applied to every concentration in
that dye's group, set where the tail has effectively no counts.

WHY EACH IS TREATED DIFFERENTLY
-------------------------------
Filter U is safe if imposed as CENSORING rather than deletion. A dwell at or
beyond the limit re-enters the likelihood as "at least this long", which is
true and costs nothing; deleting it instead throws away the information that it
was long and biases the lifetime down. The plot gets the clean axis either way,
because the axis limit is a display choice. Both variants are fitted here so
the cost of deletion is a number rather than an argument.

Filter R is NOT safe, and the reason is the rule this project already applies to
molecule-level filters: the criterion must be independent of the dwell times
being measured. Event count is not. A molecule whose one dwell is long has less
observation time left in which to show a second, so requiring a repeat
preferentially discards long dwells and biases tau_on DOWN. The bias is largest
where the unbound lifetime is a large fraction of the window -- which is exactly
this dataset's condition. So the filter is applied as asked, and its effect is
MEASURED on every condition rather than assumed small.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_smtirf as A                                      # noqa: E402
import window_select as WS                                      # noqa: E402

TAB, FT = A.TAB, A.FRAME_TIME_S
KEYS = ["fluorogen", "conc_nM", "power_mW"]

MIN_EVENTS_PER_MOLECULE = 2      # filter R; 1 = no filter
TAIL_FRACTION = 0.002            # filter U: per-condition fraction allowed beyond
N_BOOT = 200


def windowed_dwells():
    ev = WS.load_events()
    wdef = pd.read_csv(os.path.join(TAB, "window_definitions.csv"))
    wdef = wdef[wdef.rule == WS.ADOPTED]
    assert wdef.key.is_unique
    ev_w, _ = WS.apply_window(ev, wdef)
    return A.extract_dwells(ev_w, A.MAX_GAP)


def export_windowed_dwells(ON, OFFI, OFFB, keep=None, lim_on=None, lim_off=None):
    """Write the dwell-level windowed table: one row per dwell, the counterpart of
    dwell_times_long.csv.gz but inside the adopted analysis window.

    Columns follow dwell_times_long.csv.gz so the two can be compared row for
    row, with four added so a reader can reproduce any variant in
    dwell_fits_filtered.csv without re-running the filters:

      censoring          complete / right-censored, from the `censored` flag
      boundary_gap       True for the terminal dark interval after a molecule's
                         last event. These are NOT unbound waiting times - their
                         median is many times the interior median because they
                         are dominated by molecules that stopped emitting - so
                         the recommended variant excludes them. Kept in the file
                         and flagged, never silently dropped.
      passes_repeat      molecule has >= MIN_EVENTS_PER_MOLECULE bound dwells
      beyond_upper       dwell exceeds its dye's shared upper limit, i.e. the
                         rows the recommended variant enters as censored

    The file is written whole and unfiltered; every exclusion is a flag, so the
    headline analysis is a selection on this table rather than a different table.
    """
    # Censoring is set per SOURCE, not read from a shared column: extract_dwells
    # carries `censored` on the bound dwells only. Interior gaps are complete by
    # construction (both ends observed); boundary gaps are right-censored by
    # construction (one end is the window edge). Taking it from a column that is
    # absent on two of the three frames silently labels every gap censored.
    parts = []
    for d, typ, bnd, cen in [(ON, "on", False, None),
                             (OFFI, "off", False, False),
                             (OFFB, "off", True, True)]:
        if d is None or not len(d):
            continue
        x = d.copy()
        x["type"] = typ
        x["boundary_gap"] = bnd
        if cen is None:
            assert "censored" in x.columns, "bound dwells must carry `censored`"
            x["censored"] = x.censored.astype(bool)
        else:
            x["censored"] = cen
        parts.append(x)
    D = pd.concat(parts, ignore_index=True)
    assert D.censored.notna().all() and D.censored.dtype == bool
    D["seconds"] = D.frames * FT
    D["censoring"] = np.where(D.censored, "right-censored", "complete")
    D["passes_repeat"] = D.key.isin(keep) if keep is not None else np.nan
    if lim_on is not None:
        cap = D.fluorogen.map(lim_on).where(D.type == "on",
                                            D.fluorogen.map(lim_off or {}))
        D["beyond_upper"] = D.frames > cap
    else:
        D["beyond_upper"] = np.nan
    cols = [c for c in ["key", "file", "fluorogen", "conc_nM", "power_mW", "frames",
                        "seconds", "censored", "censoring", "snr", "type",
                        "boundary_gap", "passes_repeat", "beyond_upper",
                        "molecule_id", "contrast"] if c in D.columns]
    p = os.path.join(TAB, "dwell_times_windowed.csv.gz")
    D[cols].to_csv(p, index=False, compression="gzip")
    A.log(f"wrote dwell_times_windowed.csv.gz: {len(D):,} dwells "
          f"({int((D.type=='on').sum()):,} on, "
          f"{int(((D.type=='off')&(~D.boundary_gap)).sum()):,} interior off, "
          f"{int(D.boundary_gap.sum()):,} boundary), "
          f"{os.path.getsize(p)/1e6:.1f} MB")
    return D[cols]


def repeat_filter(ON, OFFI, OFFB, min_events=MIN_EVENTS_PER_MOLECULE):
    """Keep only molecules with >= min_events bound dwells (after gap bridging).

    Bridging first is deliberate: the event count that matters is the number of
    distinct occupancy episodes, not the number of segments the idealiser cut a
    single episode into.
    """
    n_per = ON.groupby("key").size()
    keep = set(n_per[n_per >= min_events].index)
    f = lambda d: d[d.key.isin(keep)] if len(d) else d
    return f(ON), f(OFFI), f(OFFB), n_per, keep


def shared_upper_limit(ON, tail=TAIL_FRACTION):
    """One limit per dye, the smallest whole frame count leaving <= `tail` of
    every condition in that dye's group beyond it.

    Shared across concentrations by request. Taking the strictest condition's
    requirement would cut the others harder than their own data warrant, so the
    rule is the most permissive value that satisfies every condition in the
    group -- i.e. the maximum over conditions of each one's own quantile.
    """
    rows = []
    for fl, g in ON.groupby("fluorogen"):
        per = {}
        for (c, pw), gg in g.groupby(["conc_nM", "power_mW"]):
            v = gg.frames.values
            per[(c, pw)] = float(np.quantile(v, 1 - tail))
        lim = int(np.ceil(max(per.values())))
        for (c, pw), q in per.items():
            v = ON[(ON.fluorogen == fl) & (ON.conc_nM == c)
                   & (ON.power_mW == pw)].frames.values
            rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw,
                             own_quantile_frames=q, shared_limit_frames=lim,
                             shared_limit_s=lim * FT,
                             n_dwells=v.size, n_beyond=int((v >= lim).sum()),
                             frac_beyond=float((v >= lim).mean()),
                             longest_s=float(v.max()) * FT))
    L = pd.DataFrame(rows)
    limits = L.groupby("fluorogen").shared_limit_frames.first().to_dict()
    return limits, L


def shared_upper_limit_off(OFFI, tail=TAIL_FRACTION):
    """Same rule as for bound dwells, on the interior unbound gaps."""
    rows = []
    for fl, g in OFFI.groupby("fluorogen"):
        per = {}
        for (c, pw), gg in g.groupby(["conc_nM", "power_mW"]):
            per[(c, pw)] = float(np.quantile(gg.frames.values, 1 - tail))
        lim = int(np.ceil(max(per.values())))
        for (c, pw), q in per.items():
            v = OFFI[(OFFI.fluorogen == fl) & (OFFI.conc_nM == c)
                     & (OFFI.power_mW == pw)].frames.values
            rows.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, dwell="off",
                             own_quantile_frames=q, shared_limit_frames=lim,
                             shared_limit_s=lim * FT, n_dwells=v.size,
                             n_beyond=int((v >= lim).sum()),
                             frac_beyond=float((v >= lim).mean()),
                             longest_s=float(v.max()) * FT))
    L = pd.DataFrame(rows)
    return L.groupby("fluorogen").shared_limit_frames.first().to_dict(), L


def fit_one(ON, OFFI, OFFB, limits, upper_mode, n_boot=N_BOOT, limits_off=None):
    """Refit single and double for both dwell types under one filter variant.

    upper_mode: "none" | "censor" | "delete".
    """
    dmin_on = A.MIN_DUR
    dmin_off = max(A.MIN_DUR, A.MAX_GAP + 1)
    limits_off = limits_off or {}
    out = []
    for (fl, c, pw), g in ON.groupby(KEYS):
        lim = limits.get(fl) if upper_mode != "none" else None
        # ---- bound dwells
        comp = g[~g.censored]
        d, kk = comp.frames.values.astype(float), comp.key.values
        dc = g[g.censored].frames.values.astype(float)
        kc = g[g.censored].key.values
        m = d >= dmin_on
        d, kk = d[m], kk[m]
        mc = dc >= dmin_on
        dc, kc = dc[mc], kc[mc]
        if lim and upper_mode == "censor":
            d, kk, dc, kc = A.apply_upper(d, kk, dc, kc, lim)
        elif lim and upper_mode == "delete":
            keep = d < lim
            d, kk = d[keep], kk[keep]
            keep_c = dc < lim
            dc, kc = dc[keep_c], kc[keep_c]
        if d.size > 50:
            r = A.fit_dwell_set(d, kk, dc, kc, dmin_on, n_boot=n_boot)
            r.update(crossval(d, kk, dc, kc, dmin_on))
            out.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, dwell="on",
                            upper_mode=upper_mode,
                            upper_limit_frames=lim or 0, **r))
        # ---- unbound dwells. The upper limit is a cap on an individual DWELL
        # duration and applies here too, with its own shared per-dye value; it is
        # not a truncation of the recording, which differs per trace and is
        # handled by the analysis window.
        oi = OFFI[(OFFI.fluorogen == fl) & (OFFI.conc_nM == c) & (OFFI.power_mW == pw)]
        ob = OFFB[(OFFB.fluorogen == fl) & (OFFB.conc_nM == c) & (OFFB.power_mW == pw)]
        od, ok_, ocd, ock, dm = A.off_arrays(oi, ob, dmin_off)
        lim_off = limits_off.get(fl) if upper_mode != "none" else None
        if lim_off and upper_mode == "censor":
            od, ok_, ocd, ock = A.apply_upper(od, ok_, ocd, ock, lim_off)
        elif lim_off and upper_mode == "delete":
            kp = od < lim_off
            od, ok_ = od[kp], ok_[kp]
            kpc = ocd < lim_off
            ocd, ock = ocd[kpc], ock[kpc]
        if od.size > 50:
            r = A.fit_dwell_set(od, ok_, ocd, ock, dmin_off, dmin_c=dm, n_boot=n_boot)
            r.update(crossval(od, ok_, ocd, ock, dmin_off, dmin_c=dm))
            out.append(dict(fluorogen=fl, conc_nM=c, power_mW=pw, dwell="off",
                            upper_mode=upper_mode,
                            upper_limit_frames=lim_off or 0, **r))
    return pd.DataFrame(out)


def crossval(d, keys, dc, keys_c, dmin, dmin_c=None, k=5, seed=0):
    """Out-of-sample test of single against double, split by MOLECULE.

    This is the test that answers "is the second component overfitting?", and
    BIC is not. BIC penalises the extra parameters but assumes independent
    observations; dwells from one molecule are correlated, so the effective
    sample size is smaller than the dwell count and BIC's penalty is too weak
    by an unknown factor. Held-out prediction has no such assumption: fit on
    one set of molecules, score the dwells of molecules never seen, and the
    extra parameters earn their place only if they predict better.

    Returned as mean log-likelihood per held-out observation, so the size of
    the advantage is interpretable and not just its sign. A positive value in
    the third decimal place means the double is better but by an amount that
    cannot matter for any reported quantity.
    """
    dmin_c = dmin if dmin_c is None else dmin_c
    mols = np.unique(np.concatenate([keys, keys_c]) if keys_c.size else keys)
    if mols.size < 2 * k:
        return dict(cv_folds=0)
    rg = np.random.default_rng(seed)
    fold_of = dict(zip(mols, rg.integers(0, k, mols.size)))
    f_d = np.array([fold_of[x] for x in keys]) if keys.size else np.array([], int)
    f_c = np.array([fold_of[x] for x in keys_c]) if keys_c.size else np.array([], int)
    gains, n_held = [], []
    for j in range(k):
        tr_d, te_d = d[f_d != j], d[f_d == j]
        tr_c, te_c = dc[f_c != j], dc[f_c == j]
        if tr_d.size < 50 or te_d.size < 10:
            continue
        a_tr = np.full(tr_d.size, float(dmin))
        ac_tr = np.full(tr_c.size, float(dmin_c))
        w_tr, wc_tr = np.ones(tr_d.size), np.ones(tr_c.size)
        s1 = A.fit_geom(tr_d, a_tr, w_tr, tr_c, ac_tr, wc_tr, False)
        s2 = A.fit_geom(tr_d, a_tr, w_tr, tr_c, ac_tr, wc_tr, True)
        th1 = [np.log(s1["tau"])]
        af = min(max(s2["a_fast"], 1e-6), 1 - 1e-6)
        th2 = [np.log(af / (1 - af)), np.log(s2["tau_fast"]), np.log(s2["tau_slow"])]
        a_te = np.full(te_d.size, float(dmin))
        ac_te = np.full(te_c.size, float(dmin_c))
        w_te, wc_te = np.ones(te_d.size), np.ones(te_c.size)
        n = te_d.size + te_c.size
        ll1 = -A.nll_geom(th1, te_d, a_te, w_te, te_c, ac_te, wc_te, False)
        ll2 = -A.nll_geom(th2, te_d, a_te, w_te, te_c, ac_te, wc_te, True)
        gains.append((ll2 - ll1) / n)
        n_held.append(n)
    if not gains:
        return dict(cv_folds=0)
    g = np.array(gains)
    return dict(cv_folds=len(g), cv_gain_per_obs=float(g.mean()),
                cv_gain_sd=float(g.std(ddof=1)) if g.size > 1 else np.nan,
                cv_folds_favouring_double=int((g > 0).sum()),
                cv_n_heldout_median=int(np.median(n_held)),
                cv_verdict=("double generalises better" if (g > 0).all()
                            else "single generalises better" if (g < 0).all()
                            else "inconsistent across folds"))


def build():
    ON0, OFFI0, OFFB0 = windowed_dwells()
    ONr, OFFIr, OFFBr, n_per, keep = repeat_filter(ON0, OFFI0, OFFB0)

    # the dwell-level table, written before any variant is fitted so the file is
    # the unfiltered record the variants select from
    lim_on, _ = shared_upper_limit(ONr)
    lim_off, _ = shared_upper_limit_off(OFFIr) if "shared_upper_limit_off" in globals() else (None, None)
    export_windowed_dwells(ON0, OFFI0, OFFB0, keep=keep,
                           lim_on=lim_on, lim_off=lim_off)

    # ---- what filter R costs, per condition, including the direction of bias
    rows = []
    for (fl, c, pw), g in ON0.groupby(KEYS):
        npm = g.groupby("key").size()
        single = set(npm[npm < MIN_EVENTS_PER_MOLECULE].index)
        gs = g[g.key.isin(single)]
        gm = g[~g.key.isin(single)]
        rows.append(dict(
            fluorogen=fl, conc_nM=c, power_mW=pw,
            n_molecules=int(npm.size),
            n_molecules_single_event=int(len(single)),
            frac_molecules_removed=len(single) / max(1, npm.size),
            n_dwells_before=len(g), n_dwells_after=len(gm),
            frac_dwells_removed=1 - len(gm) / max(1, len(g)),
            mean_dwell_removed_s=float(gs.frames.mean()) * FT if len(gs) else np.nan,
            mean_dwell_kept_s=float(gm.frames.mean()) * FT if len(gm) else np.nan,
            median_dwell_removed_s=float(gs.frames.median()) * FT if len(gs) else np.nan,
            median_dwell_kept_s=float(gm.frames.median()) * FT if len(gm) else np.nan))
    R = pd.DataFrame(rows)
    R["removed_over_kept_mean"] = R.mean_dwell_removed_s / R.mean_dwell_kept_s
    R.to_csv(os.path.join(TAB, "filter_repeat_binding_cost.csv"), index=False)

    limits, L = shared_upper_limit(ONr)
    limits_off, Loff = shared_upper_limit_off(OFFI0)
    L["dwell"] = "on"
    pd.concat([L, Loff], ignore_index=True).to_csv(
        os.path.join(TAB, "filter_shared_upper_limit.csv"), index=False)

    # The repeat filter never touches INTERIOR off-gaps (a single-event molecule
    # has none by construction). All it does to the unbound set is discard those
    # molecules' BOUNDARY gaps, which are valid right-censored observations --
    # "this molecule was unbound for at least this long" -- and the longest ones
    # available. Removing them costs a large fraction of tau_off for no
    # compensating benefit, so the recommended variant applies the filter to the
    # bound-dwell set only and keeps every molecule's boundary gaps.
    variants = {
        "baseline (no repeat filter, no upper)": (ON0, OFFI0, OFFB0, "none"),
        "repeat filter only": (ONr, OFFIr, OFFBr, "none"),
        "repeat + upper censored": (ONr, OFFIr, OFFBr, "censor"),
        "repeat + upper deleted": (ONr, OFFIr, OFFBr, "delete"),
        "RECOMMENDED: repeat on bound only + upper censored":
            (ONr, OFFI0, OFFB0, "censor"),
    }
    fits = []
    for name, (o, i, b, mode) in variants.items():
        f = fit_one(o, i, b, limits, mode, limits_off=limits_off)
        f["variant"] = name
        fits.append(f)
        A.log(f"  fitted variant: {name} ({len(f)} distributions)")
    F = pd.concat(fits, ignore_index=True)
    for col in ("tau_single", "tau_fast", "tau_slow", "tau_amp_weighted",
                "tau_single_lo", "tau_single_hi"):
        if col in F.columns:
            F[col + "_s"] = F[col] * FT
    F["dBIC_per_dwell"] = F.dBIC / F.n_dwells
    F.to_csv(os.path.join(TAB, "dwell_fits_filtered.csv"), index=False)
    A.log(f"wrote filter_repeat_binding_cost.csv, filter_shared_upper_limit.csv "
          f"and dwell_fits_filtered.csv ({len(F)} fits)")
    return R, L, Loff, F, limits, limits_off


if __name__ == "__main__":
    build()
