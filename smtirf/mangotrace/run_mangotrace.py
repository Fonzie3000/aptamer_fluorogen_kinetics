#!/usr/bin/env python
"""Raw TIRF movie -> CPD-fitted HDF5, headless, with the analysis settings embedded.

Steps (the same calls the MangoTrace GUI makes, GUI removed):
  1. localize the raw movie (net-gradient detection + Gaussian LM fit) inside the ROI
  2. AIM drift estimate from those localizations, then write <stem>_undrifted.tif
  3. localize the undrifted movie with the same settings
  4. filter localizations on sx, sy, lpx, lpy, photons
  5. DBSCAN on the filtered localizations -> binding sites (cluster centroids)
  6. annular trace extraction from the undrifted movie
  7. change-point idealization (PELT, min size 1, penalty 10, no denoiser) + events
  8. write <stem>_CPD.h5 in the MangoTrace "hmm_results" format

Runs on Linux, macOS and Windows. Uses only the bundled ``mangotrace_core`` package
(a copy of the MangoTrace numerical core, no GUI).

Usage:
  python run_mangotrace.py movie.tif [movie2.tif ...] [--outdir DIR]
                                 [--settings my_settings.json] [--skip-undrift]
  python run_mangotrace.py --print-settings > my_settings.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ----------------------------------------------------------------------------- settings
SETTINGS = {
    "detect_channel": 0,                      # 0 = single-channel movie
    "localization": dict(gradient=256.0, box_size=7, baseline=400.0, sensitivity=1.0,
                         gain=1.0, quantum_efficiency=0.82, n_workers=24, use_gpu=False),
    "roi": dict(x_min=15, y_min=15, x_max=560, y_max=560),   # None = full frame
    "frames": None,                           # [start, stop] or None = full movie
    "aim": dict(segmentation=100, intersect_d=0.154, roi_r=0.462),
    # inclusive [min, max] per localization column; columns not listed are untouched
    "filter": {"sx": [0.55, 2.7], "sy": [0.55, 2.7], "lpx": [0.0, 0.36],
               "lpy": [0.0, 0.36], "photons": [64.0, 10000.0]},
    "dbscan": dict(radius=0.4, min_locs=10),
    "dbscan_input": "filtered",               # "filtered" or "raw" pass-2 localizations
    "trace": dict(inner_radius=2, outer_radius=5, buffer=1,
                  signal_method="mean", bg_method="median"),
    "cpd": dict(min_size=1, penalty=10.0, denoise=False),
    "output_suffix": "_CPD.h5",
}


def merge_settings(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_settings(out[key], value)
        else:
            out[key] = value
    return out


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- steps
def localize(movie_path, settings, label):
    from mangotrace_core.image_io import open_frame_source
    from mangotrace_core.smlm_localize import smlm_localize

    loc = dict(settings["localization"])
    loc["n_workers"] = max(1, min(int(loc.get("n_workers", 4)), os.cpu_count() or 1))
    src = open_frame_source(str(movie_path), settings["detect_channel"])
    try:
        n_frames, height, width = map(int, src.full_shape)
        ox = oy = 0
        roi = settings["roi"]
        if roi:
            x0 = max(0, min(int(roi["x_min"]), width)); y0 = max(0, min(int(roi["y_min"]), height))
            x1 = max(0, min(int(roi["x_max"]), width)); y1 = max(0, min(int(roi["y_max"]), height))
            src.set_roi(y0, y1, x0, x1)
            ox, oy = x0, y0
        fs, fe = settings["frames"] or (0, n_frames)
        log(f"[{label}] localizing {n_frames} frames, ROI offset ({ox}, {oy}), {loc['n_workers']} workers ...")
        df = smlm_localize(src, log_callback=log, frame_start=int(fs), frame_stop=int(fe), **loc)
    finally:
        src.close()
    if df is not None and len(df) and (ox or oy):
        df = df.copy()
        df["x"] = df["x"] + ox
        df["y"] = df["y"] + oy
    log(f"[{label}] {len(df):,} localizations")
    return df, n_frames


def estimate_drift(df, n_frames, settings):
    from mangotrace_core.aim_drift import estimate_aim

    result = estimate_aim(df["x"].to_numpy(), df["y"].to_numpy(), df["frame"].to_numpy(),
                          n_frames=n_frames, **settings["aim"])
    dx = np.asarray(result.drift_x, dtype=np.float64)
    dy = np.asarray(result.drift_y, dtype=np.float64)
    log(f"AIM drift: status {result.status}, range x {np.ptp(dx):.3f} px, y {np.ptp(dy):.3f} px, "
        f"warnings {list(result.warnings)}")
    return result, dx, dy


def undrift_movie(movie_path, out_path, dx, dy, settings):
    """Chunked bilinear shift by the negative drift, written to a TIFF (GUI procedure)."""
    import tifffile
    from mangotrace_core.aim_drift import shift_movie
    from mangotrace_core.image_io import open_frame_source

    out_path = Path(out_path)
    temp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp.tif")
    src = open_frame_source(str(movie_path), settings["detect_channel"])
    try:
        n_frames, height, width = map(int, src.shape)
        if len(dx) != n_frames:
            raise ValueError("drift length does not match the movie frame count")
        dtype = np.dtype(src.dtype).newbyteorder("=")
        byte_size = n_frames * height * width * dtype.itemsize
        mapped = tifffile.memmap(temp, shape=(n_frames, height, width), dtype=dtype,
                                 photometric="minisblack", metadata={"axes": "TYX"},
                                 bigtiff=byte_size >= (4 * 1024 ** 3 - 64 * 1024 ** 2))
        chunk_frames = max(1, min(64, (32 * 1024 ** 2) // (height * width * dtype.itemsize)))
        start = 0
        while start < n_frames:
            stop = min(n_frames, start + chunk_frames)
            chunk = src.read_chunk(start, stop)
            chunk = chunk.astype(dtype) if not chunk.dtype.isnative else np.ascontiguousarray(chunk)
            mapped[start:stop] = shift_movie(chunk, dx[start:stop], dy[start:stop], order=1)
            start = stop
        mapped.flush()
        del mapped
    finally:
        src.close()
    if out_path.exists():
        out_path.unlink()
    temp.replace(out_path)
    log(f"undrifted movie written: {out_path.name}")
    return out_path


def filter_localizations(df, settings):
    from mangotrace_core.smlm_localize import filter_locs_by_ranges

    ranges = {col: (float(lo), float(hi)) for col, (lo, hi) in settings["filter"].items()}
    out = filter_locs_by_ranges(df, ranges).copy()
    log(f"filter kept {len(out):,}/{len(df):,} localizations ({100 * len(out) / max(len(df), 1):.1f}%)")
    return out


def cluster_sites(df, settings):
    """Exact grid DBSCAN + MangoTrace post-processing: drop clusters below min_locs,
    relabel contiguously, one centroid Spot per cluster."""
    try:
        from dbscan import DBSCAN
    except ImportError as exc:
        raise SystemExit("The 'dbscan' package is missing. Linux/macOS: pip install dbscan ; "
                         "Windows: pip install the wheel in <MangoTrace>/wheels/") from exc
    from mangotrace_core.smlm_localize import (cluster_labels_in_spot_order, spots_from_clustered_df,
                                            _remap_contiguous)

    labelled = df.reset_index(drop=True).copy()
    X = np.ascontiguousarray(labelled[["x", "y"]].to_numpy(dtype=np.float64))
    radius, min_locs = float(settings["dbscan"]["radius"]), int(settings["dbscan"]["min_locs"])
    labels, _ = DBSCAN(X, eps=radius, min_samples=min_locs)
    labels = np.asarray(labels, dtype=np.int64)
    if (labels >= 0).any():
        uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
        small = uniq[counts < min_locs]
        if small.size:
            labels[np.isin(labels, small)] = -1
    labelled["cluster"] = _remap_contiguous(labels)
    spots = spots_from_clustered_df(labelled)
    ids = [str(v) for v in cluster_labels_in_spot_order(labelled)]
    log(f"DBSCAN: {len(spots):,} clusters from {len(labelled):,} localizations "
        f"({int((labelled['cluster'] < 0).sum()):,} noise)")
    return labelled, spots, ids


def extract_traces(undrift_path, spots, settings):
    from mangotrace_core.image_io import load_tiff
    from mangotrace_core.trace_extractor import extract_traces_with_metadata

    image = load_tiff(str(undrift_path))
    if image.ndim == 4:
        image = image[:, max(0, int(settings["detect_channel"]) - 1), :, :]
    t = settings["trace"]
    result = extract_traces_with_metadata(image, spots, inner_radius=int(t["inner_radius"]),
                                          outer_radius=int(t["outer_radius"]), buffer=int(t["buffer"]),
                                          signal_method=t["signal_method"], bg_method=t["bg_method"])
    intensities = np.asarray(result["intensities"], dtype=np.float64)   # (frames, molecules)
    locations = np.asarray(result["locations"], dtype=np.float64).reshape(-1, 2)
    log(f"traces extracted: {intensities.shape[1]} molecules x {intensities.shape[0]} frames")
    return intensities, locations


def idealize_cpd(intensities, settings):
    from mangotrace_core.trace_cpd import detect_changepoints_batch
    from mangotrace_core.trace_events import EventExtractor

    data = [np.ascontiguousarray(intensities[:, i]) for i in range(intensities.shape[1])]
    results = detect_changepoints_batch(data, **settings["cpd"])
    extractor = EventExtractor()
    time_axis = np.arange(intensities.shape[0], dtype=float)
    states, means, events = [], [], []
    for i, (st, mu, _cov) in enumerate(results):
        st = np.asarray(st, dtype=np.int32)
        states.append(st)
        means.append(np.asarray(mu, dtype=float))
        events.append([dict(e.items()) for e in extractor.extract_events(time_axis, st, data[i])])
    log(f"CPD: {sum(len(e) for e in events):,} events on {sum(1 for e in events if e):,}/{len(events):,} molecules")
    return states, means, events


def save_fitted_h5(path, source_file, intensities, ids, locations, states, means, events):
    """MangoTrace 'hmm_results' format version 4 (one channel named 'traces')."""
    import h5py

    frames, count = intensities.shape
    state_mat = np.full((frames, count), np.nan)
    ideal_mat = np.full((frames, count), np.nan)
    for i in range(count):
        st = states[i]
        state_mat[:len(st), i] = st
        ideal_mat[:len(st), i] = means[i][np.clip(st, 0, 1)]
    time_mat = np.repeat(np.arange(frames, dtype=float)[:, None], count, axis=1)
    with h5py.File(path, "w") as f:
        f.attrs.update(mangotrace_format="hmm_results", format_version=4, source_file=str(source_file),
                       primary_channel="traces", filtered=False, idealizer="cpd")
        g = f.create_group("channel_results").create_group("0")
        g.attrs.update(channel_name="traces", channel_index=0, n_molecules=count,
                       n_frames=frames, hmm_model="pooled")
        g.create_dataset("molecule_ids", data=np.asarray(ids, dtype=h5py.string_dtype()))
        g.create_dataset("locations", data=locations)
        g.create_dataset("trace_lengths", data=np.full(count, frames, dtype=np.int64))
        g.create_dataset("time", data=time_mat)
        g.create_dataset("traces", data=intensities)
        g.create_dataset("idealized", data=ideal_mat)
        g.create_dataset("states", data=state_mat)
        ev = g.create_group("events")
        for i, mol in enumerate(ids):
            grp = ev.create_group(str(i))
            grp.attrs["molecule_id"] = mol
            grp.create_dataset("records", data=json.dumps(events[i], default=lambda v: v.item()),
                               dtype=h5py.string_dtype())
            grp.create_dataset("selected", data=np.ones(len(events[i]), dtype=bool))
    log(f"fitted file written: {Path(path).name}")


# ----------------------------------------------------------------------------- driver
def run_movie(movie, settings, outdir=None, skip_undrift=False):
    movie = Path(movie).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve() if outdir else movie.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = movie.stem
    t0 = time.time()
    record = {"movie": str(movie), "platform": sys.platform, "settings": settings,
              "timings_s": {}, "counts": {}}

    t = time.time()
    locs1, n_frames = localize(movie, settings, "pass 1")
    locs1.to_csv(outdir / f"{stem}_locs_pass1.csv", index=False)
    record["timings_s"]["localize_pass1"] = time.time() - t
    record["counts"]["localizations_pass1"] = int(len(locs1))

    undrift_path = outdir / f"{stem}_undrifted.tif"
    if skip_undrift and undrift_path.exists():
        log(f"reusing existing {undrift_path.name}")
    else:
        t = time.time()
        aim, dx, dy = estimate_drift(locs1, n_frames, settings)
        np.savetxt(outdir / f"{stem}_drift.csv", np.column_stack((np.arange(n_frames), dx, dy)),
                   delimiter=",", header="frame,drift_x_px,drift_y_px", comments="")
        record["timings_s"]["aim"] = time.time() - t
        record["aim"] = {"status": aim.status, "warnings": list(aim.warnings),
                         "drift_range_x_px": float(np.ptp(dx)), "drift_range_y_px": float(np.ptp(dy))}
        t = time.time()
        undrift_movie(movie, undrift_path, dx, dy, settings)
        record["timings_s"]["undrift"] = time.time() - t

    t = time.time()
    locs2, _ = localize(undrift_path, settings, "pass 2")
    record["timings_s"]["localize_pass2"] = time.time() - t
    record["counts"]["localizations_pass2"] = int(len(locs2))

    filtered = filter_localizations(locs2, settings)
    filtered.to_csv(outdir / f"{stem}_locs_filtered.csv", index=False)
    record["counts"]["localizations_filtered"] = int(len(filtered))

    t = time.time()
    cluster_input = filtered if settings.get("dbscan_input", "filtered") == "filtered" else locs2
    labelled, spots, ids = cluster_sites(cluster_input, settings)
    labelled.to_csv(outdir / f"{stem}_clusters.csv", index=False)
    record["timings_s"]["dbscan"] = time.time() - t
    record["counts"]["clusters"] = len(spots)

    t = time.time()
    intensities, locations = extract_traces(undrift_path, spots, settings)
    record["timings_s"]["extract"] = time.time() - t

    t = time.time()
    states, means, events = idealize_cpd(intensities, settings)
    record["timings_s"]["cpd"] = time.time() - t
    record["counts"]["events"] = int(sum(len(e) for e in events))
    record["counts"]["molecules_with_events"] = int(sum(1 for e in events if e))

    out_h5 = outdir / f"{stem}{settings['output_suffix']}"
    save_fitted_h5(out_h5, undrift_path, intensities, ids, locations, states, means, events)
    record["output"] = str(out_h5)
    record["timings_s"]["total"] = time.time() - t0
    (outdir / f"{stem}_pipeline_log.json").write_text(json.dumps(record, indent=2, default=str),
                                                      encoding="utf-8")
    log(f"done in {record['timings_s']['total'] / 60:.1f} min -> {out_h5}")
    return out_h5


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("movies", nargs="*", help="raw TIFF movie(s)")
    p.add_argument("--outdir", help="output folder (default: next to each movie)")
    p.add_argument("--settings", help="JSON file whose keys override the built-in SETTINGS")
    p.add_argument("--print-settings", action="store_true", help="print the effective settings as JSON and exit")
    p.add_argument("--skip-undrift", action="store_true",
                   help="reuse an existing <stem>_undrifted.tif instead of re-estimating drift")
    a = p.parse_args(argv)
    settings = SETTINGS
    if a.settings:
        settings = merge_settings(SETTINGS, json.loads(Path(a.settings).read_text(encoding="utf-8")))
    if a.print_settings:
        print(json.dumps(settings, indent=2))
        return
    if not a.movies:
        p.error("give at least one movie, or --print-settings")
    failures = []
    for movie in a.movies:
        try:
            run_movie(movie, settings, a.outdir, a.skip_undrift)
        except Exception as exc:
            log(f"FAILED {movie}: {exc}")
            failures.append((movie, exc))
    if failures:
        raise SystemExit(f"{len(failures)} movie(s) failed: " + "; ".join(str(m) for m, _ in failures))


if __name__ == "__main__":
    main()
