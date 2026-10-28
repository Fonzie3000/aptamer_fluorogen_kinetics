"""
SMLM / SMLM Detection Pipeline using Picasso Algorithms.

Implements bounded multi-frame spot-localization jobs using Picasso's net
gradient method, followed by exact parallel DBSCAN clustering
(wangyiqiu/dbscan-python) to define final molecular coordinates.
"""

import os
import math
import numpy as np
import pandas as pd
import numba
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Optional

from .data_types import Spot

@numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame: np.ndarray, box_size: int) -> tuple[np.ndarray, np.ndarray]:
    if box_size < 3 or box_size % 2 == 0:
        raise ValueError("box_size must be an odd integer >= 3")
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box_size / 2)
    # The upper bound is exclusive, so ``Y - box_half`` is the last legal
    # centre plus one, keeping the bottom/right row and column of every frame.
    for i in range(box_half, Y - box_half):
        for j in range(box_half, X - box_half):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box_size)
            j_local_max = int(flat_max % box_size)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x

@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame: np.ndarray, y: int, x: int) -> tuple[float, float]:
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx

@numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame: np.ndarray, y: np.ndarray, x: np.ndarray, box_size: int, uy: np.ndarray, ux: np.ndarray) -> np.ndarray:
    box_half = int(box_size / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m)
                    ng[i] += (gy * uy[k_index, l_index] + gx * ux[k_index, l_index])
    return ng

@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image: np.ndarray, minimum_ng: float, box_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if box_size < 3 or box_size % 2 == 0:
        raise ValueError("box_size must be an odd integer >= 3")
    y, x = local_maxima(image, box_size)
    box_half = int(box_size / 2)
    # net_gradient uses a central difference at every non-centre pixel in the
    # fit box, so its outer ring needs one additional source pixel beyond the
    # patch. Filter before gradient_at can wrap a negative index at the top/left.
    valid = ((y >= box_half + 1) & (y < image.shape[0] - box_half - 1)
             & (x >= box_half + 1) & (x < image.shape[1] - box_half - 1))
    y = y[valid]
    x = x[valid]
    ux = np.zeros((box_size, box_size), dtype=np.float32)
    uy = np.zeros((box_size, box_size), dtype=np.float32)
    for i in range(box_size):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    unorm[box_half, box_half] = 1.0
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box_size, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


class LocalizationCancelled(RuntimeError):
    """Internal control-flow signal for a cooperatively cancelled localization.

    A successful movie with no detections also returns an empty DataFrame, so
    cancellation must not be encoded as an empty result.
    """


@dataclass(frozen=True)
class _LocalizationJob:
    sequence: int
    start: int
    stop: int

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class _LocalizationPlan:
    workers: int
    max_inflight: int
    frames_per_chunk: int
    min_chunk_frames: int
    budget_bytes: int
    estimated_bytes_per_frame: int


_MIN_CHUNK_FRAMES = 4
_MAX_CHUNK_FRAMES = 64
_MIN_MEMORY_BUDGET = 64 * 1024 ** 2
_MAX_MEMORY_BUDGET = 2 * 1024 ** 3
_WORKING_SET_FACTOR = 8


def _localization_memory_budget(override: Optional[int] = None) -> int:
    """Transient localization budget: 25% of available RAM, capped 64 MiB–2 GiB.

    ``override`` is intentionally supported for deterministic tests and advanced
    callers; positive overrides are used verbatim.
    """
    if override is not None:
        value = int(override)
        if value <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        return value
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        available = 2 * 1024 ** 3
    return max(_MIN_MEMORY_BUDGET,
               min(_MAX_MEMORY_BUDGET, int(available * 0.25)))


def _plan_localization_jobs(video, n_frames: int, requested_workers: int,
                            memory_budget_bytes: Optional[int] = None,
                            *, bytes_per_frame: Optional[int] = None,
                            never_exceed_chunk_frames: Optional[int] = None,
                            never_exceed_inflight: Optional[int] = None,
                            ) -> _LocalizationPlan:
    """Choose multi-frame job size and concurrency under a transient RAM budget.

    The budget is soft with respect to the final output DataFrames (which are the
    requested result), but hard-bounds the estimated resident input/fit work.
    Worker count is reduced before normal jobs fall below four frames.
    """
    n_frames = max(1, int(n_frames))
    requested_workers = max(1, int(requested_workers))
    cpu_count = max(1, os.cpu_count() or 1)
    budget = _localization_memory_budget(memory_budget_bytes)
    if bytes_per_frame is None:
        shape = tuple(int(v) for v in video.shape)
        dtype = np.dtype(video.dtype)
        bytes_per_frame = int(np.prod(shape[-2:], dtype=np.int64)) * dtype.itemsize
    estimated = max(1, int(bytes_per_frame) * _WORKING_SET_FACTOR)

    preferred_min = min(_MIN_CHUNK_FRAMES, n_frames)
    # If even one preferred job cannot fit, degrade only as far as memory forces
    # us.  A one-frame job is therefore an exceptional low-memory/short-range case.
    affordable_min = max(1, min(preferred_min, budget // estimated))
    slots_at_min = max(1, budget // max(estimated * affordable_min, 1))
    jobs_at_min = max(1, math.ceil(n_frames / affordable_min))
    workers = min(requested_workers, cpu_count, jobs_at_min, slots_at_min)
    workers = max(1, workers)
    if never_exceed_inflight is not None:
        slots_at_min = min(slots_at_min, max(1, int(never_exceed_inflight)))
        workers = min(workers, slots_at_min)
    max_inflight = max(1, min(workers + 1, slots_at_min))

    by_budget = max(1, budget // max(max_inflight * estimated, 1))
    enough_jobs = max(1, math.ceil(n_frames / workers))
    frames_per_chunk = min(_MAX_CHUNK_FRAMES, by_budget, enough_jobs)
    frames_per_chunk = max(affordable_min, frames_per_chunk)
    if never_exceed_chunk_frames is not None:
        frames_per_chunk = min(frames_per_chunk,
                               max(1, int(never_exceed_chunk_frames)))
        frames_per_chunk = max(affordable_min, frames_per_chunk)

    return _LocalizationPlan(
        workers=workers,
        max_inflight=max_inflight,
        frames_per_chunk=frames_per_chunk,
        min_chunk_frames=affordable_min,
        budget_bytes=budget,
        estimated_bytes_per_frame=estimated,
    )


def _read_frame_chunk(video, start: int, stop: int) -> np.ndarray:
    """Materialise exactly one requested multi-frame chunk from an array/source."""
    reader = getattr(video, "read_chunk", None)
    chunk = reader(start, stop) if callable(reader) else video[start:stop]
    chunk = np.asarray(chunk)
    if chunk.ndim == 2:
        chunk = chunk[np.newaxis, ...]
    return chunk


def gpu_fit_available() -> bool:
    """Return whether pygpufit is importable *and* reports a usable CUDA device."""
    try:
        from pygpufit import gpufit as gf
        return bool(gf.cuda_available())
    except Exception:
        return False


# Learned GPU-fit batch ceiling. Set when a gpufit call OOMs and is retried with
# a smaller batch, so later chunks in the same run start at a size known to fit
# instead of re-discovering the limit (and re-spamming OOM warnings) every chunk.
_GPU_FIT_MAX_BATCH: Optional[int] = None


def _gpu_fit_batch_size(n_spots: int, box_size: int, default: int = 100000) -> int:
    """Largest spot batch to fit on the GPU at once, bounded by free VRAM.

    Queries free GPU memory via CuPy when available; otherwise falls back to a
    conservative fixed cap. gpufit needs the spot data plus several working
    buffers, so budget generously (~24x the raw spot bytes, ≤30% of free VRAM)
    and never exceed any ceiling learned from a previous OOM this run.
    """
    try:
        import cupy as cp
        # Reclaim CuPy's cached pool first so the free-memory query is realistic.
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        free, _total = cp.cuda.Device().mem_info
        bytes_per_spot = box_size * box_size * 4 * 24
        n = int(free * 0.30) // max(bytes_per_spot, 1)
        n = max(2048, min(n_spots, n))
    except Exception:
        n = min(n_spots, default)
    if _GPU_FIT_MAX_BATCH is not None:
        n = min(n, _GPU_FIT_MAX_BATCH)
    return int(max(1, n))


def _free_gpu_pool() -> None:
    """Release CuPy's cached GPU memory pool so freed VRAM is returned to the
    driver — gives gpufit room before/after a fit. No-op without CuPy."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _is_gpu_oom(err: Exception) -> bool:
    """True if ``err`` looks like a GPU out-of-memory error (gpufit raises a
    RuntimeError whose message contains 'not enough free GPU memory')."""
    s = str(err).lower()
    return ('memory' in s) or ('out of memory' in s) or ('cudaerrormemory' in s)


@numba.jit(nopython=True, nogil=True, cache=False)
def _extract_spots(chunk, ts, ys, xs, box_size, baseline, sensitivity, gain,
                   quantum_efficiency):
    """Cut box_size×box_size patches around (ys, xs) on local chunk frames ``ts`` and
    convert to photons: (patch - baseline) * sensitivity / (gain * qe).

    This is Picasso's camera conversion, quantum efficiency included, so photon
    counts and the localization precisions derived from them are directly
    comparable with a Picasso analysis of the same movie.

    Runs in nopython/nogil mode so it parallelises across worker threads (the
    Python slicing loop it replaces held the GIL and serialised patch extraction
    across chunks)."""
    r = box_size // 2
    n = ys.shape[0]
    scale = sensitivity / (gain * quantum_efficiency)
    spots = np.empty((n, box_size, box_size), dtype=np.float32)
    for i in range(n):
        t = ts[i]; yc = ys[i]; xc = xs[i]
        for a in range(box_size):
            row = yc - r + a
            for b in range(box_size):
                spots[i, a, b] = np.float32((chunk[t, row, xc - r + b] - baseline) * scale)
    return spots


def _identify_and_extract(
    chunk: np.ndarray,
    gradient: float,
    box_size: int,
    baseline: float,
    sensitivity: float,
    gain: float,
    quantum_efficiency: float = 1.0,
    frame_offset: int = 0,
    progress_hook: Optional[Callable[[float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    """CPU phase of localization: per-frame net-gradient identification + patch
    extraction. Returns ``(identifications_df, spots)`` or ``(empty_df, None)``.

    The heavy inner work (``identify_in_image`` and ``_extract_spots``) is nogil,
    so calling this from several worker threads identifies and extracts on
    multiple cores in true parallel — this is the dominant cost of localization
    (the Gaussian fit, CPU or GPU, is comparatively cheap)."""
    # numba kernels (_extract_spots) require native byte order, but some
    # MicroManager OME-TIFFs are big-endian (>u2). Normalize the chunk once
    # (pixel values unchanged); it is already a RAM-sized slice, so this is cheap.
    if not chunk.dtype.isnative:
        chunk = np.ascontiguousarray(chunk.astype(chunk.dtype.newbyteorder('=')))

    n_frames = chunk.shape[0]
    r = int(box_size / 2)
    all_y, all_x, all_ng, all_t = [], [], [], []

    for t in range(n_frames):
        # Cooperative cancellation: bail between frames so a Terminate stops this
        # chunk within ~a frame (never orphans the worker-thread pool). identify
        # is the dominant per-frame cost, so a mid-run cancel is near-immediate.
        if should_cancel is not None and should_cancel():
            return pd.DataFrame(), None
        frame = chunk[t].astype(np.float32)
        y, x, ng = identify_in_image(frame, gradient, box_size)
        if progress_hook is not None and (t % 8 == 0 or t == n_frames - 1):
            progress_hook((t + 1) / n_frames)

        # Filter edge spots to avoid bounds errors during patch extraction
        valid = (y >= r) & (y < frame.shape[0] - r) & (x >= r) & (x < frame.shape[1] - r)
        y, x, ng = y[valid], x[valid], ng[valid]

        if len(x) > 0:
            all_y.extend(y)
            all_x.extend(x)
            all_ng.extend(ng)
            all_t.extend([t] * len(x))

    if len(all_x) == 0:
        return pd.DataFrame(), None

    y_arr = np.asarray(all_y, dtype=np.int64)
    x_arr = np.asarray(all_x, dtype=np.int64)
    ng_arr = np.asarray(all_ng, dtype=np.float32)
    t_arr = np.asarray(all_t, dtype=np.int64)        # local (within-chunk) frame index

    spots = _extract_spots(chunk, t_arr, y_arr, x_arr, box_size, baseline,
                           sensitivity, gain, quantum_efficiency)
    identifications = pd.DataFrame(
        {'frame': t_arr + frame_offset, 'x': x_arr, 'y': y_arr, 'net_gradient': ng_arr})
    return identifications, spots


def _gpu_fit_spots(spots, identifications, box_size, log_callback=None,
                   progress_hook=None, should_cancel=None, stats=None):
    """GPU least-squares elliptical-Gaussian fit (pygpufit / Przybylski et al.,
    Sci. Rep. 2017).

    This is the SAME fit as the CPU path (core.gausslq) — the same elliptical 2D
    Gaussian model, the same least-squares objective and, since the box
    containment check moved into ``locs_from_fits``, the same acceptance rules —
    just run on the GPU.
    gpufit's least-squares estimator (``EstimatorID.LSE``) is requested
    explicitly below; gpufit can also do maximum-likelihood (MLE), but MangoTrace
    does not use it — localization is always least-squares Gaussian fitting.

    Returns a localizations DataFrame, or ``None`` if the GPU fit fails (the
    caller then falls back to the CPU fit). The CUDA context used by gpufit is
    thread-affine, so smlm_localize calls this only from the one calling thread.

    Spots whose GPU fit does not converge (or converges to an unphysical
    Gaussian) are DISCARDED, not refitted on the CPU: the CPU refit ran the same
    least-squares model on the same patch, so it recovered almost nothing while
    serializing every chunk behind a slow single-threaded fit. ``stats`` is an
    optional dict; the discarded count accumulates under ``'discarded'`` so the
    caller can report one total instead of a line per chunk.
    """
    try:
        from pygpufit import gpufit as gf

        size = box_size
        center = (size / 2.0) - 0.5
        initial_width = max(size / 5.0, 1.0)
        spot_max = spots.max(axis=(1, 2))
        spot_min = spots.min(axis=(1, 2))
        init_params = np.empty((len(spots), 6), dtype=np.float32)
        init_params[:, 0] = spot_max - spot_min   # Gaussian peak amplitude
        init_params[:, 1] = center                 # x
        init_params[:, 2] = center                 # y
        init_params[:, 3] = initial_width          # sx
        init_params[:, 4] = initial_width          # sy
        init_params[:, 5] = spot_min               # bg

        spots_flat = np.ascontiguousarray(
            spots.reshape(len(spots), size * size), dtype=np.float32)
        n_spots = len(spots_flat)

        # Always fit in VRAM-bounded sub-batches, and on an out-of-memory error
        # free the pool, halve the batch, and retry the SAME range — so a chunk
        # that is momentarily too big for free VRAM still completes on the GPU
        # instead of dumping a traceback and falling back to the CPU. The first
        # batch that OOMs lowers a process-wide ceiling so later chunks start
        # small (no repeated OOM spam).
        global _GPU_FIT_MAX_BATCH
        _free_gpu_pool()
        batch = _gpu_fit_batch_size(n_spots, size)
        MIN_BATCH = 512
        parts = []
        fit_states = []
        s0 = 0
        while s0 < n_spots:
            if should_cancel is not None and should_cancel():
                raise LocalizationCancelled()
            n = min(batch, n_spots - s0)
            try:
                p, states, _cs, _ni, _et = gf.fit(
                    spots_flat[s0:s0 + n], None,
                    gf.ModelID.GAUSS_2D_ELLIPTIC,
                    init_params[s0:s0 + n],
                    tolerance=1e-2,
                    max_number_iterations=20,
                    estimator_id=gf.EstimatorID.LSE,   # least squares (same as CPU path)
                )
            except RuntimeError as e:
                if _is_gpu_oom(e) and batch > MIN_BATCH:
                    _free_gpu_pool()
                    batch = max(MIN_BATCH, batch // 2)
                    _GPU_FIT_MAX_BATCH = batch
                    if log_callback is not None:
                        log_callback(
                            "GPU low on memory — retrying localization fit with a "
                            f"smaller batch ({batch} spots).")
                    continue  # retry the same range with the smaller batch
                raise          # not OOM (or already minimal) → CPU fallback
            parts.append(p)
            fit_states.append(states)
            s0 += n
            if progress_hook is not None:
                progress_hook(s0 / n_spots)
        all_parameters = np.vstack(parts)
        converged = (np.concatenate(fit_states) == 0)
        converged &= np.all(np.isfinite(all_parameters), axis=1)
        # Gpufit's unconstrained Gaussian can report convergence with an
        # inverted width or a centre outside the fitted patch. Such a result is
        # not a localization and is dropped exactly like a failed fit.
        converged &= ((all_parameters[:, 0] > 0)
                      & (all_parameters[:, 3] > 0) & (all_parameters[:, 4] > 0)
                      & (all_parameters[:, 1] >= 0) & (all_parameters[:, 1] <= size - 1)
                      & (all_parameters[:, 2] >= 0) & (all_parameters[:, 2] <= size - 1))
        parameters = all_parameters[converged].copy()
        successful_ids = identifications.iloc[np.flatnonzero(converged)]

        # Convert integrated photons: photons *= 2*pi*sx*sy
        parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

        # p[0] above has now been converted from amplitude to integrated photons.
        # Translate to absolute pixel coords (gpufit centres on box_size centre),
        # then assemble the locs DataFrame via the shared core helper so the
        # precision / column schema match the CPU path exactly.
        box_offset = int(box_size / 2)
        x_abs = parameters[:, 1] + successful_ids["x"].values - box_offset
        y_abs = parameters[:, 2] + successful_ids["y"].values - box_offset

        from .gausslq import build_locs_dataframe
        df = build_locs_dataframe(
            frame=successful_ids['frame'].to_numpy(),
            x=x_abs, y=y_abs,
            photons=parameters[:, 0], bg=parameters[:, 5],
            sx=parameters[:, 3], sy=parameters[:, 4],
            net_gradient=successful_ids['net_gradient'].to_numpy(), em=False,
        )
        df = _repair_negative_background_precision(df)
        if stats is not None:
            stats['discarded'] = int(stats.get('discarded', 0)) + int(
                np.count_nonzero(~converged))
        df.sort_values(by='frame', inplace=True)
        df.dropna(subset=['x', 'y', 'sx', 'sy'], inplace=True)
        return df

    except LocalizationCancelled:
        raise
    except Exception as e:
        # Free any GPU memory we managed to grab, then fall back to the CPU fit.
        # Report a CONCISE one-line reason (not a full traceback) so a genuine
        # GPU issue is still visible without flooding the log on every chunk.
        _free_gpu_pool()
        if log_callback is not None:
            log_callback(
                "GPU localization (pygpufit) unavailable — falling back to CPU. "
                f"Reason: {type(e).__name__}: {e}")
        return None


def _repair_negative_background_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve a fitted negative offset but clamp precision's variance input."""
    required = {'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy'}
    if df is None or len(df) == 0 or not required.issubset(df.columns):
        return df
    from .gausslq import localization_precision
    bg_for_variance = np.maximum(df['bg'].to_numpy(dtype=np.float64), 0.0)
    photons = df['photons'].to_numpy(dtype=np.float64)
    sx = df['sx'].to_numpy(dtype=np.float64)
    sy = df['sy'].to_numpy(dtype=np.float64)
    df = df.copy()
    df['lpx'] = localization_precision(
        photons, sx, sy, bg_for_variance, em=False).astype(np.float32)
    df['lpy'] = localization_precision(
        photons, sy, sx, bg_for_variance, em=False).astype(np.float32)
    return df


def _cpu_fit_spots(spots, identifications, box_size, should_cancel=None):
    """CPU native LM Gaussian fit (core.gausslq) → localizations DataFrame."""
    from .gausslq import fit_spots, locs_from_fits
    def _fit_progress(_index):
        if should_cancel is not None and should_cancel():
            raise LocalizationCancelled()

    theta = fit_spots(
        spots, progress_callback=_fit_progress if should_cancel is not None else None
    )  # [x, y, photons, bg, sx, sy] per spot
    if should_cancel is not None and should_cancel():
        raise LocalizationCancelled()
    df = locs_from_fits(identifications, theta, box_size, em=False)
    df = _repair_negative_background_precision(df)
    df.dropna(subset=['x', 'y', 'sx', 'sy'], inplace=True)
    return df


def _dataframe_bytes(df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    try:
        return int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        return 0


def _identified_payload_bytes(chunk, identifications, spots) -> int:
    total = int(getattr(chunk, 'nbytes', 0)) + _dataframe_bytes(identifications)
    if spots is not None:
        total += int(getattr(spots, 'nbytes', 0))
    return total


def _identify_job(chunk, gradient, box_size, baseline, sensitivity, gain,
                  quantum_efficiency, frame_offset, should_cancel):
    identifications, spots = _identify_and_extract(
        chunk, gradient, box_size, baseline, sensitivity, gain,
        quantum_efficiency, frame_offset, should_cancel=should_cancel)
    if should_cancel is not None and should_cancel():
        raise LocalizationCancelled()
    return identifications, spots, _identified_payload_bytes(
        chunk, identifications, spots)


def _cpu_localization_job(chunk, gradient, box_size, baseline, sensitivity, gain,
                          quantum_efficiency, frame_offset, should_cancel):
    identifications, spots, payload = _identify_job(
        chunk, gradient, box_size, baseline, sensitivity, gain,
        quantum_efficiency, frame_offset, should_cancel)
    if spots is None:
        return pd.DataFrame(), payload
    df = _cpu_fit_spots(
        spots, identifications, box_size, should_cancel=should_cancel)
    return df, payload + _dataframe_bytes(df)


def _run_owned_chunk(chunk_owner, worker, *args):
    """Run ``worker`` after taking sole ownership of one materialised chunk.

    ``ThreadPoolExecutor`` keeps its submitted argument tuple alive until after
    ``Future.set_result``.  Submitting the ndarray directly therefore leaves a
    short window in which the caller can observe a completed future, read its
    replacement chunk, and exceed ``max_inflight`` by one allocation.  The
    mutable one-item owner breaks that reference before work starts; the worker's
    local reference is then destroyed before the future becomes observable as
    complete.
    """
    chunk = chunk_owner.pop()
    try:
        return worker(chunk, *args)
    finally:
        del chunk


def _shrink_localization_plan(plan: _LocalizationPlan,
                              observed_bytes_per_frame: int) -> _LocalizationPlan:
    """Adapt downward after real data; chunk size/concurrency never grow."""
    estimated = max(plan.estimated_bytes_per_frame,
                    int(observed_bytes_per_frame))
    # Preserve normal four-frame work by reducing concurrency first.
    slots_for_min = max(
        1, plan.budget_bytes // max(estimated * plan.min_chunk_frames, 1))
    max_inflight = max(1, min(plan.max_inflight, slots_for_min))
    workers = max(1, min(plan.workers, max_inflight))
    max_inflight = min(max_inflight, workers + 1)
    by_budget = max(
        1, plan.budget_bytes // max(max_inflight * estimated, 1))
    frames_per_chunk = max(
        1, min(plan.frames_per_chunk, by_budget, _MAX_CHUNK_FRAMES))
    return _LocalizationPlan(
        workers=workers,
        max_inflight=max_inflight,
        frames_per_chunk=frames_per_chunk,
        min_chunk_frames=min(plan.min_chunk_frames, frames_per_chunk),
        budget_bytes=plan.budget_bytes,
        estimated_bytes_per_frame=estimated,
    )




def smlm_localize(
    video: np.ndarray,
    gradient: float = 5000.0,
    box_size: int = 5,
    baseline: float = 100.0,
    sensitivity: float = 1.0,
    gain: float = 1.0,
    quantum_efficiency: float = 1.0,
    n_workers: int = 4,
    use_gpu: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
) -> pd.DataFrame:
    """Localize a movie through a bounded rolling queue of multi-frame jobs.

    Only ``max_inflight <= workers + 1`` chunks are materialised at once. CUDA
    fitting remains on this calling thread because pygpufit's context is
    thread-affine; CPU identification/fitting runs in the bounded executor.
    Returned frame numbers are absolute source-frame indices.
    """
    if video.ndim == 2:
        video = video[np.newaxis, :, :]
    if int(box_size) < 3 or int(box_size) % 2 == 0:
        raise ValueError("box_size must be an odd integer >= 3")
    if float(gain) == 0.0:
        raise ValueError("gain must be non-zero")
    if not 0.0 < float(quantum_efficiency) <= 1.0:
        raise ValueError("quantum_efficiency must lie in (0, 1]")

    source_frames = int(video.shape[0])
    start = max(0, min(int(frame_start), source_frames))
    stop = source_frames if frame_stop is None else int(frame_stop)
    stop = max(0, min(stop, source_frames))
    if start >= stop:
        raise ValueError("frame range is empty")
    selected_frames = stop - start
    plan = _plan_localization_jobs(
        video, selected_frames, n_workers, memory_budget_bytes)

    gpu_requested = bool(use_gpu)
    gpu_active = gpu_requested and gpu_fit_available()
    if gpu_requested and not gpu_active and log_callback is not None:
        log_callback(
            "GPU localization unavailable (no usable CUDA gpufit device); "
            "using CPU.")
    global _GPU_FIT_MAX_BATCH
    if gpu_active:
        _GPU_FIT_MAX_BATCH = None

    if log_callback is not None:
        log_callback(
            f"Localization scheduler: {plan.frames_per_chunk}-frame chunks, "
            f"{plan.workers} worker(s), max {plan.max_inflight} in flight, "
            f"{plan.budget_bytes / 1024 ** 2:.0f} MiB transient budget.")

    def _cancelled():
        return should_cancel is not None and should_cancel()

    def _raise_if_cancelled():
        if _cancelled():
            raise LocalizationCancelled()

    results = {}
    completed_frames = 0
    gpu_spots = 0
    gpu_used = False
    # Nonconverged GPU fits accumulate here and are reported once at the end,
    # instead of one log line per chunk.
    gpu_stats = {'discarded': 0}

    def _commit(job, df, payload_bytes):
        nonlocal completed_frames, plan
        results[job.sequence] = df
        completed_frames += job.n_frames
        observed = math.ceil(
            max(1, int(payload_bytes)) * 1.25 / max(job.n_frames, 1))
        plan = _shrink_localization_plan(plan, observed)
        if progress_callback is not None:
            progress_callback(min(
                99, int(100 * completed_frames / selected_frames)))

    # First real 4-8 frame job: warm kernels and observe emitter-dependent
    # payload before filling the rolling queue. It is part of the final result.
    _raise_if_cancelled()
    warm_frames = min(selected_frames, plan.frames_per_chunk,
                      max(plan.min_chunk_frames, 8))
    warm_job = _LocalizationJob(0, start, start + warm_frames)
    warm_chunk = _read_frame_chunk(video, warm_job.start, warm_job.stop)
    identifications, spots, payload = _identify_job(
        warm_chunk, gradient, box_size, baseline, sensitivity, gain,
        quantum_efficiency,
        warm_job.start, should_cancel)
    del warm_chunk
    _raise_if_cancelled()
    if spots is None:
        warm_df = pd.DataFrame()
    elif gpu_active:
        warm_df = _gpu_fit_spots(
            spots, identifications, box_size, log_callback=log_callback,
            should_cancel=should_cancel, stats=gpu_stats)
        if warm_df is None:
            gpu_active = False
            warm_df = _cpu_fit_spots(
                spots, identifications, box_size, should_cancel=should_cancel)
        else:
            gpu_used = True
            gpu_spots += len(spots)
    else:
        warm_df = _cpu_fit_spots(
            spots, identifications, box_size, should_cancel=should_cancel)
    payload += _dataframe_bytes(warm_df)
    del identifications, spots
    _commit(warm_job, warm_df, payload)

    next_frame = warm_job.stop
    next_sequence = 1
    pending = {}
    executor = None
    try:
        if next_frame < stop:
            executor = ThreadPoolExecutor(max_workers=plan.workers)
        while next_frame < stop or pending:
            _raise_if_cancelled()
            while (next_frame < stop
                   and len(pending) < plan.max_inflight
                   and not _cancelled()):
                chunk_stop = min(stop, next_frame + plan.frames_per_chunk)
                job = _LocalizationJob(next_sequence, next_frame, chunk_stop)
                chunk_owner = [_read_frame_chunk(video, job.start, job.stop)]
                if gpu_active:
                    future = executor.submit(
                        _run_owned_chunk, chunk_owner, _identify_job,
                        gradient, box_size, baseline, sensitivity, gain,
                        quantum_efficiency,
                        job.start, should_cancel)
                    mode = 'identify'
                else:
                    future = executor.submit(
                        _run_owned_chunk, chunk_owner, _cpu_localization_job,
                        gradient, box_size, baseline, sensitivity, gain,
                        quantum_efficiency,
                        job.start, should_cancel)
                    mode = 'cpu'
                pending[future] = (job, mode)
                next_frame = chunk_stop
                next_sequence += 1

            _raise_if_cancelled()
            if not pending:
                continue
            done, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
            # Consume one result and refill immediately, keeping identification
            # supplied while serialized CUDA fitting runs on this thread.
            future = min(done, key=lambda f: pending[f][0].sequence)
            job, mode = pending.pop(future)
            if mode == 'cpu':
                df_chunk, payload = future.result()
            else:
                identifications, spots, payload = future.result()
                _raise_if_cancelled()
                if spots is None:
                    df_chunk = pd.DataFrame()
                elif gpu_active:
                    df_chunk = _gpu_fit_spots(
                        spots, identifications, box_size,
                        log_callback=log_callback,
                        should_cancel=should_cancel, stats=gpu_stats)
                    if df_chunk is None:
                        # Do not retry a broken CUDA path on every later chunk.
                        gpu_active = False
                        # Both fit paths now accept the same localizations
                        # (same physicality rules, same box containment), so the
                        # switch changes speed and not which spots survive — but
                        # say where it happened, so a movie's provenance is not
                        # a single "fit on the GPU" line hiding a mixed run.
                        if log_callback is not None:
                            log_callback(
                                "GPU fitting stopped at frame "
                                f"{job.start}; the rest of this movie is fitted "
                                "on the CPU.")
                        df_chunk = _cpu_fit_spots(
                            spots, identifications, box_size,
                            should_cancel=should_cancel)
                    else:
                        gpu_used = True
                        gpu_spots += len(spots)
                else:
                    df_chunk = _cpu_fit_spots(
                        spots, identifications, box_size,
                        should_cancel=should_cancel)
                payload += _dataframe_bytes(df_chunk)
                del identifications, spots
            _commit(job, df_chunk, payload)
    except BaseException:
        for future in pending:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    _raise_if_cancelled()
    if gpu_used and log_callback is not None:
        discarded = int(gpu_stats.get('discarded', 0))
        note = (f" {discarded} nonconverged fits discarded."
                if discarded else "")
        log_callback(
            f"GPU localization (pygpufit): fit {gpu_spots} spots on the GPU."
            f"{note}")
    if progress_callback is not None:
        progress_callback(100)
    nonempty = [results[i] for i in sorted(results)
                if results[i] is not None and not results[i].empty]
    if not nonempty:
        return pd.DataFrame()
    df_final = pd.concat(nonempty, ignore_index=True)
    df_final.sort_values(by=['frame'], kind='stable', inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    return df_final


def _remap_contiguous(labels: np.ndarray) -> np.ndarray:
    """Remap non-negative cluster labels to a contiguous 0..K-1 range (noise
    stays -1), so ``labels.max() + 1`` equals the cluster count."""
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.unique(labels[labels >= 0])
    if valid.size == 0:
        return labels
    remap = np.full(int(valid.max()) + 1, -1, dtype=np.int64)
    remap[valid] = np.arange(valid.size, dtype=np.int64)
    return np.where(labels < 0, -1, remap[labels])


def fast_dbscan_available() -> bool:
    """True when the exact parallel DBSCAN package (wangyiqiu/dbscan-python) is
    importable. SMLM clustering requires it (there is no local-maxima fallback);
    the GUI checks this up front and tells the user to install the bundled wheel
    if it is missing."""
    try:
        from dbscan import DBSCAN  # noqa: F401
        return True
    except Exception:
        return False


# Message lives in core.dependencies so the GUI raises the identical text.
from .dependencies import DBSCAN_MISSING_MSG  # noqa: F401




def _dbscan_run(X: np.ndarray, eps: float, min_samples: int,
                cancel_check=None) -> np.ndarray:
    """Standalone pipeline: run DBSCAN in-process (no Qt, no child process)."""
    from dbscan import DBSCAN
    labels, _ = DBSCAN(np.ascontiguousarray(X, dtype=np.float64),
                       eps=float(eps), min_samples=int(min_samples))
    return np.asarray(labels)


def _dbscan_labels(X: np.ndarray, radius: float, min_locs: int,
                   cancel_check=None) -> np.ndarray:
    """Cluster labels from the exact parallel DBSCAN (wangyiqiu/dbscan-python).

    ``eps = radius`` and ``min_samples = min_locs``. Labels use -1 for noise;
    clusters below ``min_locs`` members are dropped and the survivors remapped
    to a contiguous 0..K-1 range. Raises ``RuntimeError`` with an install hint
    if the package is unavailable — there is no local-maxima fallback. The DBSCAN
    call itself runs in an isolated child process (see ``_dbscan_run``) so its
    OpenMP pool cannot spin the CPU at idle afterward."""
    import importlib.util
    if importlib.util.find_spec("dbscan") is None:
        raise RuntimeError(DBSCAN_MISSING_MSG)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError('DBSCAN requires finite coordinates with shape (N, 2)')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('DBSCAN radius must be finite and positive')
    if (isinstance(min_locs, (bool, np.bool_)) or not np.isfinite(min_locs)
            or int(min_locs) != min_locs or min_locs < 1):
        raise ValueError('DBSCAN minimum size must be a positive integer')
    if len(X) > 100_000_000:
        raise ValueError('DBSCAN input exceeds the native index limit')
    if len(X) == 0:
        return np.empty(0, dtype=np.int64)
    if np.max(np.ptp(X, axis=0)) / float(radius) > 2**30:
        raise ValueError('DBSCAN field span / radius exceeds the native grid limit')
    if cancel_check is not None and cancel_check():
        raise InterruptedError('Clustering cancelled')
    raw_labels = _dbscan_run(X, radius, min_locs, cancel_check=cancel_check)
    if (raw_labels.shape != (len(X),) or raw_labels.dtype.kind not in 'iu'
            or np.any(raw_labels < -1) or np.any(raw_labels >= len(X))):
        raise RuntimeError('DBSCAN returned invalid cluster labels')
    labels = np.asarray(raw_labels, dtype=np.int64)
    # Enforce a minimum cluster size of min_locs (drop smaller clusters).
    if (labels >= 0).any():
        uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
        small = uniq[counts < int(min_locs)]
        if small.size:
            labels[np.isin(labels, small)] = -1
    return _remap_contiguous(labels)


def smlm_cluster(
    loc_df: pd.DataFrame,
    radius: float = 0.5,
    min_locs: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
    cancel_check=None,
) -> tuple:
    """
    Step 2: Cluster localizations with exact parallel DBSCAN.

    Uses the wangyiqiu/dbscan-python package (SIGMOD'20 grid DBSCAN):
    ``eps = radius``, ``min_samples = min_locs``. Noise points get cluster -1,
    clusters below ``min_locs`` members are dropped, and the survivors are
    remapped to a contiguous 0, 1, 2 … sequence. The package is REQUIRED — if it
    is not installed a ``RuntimeError`` is raised with an install hint (there is
    no local-maxima fallback). Install it from the bundled wheel in ./wheels.

    Parameters
    ----------
    loc_df : pd.DataFrame
        Localizations with columns 'x', 'y' (camera pixels).
    radius : float
        DBSCAN eps in camera pixels. Typical starting point: 0.5 px.
    min_locs : int
        DBSCAN min_samples / minimum localizations for a valid cluster.
    progress_callback : callable, optional
        Called with integer percentages 0-100.

    Returns
    -------
    spots : list[Spot]
        Centroid Spot objects for each cluster.
    labelled_df : pd.DataFrame
        Copy of loc_df with a 'cluster' column (-1 = noise / unclustered).
    """
    if loc_df is None or loc_df.empty:
        if progress_callback is not None:
            progress_callback(100)
        return [], loc_df if loc_df is not None else pd.DataFrame()

    # Clean 0..N-1 index so the numpy label array aligns exactly.
    labelled_df = loc_df.reset_index(drop=True).copy()
    X = labelled_df[["x", "y"]].to_numpy()  # camera-pixel units

    labelled_df["cluster"] = _dbscan_labels(
        X, radius, min_locs, cancel_check=cancel_check)

    if progress_callback is not None:
        progress_callback(100)

    if cancel_check is not None and cancel_check():
        raise InterruptedError('Clustering cancelled')
    return spots_from_clustered_df(labelled_df), labelled_df


def spots_from_clustered_df(df, label_col: str = "cluster") -> list:
    """Build one centroid Spot per cluster from a labelled localization df.

    Single source of the cluster-centroid Spot definition, shared by
    smlm_cluster (Step 6) and the GUI's clustered-CSV load path. ``df`` must
    have 'x'/'y' columns and a ``label_col`` cluster column (-1 = noise,
    skipped). Optional 'photons' -> intensity (mean) and 'frame' -> r_squared
    (frame span). Iterates the non-negative cluster ids actually present, so it
    is robust to both contiguous (smlm_cluster) and arbitrary (loaded CSV)
    label sets.
    """
    if df is None or label_col not in df.columns:
        return []
    labels = df[label_col].to_numpy()
    mask = labels >= 0
    if not mask.any():
        return []

    # Single grouped pass over ascending cluster ids keeps the cost linear in
    # the number of localizations rather than O(n_clusters x n_locs), which
    # matters once cluster counts reach the tens of thousands.
    has_ph = "photons" in df.columns
    has_fr = "frame" in df.columns
    cols = ["x", "y"] + (["photons"] if has_ph else []) + (["frame"] if has_fr else [])
    sub = df.loc[mask, cols]
    sub = sub.assign(_cid=labels[mask])
    grp = sub.groupby("_cid", sort=True)

    xm = grp["x"].mean().to_numpy()
    ym = grp["y"].mean().to_numpy()
    inten = grp["photons"].mean().to_numpy() if has_ph else None
    if has_fr:
        fagg = grp["frame"].agg(["min", "max"])
        span = (fagg["max"] - fagg["min"] + 1).to_numpy()

    spots = []
    for k in range(len(xm)):
        spots.append(Spot(
            x=float(xm[k]),
            y=float(ym[k]),
            intensity=float(inten[k]) if has_ph else 1.0,
            sigma=0.0,
            r_squared=float(span[k]) if has_fr else 1.0,
        ))
    return spots


def cluster_labels_in_spot_order(df, label_col: str = "cluster") -> list:
    """Return the DBSCAN cluster labels in the SAME order ``spots_from_clustered_df``
    emits its centroid Spots (one label per spot).

    Both this and ``spots_from_clustered_df`` iterate the non-negative cluster
    ids in ascending order (``groupby(..., sort=True)`` / ``np.unique``), so the
    k-th returned label is the cluster id of the k-th Spot. Used to key each
    extracted molecule by its real cluster label instead of a rounded ``(x, y)``
    string — the latter collides for dense fields (dropping/merging molecules at
    scale) and across movies.
    """
    if df is None or label_col not in df.columns:
        return []
    labels = df[label_col].to_numpy()
    pos = labels[labels >= 0]
    if pos.size == 0:
        return []
    return np.unique(pos).tolist()


def spots_from_raw_locs(df) -> list:
    """Build one Spot per localization (no clustering).

    Used when a reference/anchor movie is localized but not clustered — each
    localization is treated as a distinct molecule position (e.g. a single-frame
    reference image). ``df`` must have 'x'/'y' columns; optional 'photons' ->
    intensity and 'sx'/'sy' -> sigma. Returns [] when x/y are absent.
    """
    if df is None or "x" not in df.columns or "y" not in df.columns:
        return []
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return []
    ph = df["photons"].to_numpy(dtype=float) if "photons" in df.columns else np.ones(n)
    if "sx" in df.columns and "sy" in df.columns:
        sig = 0.5 * (df["sx"].to_numpy(dtype=float) + df["sy"].to_numpy(dtype=float))
    else:
        sig = np.zeros(n)
    return [Spot(x=float(x[i]), y=float(y[i]), intensity=float(ph[i]),
                 sigma=float(sig[i]), r_squared=1.0) for i in range(n)]


# ── Temporal linking (Picasso-style) ─────────────────────────────────────────
# Links localizations of the SAME molecule across consecutive (or near-
# consecutive) frames into one track, the temporal analogue of the spatial
# DBSCAN clustering above. Ported from Picasso's postprocess.link /
# _get_link_groups / _get_next_loc_index_in_link_group
# (github.com/jungmannlab/picasso). The output is written into the same
# ``cluster`` column DBSCAN uses, so tracks are interchangeable with clusters
# for every downstream step (trace extraction, HMM, kinetics, AIM, export).

@numba.njit(cache=True, nogil=True)
def _build_link_grid(x, y, x_min, y_min, inv_cell, nx, ny):
    """Bin localizations into a uniform spatial grid (CSR layout).

    Returns ``(cell_start, cell_locs)`` where ``cell_locs[cell_start[c]:
    cell_start[c+1]]`` are the indices of the localizations in cell ``c``. The
    indices within each cell stay ascending (the global arrays are frame-sorted,
    so ascending index == ascending frame), which lets the query binary-search a
    cell by frame.
    """
    N = len(x)
    ncells = nx * ny
    cell_of = np.empty(N, dtype=np.int64)
    cell_start = np.zeros(ncells + 1, dtype=np.int64)
    for i in range(N):
        cx = int((x[i] - x_min) * inv_cell)
        if cx < 0:
            cx = 0
        elif cx >= nx:
            cx = nx - 1
        cy = int((y[i] - y_min) * inv_cell)
        if cy < 0:
            cy = 0
        elif cy >= ny:
            cy = ny - 1
        c = cy * nx + cx
        cell_of[i] = c
        cell_start[c + 1] += 1
    for c in range(ncells):
        cell_start[c + 1] += cell_start[c]
    cell_locs = np.empty(N, dtype=np.int64)
    wptr = cell_start[:ncells].copy()
    for i in range(N):
        c = cell_of[i]
        cell_locs[wptr[c]] = i
        wptr[c] += 1
    return cell_start, cell_locs


@numba.njit(cache=True, nogil=True)
def _link_next_index_grid(ci, link_group, frame, x, y, d_max_2, max_gap, group,
                          x_min, y_min, inv_cell, nx, ny, cell_start, cell_locs):
    """Index of the next localization that continues the current track, or -1.

    Same selection rule as Picasso's ``_get_next_loc_index_in_link_group`` (the
    earliest-index unlinked, same-group localization within ``sqrt(d_max_2)`` of
    the current position, in frames ``[cf+1 .. cf+max_gap+1]``), but found via the
    spatial grid: only the 3x3 neighbouring cells are examined (the cell size is
    >= the link distance, so no in-range candidate lies outside them), and each
    cell is binary-searched for the frame window. Tracking the minimum qualifying
    index makes the result IDENTICAL to the linear scan while costing O(1) per
    step instead of O(localizations-per-frame). Assumes ``frame`` ascending.
    """
    cf = frame[ci]
    px = x[ci]
    py = y[ci]
    cg = group[ci]
    lo_frame = cf + 1
    hi_frame = cf + max_gap + 1

    cx = int((px - x_min) * inv_cell)
    cy = int((py - y_min) * inv_cell)
    best = -1
    for ddy in range(-1, 2):
        gy = cy + ddy
        if gy < 0 or gy >= ny:
            continue
        for ddx in range(-1, 2):
            gx = cx + ddx
            if gx < 0 or gx >= nx:
                continue
            c = gy * nx + gx
            s = cell_start[c]
            e = cell_start[c + 1]
            if s >= e:
                continue
            # First index in the cell whose frame is within the window.
            lo = s
            hi = e
            while lo < hi:
                mid = (lo + hi) >> 1
                if frame[cell_locs[mid]] < lo_frame:
                    lo = mid + 1
                else:
                    hi = mid
            k = lo
            while k < e:
                j = cell_locs[k]
                if frame[j] > hi_frame:
                    break
                if group[j] == cg and link_group[j] == -1:
                    dx = px - x[j]
                    dy = py - y[j]
                    if dx * dx + dy * dy <= d_max_2:
                        if best == -1 or j < best:
                            best = j
                        break  # first qualifying in this cell is its minimum index
                k += 1
    return best


@numba.njit(cache=True, nogil=True)
def _link_next_index_linear(ci, link_group, N, frame, x, y, d_max_2, max_gap, group):
    """Earliest in-range successor via Picasso's plain linear scan over the frame
    window. Fastest when there are few localizations per frame (the window is
    tiny), so the adaptive selector uses it for sparse data."""
    cf = frame[ci]
    px = x[ci]
    py = y[ci]
    cg = group[ci]
    min_frame = cf + 1
    min_index = N
    for mi in range(ci + 1, N):
        if frame[mi] >= min_frame:
            min_index = mi
            break
    max_frame = cf + max_gap + 1
    max_index = N
    for mx in range(min_index, N):
        if frame[mx] > max_frame:
            max_index = mx
            break
    for j in range(min_index, max_index):
        if group[j] == cg and link_group[j] == -1:
            dx = px - x[j]
            if dx * dx <= d_max_2:
                dy = py - y[j]
                if dy * dy <= d_max_2 and dx * dx + dy * dy <= d_max_2:
                    return j
    return -1


@numba.njit(cache=True, nogil=True)
def _link_groups_linear(frame, x, y, d_max, max_gap, group):
    """Chain frame-sorted localizations into tracks via the linear-scan window
    (no grid). Same result as the grid kernel; faster for sparse movies."""
    N = len(x)
    link_group = -np.ones(N, dtype=np.int32)
    d_max_2 = d_max * d_max
    current = -1
    for i in range(N):
        if link_group[i] == -1:
            current += 1
            link_group[i] = current
            idx = i
            nxt = _link_next_index_linear(idx, link_group, N, frame, x, y, d_max_2,
                                          max_gap, group)
            while nxt != -1:
                link_group[nxt] = current
                idx = nxt
                nxt = _link_next_index_linear(idx, link_group, N, frame, x, y,
                                              d_max_2, max_gap, group)
    return link_group


@numba.njit(cache=True, nogil=True)
def _link_groups_kernel(frame, x, y, d_max, max_gap, group, x_min, y_min,
                        inv_cell, nx, ny, cell_start, cell_locs):
    """Chain frame-sorted localizations into tracks using the spatial grid."""
    N = len(x)
    link_group = -np.ones(N, dtype=np.int32)
    d_max_2 = d_max * d_max
    current = -1
    for i in range(N):
        if link_group[i] == -1:
            current += 1
            link_group[i] = current
            idx = i
            nxt = _link_next_index_grid(idx, link_group, frame, x, y, d_max_2,
                                        max_gap, group, x_min, y_min, inv_cell,
                                        nx, ny, cell_start, cell_locs)
            while nxt != -1:
                link_group[nxt] = current
                idx = nxt
                nxt = _link_next_index_grid(idx, link_group, frame, x, y, d_max_2,
                                            max_gap, group, x_min, y_min, inv_cell,
                                            nx, ny, cell_start, cell_locs)
    return link_group


# Cap the grid resolution so a tiny link distance over a large field of view does
# not allocate a huge cell table; coarser cells just hold a few more candidates.
_LINK_GRID_MAX_AXIS = 2048
# Above ~this many localizations in the frame window (density x (max_gap+1)), the
# spatial grid beats the linear scan; below it the linear scan's tiny constant
# wins (Picasso uses linear for the typically-sparse PAINT data). Tuned to the
# measured crossover.
_LINK_GRID_WINDOW_THRESHOLD = 48


def _link_groups(frame, x, y, d_max, max_gap, group):
    """Assign every localization a track id by chaining across frames.

    Same chaining as Picasso's ``_get_link_groups`` (each unassigned loc seeds a
    track extended forward to its earliest in-range successor) — and the SAME
    result either way — but the search is chosen adaptively: a plain linear scan
    of the frame window for sparse movies (fewest constant factors, like Picasso),
    or a uniform spatial grid for dense movies so the cost stays O(N) instead of
    O(N x localizations-per-frame). Every localization ends with a non-negative
    track id (singletons get their own). Returns the per-row track ids.
    """
    N = len(x)
    if N == 0:
        return np.empty(0, dtype=np.int32)

    # Estimate localizations per frame window. ``frame`` is sorted ascending, so
    # the number of distinct frames is 1 + (number of value changes).
    n_frames = 1 + int(np.count_nonzero(np.diff(frame))) if N > 1 else 1
    window = (N / n_frames) * (int(max_gap) + 1)
    if window <= _LINK_GRID_WINDOW_THRESHOLD:
        return _link_groups_linear(frame, x, y, float(d_max), int(max_gap), group)

    x_min = float(x.min())
    y_min = float(y.min())
    span_x = float(x.max()) - x_min
    span_y = float(y.max()) - y_min
    cell = float(d_max)
    if span_x > 0:
        cell = max(cell, span_x / _LINK_GRID_MAX_AXIS)
    if span_y > 0:
        cell = max(cell, span_y / _LINK_GRID_MAX_AXIS)
    if cell <= 0:
        cell = 1.0
    inv_cell = 1.0 / cell
    nx = int(span_x * inv_cell) + 1
    ny = int(span_y * inv_cell) + 1
    cell_start, cell_locs = _build_link_grid(x, y, x_min, y_min, inv_cell, nx, ny)
    return _link_groups_kernel(frame, x, y, float(d_max), int(max_gap), group,
                               x_min, y_min, inv_cell, nx, ny, cell_start, cell_locs)


def smlm_link(
    loc_df: pd.DataFrame,
    max_distance: float = 1.0,
    max_gap: int = 1,
    min_length: int = 1,
    progress_callback: Optional[Callable[[int], None]] = None,
    build_spots: bool = False,
    cancel_check=None,
) -> tuple:
    """Link localizations across frames and COMBINE each track into one row.

    Picasso-style temporal linking: a molecule's localizations are chained across
    consecutive frames (tolerating ``max_gap`` blank frames) when each step stays
    within ``max_distance`` of the previous localization, then every track is
    collapsed into a single combined localization. Unlike DBSCAN clustering, this
    *replaces* the per-frame localizations with the combined ones and adds two
    Picasso columns:

      * ``n``   — number of localizations linked into the track.
      * ``len`` — track duration (frame span = last_frame - first_frame + 1).

    Parameters
    ----------
    loc_df : pd.DataFrame
        Localizations with at least 'frame', 'x', 'y' columns (camera pixels).
        Optional 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'net_gradient',
        'group'. A 'group' column partitions linking (links never cross groups).
    max_distance : float
        Maximum distance (camera pixels) between a localization and its successor
        in a later frame for them to belong to the same track.
    max_gap : int
        Maximum number of fully-blank (dark) frames bridged within a track.
        0 = strictly consecutive frames; 1 tolerates a single missed frame.
    min_length : int
        Minimum number of localizations (``n``) a track must have to be kept.
        The default of 1 keeps every track, including unlinked single
        localizations; higher values drop short tracks (e.g. 2 keeps only tracks
        that actually linked ≥ 2 localizations).
    progress_callback : callable, optional
        Called with integer percentages 0-100.

    Returns
    -------
    spots : list[Spot]
        One Spot per combined localization (intensity = summed photons,
        r_squared = track duration ``len``).
    linked_df : pd.DataFrame
        Combined localizations — one row per track — with the original columns
        aggregated plus 'n' and 'len'.
    """
    if loc_df is None or loc_df.empty:
        if progress_callback is not None:
            progress_callback(100)
        return [], loc_df if loc_df is not None else pd.DataFrame()

    if (not np.isfinite(max_distance) or max_distance < 0
            or int(max_gap) != max_gap or max_gap < 0
            or int(min_length) != min_length or min_length < 1):
        raise ValueError('Link distance, gap, and minimum length are invalid')
    if not np.isfinite(loc_df[['frame', 'x', 'y']].to_numpy()).all():
        raise ValueError('Linking requires finite frames and coordinates')
    if cancel_check is not None and cancel_check():
        raise InterruptedError('Linking cancelled')
    # Clean 0..N-1 index so numpy label arrays align with the rows exactly.
    src = loc_df.reset_index(drop=True)
    N = len(src)

    if progress_callback is not None:
        progress_callback(10)

    # The link kernel assumes frame-ascending input. Sort positions (stable, so
    # ties keep their row order as Picasso's quicksort-after-sort does), run the
    # chaining in sorted order, then scatter the track ids back to row order.
    frame_all = src["frame"].to_numpy()
    order = np.argsort(frame_all, kind="stable")
    frame_s = np.ascontiguousarray(frame_all[order]).astype(np.int64)
    x_s = np.ascontiguousarray(src["x"].to_numpy()[order]).astype(np.float64)
    y_s = np.ascontiguousarray(src["y"].to_numpy()[order]).astype(np.float64)
    if "group" in src.columns:
        group_s = np.ascontiguousarray(
            src["group"].to_numpy()[order]).astype(np.int64)
    else:
        group_s = np.zeros(N, dtype=np.int64)

    if progress_callback is not None:
        progress_callback(40)

    if cancel_check is None:
        link_group_s = _link_groups(
            frame_s, x_s, y_s, float(max_distance), int(max_gap), group_s)
    else:
        from .native_jobs import run_array_job
        link_group_s = run_array_job(
            'link', {'frame': frame_s, 'x': x_s, 'y': y_s, 'group': group_s},
            {'max_distance': float(max_distance), 'max_gap': int(max_gap)},
            cancel_check=cancel_check)
    if (link_group_s.shape != (N,) or link_group_s.dtype.kind not in 'iu'
            or np.any(link_group_s < 0) or np.any(link_group_s >= N)):
        raise RuntimeError('Linking returned invalid track labels')

    # Scatter track ids back to the original row order. Every localization gets a
    # non-negative track id (singletons get their own), so nothing is discarded.
    track = np.empty(N, dtype=np.int64)
    track[order] = link_group_s

    if progress_callback is not None:
        progress_callback(70)

    linked_df = _combine_linked_locs(src, track)
    if cancel_check is not None and cancel_check():
        raise InterruptedError('Linking cancelled')

    # Drop tracks shorter than min_length. 0 or 1 keeps everything (every track
    # has n >= 1), so unlinked single localizations are preserved by default;
    # higher values keep only tracks that linked enough localizations.
    if min_length > 1:
        linked_df = linked_df[linked_df["n"] >= int(min_length)].reset_index(drop=True)

    if progress_callback is not None:
        progress_callback(90)

    # One Spot per combined localization (intensity = summed photons, r_squared =
    # track duration ``len``). Building a Spot per row is a pure-Python loop and
    # the GUI shows linked localizations through the fast per-frame overlay, not
    # a Spot cloud — so this is off by default and opted into explicitly.
    spots = []
    if build_spots:
        has_ph = "photons" in linked_df.columns
        has_s = "sx" in linked_df.columns and "sy" in linked_df.columns
        xs = linked_df["x"].to_numpy()
        ys = linked_df["y"].to_numpy()
        lens = linked_df["len"].to_numpy()
        phot = linked_df["photons"].to_numpy() if has_ph else None
        sig = (((linked_df["sx"].to_numpy() + linked_df["sy"].to_numpy()) / 2.0)
               if has_s else None)
        for k in range(len(linked_df)):
            spots.append(Spot(
                x=float(xs[k]),
                y=float(ys[k]),
                intensity=float(phot[k]) if has_ph else 1.0,
                sigma=float(sig[k]) if has_s else 0.0,
                r_squared=float(lens[k]),
            ))

    if progress_callback is not None:
        progress_callback(100)

    return spots, linked_df


@numba.njit(cache=True, nogil=True)
def _combine_kernel(track, K, frame, x, y, photons, sx, sy, bg, lpx, lpy, ng,
                    group, has_lp):
    """Single-pass per-track aggregation (counts, frame min/max, sums, inverse-
    variance weighted x/y sums, last group). One sweep over the localizations,
    float32 accumulators (like Picasso) to halve memory traffic / cache pressure
    — far faster than a stack of ``np.bincount`` + ``np.minimum.at`` calls."""
    n = np.zeros(K, np.int64)
    BIG = np.int64(1) << 62
    fmin = np.full(K, BIG, np.int64)
    fmax = np.full(K, -BIG, np.int64)
    s_ph = np.zeros(K, np.float32); s_bg = np.zeros(K, np.float32)
    s_sx = np.zeros(K, np.float32); s_sy = np.zeros(K, np.float32)
    s_ng = np.zeros(K, np.float32)
    s_x = np.zeros(K, np.float32); s_y = np.zeros(K, np.float32)
    s_wx = np.zeros(K, np.float32); s_xwx = np.zeros(K, np.float32)
    s_wy = np.zeros(K, np.float32); s_ywy = np.zeros(K, np.float32)
    s_lpx = np.zeros(K, np.float32); s_lpy = np.zeros(K, np.float32)
    glast = np.zeros(K, np.int64)
    for i in range(len(track)):
        g = track[i]
        n[g] += 1
        f = frame[i]
        if f < fmin[g]:
            fmin[g] = f
        if f > fmax[g]:
            fmax[g] = f
        s_ph[g] += photons[i]
        s_bg[g] += bg[i]
        s_sx[g] += sx[i]
        s_sy[g] += sy[i]
        s_ng[g] += ng[i]
        s_x[g] += x[i]
        s_y[g] += y[i]
        glast[g] = group[i]
        if has_lp:
            lx = lpx[i]
            if lx > 0.0:
                w = 1.0 / (lx * lx)
                s_wx[g] += w
                s_xwx[g] += x[i] * w
            s_lpx[g] += lx
            ly = lpy[i]
            if ly > 0.0:
                w = 1.0 / (ly * ly)
                s_wy[g] += w
                s_ywy[g] += y[i] * w
            s_lpy[g] += ly
    return (n, fmin, fmax, s_ph, s_bg, s_sx, s_sy, s_ng, s_x, s_y,
            s_wx, s_xwx, s_wy, s_ywy, s_lpx, s_lpy, glast)


def _combine_linked_locs(src: pd.DataFrame, track: np.ndarray) -> pd.DataFrame:
    """Collapse each link group into one combined localization (Picasso-style).

    ``track`` is the per-row track id, contiguous ``0..K-1`` as produced by
    :func:`_link_groups`. All reductions are done in ONE numba pass
    (:func:`_combine_kernel`): positions use an inverse-variance (1/lp**2)
    weighted mean when precision columns are present, else a plain mean; photons
    and background are summed; PSF widths and quality metrics are averaged. Adds
    'n' (localizations per track) and 'len' (frame span). Output rows are ordered
    by ascending track id, matching the per-track Spot order.
    """
    cols = set(src.columns)
    N = len(track)
    K = int(track.max()) + 1 if N else 0
    if K == 0:
        return pd.DataFrame({"frame": [], "x": [], "y": [], "n": [], "len": []})
    inv = track.astype(np.int64, copy=False)  # already 0..K-1 contiguous

    def _col(name):
        return (src[name].to_numpy(np.float32) if name in cols
                else np.zeros(N, np.float32))

    frame = src["frame"].to_numpy().astype(np.int64)
    x = src["x"].to_numpy(np.float32)
    y = src["y"].to_numpy(np.float32)
    has_lp = ("lpx" in cols) and ("lpy" in cols)
    grp = (src["group"].to_numpy().astype(np.int64) if "group" in cols
           else np.zeros(N, np.int64))

    (n_, fmin, fmax, s_ph, s_bg, s_sx, s_sy, s_ng, s_x, s_y,
     s_wx, s_xwx, s_wy, s_ywy, s_lpx, s_lpy, glast) = _combine_kernel(
        inv, K, frame, x, y, _col("photons"), _col("sx"), _col("sy"),
        _col("bg"), _col("lpx"), _col("lpy"), _col("net_gradient"), grp, has_lp)

    nf = n_.astype(np.float64)
    out = {"frame": fmin}

    if has_lp:
        gx = s_wx > 0
        out["x"] = np.where(gx, np.divide(s_xwx, s_wx, out=np.zeros(K), where=gx),
                            s_x / nf)
        gy = s_wy > 0
        out["y"] = np.where(gy, np.divide(s_ywy, s_wy, out=np.zeros(K), where=gy),
                            s_y / nf)
    else:
        out["x"] = s_x / nf
        out["y"] = s_y / nf

    if "photons" in cols:
        out["photons"] = s_ph
    if "sx" in cols:
        out["sx"] = s_sx / nf
    if "sy" in cols:
        out["sy"] = s_sy / nf
    if "bg" in cols:
        out["bg"] = s_bg
    if has_lp:
        out["lpx"] = np.where(gx, np.sqrt(np.divide(1.0, s_wx, out=np.ones(K), where=gx)),
                              s_lpx / nf)
        out["lpy"] = np.where(gy, np.sqrt(np.divide(1.0, s_wy, out=np.ones(K), where=gy)),
                              s_lpy / nf)
    if "net_gradient" in cols:
        out["net_gradient"] = s_ng / nf
    if "group" in cols:
        out["group"] = glast

    out["n"] = n_.astype(np.int32)
    out["len"] = (fmax - fmin + 1).astype(np.int32)

    return pd.DataFrame(out)


def filter_locs_by_ranges(df, col_ranges):
    """Filter a localizations DataFrame to rows within each column's [lo, hi].

    ``col_ranges`` maps column name -> (lo, hi); columns absent from df are
    skipped. Returns the filtered DataFrame (a new object via boolean indexing;
    does not mutate df in place). The caller decides whether to pass a copy.
    Single source of the per-column range-filter loop used by both the
    interactive localization filter and the pipeline filter path.
    """
    for col, (lo, hi) in col_ranges.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
    return df
