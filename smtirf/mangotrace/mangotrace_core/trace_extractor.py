"""
Trace extraction with background subtraction.
Based on the MARS MarsImageUtils.integratePeaks algorithm.
"""

import numpy as np
import numba
from typing import List
from .data_types import Spot


def get_circular_mask(radius: int) -> np.ndarray:
    """
    Create a circular mask of given radius.
    
    Args:
        radius: Circle radius in pixels
        
    Returns:
        Boolean mask array of shape (2*radius+1, 2*radius+1)
    """
    if int(radius) != radius or radius < 1:
        raise ValueError("radius must be an integer >= 1")
    radius = int(radius)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x + y*y <= radius*radius


def get_ring_mask(inner_radius: int, outer_radius: int,
                  buffer: int = 0) -> np.ndarray:
    """
    Create a ring/annulus mask.

    Args:
        inner_radius: Radius of the signal disk this ring surrounds
        outer_radius: Outer radius of ring
        buffer: Gap in pixels between the signal disk and the ring. The PSF
            does not stop at ``inner_radius``, so without a gap the innermost
            background pixels still carry signal and the background estimate is
            biased high. The ring therefore starts at ``inner_radius + buffer``.

    Returns:
        Boolean mask array
    """
    if (int(inner_radius) != inner_radius or int(outer_radius) != outer_radius
            or int(buffer) != buffer):
        raise ValueError("trace radii must be integers")
    if buffer < 0:
        raise ValueError("buffer must not be negative")
    if inner_radius < 1 or outer_radius <= inner_radius:
        raise ValueError("outer_radius must be greater than inner_radius >= 1")
    hole_radius = inner_radius + buffer
    if outer_radius <= hole_radius:
        raise ValueError(
            "outer_radius must exceed inner_radius + buffer, otherwise the "
            "background annulus is empty")
    outer_mask = get_circular_mask(outer_radius)
    hole_mask = get_circular_mask(hole_radius)

    # Pad the excluded core to match the outer size
    pad = outer_radius - hole_radius
    hole_padded = np.pad(hole_mask, pad, mode='constant', constant_values=False)

    return outer_mask & ~hole_padded




@numba.njit(parallel=True, nogil=True, cache=False)
def _extract_traces_chunk(chunk, sy, sx, idy, idx, ody, odx,
                     sig_med, bg_mode, out):
    """Annular spot trace extraction for one frame-chunk. ``prange`` over frames;
    each frame reuses two small scratch buffers across all spots (so the only
    allocations are per-frame, not per-(frame, spot)). Out-of-bounds samples
    are skipped, so border pixels are never duplicated by clamping.

    ``bg_mode`` selects how the local background under a spot is estimated from
    the annulus pixels: 0 = mean, 1 = median."""
    nf = chunk.shape[0]
    H = chunk.shape[1]
    W = chunk.shape[2]
    N = sy.shape[0]
    ni = idy.shape[0]
    no = ody.shape[0]
    for t in numba.prange(nf):
        inbuf = np.empty(ni, dtype=np.float64)
        obuf = np.empty(no, dtype=np.float64)
        for s in range(N):
            ys = sy[s]
            xs = sx[s]
            n_in = 0
            for k in range(ni):
                yy = ys + idy[k]
                xx = xs + idx[k]
                if 0 <= yy < H and 0 <= xx < W:
                    inbuf[n_in] = chunk[t, yy, xx]
                    n_in += 1
            if sig_med:
                sig = np.median(inbuf[:n_in])
            else:
                acc = 0.0
                for k in range(n_in):
                    acc += inbuf[k]
                sig = acc / n_in
            n_out = 0
            for k in range(no):
                yy = ys + ody[k]
                xx = xs + odx[k]
                if 0 <= yy < H and 0 <= xx < W:
                    obuf[n_out] = chunk[t, yy, xx]
                    n_out += 1
            if n_out == 0:
                out[t, s] = np.nan
                continue
            if bg_mode == 1:
                bg = np.median(obuf[:n_out])
            else:
                acc = 0.0
                for k in range(n_out):
                    acc += obuf[k]
                bg = acc / n_out
            out[t, s] = (sig - bg) * n_in


def extract_traces_all_frames(stack: np.ndarray,
                         spots: List[Spot],
                         inner_radius: int = 2,
                         outer_radius: int = 4,
                         signal_method: str = 'mean',
                         bg_method: str = 'median',
                         buffer: int = 1) -> np.ndarray:
    """
    Extract traces for spots across all frames of a time series.

    Parallelized with a numba ``prange`` kernel (``_extract_traces_chunk``): the per-
    frame work runs on all cores with the GIL released. Frames are processed in
    RAM-bounded chunks
    converted to native byte order (numba rejects big-endian dtypes), so a
    multi-GB / big-endian movie is never materialized whole.

    Args:
        stack: 3D image stack (T, Y, X)
        spots: List of Spot objects (detected on max projection)
        inner_radius: Extraction region radius
        outer_radius: Background region outer radius
        signal_method: 'mean' or 'median' — how inner pixels are combined
        bg_method: 'mean' or 'median' — how the local background is estimated
            from the annulus.
        buffer: Gap in pixels between the signal disk and the background
            annulus, so PSF tails do not contaminate the background.

    Returns:
        2D array of shape (T, N) where T is frames and N is number of spots
    """
    if signal_method not in {'mean', 'median'}:
        raise ValueError("signal_method must be 'mean' or 'median'")
    if str(bg_method).lower() not in {'mean', 'median'}:
        raise ValueError("bg_method must be 'mean' or 'median'")
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"stack must have TYX shape; got {stack.shape!r}")

    n_frames, height, width = stack.shape
    n_spots = len(spots)

    if n_spots == 0:
        return np.zeros((n_frames, 0), dtype=np.float64)

    inner_mask = get_circular_mask(inner_radius)
    ring_mask  = get_ring_mask(inner_radius, outer_radius, buffer)

    inner_dy, inner_dx = np.where(inner_mask)
    inner_dy = (inner_dy - inner_radius).astype(np.int64)
    inner_dx = (inner_dx - inner_radius).astype(np.int64)

    outer_dy, outer_dx = np.where(ring_mask)
    outer_dy = (outer_dy - outer_radius).astype(np.int64)
    outer_dx = (outer_dx - outer_radius).astype(np.int64)
    if outer_dy.size == 0:
        raise ValueError("the background annulus contains no pixels")

    spot_y = np.array([int(s.y + 0.5) for s in spots], dtype=np.int64)
    spot_x = np.array([int(s.x + 0.5) for s in spots], dtype=np.int64)
    if (np.any(~np.isfinite([s.y for s in spots]))
            or np.any(~np.isfinite([s.x for s in spots]))):
        raise ValueError("spot coordinates must be finite")
    if (np.any(spot_y < 0) or np.any(spot_y >= height)
            or np.any(spot_x < 0) or np.any(spot_x >= width)):
        raise ValueError("spot centres must lie inside the movie")

    sig_med = signal_method == 'median'
    bm = str(bg_method).lower()
    bg_mode = 1 if 'median' in bm else 0

    intensities = np.zeros((n_frames, n_spots), dtype=np.float64)

    # Process in RAM-bounded frame-chunks; convert each to native, C-contiguous
    # layout (numba can't type big-endian / non-contiguous views) without ever
    # holding the whole movie in memory.
    frame_bytes = int(height) * int(width) * int(stack.dtype.itemsize)
    chunk_frames = max(1, min(n_frames,
                              (256 * 1024 * 1024) // max(frame_bytes, 1)))
    for c0 in range(0, n_frames, chunk_frames):
        c1 = min(c0 + chunk_frames, n_frames)
        chunk = stack[c0:c1]
        if not chunk.dtype.isnative:
            chunk = chunk.astype(chunk.dtype.newbyteorder('='))
        chunk = np.ascontiguousarray(chunk)
        _extract_traces_chunk(chunk, spot_y, spot_x, inner_dy, inner_dx,
                         outer_dy, outer_dx, sig_med, bg_mode,
                         intensities[c0:c1])

    return intensities



def warmup() -> None:
    """Force JIT compilation of the extraction kernel on a tiny array.

    The ``_extract_traces_chunk`` numba kernel is ``cache=False`` (consistent with
    the other heavy parallel kernels in this package), so it recompiles once per
    process — a ~2–3s cost that otherwise lands on the first Run Trace Extraction.
    Calling this from a background thread at startup hides that compile behind app
    init, so the first extraction is fast too. Pure NumPy/numba; safe off-thread.
    """
    try:
        dummy = np.zeros((2, 8, 8), dtype=np.float64)
        spots = [Spot(x=4.0, y=4.0, intensity=1.0, sigma=1.0, r_squared=1.0)]
        extract_traces_all_frames(dummy, spots, inner_radius=2, outer_radius=4)
    except Exception:
        pass


def extract_traces_with_metadata(stack: np.ndarray,
                            spots: List[Spot],
                            inner_radius: int = 2,
                            outer_radius: int = 4,
                            signal_method: str = 'mean',
                            bg_method: str = 'median',
                            buffer: int = 1) -> dict:
    """
    Extract traces for spots and return full metadata.

    Args:
        stack: 3D image stack (T, Y, X)
        spots: List of Spot objects
        inner_radius: Extraction region radius
        outer_radius: Background region outer radius
        signal_method: 'mean' or 'median' for inner-circle pixel aggregation
        bg_method: 'mean' or 'median' for the local background estimate
        buffer: Gap in pixels between the signal disk and the background ring

    Returns:
        Dict with 'intensities' and 'locations'.
    """
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]

    intensities = extract_traces_all_frames(
        stack, spots, inner_radius, outer_radius,
        signal_method=signal_method, bg_method=bg_method, buffer=buffer
    )

    return {
        'intensities': intensities,
        'locations':   [(spot.x, spot.y) for spot in spots],
    }
