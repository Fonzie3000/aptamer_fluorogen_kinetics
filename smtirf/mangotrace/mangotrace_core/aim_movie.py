"""Bounded NumPy movie translation with zero boundaries and exact integer data."""

from __future__ import annotations

import math
import os

import numpy as np


# Conservative live scratch allowance, shared with the streaming worker before
# it reads a source chunk. Ordinary tiles hold <=65,536 numeric pixels; exact
# 64-bit integer tiles hold <=2,048 Python integers, including subnormal-weight
# denominators up to 2**2148 and their wider numerator intermediates.
MOVIE_SCRATCH_BYTES = 64 * 1024**2


def movie_memory_budget():
    """Reserve headroom against both physical RAM and Windows commit limits."""
    import psutil
    available = int(psutil.virtual_memory().available)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_commit", "available_commit",
                    "total_virtual", "available_virtual", "available_extended_virtual")]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot establish movie correction memory budget")
        available = min(available, status.available_commit)
    return int(available * .8)


def _integer_bilinear(planes, weights, denominator, dtype):
    # Python integer arithmetic avoids losing a uint64/int64 baseline or
    # overflowing a signed difference. Tiles bound these object temporaries.
    numerator = np.zeros(planes[0].shape, dtype=object)
    for plane, weight in zip(planes, weights):
        if weight:
            numerator += plane.astype(object) * weight
    whole = numerator // denominator
    twice_remainder = (numerator - whole * denominator) * 2
    whole += ((twice_remainder > denominator)
              | ((twice_remainder == denominator) & (whole >= 0)))
    return np.asarray(whole, dtype=dtype)


def shift_pixels(movie, dx, dy, *, order, progress_callback, cancel_check):
    """Translate frames by negative drift with bounded numeric/object tiles."""
    dtype = movie.dtype.newbyteorder("=")
    requested = int(movie.size) * dtype.itemsize + MOVIE_SCRATCH_BYTES
    if requested > movie_memory_budget():
        raise MemoryError("Movie correction output exceeds 80% of available memory; use smaller chunks")
    cancel_check()
    result = np.zeros(movie.shape, dtype=dtype)
    source = movie[None] if movie.ndim == 2 else movie
    destination = result[None] if result.ndim == 2 else result
    frames, height, width = source.shape
    integer = dtype.kind in "iu"
    limits = np.iinfo(dtype) if integer else None
    if progress_callback is not None:
        progress_callback(0)
    for frame in range(frames):
        cancel_check()
        ix, iy = math.floor(float(dx[frame])), math.floor(float(dy[frame]))
        fx, fy = float(dx[frame]) - ix, float(dy[frame]) - iy
        # Bounds apply to the continuous sample, before nearest rounding. This
        # prevents a slightly out-of-image sample from acquiring an edge pixel.
        x_begin, x_end = max(0, -ix), min(width, width - ix - bool(fx))
        y_begin, y_end = max(0, -iy), min(height, height - iy - bool(fy))
        if x_begin < x_end and y_begin < y_end:
            exact_integer = order == 1 and integer and dtype.itemsize == 8 and (fx or fy)
            if exact_integer:
                nx, denominator_x = fx.as_integer_ratio()
                ny, denominator_y = fy.as_integer_ratio()
                denominator = denominator_x * denominator_y
                weights = ((denominator_x-nx)*(denominator_y-ny),
                           nx*(denominator_y-ny), (denominator_x-nx)*ny, nx*ny)
            tile_rows, tile_cols = (8, 256) if exact_integer else (64, 1024)
            for row in range(y_begin, y_end, tile_rows):
                rows = min(tile_rows, y_end-row)
                for col in range(x_begin, x_end, tile_cols):
                    cancel_check()
                    cols = min(tile_cols, x_end-col)
                    sy, sx = row+iy, col+ix
                    if order == 0 or (fx == 0 and fy == 0):
                        if order == 0:
                            sy += fy >= .5
                            sx += fx >= .5
                        values = source[frame, sy:sy+rows, sx:sx+cols]
                    else:
                        p00 = source[frame, sy:sy+rows, sx:sx+cols]
                        p10 = source[frame, sy:sy+rows, sx+1:sx+cols+1] if fx else p00
                        p01 = source[frame, sy+1:sy+rows+1, sx:sx+cols] if fy else p00
                        p11 = source[frame, sy+1:sy+rows+1, sx+1:sx+cols+1] if fx and fy else p10 if fx else p01
                        if integer and dtype.itemsize == 8:
                            values = _integer_bilinear((p00, p10, p01, p11), weights, denominator, dtype)
                        else:
                            # Omit zero-weight neighbors so NaNs do not corrupt
                            # an exact integer sample or an unshifted axis.
                            top = p00.astype(np.float64)
                            if fx:
                                top = (1-fx)*top+fx*p10.astype(np.float64)
                            if fy:
                                bottom = p01.astype(np.float64)
                                if fx:
                                    bottom = (1-fx)*bottom+fx*p11.astype(np.float64)
                                values = (1-fy)*top+fy*bottom
                            else:
                                values = top
                            if integer:
                                integral = np.trunc(values)
                                increment = np.abs(values-integral) >= .5
                                values = integral+np.copysign(increment, values)
                                values = np.clip(values, limits.min, limits.max)
                    destination[frame, row:row+rows, col:col+cols] = values
        if progress_callback is not None:
            progress_callback(int(100*(frame+1)/frames))
        cancel_check()
    return result
