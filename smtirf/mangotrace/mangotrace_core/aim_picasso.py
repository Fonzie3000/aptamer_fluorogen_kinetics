"""Guarded AIM using Picasso's compiled row-intersection method.

The compiled counting loop adapts Picasso's MIT-licensed row-search method
(commit 87c2a2a15d445b65cb32bea0604b34922a5952e6).
The two-pass scientific conventions follow MangoTrace's independent equation
oracle, rather than Picasso's different phase scaling and temporal origin.

MIT License
Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry (jungmannlab.org)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import operator
import os
from types import SimpleNamespace

import numba
import numpy as np
from scipy.interpolate import CubicSpline


ARRAY_CHUNK = 262_144
KERNEL_WORK_BUDGET = 262_144
EXACT_COORDINATE_LIMIT = 2**52
PICASSO_COMMIT = '87c2a2a15d445b65cb32bea0604b34922a5952e6'
ALGORITHM_VERSION = 'mangotrace-aim-2d-picasso-guarded-v1'


def backend_info():
    return {'backend': 'picasso-numba', 'algorithm': ALGORITHM_VERSION,
            'version': '1', 'implementation': 'guarded_picasso_row_intersections',
            'reference_repository': 'https://github.com/jungmannlab/picasso',
            'reference_commit': PICASSO_COMMIT, 'license': 'MIT',
            'scientific_contract': 'mangotrace-2d-frame-zero-cell-phase-v2',
            'numba_version': numba.__version__, 'numpy_version': np.__version__}


@numba.njit(cache=True, nogil=True)
def _count_compact(reference, ref_counts, target, target_counts, stride,
                   offset_x, offset_y, radius, begin, end, scores):
    """Adapted Picasso row lower-bound plus contiguous short scan.

    Keys use checked actual-coordinate bounds and enough halo for every
    possible adaptive shift. All arithmetic is int64 even for int32 storage.
    """
    side = 2 * radius + 1
    for index in range(begin, end):
        base = np.int64(target[index]) + offset_x + offset_y * stride
        count = target_counts[index]
        for iy in range(side):
            low = base + (iy - radius) * stride - radius
            high = low + side - 1
            left, right = 0, len(reference)
            while left < right:
                middle = (left + right) // 2
                if reference[middle] < low:
                    left = middle + 1
                else:
                    right = middle
            while left < len(reference) and reference[left] <= high:
                ix = reference[left] - low
                scores[iy, ix] += min(ref_counts[left], count)
                left += 1


@numba.njit(cache=True, nogil=True)
def _count_pairs(ref_x, ref_y, ref_counts, target_x, target_y, target_counts,
                 offset_x, offset_y, radius, begin, end, scores):
    """Exact signed-row fallback when a reversible int64 packing is impossible."""
    side = 2 * radius + 1
    for index in range(begin, end):
        low = target_x[index] + offset_x - radius
        high = low + side - 1
        count = target_counts[index]
        for iy in range(side):
            wanted_y = target_y[index] + offset_y + iy - radius
            left, right = 0, len(ref_x)
            while left < right:
                middle = (left + right) // 2
                if (ref_y[middle] < wanted_y or
                        (ref_y[middle] == wanted_y and ref_x[middle] < low)):
                    left = middle + 1
                else:
                    right = middle
            while (left < len(ref_x) and ref_y[left] == wanted_y
                   and ref_x[left] <= high):
                scores[iy, ref_x[left] - low] += min(ref_counts[left], count)
                left += 1


def _check(cancel_check):
    if cancel_check is not None:
        cancel_check()


def _positive_index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a positive integer')
    value = operator.index(value)
    if not 0 < value <= np.iinfo(np.int64).max:
        raise ValueError(f'{name} must be a positive signed 64-bit integer')
    return value


def _memory_preflight(observations, frames, segments):
    estimated = observations * 256 + frames * 128 + segments * 256 + 32 * 1024**2
    import psutil
    available = psutil.virtual_memory().available
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    'total_physical', 'available_physical', 'total_commit', 'available_commit',
                    'total_virtual', 'available_virtual', 'available_extended_virtual')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot establish AIM host memory budget')
        available = min(available, status.available_commit)
    if estimated > .8 * available:
        raise MemoryError('AIM estimated working memory exceeds 80% of available host memory')


def _round_scalar(value):
    if not math.isfinite(value) or abs(value) > EXACT_COORDINATE_LIMIT:
        raise ValueError('AIM adaptive shift exceeds exact coordinate range')
    fraction, integral = math.modf(abs(value))
    rounded = int(integral) + (fraction >= .5)
    return -rounded if value < 0 else rounded


def _quantize(values, distance, cancel_check):
    result = np.empty(len(values), dtype=np.int64)
    for begin in range(0, len(values), ARRAY_CHUNK):
        _check(cancel_check)
        end = min(begin + ARRAY_CHUNK, len(values))
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = values[begin:end] / distance
        if not np.isfinite(scaled).all() or np.max(np.abs(scaled), initial=0) > EXACT_COORDINATE_LIMIT:
            raise ValueError('AIM coordinate/intersection-distance ratio exceeds exact integer range')
        magnitude = np.abs(scaled)
        integral = np.floor(magnitude)
        integral += magnitude - integral >= .5
        result[begin:end] = np.copysign(integral, scaled).astype(np.int64)
    return result


def _cell_storage(x, y, distance, segments, radius, cancel_check):
    qx, qy = _quantize(x, distance, cancel_check), _quantize(y, distance, cancel_check)
    min_x, max_x = int(qx.min()), int(qx.max())
    min_y, max_y = int(qy.min()), int(qy.max())
    # Rounding the previous correction cannot move by more than radius cells
    # per accepted segment. The extra margin covers the current search itself.
    halo = radius * (segments + 1) + 1
    x_base, y_base = min_x - halo, min_y - halo
    stride = max_x - min_x + 1 + 2 * halo
    height = max_y - min_y + 1 + 2 * halo
    maximum = stride * height - 1
    if (maximum > np.iinfo(np.int64).max or halo > EXACT_COORDINATE_LIMIT
            or max(abs(x_base), abs(y_base)) > np.iinfo(np.int64).max):
        return SimpleNamespace(compact=False, x=qx, y=qy)
    dtype = np.int32 if maximum <= np.iinfo(np.int32).max else np.int64
    keys = np.empty(len(qx), dtype=dtype)
    for begin in range(0, len(keys), ARRAY_CHUNK):
        _check(cancel_check)
        end = min(begin + ARRAY_CHUNK, len(keys))
        keys[begin:end] = (qy[begin:end] - y_base) * stride + (qx[begin:end] - x_base)
    return SimpleNamespace(compact=True, keys=keys, stride=stride, halo=halo)


def _counts(storage, begin, end, cancel_check):
    _check(cancel_check)
    if storage.compact:
        keys, counts = np.unique(storage.keys[begin:end], return_counts=True)
        result = (keys, counts)
    else:
        pairs = np.empty(end - begin, dtype=[('y', '<i8'), ('x', '<i8')])
        pairs['x'], pairs['y'] = storage.x[begin:end], storage.y[begin:end]
        pairs, counts = np.unique(pairs, return_counts=True)
        result = (np.ascontiguousarray(pairs['x']), np.ascontiguousarray(pairs['y']), counts)
    # NumPy's sort is synchronous. Cancellation is checked on both sides; the
    # counting kernels below are explicitly divided into bounded work batches.
    _check(cancel_check)
    return result


def _measure(reference, target, storage, offset_x, offset_y, radius, cancel_check):
    side = 2 * radius + 1
    scores = np.zeros((side, side), dtype=np.int64)
    work_per_cell = side * (max(1, len(reference[0]).bit_length()) + side)
    batch = max(1, min(8192, KERNEL_WORK_BUDGET // work_per_cell))
    if storage.compact and max(abs(offset_x), abs(offset_y)) + radius >= storage.halo:
        raise ValueError('AIM adaptive search exceeds verified compact-grid bounds')
    for begin in range(0, len(target[0]), batch):
        _check(cancel_check)
        end = min(begin + batch, len(target[0]))
        if storage.compact:
            _count_compact(*reference, *target, storage.stride,
                           offset_x, offset_y, radius, begin, end, scores)
        else:
            _count_pairs(*reference, *target, offset_x, offset_y, radius, begin, end, scores)
    _check(cancel_check)
    evidence = int(scores.max(initial=0))
    total = float(scores.sum(dtype=np.float64))
    if total == 0:
        return evidence, 0., None
    spectrum = np.fft.fft2(scores)
    harmonics = np.array([spectrum[0, 1], spectrum[1, 0]])
    strength = float(np.min(np.abs(harmonics)) / total)
    if not np.isfinite(strength) or strength < 1e-8:
        return evidence, strength, None
    position = np.mod(-np.angle(harmonics), 2*np.pi) * side / (2*np.pi) - radius
    if np.max(np.abs(position)) > radius + 1e-9:
        return evidence, strength, None
    return evidence, strength, position


def _interpolate(centers, values, n_frames, cancel_check):
    times = np.r_[2*centers[0]-centers[1], centers, 2*centers[-1]-centers[-2]]
    extended = np.vstack([2*values[0]-values[1], values, 2*values[-1]-values[-2]])
    spline = CubicSpline(times, extended)
    result = np.empty((n_frames, 2), dtype=np.float64)
    for begin in range(0, n_frames, ARRAY_CHUNK):
        _check(cancel_check)
        end = min(begin + ARRAY_CHUNK, n_frames)
        result[begin:end] = spline(np.arange(begin, end))
    if not np.isfinite(result).all():
        raise ValueError('AIM spline overflow')
    return result


def estimate_2d(x, y, frame, *, n_frames, segmentation=100, intersect_d=.2, roi_r=.6,
                cancel_check=None, progress_callback=None):
    """Estimate immutable drift arrays with bounded, cancellable counting batches.

    ``cancel_check`` raises the caller's cancellation exception when requested.
    Input coordinates are never modified; frames are zero-based movie indices.
    """
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError('cancel_check must be callable')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable')
    _check(cancel_check)
    if any(np.iscomplexobj(value) for value in (x, y, frame)):
        raise ValueError('AIM inputs must be real')
    x, y, frame = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(frame)
    if (x.ndim != 1 or y.ndim != 1 or frame.ndim != 1 or not len(x)
            or len(x) != len(y) or len(x) != len(frame)):
        raise ValueError('AIM inputs must be nonempty equal-length vectors')
    n_frames = _positive_index(n_frames, 'n_frames')
    segmentation = _positive_index(segmentation, 'segmentation')
    if n_frames <= segmentation:
        raise ValueError('AIM needs at least two temporal segments')
    distance, requested_radius = float(intersect_d), float(roi_r)
    if not all(math.isfinite(value) and value > 0 for value in (distance, requested_radius)):
        raise ValueError('AIM distances must be finite and positive')
    ratio = requested_radius / distance
    if not math.isfinite(ratio) or ratio > 512:
        raise ValueError('AIM search radius must not exceed 512 intersection cells')
    radius = max(1, math.ceil(ratio))
    if not math.isfinite(radius * distance):
        raise ValueError('AIM effective search radius exceeds the finite coordinate range')
    segments = 1 + (n_frames - 1) // segmentation
    _memory_preflight(len(x), n_frames, segments)
    if (not np.isfinite(frame).all() or not np.equal(frame, np.floor(frame)).all()
            or np.min(frame) < 0 or np.max(frame) >= n_frames):
        raise ValueError('AIM frames must be exact integers within the movie')
    frame = frame.astype(np.int64, copy=False)
    _check(cancel_check)
    original_x, original_y, original_frame = x, y, frame
    if not np.all(frame[1:] >= frame[:-1]):
        order = np.argsort(frame, kind='stable')
        frame, x, y = frame[order], x[order], y[order]
    _check(cancel_check)
    bounds = np.r_[np.arange(0, n_frames, segmentation, dtype=np.int64), n_frames]
    positions = np.searchsorted(frame, bounds, side='left')
    first = int(np.flatnonzero(np.diff(positions))[0])
    centers = (bounds[:-1] + bounds[1:] - 1) / 2
    warnings = []
    modes = []

    def report(percent):
        _check(cancel_check)
        if progress_callback is not None:
            progress_callback(percent)
        _check(cancel_check)

    def one_pass(a, b, initial):
        storage = _cell_storage(a, b, distance, segments, radius, cancel_check)
        modes.append('checked-compact' if storage.compact else 'signed-pairs')
        reference = (_counts(storage, positions[first], positions[first+1], cancel_check)
                     if initial else _counts(storage, 0, len(a), cancel_check))
        values = np.zeros((segments, 2))
        evidence = np.zeros(segments, dtype=np.uint64)
        strength = np.zeros(segments)
        valid = np.zeros(segments, dtype=bool)
        offset = np.zeros(2)
        for segment in range(segments):
            _check(cancel_check)
            begin, end = positions[segment:segment+2]
            if initial and segment == first:
                evidence[segment], strength[segment], valid[segment] = end-begin, 1., True
            elif begin < end:
                target = _counts(storage, begin, end, cancel_check)
                ox, oy = _round_scalar(offset[0]), _round_scalar(offset[1])
                ev, quality, position = _measure(reference, target, storage, ox, oy, radius, cancel_check)
                evidence[segment], strength[segment] = ev, quality
                if position is not None:
                    offset = np.array([ox, oy]) + position
                    values[segment], valid[segment] = -offset * distance, True
            report((0 if initial else 50) + 50*(segment+1)//segments)
        number = int(valid.sum())
        if number < 2:
            raise ValueError('AIM has fewer than two segments with identifiable overlapping evidence')
        stage = 1 if initial else 2
        if number < segments:
            warnings.append(f'stage {stage}: {segments-number} empty/unidentifiable segment(s) '
                            'interpolated from measured segments')
        if not valid[0] or not valid[-1]:
            warnings.append(f'stage {stage}: drift at an unmeasured movie boundary is extrapolated')
        drift = _interpolate(centers[valid], values[valid], n_frames, cancel_check)
        return drift, evidence, strength, valid

    report(0)
    stage1 = one_pass(x, y, True)
    corrected_x, corrected_y = np.empty_like(x), np.empty_like(y)
    for begin in range(0, len(x), ARRAY_CHUNK):
        _check(cancel_check)
        end = min(begin + ARRAY_CHUNK, len(x))
        corrected_x[begin:end] = x[begin:end] - stage1[0][frame[begin:end], 0]
        corrected_y[begin:end] = y[begin:end] - stage1[0][frame[begin:end], 1]
    stage2 = one_pass(corrected_x, corrected_y, False)
    drift = stage1[0] + stage2[0]
    drift -= drift[0]
    # Reuse these buffers, restoring the input's observation order explicitly.
    for begin in range(0, len(x), ARRAY_CHUNK):
        _check(cancel_check)
        end = min(begin + ARRAY_CHUNK, len(x))
        selected = original_frame[begin:end]
        corrected_x[begin:end] = original_x[begin:end] - drift[selected, 0]
        corrected_y[begin:end] = original_y[begin:end] - drift[selected, 1]
    fields = dict(corrected_x=corrected_x, corrected_y=corrected_y,
                  drift_x=drift[:, 0], drift_y=drift[:, 1])
    for number, stage in ((1, stage1), (2, stage2)):
        fields.update({f'stage{number}_drift_x': stage[0][:, 0],
                       f'stage{number}_drift_y': stage[0][:, 1],
                       f'stage{number}_evidence': stage[1],
                       f'stage{number}_harmonic_strength': stage[2],
                       f'stage{number}_valid': stage[3]})
    for array in fields.values():
        if not np.isfinite(array).all():
            raise ValueError('AIM produced a non-finite result')
        array.setflags(write=False)
    report(100)
    return SimpleNamespace(**fields, n_frames=n_frames, n_segments=segments,
                           segmentation=segmentation, intersect_d=distance,
                           requested_roi_r=requested_radius, effective_roi_r=radius*distance,
                           algorithm_version=ALGORITHM_VERSION,
                           stage1_reference_segment=first,
                           status='interpolated' if warnings else 'ok', warnings=tuple(warnings),
                           provenance={**backend_info(), 'count_storage': modes})
