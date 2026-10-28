"""Validated AIM estimation, immutable results, and movie correction.

Drift estimation uses the guarded Picasso/Numba method in ``aim_picasso``.
Movie correction uses bounded NumPy tiles in ``aim_movie``. Both operations
share a thread-safe cancellation token and never publish partial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
import threading
from typing import Callable, Optional

import numpy as np


class AimCancelled(RuntimeError):
    """An AIM operation was cancelled before publishing a result."""


class CancellationToken:
    """Thread-safe cooperative cancellation shared by AIM workers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _cancel_check(progress_callback, cancel_token):
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if cancel_token is not None and not isinstance(cancel_token, CancellationToken):
        raise TypeError("cancel_token must be a CancellationToken or None")

    def check():
        if cancel_token is not None and cancel_token.is_cancelled():
            raise AimCancelled("AIM operation cancelled")

    check()
    return check


_ARRAY_FIELDS = (
    "corrected_x", "corrected_y", "drift_x", "drift_y",
    "stage1_drift_x", "stage1_drift_y", "stage2_drift_x", "stage2_drift_y",
    "stage1_evidence", "stage2_evidence",
    "stage1_harmonic_strength", "stage2_harmonic_strength",
    "stage1_valid", "stage2_valid",
)


def _readonly_array(value, dtype) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    if array.ndim != 1:
        raise ValueError("AimResult arrays must be one-dimensional")
    array.setflags(write=False)
    return array


def _freeze_metadata(value):
    if isinstance(value, dict):
        return tuple((str(key), _freeze_metadata(item))
                     for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw_metadata(value):
    if (isinstance(value, tuple) and value
            and all(isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[0], str) for item in value)):
        return {key: _thaw_metadata(item) for key, item in value}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


@dataclass(frozen=True)
class AimResult:
    """Immutable, session-serializable view of one complete AIM estimate."""

    corrected_x: np.ndarray
    corrected_y: np.ndarray
    drift_x: np.ndarray
    drift_y: np.ndarray
    stage1_drift_x: np.ndarray
    stage1_drift_y: np.ndarray
    stage2_drift_x: np.ndarray
    stage2_drift_y: np.ndarray
    stage1_evidence: np.ndarray
    stage2_evidence: np.ndarray
    stage1_harmonic_strength: np.ndarray
    stage2_harmonic_strength: np.ndarray
    stage1_valid: np.ndarray
    stage2_valid: np.ndarray
    n_frames: int
    n_segments: int
    segmentation: int
    intersect_d: float
    requested_roi_r: float
    effective_roi_r: float
    algorithm_version: str
    stage1_reference_segment: int
    status: str = "ok"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance_data: tuple = field(default_factory=tuple, repr=False)

    def __post_init__(self):
        float_fields = (
            "corrected_x", "corrected_y", "drift_x", "drift_y",
            "stage1_drift_x", "stage1_drift_y",
            "stage2_drift_x", "stage2_drift_y",
            "stage1_harmonic_strength", "stage2_harmonic_strength",
        )
        for name in float_fields:
            object.__setattr__(self, name, _readonly_array(getattr(self, name), np.float64))
        for name in ("stage1_evidence", "stage2_evidence"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), np.uint64))
        for name in ("stage1_valid", "stage2_valid"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), np.bool_))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "provenance_data", _freeze_metadata(self.provenance_data))

        if self.n_frames <= 0 or self.n_segments < 2 or self.segmentation <= 0:
            raise ValueError("AimResult has invalid frame/segment dimensions")
        for name in ("drift_x", "drift_y", "stage1_drift_x", "stage1_drift_y",
                     "stage2_drift_x", "stage2_drift_y"):
            if len(getattr(self, name)) != self.n_frames:
                raise ValueError(f"AimResult {name} length does not match n_frames")
        if len(self.corrected_x) != len(self.corrected_y):
            raise ValueError("AimResult corrected coordinate lengths differ")
        for name in ("stage1_evidence", "stage2_evidence",
                     "stage1_harmonic_strength", "stage2_harmonic_strength",
                     "stage1_valid", "stage2_valid"):
            if len(getattr(self, name)) != self.n_segments:
                raise ValueError(f"AimResult {name} length does not match n_segments")
        if not (np.isfinite(self.intersect_d) and self.intersect_d > 0
                and np.isfinite(self.requested_roi_r) and self.requested_roi_r > 0
                and np.isfinite(self.effective_roi_r) and self.effective_roi_r > 0):
            raise ValueError("AimResult contains invalid AIM parameters")
        if not 0 <= self.stage1_reference_segment < self.n_segments:
            raise ValueError("AimResult reference segment is out of range")
        for name in float_fields:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"AimResult {name} contains non-finite values")

    @property
    def provenance(self) -> dict:
        return _thaw_metadata(self.provenance_data) if self.provenance_data else {}

    @classmethod
    def from_estimate(cls, estimate) -> "AimResult":
        return cls(
            **{name: getattr(estimate, name) for name in _ARRAY_FIELDS},
            n_frames=int(estimate.n_frames),
            n_segments=int(estimate.n_segments),
            segmentation=int(estimate.segmentation),
            intersect_d=float(estimate.intersect_d),
            requested_roi_r=float(estimate.requested_roi_r),
            effective_roi_r=float(estimate.effective_roi_r),
            algorithm_version=str(estimate.algorithm_version),
            stage1_reference_segment=int(estimate.stage1_reference_segment),
            status=str(estimate.status),
            warnings=tuple(estimate.warnings),
            provenance_data=dict(estimate.provenance),
        )


def _positive_index(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return int(result)


def estimate_aim(
    x,
    y,
    frame,
    n_frames: int,
    segmentation: int = 100,
    intersect_d: float = 20.0 / 130.0,
    roi_r: float = 60.0 / 130.0,
    progress_callback: Optional[Callable[[int], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> AimResult:
    """Validate inputs, run guarded Picasso 2-D AIM, and freeze its result."""
    cancel_check = _cancel_check(progress_callback, cancel_token)
    if any(np.iscomplexobj(values) for values in (x, y, frame)):
        raise ValueError("AIM coordinates and frames must contain real values")
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    frame_values = np.asarray(frame)
    if x_array.ndim != 1 or y_array.ndim != 1 or frame_values.ndim != 1:
        raise ValueError("AIM x, y and frame inputs must be one-dimensional")
    if not len(x_array) or len(y_array) != len(x_array) or len(frame_values) != len(x_array):
        raise ValueError("AIM x, y and frame inputs must be nonempty and equal length")
    if not (np.isfinite(x_array).all() and np.isfinite(y_array).all()):
        raise ValueError("AIM coordinates must be finite")
    if frame_values.dtype.kind not in "iub":
        frame_values = np.asarray(frame_values, dtype=np.float64)
        if (not np.isfinite(frame_values).all()
                or not np.equal(frame_values, np.floor(frame_values)).all()):
            raise ValueError("AIM frames must be finite exact integers")
    frame_count = _positive_index(n_frames, "n_frames")
    segment_size = _positive_index(segmentation, "segmentation")
    if frame_count > np.iinfo(np.int64).max:
        raise ValueError("n_frames exceeds the supported signed 64-bit range")
    if (frame_values.min() < 0 or frame_values.max() >= frame_count
            or frame_values.max() >= 2**63):
        raise ValueError("AIM frame indices must lie within the movie")
    frame_array = frame_values.astype(np.int64, copy=False)
    if len(frame_array) and (frame_array.min() < 0 or frame_array.max() >= frame_count):
        raise ValueError("AIM frame indices must lie within the movie")
    grid = float(intersect_d)
    radius = float(roi_r)
    if not (np.isfinite(grid) and grid > 0):
        raise ValueError("intersect_d must be finite and positive")
    if not (np.isfinite(radius) and radius > 0):
        raise ValueError("roi_r must be finite and positive")
    from .aim_picasso import estimate_2d
    estimate = estimate_2d(
        np.ascontiguousarray(x_array),
        np.ascontiguousarray(y_array),
        np.ascontiguousarray(frame_array),
        n_frames=frame_count,
        segmentation=segment_size,
        intersect_d=grid,
        roi_r=radius,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    cancel_check()
    result = AimResult.from_estimate(estimate)
    cancel_check()
    return result


def shift_movie(
    stack,
    drift_x,
    drift_y,
    order: int = 1,
    progress_callback: Optional[Callable[[int], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> np.ndarray:
    """Correct movie drift using bounded nearest/bilinear NumPy tiles."""
    cancel_check = _cancel_check(progress_callback, cancel_token)
    movie = np.asarray(stack)
    if movie.ndim not in (2, 3) or any(size <= 0 for size in movie.shape):
        raise ValueError("Movie must have nonempty shape (H,W) or (T,H,W)")
    frames = movie.shape[0] if movie.ndim == 3 else 1
    if np.iscomplexobj(drift_x) or np.iscomplexobj(drift_y):
        raise ValueError("Movie drift arrays must contain real values")
    dx = np.asarray(drift_x, dtype=np.float64)
    dy = np.asarray(drift_y, dtype=np.float64)
    if dx.ndim != 1 or dy.ndim != 1 or len(dx) != frames or len(dy) != frames:
        raise ValueError("Drift array lengths must exactly match movie frames")
    if not (np.isfinite(dx).all() and np.isfinite(dy).all()):
        raise ValueError("Movie drift arrays must be finite")
    if isinstance(order, (bool, np.bool_)):
        raise TypeError("order must be 0 or 1")
    try:
        interpolation = operator.index(order)
    except TypeError as exc:
        raise TypeError("order must be 0 or 1") from exc
    if interpolation not in (0, 1):
        raise ValueError("order must be 0 or 1")
    if movie.dtype.kind not in "iu" and not (movie.dtype.kind == "f" and movie.dtype.itemsize in (4, 8)):
        raise ValueError("Movie dtype must be an integer or float32/float64")
    from .aim_movie import shift_pixels
    return shift_pixels(
        movie, dx, dy, order=int(interpolation),
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


__all__ = [
    "AimCancelled",
    "AimResult",
    "CancellationToken",
    "estimate_aim",
    "shift_movie",
]
