"""
Shared data types for MangoTrace.

Small, dependency-light data types passed between the core pipeline and the
GUI: the universal ``Spot`` representation and the per-molecule HMM
``StateResult``. Tabular / JSON-serialized pipeline data (localization tables,
binding events) deliberately stays in DataFrames / dicts; image SNR helpers
live in ``core/image_io.py``.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import NamedTuple
import hashlib
import uuid

import numpy as np


class Spot(NamedTuple):
    """Detected spot with position and properties.

    Field meanings depend on the detection mode:

    Universal spot representation used across detection and trace extraction:
        intensity  – brightest-pixel intensity
        sigma      – Gaussian fit width (PSF sigma, px)
        r_squared  – Gaussian fit R² (goodness-of-fit, 0–1)

    SMLM (frame-by-frame localization + DBSCAN clustering):
        intensity  – number of localizations in the cluster
        sigma      – mean Gaussian PSF width across cluster locs
        r_squared  – **frame span** (max_frame − min_frame + 1),
                     i.e. the temporal extent of the cluster.
                     NOT a fit R² in SMLM mode.
    """
    x: float
    y: float
    intensity: float = 0.0
    sigma: float = 0.0
    r_squared: float = 0.0


class StateResult(NamedTuple):
    """Per-molecule HMM result: the per-frame integer state-index array and the
    per-state mean intensities (state 0 = background, by convention the lower
    mean). Both arrays are present once a molecule has been fitted.
    """
    states: np.ndarray
    means: np.ndarray


def file_fingerprint(path: str, sample_bytes: int = 65536) -> str:
    """Return a small, deterministic fingerprint without reading a whole movie.

    The digest covers the file size and samples from both ends.  It is intended
    for safe session relinking, not cryptographic authentication.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    size = p.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(size.to_bytes(8, "little", signed=False))
    with p.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    """Widget-independent identity and metadata for one file-list row.

    ``id`` is the durable identity used by session format v6. ``result_key`` is
    the in-memory/UI identity, while ``source_paths`` always contains physical
    paths used for I/O.  This separation is what makes expanded channel rows
    (``movie.tif::ch2``) safe to process and relink.
    """

    id: str
    result_key: str
    source_paths: tuple[str, ...]
    channel_index: int = 1
    channel_count: int = 1
    source_type: str = "movie"
    parameter: float = 0.0
    parameter_text: str = ""
    parameter_role: str = "generic"
    parameter_is_set: bool = False
    concentration: float = 0.0
    concentration_unit: str = "nM"
    concentration_source: str = "unknown"
    registration: bool = False
    base: str = ""
    active: bool = True
    processing_flags: tuple[str, ...] = field(default_factory=tuple)
    source_sizes: tuple[int | None, ...] = field(default_factory=tuple)
    source_fingerprints: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        result_key: str,
        source_paths: tuple[str, ...] | list[str],
        *,
        channel_index: int = 1,
        source_type: str = "movie",
        record_id: str | None = None,
        inspect_sources: bool = True,
        **metadata,
    ) -> "FileRecord":
        sources = tuple(str(p) for p in source_paths)
        sizes = []
        fingerprints = []
        for source in sources:
            if not inspect_sources:
                sizes.append(None)
                fingerprints.append("")
                continue
            p = Path(source)
            try:
                sizes.append(p.stat().st_size if p.is_file() else None)
            except OSError:
                sizes.append(None)
            try:
                fingerprints.append(file_fingerprint(source))
            except OSError:
                fingerprints.append("")
        return cls(
            id=record_id or str(uuid.uuid4()),
            result_key=str(result_key),
            source_paths=sources,
            channel_index=int(channel_index),
            source_type=str(source_type),
            source_sizes=tuple(sizes),
            source_fingerprints=tuple(fingerprints),
            **metadata,
        )

    def __post_init__(self):
        try:
            uuid.UUID(self.id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid FileRecord UUID: {self.id!r}") from exc
        if not self.result_key:
            raise ValueError("FileRecord.result_key cannot be empty")
        if not self.source_paths:
            raise ValueError("FileRecord.source_paths cannot be empty")
        if self.channel_index < 1:
            raise ValueError("FileRecord.channel_index must be >= 1")
        if self.channel_count < self.channel_index:
            raise ValueError("FileRecord.channel_count must be >= channel_index")
        if len(self.source_sizes) not in (0, len(self.source_paths)):
            raise ValueError("source_sizes must align with source_paths")
        if len(self.source_fingerprints) not in (0, len(self.source_paths)):
            raise ValueError("source_fingerprints must align with source_paths")
        try:
            parameter = float(self.parameter)
        except (TypeError, ValueError) as exc:
            raise ValueError("FileRecord.parameter must be numeric") from exc
        if not np.isfinite(parameter):
            raise ValueError("FileRecord.parameter must be finite")
        if self.parameter_role not in {"generic", "concentration"}:
            raise ValueError("FileRecord.parameter_role must be generic or concentration")
        if not isinstance(self.parameter_is_set, bool):
            raise ValueError("FileRecord.parameter_is_set must be boolean")
        if self.concentration_source not in {"unknown", "filename", "manual", "parameter", "legacy"}:
            raise ValueError("Invalid FileRecord.concentration_source")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(
            self, "parameter_text",
            "" if self.parameter_text is None else str(self.parameter_text).strip())

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("source_paths", "processing_flags", "source_sizes",
                    "source_fingerprints"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "FileRecord":
        if not isinstance(value, dict):
            raise TypeError("FileRecord payload must be a mapping")
        data = dict(value)
        # Additive v6 metadata. Older archives deliberately retain generic
        # Parameter semantics and do not reinterpret an untouched zero.
        data.setdefault("parameter_role", "generic")
        data.setdefault("parameter_is_set", bool(data.get("parameter", 0)))
        data.setdefault("concentration_source", "legacy" if data.get("concentration", 0) > 0 else "unknown")
        missing = set(cls.__dataclass_fields__) - set(data)
        if missing:
            raise ValueError(f"Incomplete FileRecord payload: {sorted(missing)}")
        for key in ("source_paths", "processing_flags", "source_sizes",
                    "source_fingerprints"):
            data[key] = tuple(data.get(key, ()))
        return cls(**data)


@dataclass(frozen=True)
class BindingEvent:
    """One binding event detected in a single-molecule trace.

    ``Start``/``End``/``Duration`` are in the trace's time units; ``StartIdx``/
    ``EndIdx`` are the corresponding frame indices (``-1`` when reconstructed
    from an export that omits them). Both end fields are exclusive: an event
    with ``StartIdx=3`` and ``EndIdx=5`` covers frames 3 and 4. The event is
    exposes field iteration for pandas / CSV / session serialization.

    ``StartTrunc``/``EndTrunc`` mark a BOUNDARY-TRUNCATED event — one already in
    progress when acquisition started, or still in progress when it ended. Its true
    dwell time is unknown (only a lower bound was observed), so ``core.kinetics``
    excludes it from on-times when fitting k_on / k_off / K_D. It is still a real
    detection and stays in the event list for display and for trace filtering.

    They are ``int``, not ``bool``, deliberately: ``isinstance(True, int)`` is True,
    so ``session_manager``'s numeric coercion saves a bool as ``1.0`` and
    :meth:`from_dict` would then restore a float into a bool-annotated field. Only
    truthiness is ever tested.
    """
    Start: float
    End: float
    Duration: float
    MeanIntensity: float = 0.0
    StartIdx: int = -1
    EndIdx: int = -1
    SNR: float = 0.0
    StartTrunc: int = 0
    EndTrunc: int = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def items(self):
        return asdict(self).items()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "BindingEvent":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def event_get(ev, key, default=None):
    """Read a typed event or a serialized event with optional QC annotations."""
    if ev is None:
        return default
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)
