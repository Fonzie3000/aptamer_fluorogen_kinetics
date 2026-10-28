"""
TIFF image loading utilities.
Supports multi-frame stacks, channel splitting, and quadrant extraction.

Cloud-storage note
------------------
Google Drive for Desktop, OneDrive, and similar virtual-filesystem providers
can expose files as online-only stubs.  These stubs may:
  • make Path.exists() return False even though the item is present;
  • make Path.stat() raise OSError or return st_size == 0;
  • be un-mappable (tifffile.memmap will fail or give corrupt data).

The helper _resolve_to_local_path() detects these situations and streams the
file bytes to a temporary local copy so that all tifffile calls work normally.
The temp file is cleaned up automatically after use.
"""

import shutil
import tempfile
from collections import OrderedDict
import numpy as np
import tifffile
from pathlib import Path
from threading import RLock
from typing import Optional, Union


def _resolve_to_local_path(path: Path) -> tuple:
    """
    Ensure *path* is a readable local file.

    Cloud-storage virtual-filesystem drivers (Google Drive for Desktop,
    OneDrive, Dropbox, network shares, etc.) may expose files as online-only
    stubs that cannot be stat-ed, memory-mapped, or reliably opened by
    tifffile.  This function detects those cases and streams the file to a
    temporary local copy.

    Returns
    -------
    (local_path, tmp_dir)
        *local_path* is a Path that is guaranteed to be a real, readable,
        locally-accessible file.
        *tmp_dir* is either a ``tempfile.TemporaryDirectory`` object (must be
        kept alive while *local_path* is in use) or ``None`` when no copy was
        needed.
    """
    # ------------------------------------------------------------------ #
    # 1. Try to stat the file to get its size.                            #
    #    On cloud stubs this may raise OSError or return size == 0.       #
    # ------------------------------------------------------------------ #
    try:
        file_size = path.stat().st_size
        is_cloud_stub = (file_size == 0)
    except OSError:
        is_cloud_stub = True
        file_size = 0

    if not is_cloud_stub:
        # Normal local file — use as-is.
        return path, None

    # ------------------------------------------------------------------ #
    # 2. The file looks like a cloud stub.  Try to open it for reading;   #
    #    if the cloud provider downloads it on-demand the open() call     #
    #    will succeed (possibly after a brief wait while it syncs).       #
    # ------------------------------------------------------------------ #
    tmp_dir = tempfile.TemporaryDirectory(prefix="mangotrace_cloud_")
    local_copy = Path(tmp_dir.name) / path.name
    try:
        shutil.copy2(str(path), str(local_copy))
    except Exception as exc:
        tmp_dir.cleanup()
        raise IOError(
            f"Cannot read '{path}'. "
            "The file may be an online-only cloud stub that could not be "
            f"downloaded automatically: {exc}"
        ) from exc
    return local_copy, tmp_dir


def normalize_to_255(img: np.ndarray) -> np.ndarray:
    """Stretch image pixel values to the [0, 255] range (float64).

    Returns an all-zero array for a flat image (max == min). Used to give
    registration inputs a consistent intensity scale.
    """
    img = np.asarray(img, dtype=np.float64)
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn) * 255.0
    return np.zeros_like(img, dtype=np.float64)


def subtract_gaussian_background(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian high-pass background removal: img - gaussian_blur(img), clipped at 0.

    Works in float64 to handle the intermediate negatives; returns float64.
    """
    from scipy.ndimage import gaussian_filter
    img = np.asarray(img, dtype=np.float64)
    bg = gaussian_filter(img, sigma=sigma)
    return np.clip(img - bg, 0, None)


def subtract_median_background(img: np.ndarray, radius: float) -> np.ndarray:
    """Median-filter background removal: img - median_filter(img, disk(radius)),
    clipped at 0. Works in float64; returns float64."""
    from scipy.ndimage import median_filter
    img = np.asarray(img, dtype=np.float64)
    r = int(max(1, radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    footprint = (yy * yy + xx * xx) <= r * r
    bg = median_filter(img, footprint=footprint)
    return np.clip(img - bg, 0, None)


_GAIN_BG_PERCENTILE = 10.0  # temporal percentile used as the background level
_BACKGROUND_WORKING_BYTES = 128 * 1024 ** 2


def _check_background_cancel(cancel_check):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Background correction cancelled")


def _background_level(stack3d, *, working_memory_bytes=_BACKGROUND_WORKING_BYTES,
                      cancel_check=None) -> np.ndarray:
    """Molecule-free background level per pixel: a low temporal percentile over a
    frame subsample. Each molecule is ON only a fraction of the time, so a low
    percentile is molecule-free, and unlike a pure MIN projection it is not biased
    low by read noise (the min over thousands of noisy frames sits ~3σ below the
    true background). Uses the same evenly spaced sample of up to 200 frames
    for every tile. Percentiles are exact for that sample; spatial tiling changes
    memory use, not the sample or the estimated pixel values.
    """
    T, height, width = map(int, stack3d.shape)
    if T < 1 or height < 1 or width < 1:
        raise ValueError("Background correction requires a nonempty movie")
    idx = np.unique(np.linspace(0, T - 1, min(200, T)).astype(int))
    # Reserve the result and one decoded input frame. A percentile may copy its
    # work buffer, so allow two sample buffers even with overwrite_input=True.
    reserved = height * width * (8 + np.dtype(stack3d.dtype).itemsize)
    remaining = int(working_memory_bytes) - reserved
    pixels_per_tile = remaining // (16 * len(idx) + 32)
    if pixels_per_tile < 1:
        raise MemoryError(
            "Background working-memory budget cannot hold one image plane and "
            "one temporal pixel sample; increase working_memory_bytes")
    tile_width = min(width, pixels_per_tile)
    tile_height = max(1, min(height, pixels_per_tile // tile_width))
    level = np.empty((height, width), dtype=np.float64)
    for y0 in range(0, height, tile_height):
        y1 = min(height, y0 + tile_height)
        for x0 in range(0, width, tile_width):
            x1 = min(width, x0 + tile_width)
            sample = np.empty((len(idx), y1 - y0, x1 - x0), dtype=np.float64)
            for offset, frame_index in enumerate(idx):
                _check_background_cancel(cancel_check)
                frame = stack3d[int(frame_index)]
                sample[offset] = frame[y0:y1, x0:x1]
                del frame
            level[y0:y1, x0:x1] = np.percentile(
                sample, _GAIN_BG_PERCENTILE, axis=0, overwrite_input=True)
            del sample
    return level


def _illumination_gain_map(bg_level: np.ndarray, radius: int,
                           baseline: float) -> np.ndarray:
    """Relative illumination-gain map (mean 1) from a molecule-free background
    level image (see _background_level).

    The level is spatially median-smoothed by ``radius`` (robust; removes residual
    hot pixels without the low bias of a min filter); subtracting the camera
    ``baseline`` leaves a value proportional to the local laser gain, normalized
    to mean 1 and clipped to a sane range so the later division can't blow up.

    The ``baseline`` MUST lie below the background floor for the gain to be
    meaningful. If it is at/above the floor — a wrong value for the movie, or a
    corrected channel with ~no background above offset — there is no headroom to
    estimate a gain, so a uniform (no-op) map is returned instead of amplifying
    noise into a spurious correction.
    """
    from scipy.ndimage import median_filter
    size = 2 * int(max(1, radius)) + 1
    bg = median_filter(bg_level.astype(np.float64), size=size)
    floor = float(bg.min())
    if float(baseline) >= floor:
        return np.ones_like(bg)
    g = bg - float(baseline)
    m = float(g.mean())
    if m <= 0:
        return np.ones_like(bg)
    return np.clip(g / m, 0.2, 5.0)


def illumination_gain_map(movie, radius: int, baseline: float, *,
                          working_memory_bytes=_BACKGROUND_WORKING_BYTES,
                          cancel_check=None) -> np.ndarray:
    """Estimate one channel's gain from a 2-D image or lazy 3-D movie.

    Only a bounded tile of the temporal sample and a few image planes reside in
    memory. The sample positions and percentile match the in-memory estimator.
    """
    _check_background_cancel(cancel_check)
    if len(movie.shape) == 2:
        level = np.asarray(movie, dtype=np.float64)
    elif len(movie.shape) == 3:
        level = _background_level(
            movie, working_memory_bytes=working_memory_bytes,
            cancel_check=cancel_check)
    else:
        raise ValueError("An illumination gain map requires one 2-D/3-D channel")
    _check_background_cancel(cancel_check)
    gain = _illumination_gain_map(level, radius, float(baseline))
    _check_background_cancel(cancel_check)
    return gain


class _BackgroundChannel:
    """Single-channel view which never materializes a 4-D movie."""

    def __init__(self, movie, channel):
        self.movie, self.channel = movie, channel
        self.shape = (movie.shape[0], *movie.shape[-2:])
        self.dtype = movie.dtype

    def __getitem__(self, frame):
        return self.movie[frame][self.channel]


def remove_movie_background(movie: np.ndarray, method: str = 'gaussian',
                            sigma: float = 10.0, radius: float = 7,
                            use_gpu: bool = False, baseline: float = 400.0,
                            progress_callback=None, *, output=None,
                            cancel_check=None,
                            working_memory_bytes=_BACKGROUND_WORKING_BYTES) -> np.ndarray:
    """Background / illumination correction over a movie stack.

    method:
      'gaussian'  — per-frame Gaussian high-pass with ``sigma`` (subtractive).
      'median'    — per-frame disk-median background of ``radius`` (subtractive).
      'flatfield' — multiplicative gain correction for uneven illumination: a
                    molecule-free background level (low temporal percentile)
                    median-smoothed by ``radius`` minus the camera ``baseline`` is
                    the relative gain, and each frame's baseline-subtracted signal
                    is divided by it so molecule peak amplitudes become uniform
                    across the field (edge molecules boosted); ``baseline`` must be
                    below the background floor or it is a no-op. See
                    illumination_gain_map.

    Handles (Y,X), (T,Y,X) and (T,C,Y,X) stacks and returns the same shape and
    dtype as the input (integer dtypes are rounded and clipped to their range —
    the flat-field division can amplify, so bright pixels may saturate). With
    ``use_gpu`` the per-frame subtractive filtering runs on the GPU via CuPy
    (falls back to CPU). ``progress_callback(int 0-100)`` is optional.

    Pass a writable ``output`` array/memmap of the same shape and dtype to keep
    memory independent of the number of frames. A lazy frame-indexable input
    is accepted. Without an output sink this function allocates only the final
    input-dtype result, never a whole-movie float64 working copy. Cancellation
    raises InterruptedError and leaves partial output unpublished for the
    caller to discard.
    """
    if method not in {'gaussian', 'median', 'flatfield'}:
        raise ValueError(f"Unknown background-correction method: {method}")
    if not hasattr(movie, 'shape'):
        movie = np.asarray(movie)
    shape = tuple(int(value) for value in movie.shape)
    ndim = len(shape)
    if ndim not in (2, 3, 4) or any(value < 1 for value in shape):
        raise ValueError(f"Unsupported or empty movie shape: {shape}")
    in_dtype = np.dtype(movie.dtype)
    if not (np.issubdtype(in_dtype, np.integer) or np.issubdtype(in_dtype, np.floating)):
        raise ValueError("Background correction requires real-valued image pixels")
    if (not np.isfinite(sigma) or sigma < 0 or not np.isfinite(radius) or radius < 0
            or not np.isfinite(baseline)):
        raise ValueError("Background-correction parameters must be finite and nonnegative")
    # Input frame, arithmetic/filter temporaries, gain and output conversion.
    plane_bytes = shape[-2] * shape[-1] * (8 * 6 + in_dtype.itemsize)
    if int(working_memory_bytes) < plane_bytes:
        raise MemoryError(
            "Background working-memory budget cannot hold the image-plane work "
            "buffers; increase working_memory_bytes")
    _check_background_cancel(cancel_check)
    if output is None:
        output = np.empty(shape, dtype=in_dtype)
    elif (tuple(output.shape) != shape
          or np.dtype(output.dtype).newbyteorder('=') != in_dtype.newbyteorder('=')):
        raise ValueError("Background output must have the input shape and dtype")
    if isinstance(movie, np.ndarray) and np.shares_memory(movie, output):
        raise ValueError("Background input and output must not share pixel storage")

    cp = _gpu_gauss = _gpu_median = _gpu_footprint = None
    if use_gpu and method != 'flatfield':
        try:
            import cupy as cp
            from cupyx.scipy import ndimage as _gpu_ndi
            _gpu_gauss, _gpu_median = _gpu_ndi.gaussian_filter, _gpu_ndi.median_filter
            r = int(max(1, radius))
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            _gpu_footprint = cp.asarray((yy * yy + xx * xx) <= r * r)
        except Exception:
            use_gpu = False

    def _bg2d(frame):
        if use_gpu:
            try:
                f = cp.asarray(frame, dtype=cp.float64)
                bg = (_gpu_median(f, footprint=_gpu_footprint) if method == 'median'
                      else _gpu_gauss(f, sigma=sigma))
                return cp.asnumpy(cp.clip(f - bg, 0, None))
            except (RuntimeError, MemoryError):
                # A frame that exceeds device memory still has a CPU path.
                pass
        if method == 'median':
            return subtract_median_background(frame, radius)
        return subtract_gaussian_background(frame, sigma)

    frames = 1 if ndim == 2 else shape[0]
    channels = shape[1] if ndim == 4 else 1
    info = np.iinfo(in_dtype) if np.issubdtype(in_dtype, np.integer) else None
    completed = 0
    for channel in range(channels):
        view = _BackgroundChannel(movie, channel) if ndim == 4 else movie
        gain = (illumination_gain_map(
            view, int(max(1, radius)), baseline,
            working_memory_bytes=working_memory_bytes, cancel_check=cancel_check)
            if method == 'flatfield' else None)
        for frame_index in range(frames):
            _check_background_cancel(cancel_check)
            frame = view if ndim == 2 else view[frame_index]
            if gain is None:
                corrected = _bg2d(frame)
            else:
                corrected = np.asarray(frame, dtype=np.float64).copy()
                corrected -= float(baseline)
                corrected /= gain
                corrected += float(baseline)
            if info is not None:
                np.rint(corrected, out=corrected)
                np.clip(corrected, info.min, info.max, out=corrected)
            if ndim == 2:
                output[...] = corrected
            elif ndim == 3:
                output[frame_index] = corrected
            else:
                output[frame_index, channel] = corrected
            del corrected, frame
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / (frames * channels)))
        del gain
    _check_background_cancel(cancel_check)
    return output


def strip_channel_suffix(path: str) -> str:
    """Return the real file path from a virtual '<real>::chN' key.

    Channel-expanded single-channel entries are encoded as '<real_path>::chN'
    (N is 1-based). This strips that suffix; paths without a '::ch' suffix are
    returned unchanged. Single source of truth for decoding the virtual-path
    scheme used across the spot/movie pipeline.
    """
    return path.rsplit('::ch', 1)[0] if '::ch' in path else path


def load_channel_stack(path, detect_ch: int = 1) -> np.ndarray:
    """Load a TIFF and reduce a 4D (T, C, Y, X) stack to its (T, Y, X) channel.

    Strips a trailing '::chN' virtual-path suffix, loads via load_tiff, and
    selects channel index (detect_ch - 1), falling back to channel 0 when that
    index is out of range. 3D/2D stacks are returned unchanged. Shared by the
    Tracks and Trace Filter crop-viewer load paths.
    """
    real = strip_channel_suffix(path)
    stack = load_tiff(real)
    if stack.ndim == 4:
        chi = max(0, detect_ch - 1)
        chi = chi if chi < stack.shape[1] else 0
        stack = stack[:, chi, :, :]
    return stack


def load_tiff(path: Union[str, Path], mmap_mode: bool = False) -> np.ndarray:
    """
    Load a TIFF file as a numpy array.
    
    Args:
        path: Path to the TIFF file.  Cloud-storage paths (Google Drive,
              OneDrive, …) are supported; online-only stub files are
              downloaded to a temporary local copy automatically.
        mmap_mode: If True, force memory-mapped loading (read-only, low RAM).
                   Silently ignored for cloud files (mmap requires a real
                   local file descriptor).
        
    Returns:
        Image data as numpy array, normalized to:
        - (Y, X) for single frame
        - (T, Y, X) for single-channel time series
        - (T, C, Y, X) for multi-channel time series
    """
    path = Path(path)

    # Accept cloud paths where Path.exists() may wrongly return False
    # (online-only stubs).  We check exists() first as a quick pass, but
    # fall through to _resolve_to_local_path which handles the stub case.
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    local_path, tmp_dir = _resolve_to_local_path(path)
    try:
        return _load_tiff_local(local_path, mmap_mode, is_cloud=(tmp_dir is not None))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _declared_tiff_axes(tif, series) -> str:
    """Return only axes backed by file metadata, never tifffile shape guesses.

    For an unannotated three-dimensional array tifffile may report ``SYX`` when
    the first dimension happens to contain three or four planes. That is the same
    size heuristic this module deliberately avoids. OME/ImageJ/MicroManager axes
    and an explicit shaped-TIFF ``metadata={'axes': ...}`` declaration are trusted;
    otherwise an empty string lets :func:`_normalize_axes` apply the metadata-free
    TYX contract.
    """
    axes = (getattr(series, 'axes', '') or '').upper()
    if getattr(tif, 'is_ome', False) or getattr(tif, 'is_imagej', False):
        return axes
    if getattr(tif, 'is_micromanager', False):
        return axes
    try:
        shaped = tif.shaped_metadata or ()
        for metadata in shaped:
            if isinstance(metadata, dict) and metadata.get('axes'):
                return str(metadata['axes']).upper()
    except Exception:
        pass
    return ''


def _load_tiff_local(path: Path, mmap_mode: bool, is_cloud: bool) -> np.ndarray:
    """Internal: load a *local* TIFF file (cloud stubs already resolved)."""
    import warnings

    # Memory-map large files to avoid loading everything into RAM.
    # Skip mmap for cloud copies — they are temp files that may disappear.
    MMAP_THRESHOLD = 500 * 1024 * 1024  # 500 MB
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0

    if not is_cloud and (mmap_mode or file_size > MMAP_THRESHOLD):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = tifffile.memmap(str(path), mode='r')
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes = _declared_tiff_axes(tif, series)
        except Exception:
            pass  # Fall back to normal loading
        else:
            return _normalize_axes(data, axes)

    # Fast full decode for single-file planar 3-D TIFFs (e.g. MicroManager
    # OME-TIFFs, which tifffile.memmap above cannot map): the per-frame-offset
    # reader decodes ~25x faster than tifffile.imread and returns the identical
    # (T, Y, X) native array. 4-D is left to the standard path so it keeps its
    # full (T, C, Y, X) layout. mmap_mode callers are skipped — they asked for a
    # low-RAM memmap, not a materialised array.
    if not is_cloud and not mmap_mode:
        try:
            lazy = open_lazy_stack(path, allow_4d=False)
            if lazy is not None:
                data = np.asarray(lazy[:])
                # open_lazy_stack preserves the file's byte order; normalise to
                # native to match tifffile.imread's contract.
                if not data.dtype.isnative:
                    data = data.astype(data.dtype.newbyteorder('='))
                return data
        except Exception:
            pass  # Fall back to the standard loaders below

    # Suppress tifffile warnings about metadata issues. Read every normal series
    # through its axes declaration, not only ImageJ files: OME and shaped TIFFs can
    # carry non-canonical dimension orders too.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                axes = _declared_tiff_axes(tif, series)
                data = series.asarray()
        except Exception:
            # Fallback for MicroManager files with broken metadata
            # Load pages directly — pre-allocate to avoid 2x memory
            with tifffile.TiffFile(path, _useframes=True) as tif:
                axes = ''
                n_pages = len(tif.pages)
                first_page = tif.pages[0].asarray()
                data = np.empty((n_pages, *first_page.shape), dtype=first_page.dtype)
                data[0] = first_page
                del first_page
                for i in range(1, n_pages):
                    data[i] = tif.pages[i].asarray()
                
                # Try to reshape using ImageJ metadata if available
                data = _reshape_from_imagej_meta(tif, data)
                if data.ndim == 4:
                    axes = 'TCYX'
                elif data.ndim == 3:
                    axes = 'TYX'
                elif data.ndim == 2:
                    axes = 'YX'

    return _normalize_axes(data, axes)


def _normalize_axes(data: np.ndarray, axes: str) -> np.ndarray:
    """Return TIFF data in the canonical ``YX``/``TYX``/``TCYX`` layout.

    A metadata-free three-dimensional TIFF is always a time series (``TYX``),
    regardless of how short its first dimension is. Multichannel data is always
    represented as ``TCYX``. Unknown four-dimensional layouts are rejected rather
    than guessed because their two leading dimensions cannot be distinguished from
    shape alone.

    Explicit ``CYX`` data is accepted as a single-frame multichannel image and is
    promoted to ``TCYX``. ImageJ's singleton ``Z`` axes are removed; a standalone
    ``ZYX`` page series is interpreted as time. Other non-canonical,
    non-singleton axes are unsupported.
    """
    data = np.asanyarray(data)
    axes = (axes or '').strip().upper()

    if not axes:
        if data.ndim == 2:
            axes = 'YX'
        elif data.ndim == 3:
            axes = 'TYX'
        else:
            raise ValueError(
                f"Ambiguous metadata-free {data.ndim}-D image; provide explicit "
                "axes (multichannel data must be TCYX)")
    elif len(axes) != data.ndim:
        raise ValueError(
            f"Axes {axes!r} do not match array shape {tuple(data.shape)}")

    # tifffile labels an unannotated page sequence QYX (or occasionally IYX).
    # That is precisely the metadata-free 3-D case and therefore means TYX here.
    if data.ndim == 3 and axes in ('QYX', 'IYX', 'ZYX'):
        axes = 'TYX'

    # Remove only semantically harmless singleton dimensions (notably the Z=1
    # axis in ImageJ TZCYX). Never flatten a real Z stack into time or channels.
    for idx in range(len(axes) - 1, -1, -1):
        label = axes[idx]
        if label not in 'TCYX':
            if data.shape[idx] != 1:
                raise ValueError(
                    f"Unsupported non-singleton TIFF axis {label!r} in {axes!r}")
            data = np.squeeze(data, axis=idx)
            axes = axes[:idx] + axes[idx + 1:]

    if len(set(axes)) != len(axes) or 'Y' not in axes or 'X' not in axes:
        raise ValueError(f"Unsupported TIFF axes {axes!r}")

    present = set(axes)
    if present == {'Y', 'X'} and len(axes) == 2:
        return np.transpose(data, [axes.index('Y'), axes.index('X')])

    if present == {'T', 'Y', 'X'} and len(axes) == 3:
        return np.transpose(
            data, [axes.index('T'), axes.index('Y'), axes.index('X')])

    if present == {'C', 'Y', 'X'} and len(axes) == 3:
        cyx = np.transpose(
            data, [axes.index('C'), axes.index('Y'), axes.index('X')])
        return cyx[np.newaxis, ...]

    if present == {'T', 'C', 'Y', 'X'} and len(axes) == 4:
        tcyx = np.transpose(data, [
            axes.index('T'), axes.index('C'),
            axes.index('Y'), axes.index('X'),
        ])
        # Single-channel movies use the simpler TYX contract.
        return tcyx[:, 0] if tcyx.shape[1] == 1 else tcyx

    raise ValueError(
        f"Unsupported TIFF axes {axes!r}; expected YX, TYX, CYX, or TCYX")


def _reshape_from_imagej_meta(tif, data: np.ndarray) -> np.ndarray:
    """
    Reshape page-by-page loaded data using ImageJ metadata if available.
    
    When tifffile series parsing fails (e.g. MicroManager files with
    inconsistent metadata), pages are loaded as a flat (N, Y, X) array.
    This function reads the ImageJ description tag to recover the
    correct (T, C, Y, X) shape.
    """
    try:
        if not tif.is_imagej:
            return data
        ijmeta = tif.imagej_metadata
        if ijmeta is None:
            return data
        
        n_channels = ijmeta.get('channels', 1)
        n_frames = ijmeta.get('frames', 1)
        n_slices = ijmeta.get('slices', 1)
        
        if data.ndim < 3:
            return data
        
        n_pages = data.shape[0]
        h, w = data.shape[-2], data.shape[-1]
        expected = n_channels * n_frames * n_slices
        
        if expected == n_pages and expected > 1:
            # ImageJ interleaves as frames × slices × channels
            data = data.reshape(n_frames, n_slices, n_channels, h, w)
            if n_slices == 1:
                data = data[:, 0, :, :, :]  # (T, C, Y, X)
            if n_channels == 1:
                data = data[:, 0, :, :]  # (T, Y, X)
            # Keep the singleton T axis for multichannel data: the canonical
            # representation is TCYX even when there is only one frame.
        
        return data
    except Exception:
        return data  # If anything goes wrong, return raw pages




def split_quadrants_grid(data: np.ndarray,
                          rows: int = 1,
                          cols: int = 2) -> list:
    """
    Split image into a rows × cols grid of sub-images.

    Sub-images are returned in row-major order:
      grid[0] = top-left, grid[1] = top-right, ...
      grid[cols] = second-row-left, etc.

    Args:
        data: Image data with shape (..., Y, X). Works for 2D, 3D (T,Y,X),
              and 4D (T,C,Y,X) arrays.
        rows: Number of rows in the grid (1–4).
        cols: Number of columns in the grid (1–4).

    Returns:
        List of sub-images in row-major order (length = rows × cols).
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got rows={rows}, cols={cols}")

    height = data.shape[-2]
    width = data.shape[-1]

    row_edges = [int(round(height * i / rows)) for i in range(rows + 1)]
    col_edges = [int(round(width * j / cols)) for j in range(cols + 1)]

    quadrants = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = row_edges[r], row_edges[r + 1]
            x0, x1 = col_edges[c], col_edges[c + 1]
            quadrants.append(data[..., y0:y1, x0:x1])

    return quadrants


def max_projection(stack: np.ndarray, axis: int = 0,
                   n_frames: Optional[int] = None) -> np.ndarray:
    """
    Compute maximum intensity projection.

    Args:
        stack: Image stack (T, Y, X)
        axis: Axis to project along (default: 0 for time)
        n_frames: If set, only project the first n frames

    Returns:
        Max projection (Y, X) as float64
    """
    if n_frames is not None:
        stack = stack[:n_frames]
    return np.max(stack, axis=axis).astype(np.float64)


def sum_projection(stack: np.ndarray, axis: int = 0,
                   n_frames: Optional[int] = None) -> np.ndarray:
    """
    Compute sum intensity projection.

    Args:
        stack: Image stack (T, Y, X)
        axis: Axis to project along
        n_frames: If set, only sum first n frames

    Returns:
        Sum projection (Y, X)
    """
    if n_frames is not None:
        stack = stack[:n_frames]
    return np.sum(stack, axis=axis, dtype=np.float64)


def mean_projection(stack: np.ndarray, axis: int = 0,
                    n_frames: Optional[int] = None) -> np.ndarray:
    """
    Compute mean (average) intensity projection.

    Args:
        stack: Image stack (T, Y, X)
        axis: Axis to project along (default: 0 for time)
        n_frames: If set, only average the first n frames

    Returns:
        Mean projection (Y, X) as float64
    """
    if n_frames is not None:
        stack = stack[:n_frames]
    return np.mean(stack, axis=axis, dtype=np.float64)


def median_projection(stack: np.ndarray, axis: int = 0,
                      n_frames: Optional[int] = None) -> np.ndarray:
    """
    Compute median intensity projection.

    The median is robust to transient single-frame outliers (e.g. blinking
    fluorophores or hot pixels), making it useful for estimating a stable
    background or a representative single frame from a movie.

    Args:
        stack: Image stack (T, Y, X)
        axis: Axis to project along (default: 0 for time)
        n_frames: If set, only use the first n frames

    Returns:
        Median projection (Y, X) as float64
    """
    if n_frames is not None:
        stack = stack[:n_frames]
    return np.median(stack, axis=axis).astype(np.float64)


def project_stack(stack: np.ndarray, method: str = "mean", axis: int = 0,
                  n_frames: Optional[int] = None) -> np.ndarray:
    """
    Project a stack to a single 2D frame using a named method.

    Convenience dispatcher used by the co-localization module so the GUI can
    pass a method string ("mean", "median", "max", "sum") chosen by the user.

    Args:
        stack: Image stack (T, Y, X). A 2D array is returned unchanged.
        method: One of "mean", "median", "max", "sum" (case-insensitive).
        axis: Axis to project along (default: 0 for time).
        n_frames: If set, only use the first n frames.

    Returns:
        Projected image (Y, X) as float64.

    Raises:
        ValueError: If method is not recognized.
    """
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    m = str(method).strip().lower()
    if m in ("mean", "average", "avg"):
        return mean_projection(arr, axis=axis, n_frames=n_frames)
    if m == "median":
        return median_projection(arr, axis=axis, n_frames=n_frames)
    if m in ("max", "maximum"):
        return max_projection(arr, axis=axis, n_frames=n_frames)
    if m == "sum":
        return sum_projection(arr, axis=axis, n_frames=n_frames)
    raise ValueError(
        f"Unknown projection method '{method}'. "
        "Expected one of: mean, median, max, sum."
    )


def get_tiff_metadata(path: Union[str, Path]) -> dict:
    """
    Extract timing metadata from a TIFF file.
    
    Attempts to read exposure time and frame interval from:
    - OME-XML metadata (ExposureTime)
    - ImageJ metadata (finterval — a frame INTERVAL, in seconds)
    - MicroManager Summary (Interval_ms)
    
    Args:
        path: Path to the TIFF file.  Cloud-storage paths are supported;
              online-only stub files are downloaded automatically.
        
    Returns:
        Dict with:
            'n_frames': int or None — actual number of frames in file
            'exposure_ms': float or None — camera exposure per frame (ms)
            'interval_ms': float or None — time between frame starts (ms);
                           0 means continuous acquisition (effective = exposure)
    """
    result = {'n_frames': None, 'exposure_ms': None, 'interval_ms': None}
    path = Path(path)
    if not path.exists():
        return result

    local_path, tmp_dir = _resolve_to_local_path(path)
    try:
        return _get_tiff_metadata_local(local_path)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _get_tiff_metadata_local(path: Path) -> dict:
    """Internal: extract metadata from a *local* TIFF file."""
    import warnings
    import xml.etree.ElementTree as ET

    result = {'n_frames': None, 'exposure_ms': None, 'interval_ms': None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with tifffile.TiffFile(path) as tif:
                # --- Frame count ---
                # Try series shape first
                if len(tif.series) > 0:
                    s = tif.series[0]
                    axes = _declared_tiff_axes(tif, s)
                    if 'T' in axes:
                        result['n_frames'] = s.shape[axes.index('T')]
                    elif axes in ('QYX', 'IYX', 'ZYX'):
                        result['n_frames'] = s.shape[0]
                    elif 'C' in axes and set(axes) == {'C', 'Y', 'X'}:
                        result['n_frames'] = 1
                    elif 'Z' in axes and s.shape[axes.index('Z')] > 1:
                        result['n_frames'] = s.shape[axes.index('Z')]
                    elif not axes and len(s.shape) == 3:
                        result['n_frames'] = s.shape[0]
                
                # Fallback to ImageJ metadata
                if result['n_frames'] is None and tif.is_imagej:
                    ijm = tif.imagej_metadata or {}
                    result['n_frames'] = ijm.get('frames') or ijm.get('slices')
                
                # Fallback to page count
                if result['n_frames'] is None:
                    result['n_frames'] = len(tif.pages)
                
                # --- Exposure time (camera integration, ms) ---
                # OME defaults time quantities to seconds; explicit units may
                # differ for exposure and frame interval in the same document.
                if tif.is_ome:
                    try:
                        root = ET.fromstring(tif.ome_metadata)
                        units_ms = {'s': 1000.0, 'ms': 1.0, 'µs': 0.001,
                                    'μs': 0.001, 'us': 0.001, 'ns': 1e-6,
                                    'ps': 1e-9, 'min': 60000.0, 'h': 3600000.0}
                        for node in root.iter():
                            for attribute, key in (
                                ('ExposureTime', 'exposure_ms'),
                                ('TimeIncrement', 'interval_ms'),
                            ):
                                if result[key] is not None or attribute not in node.attrib:
                                    continue
                                factor = units_ms.get(node.get(attribute + 'Unit', 's'))
                                if factor is not None:
                                    value = float(node.get(attribute)) * factor
                                    if np.isfinite(value) and value > 0:
                                        result[key] = value
                    except (ET.ParseError, TypeError, ValueError):
                        pass
                
                # 2. ImageJ finterval (seconds) is the FRAME INTERVAL, i.e. the
                #    time between frame starts — not the camera exposure. This
                #    dict distinguishes the two, so it belongs in interval_ms;
                #    storing it as exposure_ms reported an exposure the file
                #    never recorded and left the real interval unknown.
                if result['interval_ms'] is None and tif.is_imagej:
                    ijm = tif.imagej_metadata or {}
                    finterval = ijm.get('finterval')
                    if finterval and float(finterval) > 0:
                        result['interval_ms'] = float(finterval) * 1000.0
                
                # --- Frame interval (cycle time between frames, ms) ---
                # MicroManager Summary Interval_ms
                if tif.is_micromanager and result['interval_ms'] is None:
                    try:
                        mm = tif.micromanager_metadata or {}
                        summary = mm.get('Summary', {})
                        # A MISSING key means unknown, not "continuous": the old
                        # default of 0 asserted a zero cycle time the file never
                        # stated. Absent stays None.
                        if 'Interval_ms' in summary:
                            interval = summary.get('Interval_ms')
                            if interval is not None:
                                result['interval_ms'] = float(interval)
                    except Exception:
                        pass
    except Exception:
        pass

    return result


class LazyTiffStack:
    """Lazy, frame-indexable view of a single-file, uncompressed, planar TIFF /
    OME-TIFF (Picasso-style).

    Frames are materialised only on access, straight from a byte-level memmap at
    each page's stored data offset — so localizing a small ROI of a multi-GB
    movie reads *only* that ROI from disk, and there is no slow up-front OME
    decode. Exposes the minimal array-like surface ``smlm_localize`` needs:
    ``ndim``, ``shape``, ``dtype``, ``len()``, integer indexing (one frame) and
    slice indexing (a materialised chunk of frames). An optional ROI restricts
    every returned frame to ``[y0:y1, x0:x1]``.

    Build instances via :func:`open_frame_source`.  Files that cannot be mapped
    directly are represented by :class:`TiffPageStack`, which decodes only the
    bounded chunk requested by the caller.
    """

    def __init__(self, path, offsets, n, h, w, dtype):
        # Whole file mapped as raw bytes; per-frame offsets index into it. uint8
        # avoids alignment issues with odd page offsets (MicroManager IFD gaps).
        self._mm = np.memmap(str(path), dtype=np.uint8, mode='r')
        self._offsets = offsets
        self._h, self._w = int(h), int(w)
        self._dtype = np.dtype(dtype)
        self._fb = self._h * self._w * self._dtype.itemsize
        self.full_shape = (int(n), self._h, self._w)
        self._roi = (0, self._h, 0, self._w)  # y0, y1, x0, x1
        self.ndim = 3

    def set_roi(self, y0, y1, x0, x1):
        """Restrict returned frames to ``[y0:y1, x0:x1]`` (absolute pixels)."""
        self._roi = (int(y0), int(y1), int(x0), int(x1))

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        y0, y1, x0, x1 = self._roi
        return (self.full_shape[0], y1 - y0, x1 - x0)

    def __len__(self):
        return self.full_shape[0]

    def read_chunk(self, start, stop):
        """Materialise exactly ``[start:stop]`` as one contiguous frame chunk.

        Localization uses this explicit API so its bounded scheduler controls
        every allocation, without materialising an eager whole-window slice
        before jobs are submitted.
        """
        start = max(0, min(int(start), self.full_shape[0]))
        stop = max(start, min(int(stop), self.full_shape[0]))
        return self[slice(start, stop)]

    def close(self):
        """Release the underlying file mapping when the owning worker is done."""
        mm = getattr(self, '_mm', None)
        mmap_obj = getattr(mm, '_mmap', None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except (BufferError, OSError):
                # Outstanding frame views keep the mapping valid until their
                # references are released; normal GC will close it afterward.
                pass

    def _frame(self, i):
        o = self._offsets[i]
        full = np.frombuffer(self._mm[o:o + self._fb],
                             dtype=self._dtype).reshape(self._h, self._w)
        y0, y1, x0, x1 = self._roi
        return full[y0:y1, x0:x1]

    def __getitem__(self, key):
        # Tuple key (frame-axis + spatial): materialise the requested frame(s),
        # then apply the remaining numpy indexing — so stack[t, y0:y1, x0:x1],
        # stack[t, c] (no-op here, already single-channel), and stack[:, ...]
        # all work and only the requested frames are read from disk.
        if isinstance(key, tuple):
            base = self.__getitem__(key[0])
            rest = key[1:]
            if not rest:
                return base
            if isinstance(key[0], slice):
                return base[(slice(None),) + rest]
            return base[rest]
        if isinstance(key, slice):
            idx = range(*key.indices(self.full_shape[0]))
            if len(idx) == 0:
                _, rh, rw = self.shape
                return np.empty((0, rh, rw), self._dtype)
            # np.stack copies — materialising (and reading from disk) only the
            # requested frames' ROI region.
            return np.stack([self._frame(i) for i in idx])
        return self._frame(int(key))


class TiffPageStack:
    """Bounded chunk reader for compressed/tiled planar TIFF series.

    Unlike :class:`LazyTiffStack`, pages are decoded on demand rather than
    memory-mapped.  The public surface is intentionally identical, so callers
    can keep one rolling scheduler for uncompressed and compressed movies.
    Only planar YX pages are accepted; ambiguous metadata-free 4-D data is
    rejected by the same canonical-axis contract as :func:`load_tiff`.
    """

    def __init__(self, path, detect_channel=0):
        self._tif = tifffile.TiffFile(str(path))
        self._read_lock = RLock()
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_cache_size = 0
        self._series = self._tif.series[0]
        shape = tuple(int(v) for v in self._series.shape)
        axes = _declared_tiff_axes(self._tif, self._series)
        if not axes:
            if len(shape) == 2:
                axes = 'YX'
            elif len(shape) == 3:
                axes = 'TYX'
            else:
                self.close()
                raise ValueError(
                    "Ambiguous metadata-free TIFF; multichannel data requires "
                    "explicit TCYX axes")
        axes = axes.upper()
        if len(axes) != len(shape) or not axes.endswith('YX'):
            self.close()
            raise ValueError(
                f"Page-streaming requires planar YX TIFF data; got {axes!r}")
        if axes in ('QYX', 'IYX', 'ZYX'):
            axes = 'TYX'

        leading_axes = axes[:-2]
        leading_shape = shape[:-2]
        for label, size in zip(leading_axes, leading_shape):
            if label not in 'TC' and size != 1:
                self.close()
                raise ValueError(
                    f"Unsupported non-singleton TIFF axis {label!r}")
        if len(set(label for label in leading_axes if label in 'TC')) != sum(
                label in 'TC' for label in leading_axes):
            self.close()
            raise ValueError(f"Duplicate time/channel axes in {axes!r}")

        self._leading_axes = leading_axes
        self._leading_shape = leading_shape
        self._n_frames = (leading_shape[leading_axes.index('T')]
                          if 'T' in leading_axes else 1)
        self._n_channels = (leading_shape[leading_axes.index('C')]
                            if 'C' in leading_axes else 1)
        channel = max(0, int(detect_channel) - 1) if detect_channel else 0
        self._channel = channel if channel < self._n_channels else 0
        expected_pages = int(np.prod(leading_shape, dtype=np.int64)) or 1
        self._pages = self._series.pages
        if len(self._pages) != expected_pages:
            self.close()
            raise ValueError(
                "TIFF series does not expose one independently decodable page "
                "per time/channel plane")

        self._h, self._w = shape[-2:]
        self._dtype = np.dtype(self._pages[0].dtype)
        self.full_shape = (int(self._n_frames), self._h, self._w)
        self._roi = (0, self._h, 0, self._w)
        self.ndim = 3

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        y0, y1, x0, x1 = self._roi
        return (self.full_shape[0], y1 - y0, x1 - x0)

    def __len__(self):
        return self.full_shape[0]

    def set_roi(self, y0, y1, x0, x1):
        y0, y1 = int(y0), int(y1)
        x0, x1 = int(x0), int(x1)
        if not (0 <= y0 < y1 <= self._h and 0 <= x0 < x1 <= self._w):
            raise ValueError("ROI lies outside the TIFF frame")
        self._roi = (y0, y1, x0, x1)

    def set_frame_cache_size(self, size):
        """Keep at most ``size`` decoded full frames for interactive preview."""
        with self._read_lock:
            self._frame_cache_size = max(0, int(size))
            while len(self._frame_cache) > self._frame_cache_size:
                self._frame_cache.popitem(last=False)
            if self._frame_cache_size == 0:
                self._frame_cache.clear()

    def _page_index(self, frame):
        indices = []
        for label in self._leading_axes:
            if label == 'T':
                indices.append(int(frame))
            elif label == 'C':
                indices.append(self._channel)
            else:
                indices.append(0)
        if not indices:
            return 0
        return int(np.ravel_multi_index(tuple(indices), self._leading_shape))

    def _frame(self, frame):
        frame = int(frame)
        if frame < 0:
            frame += self._n_frames
        if not 0 <= frame < self._n_frames:
            raise IndexError(frame)
        with self._read_lock:
            decoded = self._frame_cache.get(frame)
            if decoded is not None:
                self._frame_cache.move_to_end(frame)
            else:
                decoded = np.asarray(
                    self._pages[self._page_index(frame)].asarray())
                if self._frame_cache_size:
                    self._frame_cache[frame] = decoded
                    self._frame_cache.move_to_end(frame)
                    while len(self._frame_cache) > self._frame_cache_size:
                        self._frame_cache.popitem(last=False)
        if decoded.shape != (self._h, self._w):
            raise ValueError(
                f"Decoded TIFF page has unexpected shape {decoded.shape}")
        y0, y1, x0, x1 = self._roi
        return decoded[y0:y1, x0:x1]

    def read_chunk(self, start, stop):
        start = max(0, min(int(start), self._n_frames))
        stop = max(start, min(int(stop), self._n_frames))
        if stop == start:
            _, height, width = self.shape
            return np.empty((0, height, width), dtype=self._dtype)
        return np.stack([self._frame(frame) for frame in range(start, stop)])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            base = self.__getitem__(key[0])
            remainder = key[1:]
            if not remainder:
                return base
            if isinstance(key[0], slice):
                return base[(slice(None),) + remainder]
            return base[remainder]
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n_frames)
            if step == 1:
                return self.read_chunk(start, stop)
            indices = range(start, stop, step)
            if len(indices) == 0:
                _, height, width = self.shape
                return np.empty((0, height, width), dtype=self._dtype)
            return np.stack([self._frame(frame) for frame in indices])
        return self._frame(key)

    def close(self):
        with self._read_lock:
            self._frame_cache.clear()
            tif = getattr(self, '_tif', None)
            self._tif = None
            if tif is not None:
                tif.close()


def open_lazy_stack(path, detect_channel=0, allow_4d=True):
    """Return a :class:`LazyTiffStack` for a single-file, uncompressed, planar
    TIFF/OME-TIFF, or ``None`` when the file is not laid out compatibly
    (compressed/tiled, multi-file OME series, >4-D), so the caller can fall back
    to ``tifffile.imread``.

    Supports both 3-D ``(T, Y, X)`` and 4-D ``(T, C, Y, X)`` series. For 4-D the
    channel selected by ``detect_channel`` (1-based; ``0`` → first channel,
    matching the movie-list ``detect_channel`` convention) is read, so only that
    channel's planes (and, with an ROI, only the cropped region) are touched.

    ``allow_4d=False`` restricts the reader to 3-D series (returns ``None`` for
    4-D). Callers that need the FULL multi-channel stack — e.g. ``load_tiff``,
    which must return all channels — pass this so a 4-D file is never silently
    reduced to a single channel."""
    try:
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            shape = series.shape
            ndim = len(shape)
            axes = _declared_tiff_axes(tf, series)
            if ndim not in (3, 4) or (ndim == 4 and not allow_4d):
                return None  # >4-D (or 4-D when disallowed) → imread fallback
            pages = tf.pages
            p0 = pages[0]
            if int(getattr(p0, 'compression', 1)) != 1:
                return None  # compressed
            if len(p0.dataoffsets) != 1 or len(p0.databytecounts) != 1:
                return None  # tiled / multi-strip
            h, w = int(p0.shape[0]), int(p0.shape[1])
            # tifffile reports page.dtype in NATIVE byte order, discarding the
            # file's endianness. Re-stamp it with the file's true byte order
            # (tf.byteorder) so the raw memmapped bytes are interpreted
            # correctly — big-endian MicroManager movies would otherwise decode
            # to byte-swapped garbage. A non-native dtype also re-arms the
            # byte-order normalisation in smlm_localize._identify_and_extract.
            dt = np.dtype(p0.dtype).newbyteorder(tf.byteorder)
            fb = h * w * np.dtype(dt).itemsize
            if int(p0.databytecounts[0]) != fb:
                return None  # not a plain contiguous plane

            if ndim == 3:
                # QYX/IYX are tifffile's labels for an unannotated page sequence;
                # by contract that is TYX. Explicit channel-first CYX must fall
                # back to the canonical eager loader rather than masquerade as T.
                if axes not in ('', 'TYX', 'QYX', 'IYX', 'ZYX'):
                    return None
                n_frames = int(shape[0])
                if len(pages) != n_frames:
                    return None  # multi-file OME series — offsets span files
                offsets = [int(p.dataoffsets[0]) for p in pages]
            else:  # ndim == 4: (frames, channels, Y, X)
                # The t*d1+ci page formula below is only valid for the canonical
                # MicroManager frames-major / channel-as-axis-1 interleave. Bail
                # on any other dimension order (→ imread fallback) so we never
                # read the wrong channel or frame.
                if axes != 'TCYX':
                    return None
                d0, d1 = int(shape[0]), int(shape[1])
                if len(pages) != d0 * d1:
                    return None  # multi-file OME / unexpected page count
                ci = max(0, detect_channel - 1) if detect_channel > 0 else 0
                if ci >= d1:
                    ci = 0
                # Pages flatten the leading (frames, channels) axes row-major, so
                # the plane for (t, ci) is page t*d1 + ci (verified on
                # MicroManager OME). Take only the selected channel's planes.
                offsets = [int(pages[t * d1 + ci].dataoffsets[0]) for t in range(d0)]
                n_frames = d0
        # Guard against truncated / cloud-stub files: the last frame's bytes
        # must actually be present on disk, else memmap would read garbage.
        if max(offsets) + fb > Path(str(path)).stat().st_size:
            return None
        return LazyTiffStack(path, offsets, n_frames, h, w, dt)
    except Exception:
        return None


def open_frame_source(path, detect_channel=0):
    """Open a bounded, single-channel frame source for a TIFF movie.

    Uncompressed planar files use a zero-copy :class:`LazyTiffStack`.  Compressed
    or tiled files use :class:`TiffPageStack` and decode pages only as each
    scheduler job requests them, without materialising the complete movie.

    The returned object owns an open file/mapping and must be closed by its
    caller.  Ambiguous axis layouts are rejected rather than guessed.
    """
    lazy = open_lazy_stack(path, detect_channel=detect_channel)
    if lazy is not None:
        return lazy
    return TiffPageStack(path, detect_channel=detect_channel)


