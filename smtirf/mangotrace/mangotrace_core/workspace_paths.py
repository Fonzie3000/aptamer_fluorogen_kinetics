"""Local retained outputs which must outlive an analysis worker or session.

These are intentionally separate from disposable caches: session archives may
refer to corrected movies here, and batch checkpoints protect earlier files.
"""

from pathlib import Path
import shutil
import uuid


def configure_startup_environment() -> None:
    """Choose writable local caches before importing numerical/GUI libraries.

    Explicit environment settings are left intact. A read-only installation
    falls back to the user's cache, then a temporary cache directory.
    """
    import os
    import tempfile

    def cache_roots():
        yield Path(__file__).resolve().parents[1] / '_workspace' / 'cache'
        if os.environ.get('LOCALAPPDATA'):
            yield Path(os.environ['LOCALAPPDATA']) / 'MangoTrace' / 'cache'
        try:
            yield Path(tempfile.gettempdir()) / 'MangoTrace' / 'cache'
        except OSError:
            return

    for variable, subdirectory in (
        ('NUMBA_CACHE_DIR', 'numba'),
        ('CUPY_CACHE_DIR', 'cupy'),
        ('CUDA_CACHE_PATH', 'cuda'),
        ('TORCH_HOME', 'torch'),
        ('MPLCONFIGDIR', 'matplotlib'),
    ):
        if variable in os.environ:
            continue
        for root in cache_roots():
            destination = root / subdirectory
            try:
                destination.mkdir(parents=True, exist_ok=True)
                # mkdir alone succeeds for an existing read-only directory.
                # Probe only a new owned temporary file; never touch user files.
                with tempfile.TemporaryFile(prefix='.mangotrace-cache-', dir=destination):
                    pass
            except OSError:
                continue
            os.environ[variable] = str(destination)
            break


def support_directory() -> Path:
    root = Path(__file__).resolve().parents[1] / '_workspace'
    root.mkdir(parents=True, exist_ok=True)
    return root


def temporary_directory(category: str) -> Path:
    if not category.isidentifier():
        raise ValueError('Invalid temporary directory category')
    path = support_directory() / 'tmp' / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_disk_space(path, needed_bytes, *, reserve_bytes=16 * 1024**2):
    """Reject a known oversized write before replacing or allocating results."""
    target = Path(path).resolve()
    while not target.exists():
        target = target.parent
    needed = max(0, int(needed_bytes)) + int(reserve_bytes)
    free = shutil.disk_usage(target).free
    if free < needed:
        raise OSError(
            f'Insufficient free disk space at {target}: '
            f'{needed / 1024**3:.2f} GiB required including reserve, '
            f'{free / 1024**3:.2f} GiB available. Existing results were kept.')


def retained_output_directory(category: str) -> Path:
    if category not in {'correction', 'pipeline'}:
        raise ValueError(f'Unknown retained-output category: {category}')
    root = support_directory() / 'retained'
    directory = root / category / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory
