"""User-facing guidance for required and optional dependencies.

MangoTrace uses required native wheels plus optional GPU packages that a
plain ``pip install`` cannot provide. Their UI and headless error messages live
here so install guidance cannot drift between call sites.

GUI-independent by design (``core/`` never imports PySide6) so the same strings
back both headless ``RuntimeError`` exceptions and Qt dialogs/tooltips.
"""

# Vendored wheel paths - keep in lockstep with ./wheels and requirements.txt.
DBSCAN_WHEEL = "dbscan"
GPUFIT_WHEEL = "wheels/pyGpufit-1.2.0-py2.py3-none-win_amd64.whl"

# CUDA build MangoTrace is validated against. PyTorch does not publish its CUDA
# Windows binaries to PyPI, so a plain `pip install torch` yields a CPU-only
# wheel there and this index URL is the only way to get CUDA.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_COMMAND = f"pip install torch==2.12.0 --index-url {TORCH_CUDA_INDEX}"


def missing_dependency_message(summary: str, install_hint: str) -> str:
    """Combine a dependency summary with actionable install guidance."""
    return f"{summary}\n\n{install_hint}"


DBSCAN_MISSING_MSG = missing_dependency_message(
    "SMLM clustering requires the 'dbscan' package (wangyiqiu/dbscan-python), "
    "which is not installed. It is the only clustering path - there is no "
    "local-maxima fallback.",
    "Install it:\n"
    f"    pip install {DBSCAN_WHEEL}\n"
    "(on Windows, build wangyiqiu/dbscan-python from source if pip fails).",
)

GPUFIT_MISSING_MSG = missing_dependency_message(
    "pygpufit is not installed - GPU fitting unavailable (the CPU fit is used "
    "instead).",
    "Install the bundled wheel:\n"
    f"    pip install {GPUFIT_WHEEL}\n"
    "(not on PyPI; other platforms must build Gpufit from source). "
    "See wheels/README.md.",
)

TORCH_NO_CUDA_MSG = missing_dependency_message(
    "GPU not available - the HMM will be fitted on the CPU.",
    "PyTorch does not publish its CUDA Windows builds to PyPI, so a plain\n"
    "'pip install torch' gives the CPU-only wheel. Install the CUDA build:\n"
    f"    {TORCH_CUDA_COMMAND}\n"
    "(cu130 = CUDA 13.0; see requirements.txt for other toolkits.)",
)
