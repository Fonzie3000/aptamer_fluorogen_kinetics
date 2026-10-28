"""Standalone copy of the MangoTrace numerical core used by run_mangotrace.py.

Modules: image_io (TIFF reading), gausslq + smlm_localize (detection, Gaussian
fitting, DBSCAN, filtering), aim_drift + aim_picasso + aim_movie (drift estimate and
movie correction), trace_extractor (annular traces), trace_cpd (change-point
idealization), trace_events (event extraction), data_types, workspace_paths,
dependencies. No GUI code. Platform-specific branches inside these
modules are guarded (Windows memory queries fall back to psutil elsewhere).
"""
from .workspace_paths import configure_startup_environment as _configure

# Redirects numba / joblib caches to a writable per-user location on every OS so a
# read-only install or a synced folder (OneDrive, Dropbox) does not break compilation.
_configure()

__version__ = "1.0"
