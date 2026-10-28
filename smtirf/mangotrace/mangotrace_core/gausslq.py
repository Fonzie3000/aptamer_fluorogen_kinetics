"""
Least-squares 2D Gaussian fitting for single-molecule localization.

This is a self-contained adaptation of the Picasso ``gausslq`` method
(Schnitzbauer/Strauss, Jungmann Lab) ported natively into MangoTrace — Picasso is
NOT a runtime dependency. Each detected spot (a small square image patch) is
fitted with an elliptical 2D Gaussian by non-linear least squares — a numba
Levenberg-Marquardt solver with an analytic Jacobian, backed by
``scipy.optimize.leastsq`` as the divergence-guard reference:

    spot(i, j) ≈ photons · Gx(j; x, sx) · Gy(i; y, sy) + bg

The model is separable (outer product of two 1D Gaussians), so residuals are
cheap to evaluate. Parameters are initialized from the spot's center of mass
and second moments. Theoretical localization precision (lpx, lpy) follows
Mortensen et al., Nat. Methods 2010.

Returned parameter order per spot: ``[x, y, photons, bg, sx, sy]`` (offsets
relative to the patch centre for x, y).
"""

from __future__ import annotations

from typing import Callable, Optional

import numba
import numpy as np
import pandas as pd
from scipy import optimize


@numba.jit(nopython=True, nogil=True)
def _gaussian(mu: float, sigma: float, grid: np.ndarray) -> np.ndarray:
    """1D Gaussian PDF on ``grid``."""
    norm = 0.3989422804014327 / sigma  # 1 / sqrt(2*pi) / sigma
    return norm * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


@numba.jit(nopython=True, nogil=True)
def _sum_and_center_of_mass(spot: np.ndarray, size: int):
    """Total intensity and intensity-weighted centroid of a 2D spot."""
    y = 0.0
    x = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            y += spot[i, j] * i
            x += spot[i, j] * j
            _sum_ += spot[i, j]
    if _sum_ <= 0.0:
        # Degenerate (flat) spot: fall back to the geometric centre.
        return 0.01, (size - 1) / 2.0, (size - 1) / 2.0
    y /= _sum_
    x /= _sum_
    return _sum_, y, x


@numba.jit(nopython=True, nogil=True)
def _initial_sigmas(spot: np.ndarray, y: float, x: float, sum: float, size: int):
    """Initial sx, sy from the second moments of the (bg-subtracted) spot."""
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    for i in range(size):
        for j in range(size):
            sum_deviation_y += spot[i, j] * (i - y) ** 2
            sum_deviation_x += spot[i, j] * (j - x) ** 2
    sy = np.sqrt(np.abs(sum_deviation_y) / sum)
    sx = np.sqrt(np.abs(sum_deviation_x) / sum)
    if sx <= 0.0:
        sx = 1.0
    if sy <= 0.0:
        sy = 1.0
    return sy, sx


@numba.jit(nopython=True, nogil=True)
def _initial_parameters(spot: np.ndarray, size: int, size_half: int) -> np.ndarray:
    """Initialize [x, y, photons, bg, sx, sy] for the least-squares fit."""
    theta = np.zeros(6, dtype=np.float32)
    theta[3] = np.min(spot)              # bg
    spot_without_bg = spot - theta[3]
    sum_, theta[1], theta[0] = _sum_and_center_of_mass(spot_without_bg, size)
    theta[2] = max(1.0, sum_)            # photons
    theta[5], theta[4] = _initial_sigmas(spot - theta[3], theta[1], theta[0], sum_, size)
    theta[0:2] -= size_half             # centre x, y on the patch middle
    return theta


@numba.jit(nopython=True, nogil=True)
def _outer(a: np.ndarray, b: np.ndarray, size: int, model: np.ndarray,
           n: float, bg: float) -> None:
    """model[i, j] = n * a[i] * b[j] + bg."""
    for i in range(size):
        for j in range(size):
            model[i, j] = n * a[i] * b[j] + bg


@numba.jit(nopython=True, nogil=True)
def _compute_model(theta: np.ndarray, grid: np.ndarray, size: int,
                   model_x: np.ndarray, model_y: np.ndarray,
                   model: np.ndarray) -> np.ndarray:
    """Separable elliptical 2D Gaussian model from theta."""
    model_x[:] = _gaussian(theta[0], theta[4], grid)
    model_y[:] = _gaussian(theta[1], theta[5], grid)
    _outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@numba.jit(nopython=True, nogil=True)
def _compute_residuals(theta: np.ndarray, spot: np.ndarray, grid: np.ndarray,
                       size: int, model_x: np.ndarray, model_y: np.ndarray,
                       model: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Flattened (observed - model) residual vector for ``leastsq``."""
    _compute_model(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Native LM solver: numba-jitted Levenberg–Marquardt with an analytic Jacobian.
#
# This is the production fit path:
#   • analytic Jacobian for the separable elliptical Gaussian (one model and one
#     Jacobian evaluation per iteration);
#   • a self-contained LM step in pure nopython/nogil numba, so calling
#     ``fit_spots`` from N worker threads fits on N cores in true parallel — the
#     GIL is released for the whole batch, exactly like
#     ``core.smlm_localize.identify_in_image``;
#   • physical bounds (centre inside the patch, positive widths/photons) so a
#     few low-SNR spots cannot run away the way an unbounded fit can.
# Same model, same ``[x, y, photons, bg, sx, sy]`` output, and the same grid /
# coordinate convention as the ``scipy.optimize.leastsq`` reference path below,
# so locs_from_fits and build_locs_dataframe consume either interchangeably.
# ─────────────────────────────────────────────────────────────────────────────

# Convergence is governed by the relative SSE improvement (ftol). The |Δx|+|Δy|
# criterion (xtol) is disabled by default (0.0): x, y lock in well before the
# photon count and PSF widths settle, so stopping on position change alone leaves
# the fit short of the least-squares minimum (measurably higher SSE than scipy).
_LM_MAXITER = 50
_LM_FTOL = 1e-3      # relative SSE improvement to declare convergence
_LM_XTOL = 0.0       # |Δx| + |Δy| (px) early-stop; 0 disables (rely on ftol)
_LM_LAMBDA0 = 1.0    # initial LM damping; trust-region-like, annealed each step


@numba.jit(nopython=True, nogil=True)
def _solve6(H, g, out) -> bool:
    """Solve the 6x6 system H·x = g by Gaussian elimination with partial
    pivoting. ``H`` and ``g`` are overwritten; the solution is written to
    ``out``. Returns False if the system is (numerically) singular."""
    n = 6
    for c in range(n):
        p = c
        maxv = abs(H[c, c])
        for r in range(c + 1, n):
            v = abs(H[r, c])
            if v > maxv:
                maxv = v
                p = r
        if p != c:
            for k in range(n):
                tmp = H[c, k]; H[c, k] = H[p, k]; H[p, k] = tmp
            tmp = g[c]; g[c] = g[p]; g[p] = tmp
        piv = H[c, c]
        if abs(piv) < 1e-30:
            return False
        for r in range(c + 1, n):
            f = H[r, c] / piv
            if f != 0.0:
                for k in range(c, n):
                    H[r, k] -= f * H[c, k]
                g[r] -= f * g[c]
    for c in range(n - 1, -1, -1):
        s = g[c]
        for k in range(c + 1, n):
            s -= H[c, k] * out[k]
        out[c] = s / H[c, c]
    return True


@numba.jit(nopython=True, nogil=True)
def _fit_one(spot, grid, size, size_half, out, maxiter, ftol, xtol, lam0,
             H, g, Hd, gd, delta, jac,
             gx, gy, dgx_dx, dgx_ds, dgy_dy, dgy_ds):
    """Levenberg–Marquardt fit of one (size, size) spot. Writes the optimized
    ``[x, y, photons, bg, sx, sy]`` (x, y are offsets from the patch centre) to
    ``out`` (length 6). The remaining arguments are caller-owned scratch buffers
    reused across spots to avoid per-spot allocation."""
    theta0 = _initial_parameters(spot, size, size_half)
    x = theta0[0]; y = theta0[1]; N = theta0[2]
    b = theta0[3]; sx = theta0[4]; sy = theta0[5]

    lim = (size - 1) / 2.0           # centre must stay within the patch
    smin = 0.3; smax = size * 1.0    # PSF width bounds (px)
    if sx < smin: sx = smin
    if sx > smax: sx = smax
    if sy < smin: sy = smin
    if sy > smax: sy = smax
    if x < -lim: x = -lim
    if x > lim: x = lim
    if y < -lim: y = -lim
    if y > lim: y = lim

    c0 = 0.3989422804014327          # 1 / sqrt(2*pi)
    lam = lam0

    # initial SSE
    inv_sx = 1.0 / sx; inv_sy = 1.0 / sy
    Ax = c0 * inv_sx; Ay = c0 * inv_sy
    for j in range(size):
        ex = (grid[j] - x) * inv_sx
        gx[j] = Ax * np.exp(-0.5 * ex * ex)
    for i in range(size):
        ey = (grid[i] - y) * inv_sy
        gy[i] = Ay * np.exp(-0.5 * ey * ey)
    sse = 0.0
    for i in range(size):
        for j in range(size):
            r = spot[i, j] - (N * gy[i] * gx[j] + b)
            sse += r * r

    for _it in range(maxiter):
        inv_sx = 1.0 / sx; inv_sy = 1.0 / sy
        Ax = c0 * inv_sx; Ay = c0 * inv_sy
        for j in range(size):
            ex = (grid[j] - x) * inv_sx
            val = Ax * np.exp(-0.5 * ex * ex)
            gx[j] = val
            dgx_dx[j] = val * ex * inv_sx               # ∂Gx/∂x
            dgx_ds[j] = val * (ex * ex - 1.0) * inv_sx  # ∂Gx/∂sx
        for i in range(size):
            ey = (grid[i] - y) * inv_sy
            val = Ay * np.exp(-0.5 * ey * ey)
            gy[i] = val
            dgy_dy[i] = val * ey * inv_sy               # ∂Gy/∂y
            dgy_ds[i] = val * (ey * ey - 1.0) * inv_sy  # ∂Gy/∂sy

        for a in range(6):
            g[a] = 0.0
            for cc in range(6):
                H[a, cc] = 0.0
        # normal equations: H = JᵀJ, g = Jᵀr  (r = data − model)
        for i in range(size):
            for j in range(size):
                r = spot[i, j] - (N * gy[i] * gx[j] + b)
                jac[0] = N * gy[i] * dgx_dx[j]   # ∂m/∂x
                jac[1] = N * gx[j] * dgy_dy[i]   # ∂m/∂y
                jac[2] = gy[i] * gx[j]           # ∂m/∂N
                jac[3] = 1.0                     # ∂m/∂bg
                jac[4] = N * gy[i] * dgx_ds[j]   # ∂m/∂sx
                jac[5] = N * gx[j] * dgy_ds[i]   # ∂m/∂sy
                for a in range(6):
                    g[a] += jac[a] * r
                    for cc in range(a, 6):
                        H[a, cc] += jac[a] * jac[cc]
        for a in range(6):
            for cc in range(a):
                H[a, cc] = H[cc, a]

        improved = False
        for _try in range(8):
            # (H + λ·diag(H))·δ = g   — Marquardt damping
            for a in range(6):
                gd[a] = g[a]
                for cc in range(6):
                    Hd[a, cc] = H[a, cc]
                Hd[a, a] = H[a, a] * (1.0 + lam) + 1e-12
            if not _solve6(Hd, gd, delta):
                lam *= 10.0
                continue
            nx = x + delta[0]; ny = y + delta[1]; nN = N + delta[2]
            nb = b + delta[3]; nsx = sx + delta[4]; nsy = sy + delta[5]
            if nsx < smin: nsx = smin
            if nsx > smax: nsx = smax
            if nsy < smin: nsy = smin
            if nsy > smax: nsy = smax
            if nx < -lim: nx = -lim
            if nx > lim: nx = lim
            if ny < -lim: ny = -lim
            if ny > lim: ny = lim
            if nN < 0.0: nN = 0.0

            isx = 1.0 / nsx; isy = 1.0 / nsy
            nAx = c0 * isx; nAy = c0 * isy
            for j in range(size):
                ex = (grid[j] - nx) * isx
                gx[j] = nAx * np.exp(-0.5 * ex * ex)
            for i in range(size):
                ey = (grid[i] - ny) * isy
                gy[i] = nAy * np.exp(-0.5 * ey * ey)
            nsse = 0.0
            for i in range(size):
                for j in range(size):
                    r = spot[i, j] - (nN * gy[i] * gx[j] + nb)
                    nsse += r * r

            if nsse < sse:
                dpar = abs(delta[0]) + abs(delta[1])
                rel = (sse - nsse) / (sse + 1e-30)
                x = nx; y = ny; N = nN; b = nb; sx = nsx; sy = nsy
                sse = nsse
                lam *= 0.5
                if lam < 1e-9:
                    lam = 1e-9
                improved = True
                if rel < ftol or dpar < xtol:
                    out[0] = x; out[1] = y; out[2] = N
                    out[3] = b; out[4] = sx; out[5] = sy
                    return
                break
            else:
                lam *= 10.0
                if lam > 1e8:
                    break
        if not improved:
            break

    out[0] = x; out[1] = y; out[2] = N
    out[3] = b; out[4] = sx; out[5] = sy


@numba.jit(nopython=True, nogil=True)
def _fit_batch(spots, grid, size, size_half, thetas, maxiter, ftol, xtol, lam0):
    """Fit a batch of spots, serially within this call. Runs entirely in
    nopython/nogil mode so concurrent worker threads (one batch each) fit on
    separate cores in true parallel. Do NOT add ``prange`` here: ``fit_spots``
    is already called from smlm_localize's thread pool, and an inner parallel
    region would oversubscribe the cores."""
    H = np.zeros((6, 6)); g = np.zeros(6)
    Hd = np.zeros((6, 6)); gd = np.zeros(6)
    delta = np.zeros(6); jac = np.zeros(6)
    gx = np.zeros(size); gy = np.zeros(size)
    dgx_dx = np.zeros(size); dgx_ds = np.zeros(size)
    dgy_dy = np.zeros(size); dgy_ds = np.zeros(size)
    for k in range(spots.shape[0]):
        _fit_one(spots[k], grid, size, size_half, thetas[k],
                 maxiter, ftol, xtol, lam0,
                 H, g, Hd, gd, delta, jac,
                 gx, gy, dgx_dx, dgx_ds, dgy_dy, dgy_ds)


def _fit_spot_leastsq(spot: np.ndarray) -> np.ndarray:
    """Reference scipy.optimize.leastsq fit (the original Picasso gausslq path).

    Used by the divergence guard in :func:`fit_spots`: a native LM fit that ends
    on one of its clamps is non-physical for a single emitter, and this
    unbounded solver recovers the central spot for those few patches.
    Returns ``[x, y, photons, bg, sx, sy]``.
    """
    size = spot.shape[0]
    size_half = int(size / 2)
    grid = (np.arange(size) - (size - 1) / 2.0).astype(np.float32)
    model_x = np.empty(size, dtype=np.float32)
    model_y = np.empty(size, dtype=np.float32)
    model = np.empty((size, size), dtype=np.float32)
    residuals = np.empty((size, size), dtype=np.float32)
    theta0 = _initial_parameters(spot, size, size_half)
    args = (spot, grid, size, model_x, model_y, model, residuals)
    result = optimize.leastsq(
        _compute_residuals, theta0, args=args, ftol=1e-2, xtol=1e-2
    )
    return result[0]


def fit_spots(
    spots: np.ndarray,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Fit an array of spots, shape (n_spots, size, size), with the native LM
    solver (analytic Jacobian).

    Because the inner fit is nopython/nogil, calling this from several worker
    threads (as smlm_localize does, one frame-chunk per thread) fits spots on
    multiple cores in true parallel.

    Divergence guard: on real (noisy / neighbour-contaminated) spots the bounded
    LM can settle in a degenerate basin — centre pushed to the patch edge and
    sigma to the max ("one wide Gaussian covers the whole patch") — that the
    unbounded scipy reference avoids. Such fits are detected by their clamp hits
    and re-fitted with the scipy reference, so contaminated spots are localized by
    the unbounded solver; well-conditioned spots never hit a clamp and keep the
    fast numba path. Verified against scipy on real movies:
    bright isolated spots agree to ~0.016 px median.

    Returns an (n_spots, 6) array of ``[x, y, photons, bg, sx, sy]``.
    """
    n = len(spots)
    theta = np.empty((n, 6), dtype=np.float32)
    if n == 0:
        return theta
    size = spots.shape[1]
    size_half = int(size / 2)
    grid = (np.arange(size) - (size - 1) / 2.0).astype(np.float64)
    spots32 = np.ascontiguousarray(spots, dtype=np.float32)
    thetas = np.empty((n, 6), dtype=np.float64)

    if callable(progress_callback):
        # Fit in blocks so progress can be reported between them; each block is
        # still a single nogil call (GIL released for the duration of the block).
        block = 4096
        for s0 in range(0, n, block):
            s1 = min(s0 + block, n)
            _fit_batch(spots32[s0:s1], grid, size, size_half, thetas[s0:s1],
                       _LM_MAXITER, _LM_FTOL, _LM_XTOL, _LM_LAMBDA0)
            progress_callback(s1 - 1)
    else:
        _fit_batch(spots32, grid, size, size_half, thetas,
                   _LM_MAXITER, _LM_FTOL, _LM_XTOL, _LM_LAMBDA0)

    # ── Divergence guard: re-fit clamp-hitting (degenerate) fits with scipy ──
    # Thresholds mirror the clamps in _fit_one (centre within +/-(size-1)/2,
    # sigma within [0.3, size]). A fit touching any clamp is non-physical for a
    # single emitter; the scipy reference recovers the central spot for those.
    lim = (size - 1) / 2.0
    smin, smax, eps = 0.3, float(size), 1e-3
    deg = (
        (np.abs(thetas[:, 0]) >= lim - eps) | (np.abs(thetas[:, 1]) >= lim - eps)
        | (thetas[:, 4] >= smax - eps) | (thetas[:, 5] >= smax - eps)
        | (thetas[:, 4] <= smin + eps) | (thetas[:, 5] <= smin + eps)
    )
    for i in np.nonzero(deg)[0]:
        thetas[i] = _fit_spot_leastsq(spots32[i])

    theta[:] = thetas
    return theta


def localization_precision(
    photons: np.ndarray, s: np.ndarray, s_orth: np.ndarray,
    bg: np.ndarray, em: bool,
) -> np.ndarray:
    """Theoretical localization precision (Mortensen et al., Nat. Methods 2010)
    for a 2D unweighted Gaussian fit, with the diagonal-covariance correction."""
    s2 = s ** 2
    sa2 = s2 + 1 / 12
    sa = sa2 ** 0.5
    sa_orth2 = s_orth ** 2 + 1 / 12
    sa_orth = sa_orth2 ** 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        v = sa2 * (16 / 9 + (8 * np.pi * sa * sa_orth * bg) / photons) / photons
        if em:
            v = v * 2
        return np.sqrt(v)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: np.ndarray,
    box: int,
    em: bool = False,
) -> pd.DataFrame:
    """Assemble a localizations DataFrame from least-squares fit results.

    Parameters
    ----------
    identifications : pd.DataFrame
        Per-spot identifications with columns 'frame', 'x', 'y',
        'net_gradient' (x, y are the absolute integer pixel centres).
    theta : np.ndarray
        (n_spots, 6) fit results [x, y, photons, bg, sx, sy] with x, y as
        sub-pixel offsets relative to the patch centre.
    box : int
        Fit box size. Used to reject fits whose centre left the patch.
    em : bool
        Whether an EMCCD camera was used (doubles the precision variance).

    Returns
    -------
    pd.DataFrame
        Columns: frame, x, y, photons, sx, sy, bg, lpx, lpy, net_gradient.
    """
    # A fit whose centre left its own box is not a localization of that spot.
    # The divergence guard above re-fits clamp-hitting spots with an UNBOUNDED
    # solver, which can push the centre outside the patch, so the check belongs
    # here rather than relying on the clamps. The GPU path applies the identical
    # constraint before it converts amplitudes to photons, so both fit paths now
    # accept exactly the same localizations.
    half_box = (int(box) - 1) / 2.0
    inside = ((np.abs(theta[:, 0]) <= half_box)
              & (np.abs(theta[:, 1]) <= half_box))
    if not inside.all():
        identifications = identifications[inside]
        theta = theta[inside]

    x = theta[:, 0] + identifications["x"].to_numpy()
    y = theta[:, 1] + identifications["y"].to_numpy()
    locs = build_locs_dataframe(
        frame=identifications["frame"].to_numpy(),
        x=x, y=y,
        photons=theta[:, 2], bg=theta[:, 3],
        sx=theta[:, 4], sy=theta[:, 5],
        net_gradient=identifications["net_gradient"].to_numpy(),
        em=em,
    )
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


# Physical validity of a localization: PSF widths and the integrated photon
# count must be strictly positive, and the fitted centre must sit on the
# detector (x, y >= 0). The fitted BACKGROUND is deliberately NOT screened —
# see drop_nonphysical_locs for why a negative bg is legitimate.
_POSITIVE_LOC_COLS = ("sx", "sy", "photons")   # must be finite and > 0
_NONNEGATIVE_LOC_COLS = ("x", "y")             # must be finite and >= 0


def drop_nonphysical_locs(df: pd.DataFrame) -> pd.DataFrame:
    """Silently drop localizations whose fitted parameters are physically
    impossible.

    On a small fraction of low-SNR spots the least-squares fit settles on
    non-physical values — chiefly the UNBOUNDED GPU fit (pygpufit places no
    bounds on the elliptical-Gaussian parameters): negative PSF widths (which
    even flip the sign of the integrated photon count, since
    ``photons = amp * 2*pi*sx*sy``), non-positive photons, or a centre pushed off
    the detector. Such localizations are meaningless, so they are removed here —
    the single exit shared by the CPU and GPU fit paths — before any downstream
    step (the localization filter/histogram, clustering, linking, trace
    extraction, export) can see them.

    A row is kept only when every present column of ``sx``/``sy``/``photons`` is
    finite and > 0 and every present column of ``x``/``y`` is finite and >= 0;
    missing columns are ignored.

    The fitted BACKGROUND (``bg``) is intentionally NOT screened. bg is a
    least-squares OFFSET, not a photon count: when the true background sits near
    zero (a clean sample, a baseline tuned to the background floor, or a
    background-subtracted movie), a perfectly good dim emitter fits a slightly
    NEGATIVE bg — so a ``bg >= 0`` rule would silently strip roughly half of the
    real localizations in that regime. Genuinely broken fits that also drive bg
    wild are already removed by the width/photons/coordinate checks; a user who
    wants to exclude negative-background localizations in their own data does so
    with the interactive localization filter (set the ``bg`` minimum), which is
    the right home for a data-dependent preference.

    Returns a filtered DataFrame with a fresh 0..N-1 index (``df`` is not
    mutated); the SAME object is returned when nothing is dropped. This is an
    integrity guard, NOT the user-facing range filter (see
    ``core.smlm_localize.filter_locs_by_ranges``).
    """
    if df is None or len(df) == 0:
        return df
    keep = np.ones(len(df), dtype=bool)
    for col in _POSITIVE_LOC_COLS:
        if col in df.columns:
            v = df[col].to_numpy()
            keep &= np.isfinite(v) & (v > 0.0)
    for col in _NONNEGATIVE_LOC_COLS:
        if col in df.columns:
            v = df[col].to_numpy()
            keep &= np.isfinite(v) & (v >= 0.0)
    if keep.all():
        return df
    return df.loc[keep].reset_index(drop=True)


def build_locs_dataframe(frame, x, y, photons, sx, sy, bg, net_gradient, em=False):
    """Assemble the canonical localizations DataFrame from absolute coordinates.

    Single source of the Mortensen precision (lpx/lpy) and the 10-column schema,
    shared by the CPU least-squares path (locs_from_fits) and the GPU pygpufit
    path (core.smlm_localize). x/y must already be absolute pixel coordinates.
    Does not sort — callers handle that. Physically-impossible localizations
    (negative width/photons or off-detector coordinates — mostly from the
    unbounded GPU fit) are silently removed by the integrity guard
    ``drop_nonphysical_locs``, so callers never see them.

    Columns: frame, x, y, photons, sx, sy, bg, lpx, lpy, net_gradient.
    """
    photons = np.asarray(photons)
    sx = np.asarray(sx)
    sy = np.asarray(sy)
    bg = np.asarray(bg)
    lpx = localization_precision(photons, sx, sy, bg, em=em)
    lpy = localization_precision(photons, sy, sx, bg, em=em)
    df = pd.DataFrame({
        "frame": np.asarray(frame, dtype=np.uint32),
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "photons": photons.astype(np.float32),
        "sx": sx.astype(np.float32),
        "sy": sy.astype(np.float32),
        "bg": bg.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "net_gradient": np.asarray(net_gradient, dtype=np.float32),
    })
    return drop_nonphysical_locs(df)
