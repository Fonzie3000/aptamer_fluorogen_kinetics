"""
Change-point detection (CPD) idealization of single-molecule fluorescence traces
— a GUI-free core engine, an alternative to the HMM in ``core.hmm_engine``.

Pipeline (per trace):
    raw trace
      → BOCD Bayesian denoise (average one-step predictive mean; Adams & MacKay,
        "Bayesian Online Changepoint Detection", 2007)
      → per-trace noise-normalization (so one penalty value transfers across dim
        and bright molecules)
      → ruptures KernelCPD(kernel="linear") penalized segmentation (a fast C
        PELT-style pruned dynamic program) with user ``min_size`` + ``penalty``
      → segment means → classify each segment bound / background
      → per-frame binary state path (0 = background, 1 = bound)

The returned ``(states, means, covars)`` tuple has the SAME shape contract as
``hmm_engine.fit_hmm_batch`` results, so ``EventExtractor.extract_events`` and the
whole downstream Trace-Filter / Kinetics chain consume it unchanged.

Change-point + clustering idealization (CPD → segment → cluster) is the modern,
often faster alternative to HMMs for single-molecule traces (cf. DISC, White et
al. 2020; STaSI, Shuang et al. 2014).
"""

import os

import numpy as np
import numba

# Minimum separation (in robust-noise σ units) between the background and bound
# LEVELS for a trace to be called a real two-level binder. Internal fixed gate
# (the CPD analog of the HMM's min_state_sep) — the UI exposes only min_size +
# penalty, as requested; this suppresses false events on pure-noise traces.
_BOUND_SEP_SIGMA = 3.0
# Constant-hazard expected run length for the BOCD denoiser (frames). Controls the
# smoothness of the Bayesian predictive mean; not user-critical.
_BOCD_HAZARD_LAMBDA = 200.0
# Run-length truncation for the BOCD recursion: keep at most this many run lengths
# (drop the longest), turning the exact O(T²) recursion into O(T·R_MAX). Large
# enough that typical binding dwells are represented; the denoise is unaffected in
# practice (a very long dwell just re-anchors its Bayesian mean every R_MAX frames).
_BOCD_R_MAX = 150


def _robust_noise(x):
    """Robust per-frame noise σ from the median absolute first difference
    (MAD/√2 estimator — insensitive to the steps/events in the trace)."""
    if x.size < 2:
        return 1e-9
    d = np.diff(x)
    s = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    # A zero MAD is valid for noiseless/quantized plateaus. Their total standard
    # deviation measures the binding levels, not acquisition noise. Keep only a
    # numerical floor; using the trace spread here erases longer events.
    numerical_floor = max(float(np.ptp(x)) * 1e-6, 1e-9)
    return max(s, numerical_floor)


@numba.njit(cache=True, nogil=True)
def _bocd_denoise_kernel(x, mean0, var0, varx, logH, log1mH, r_max):
    """BOCD recursion (Adams & MacKay 2007) with a Gaussian prior on the segment
    mean and known per-trace noise variance ``varx``. Run length is truncated to
    ``r_max`` (drop the longest) → O(T·r_max). Returns the average one-step
    predictive mean per frame (the Bayesian-denoised trace)."""
    T = x.shape[0]
    inv_varx = 1.0 / varx
    prec0 = 1.0 / var0
    cap = r_max + 1
    mean_p = np.empty(cap, dtype=np.float64)
    prec_p = np.empty(cap, dtype=np.float64)
    logm = np.empty(cap, dtype=np.float64)
    new_mean = np.empty(cap, dtype=np.float64)
    new_prec = np.empty(cap, dtype=np.float64)
    new_logm = np.empty(cap, dtype=np.float64)
    lp = np.empty(cap, dtype=np.float64)
    mean_p[0] = mean0; prec_p[0] = prec0; logm[0] = 0.0
    n = 1
    denoised = np.empty(T, dtype=np.float64)

    for t in range(T):
        xt = x[t]
        # Normalize run-length log-posterior; denoised = Σ P(r)·mean_r.
        mx = logm[0]
        for i in range(1, n):
            if logm[i] > mx:
                mx = logm[i]
        Z = 0.0
        for i in range(n):
            Z += np.exp(logm[i] - mx)
        logZ = mx + np.log(Z)
        dv = 0.0
        for i in range(n):
            dv += np.exp(logm[i] - logZ) * mean_p[i]
        denoised[t] = dv

        # Predictive log-prob per run length + changepoint mass (log_cp).
        mxc = -1.0e300
        for i in range(n):
            pv = varx + 1.0 / prec_p[i]
            lp[i] = -0.5 * np.log(2.0 * np.pi * pv) - 0.5 * (xt - mean_p[i]) ** 2 / pv
            v = lp[i] + logm[i] + logH
            if v > mxc:
                mxc = v
        sc = 0.0
        for i in range(n):
            sc += np.exp(lp[i] + logm[i] + logH - mxc)
        log_cp = mxc + np.log(sc)

        newn = n + 1
        if newn > cap:
            newn = cap
        # index 0 = changepoint (run length reset); index j = grown run i=j-1.
        new_logm[0] = log_cp; new_mean[0] = mean0; new_prec[0] = prec0
        for j in range(1, newn):
            i = j - 1
            new_logm[j] = lp[i] + logm[i] + log1mH
            npr = prec_p[i] + inv_varx
            new_prec[j] = npr
            new_mean[j] = (mean_p[i] * prec_p[i] + xt * inv_varx) / npr
        # Normalize and copy back.
        mx3 = new_logm[0]
        for j in range(1, newn):
            if new_logm[j] > mx3:
                mx3 = new_logm[j]
        z3 = 0.0
        for j in range(newn):
            z3 += np.exp(new_logm[j] - mx3)
        lz3 = mx3 + np.log(z3)
        for j in range(newn):
            logm[j] = new_logm[j] - lz3
            mean_p[j] = new_mean[j]
            prec_p[j] = new_prec[j]
        n = newn

    return denoised


def bocd_denoise(x, hazard_lambda=_BOCD_HAZARD_LAMBDA, r_max=_BOCD_R_MAX):
    """Gaussian BOCD (Adams & MacKay 2007): returns the **average one-step
    predictive mean** (run-length-posterior-weighted expected value at each frame)
    as a Bayesian-denoised trace. Causal (filtering) estimate — uses only past
    data, so it lags a few frames at a step, an inherent property of online BOCD.
    Runs the truncated numba kernel (O(T·r_max))."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    T = x.size
    if T == 0:
        return x.copy()
    mean0 = float(np.median(x))
    var0 = float(max(np.var(x), 1e-12))
    varx = _robust_noise(x) ** 2
    H = 1.0 / float(hazard_lambda)
    return _bocd_denoise_kernel(x, mean0, var0, varx,
                                np.log(H), np.log(1.0 - H), int(r_max))


def _weighted_otsu(values, weights):
    """Length-weighted 1-D Otsu threshold that maximizes between-class variance —
    the split of the (few) segment means into a background vs bound level. Returns
    a threshold value between the two adjacent sorted segment means."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total_w = float(w.sum())
    if total_w <= 0 or v.size < 2:
        return float(v[-1]) if v.size else 0.0
    cw = np.cumsum(w)
    cwv = np.cumsum(w * v)
    total_wv = float(cwv[-1])
    best_between = -1.0
    best_thr = 0.5 * (v[0] + v[-1])
    for i in range(v.size - 1):
        w0 = float(cw[i])
        w1 = total_w - w0
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = cwv[i] / w0
        m1 = (total_wv - cwv[i]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_between:
            best_between = between
            best_thr = 0.5 * (v[i] + v[i + 1])
    return float(best_thr)


def _classify_segments(x, bounds, sigma_n, sep):
    """Turn segment boundaries into a binary 0/1 state path via a 2-level split of
    the segment means (length-weighted Otsu) plus a fixed separation gate: if the
    background and bound LEVELS are closer than ``sep`` noise-σ (or there is only
    one segment / one cluster), the trace is treated as pure noise → all-background
    (no events). This is what stops CPD from flagging events on noise-only traces.
    Returns (states, means, covars) with state 0 = background."""
    T = x.size
    states = np.zeros(T, dtype=np.int32)
    med = float(np.median(x))
    flat_means = np.array([med, med])
    flat_covars = np.array([sigma_n * sigma_n, sigma_n * sigma_n])

    starts = np.concatenate(([0], bounds[:-1])).astype(np.int64)
    ends = bounds.astype(np.int64)
    seg_means = np.array([float(x[s:e].mean()) if e > s else med
                          for s, e in zip(starts, ends)])
    seg_len = (ends - starts).astype(np.float64)
    if seg_means.size < 2:
        return states, flat_means, flat_covars   # single segment → all background

    thr = _weighted_otsu(seg_means, seg_len)
    lo_mask = seg_means <= thr
    hi_mask = ~lo_mask
    if not lo_mask.any() or not hi_mask.any():
        return states, flat_means, flat_covars
    bg_level = float(np.average(seg_means[lo_mask], weights=seg_len[lo_mask]))
    bound_level = float(np.average(seg_means[hi_mask], weights=seg_len[hi_mask]))

    # Separation gate: the two levels must be genuinely distinct.
    if (bound_level - bg_level) / max(sigma_n, 1e-9) < sep:
        return states, flat_means, flat_covars

    for s, e, m in zip(starts, ends, seg_means):
        if m > thr:
            states[s:e] = 1

    m0mask = states == 0
    m1mask = states == 1
    if m0mask.any() and m1mask.any():
        means = np.array([float(x[m0mask].mean()), float(x[m1mask].mean())])
        covars = np.array([max(float(x[m0mask].var()), 1e-6),
                           max(float(x[m1mask].var()), 1e-6)])
    else:
        means = flat_means
        covars = flat_covars
    return states, means, covars


def _segment_bounds(x, min_size=5, penalty=None, denoise=True):
    """Change-point segmentation of one ALREADY-SANITIZED trace.

    Shared core of :func:`detect_changepoints` (which binarizes the result) and
    :func:`segment_levels` (which does not). Returns ``(bounds, sigma_n)`` where
    ``bounds`` is the int64 array of segment END indices with ``bounds[-1] == len(x)``,
    or ``(None, sigma_n)`` when the trace is too short / constant / ruptures fails.
    """
    import ruptures as rpt

    T = x.size
    if T < max(4, 2 * int(min_size)):
        return None, 1.0
    sigma_n = _robust_noise(x)
    if x.max() - x.min() < 1e-9:
        return None, sigma_n

    sig = bocd_denoise(x) if denoise else x
    # Per-trace normalization so a single penalty behaves the same on dim and
    # bright molecules (KernelCPD's linear cost is in these normalized units).
    sig_norm = (sig - np.median(sig)) / sigma_n

    pen = float(3.0 * np.log(T)) if penalty is None else float(penalty)
    algo = rpt.KernelCPD(kernel="linear", min_size=int(min_size)).fit(
        np.ascontiguousarray(sig_norm.reshape(-1, 1)))
    try:
        bkps = algo.predict(pen=pen)   # list of segment ends, last == T
    except Exception:
        return None, sigma_n
    bounds = np.asarray(bkps, dtype=np.int64)
    if bounds.size == 0 or bounds[-1] != T:
        bounds = np.append(bounds, T)
    return bounds, sigma_n


def detect_changepoints(x, min_size=5, penalty=None, denoise=True,
                        sep=_BOUND_SEP_SIGMA):
    """Idealize one trace by change-point detection.

    Args:
        x: 1-D intensity trace.
        min_size: minimum segment length (frames) — ruptures ``min_size``.
        penalty: ruptures penalty (``pen``); larger → fewer change points.
            None → a BIC-style default ``3·log(T)`` (the signal is σ-normalized,
            so this transfers across traces).
        denoise: BOCD Bayesian denoise before segmentation (default True).
        sep: bound/background separation threshold in noise-σ units.

    Returns:
        (states, means, covars) — state 0 = background, 1 = bound; same shape
        contract as ``hmm_engine.fit_hmm_batch`` results.
    """
    x = np.ascontiguousarray(np.nan_to_num(np.asarray(x, dtype=np.float64),
                                           nan=0.0, posinf=0.0, neginf=0.0))
    T = x.size
    med = float(np.median(x)) if T else 0.0
    flat = (np.zeros(T, dtype=np.int32), np.array([med, med]), np.array([1.0, 1.0]))
    bounds, sigma_n = _segment_bounds(x, min_size, penalty, denoise)
    if bounds is None:
        return flat
    return _classify_segments(x, bounds, sigma_n, sep)


def segment_levels(x, min_size=5, penalty=None, denoise=True):
    """Multi-level piecewise-constant segmentation of one trace — NOT binarized.

    Same change-point segmentation as :func:`detect_changepoints`, but returns the
    raw per-segment levels instead of a 2-state path. Used by the segmented detrend
    (``core.trace_detrend``) to build a bound/unbound mask that is derived from the
    DATA rather than from a baseline fit. That independence is the whole point: a
    mask computed from the residual to a smooth fit is circular and re-derives the
    duration confound (an event long enough for the baseline to follow stops looking
    like an outlier), which is what makes penalized-least-squares baselines eat long
    binding events.

    ``seg_means`` are computed on the RAW trace, never on the BOCD-denoised signal —
    the denoiser is causal and biases segment means near a step.

    Returns:
        (starts, ends, seg_means, sigma_n) — int64[K], int64[K], float64[K], float.
        K == 0 (empty arrays) when the trace is degenerate or segmentation fails.
    """
    x = np.ascontiguousarray(np.nan_to_num(np.asarray(x, dtype=np.float64),
                                           nan=0.0, posinf=0.0, neginf=0.0))
    bounds, sigma_n = _segment_bounds(x, min_size, penalty, denoise)
    if bounds is None:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64), sigma_n)
    med = float(np.median(x))
    starts = np.concatenate(([0], bounds[:-1])).astype(np.int64)
    ends = bounds.astype(np.int64)
    seg_means = np.array([float(x[s:e].mean()) if e > s else med
                          for s, e in zip(starts, ends)], dtype=np.float64)
    return starts, ends, seg_means, sigma_n


def detect_changepoints_batch(data_list, min_size=5, penalty=None, denoise=True,
                              n_workers=None):
    """Idealize a list of traces by CPD, in parallel across a thread pool: the
    BOCD kernel is numba ``nogil`` (releases the GIL) and the ruptures fit is a
    short C call, so threading gives near-linear speed-up. Returns a list of
    (states, means, covars), one per trace."""
    if not data_list:
        return []
    n = len(data_list)
    if n_workers is None:
        n_workers = min(32, (os.cpu_count() or 4))
    n_workers = max(1, int(n_workers))

    def _job(i):
        return i, detect_changepoints(data_list[i], min_size=min_size,
                                      penalty=penalty, denoise=denoise)

    if n_workers == 1 or n == 1:
        return [_job(i)[1] for i in range(n)]

    results = [None] * n
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, res in ex.map(_job, range(n)):
            results[i] = res
    return results
