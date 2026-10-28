"""
TCSPC estimator, reimplemented from the documented specification of the Chapter 4
pipeline (the original das_reader/das_fit/das_global are not available).

Specification being reproduced, unchanged from the handover
----------------------------------------------------------
* iterative reconvolution with the MEASURED IRF (one prompt per acquisition), not
  tail fitting, because tau is comparable to the IRF width
* Poisson maximum likelihood, not weighted least squares, because tail channels
  carry <1 count
* a fitted colour shift (the prompt peaks 3-5 channels ahead of the decay)
* a zero-lifetime scatter term s*IRF, physically justified because the IRF was
  acquired on a Ludox scattering solution detected at the 467 nm excitation wavelength
* lifetimes globally linked within each fluorophore series, amplitudes free per
  acquisition
* algorithmically determined fit windows

Deliberately NOT included: a fitted constant background. The specification does not
list one, and its absence is mechanistically why the far tail matters - with no
baseline term the model tends to zero in the tail while the data sit at ~0.4 counts,
so including those channels pushes the fit to raise a long, low-amplitude component
to cover them. `background` is exposed as a diagnostic switch only.

Model
-----
For acquisition q with measured prompt P_q and shared lifetimes tau_1..tau_k,

    m_q[n] = sum_j B_qj * (P_q(shift_q) * exp(-t/tau_j))[n]  +  s_q * P_q(shift_q)[n]  (+ bg_q)

The convolution of a sampled IRF with a single exponential is the exact discrete sum
c[n] = sum_{m<=n} P[m] exp(-(n-m)dt/tau), which satisfies the one-pole recursion
c[n] = P[n] + exp(-dt/tau) c[n-1] and is therefore computed in O(N) with lfilter
rather than by FFT.

Fitting is separable and is done as a profile likelihood: an OUTER optimisation over
the k shared log-lifetimes, and for each trial an INNER per-acquisition optimisation
of (log B_qj, log s_q, shift_q), which is 4-5 well-conditioned parameters. This is the
standard way to run a global analysis and is far better conditioned than throwing all
~90 parameters at one optimiser.

Objective: Poisson deviance D = 2 * sum [ y ln(y/m) - (y - m) ], with the y ln y term
taken as 0 where y = 0. Minimising D is equivalent to maximising the Poisson likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

NS_PER_CHANNEL = 0.055176
TINY = 1e-12


# ----------------------------------------------------------------------------------
# core model
# ----------------------------------------------------------------------------------
def shift_irf(irf: np.ndarray, shift: float) -> np.ndarray:
    """Sub-channel shift by linear interpolation. Positive `shift` moves the IRF later.

    The bulk of the 3-5 channel prompt/decay offset is produced by the convolution
    itself, so the fitted residual shift should be a fraction of a channel.
    """
    n = irf.size
    idx = np.arange(n, dtype=float)
    return np.interp(idx - shift, idx, irf, left=0.0, right=0.0)


_CONV_CACHE: dict = {}
_CACHE_CAP = 4000


def conv_base(prompt: np.ndarray, tau_ch: float, pkey) -> np.ndarray:
    """UNSHIFTED convolution of the IRF with exp(-t/tau_ch), memoised on (prompt, tau).

    Convolution with a time-invariant kernel commutes with a time shift, and the IRF is
    zero over the first ~240 channels, so

        conv(shift(P, d), K) == shift(conv(P, K), d)

    to within the interpolation used for the sub-channel shift. Exploiting this means the
    O(N) recursion runs ONCE per trial lifetime instead of once per function evaluation,
    and - unlike memoising on a rounded shift - it leaves the objective smooth in the
    shift, which the finite-difference gradients of the inner optimiser require.
    """
    key = (pkey, float(tau_ch))
    hit = _CONV_CACHE.get(key)
    if hit is None:
        if len(_CONV_CACHE) > _CACHE_CAP:
            _CONV_CACHE.clear()
        hit = conv_exp(prompt, tau_ch)
        _CONV_CACHE[key] = hit
    return hit


def conv_exp(irf: np.ndarray, tau_ch: float) -> np.ndarray:
    """Discrete convolution of a sampled IRF with exp(-t/tau) via the one-pole recursion.

    tau_ch is the lifetime IN CHANNELS. Exact for the discrete sum, O(N).
    """
    a = np.exp(-1.0 / tau_ch)
    return lfilter([1.0], [1.0, -a], irf)


def poisson_deviance(y: np.ndarray, m: np.ndarray) -> float:
    """2*sum[y ln(y/m) - (y-m)]; 0 <= D, and D/dof ~ 1 for a correct model."""
    m = np.clip(m, TINY, None)
    term = np.where(y > 0, y * np.log(y / m), 0.0)
    return float(2.0 * np.sum(term - (y - m)))


@dataclass
class Acquisition:
    dye: str
    stem_bp: int
    acq: int
    decay: np.ndarray
    prompt: np.ndarray
    lo: int = 0          # inclusive fit-window start channel
    hi: int = 4096       # exclusive fit-window end channel
    pkey: int = 0        # hash of the prompt, for the convolution cache

    def __post_init__(self):
        self.pkey = hash(self.prompt.tobytes())

    @property
    def slc(self):
        return slice(self.lo, self.hi)

    @property
    def label(self):
        return f"{self.dye}_{self.stem_bp}bp_acq{self.acq}"


def _model(prompt: np.ndarray, taus_ch, B, s, shift, bg=0.0, pkey=None):
    if pkey is None:
        p = shift_irf(prompt, shift)
        m = s * p
        for tau, b in zip(taus_ch, B):
            m = m + b * conv_exp(p, tau)
        return m + bg
    m = s * shift_irf(prompt, shift)
    for tau, b in zip(taus_ch, B):
        m = m + b * shift_irf(conv_base(prompt, tau, pkey), shift)
    return m + bg


def _fit_one_acquisition(a: Acquisition, taus_ch, x0=None, background=False, scatter=True,
                         bg_fixed=None):
    """`bg_fixed` holds the baseline at a supplied value instead of fitting it, which
    distinguishes 'the model had a free baseline parameter' from 'a constant background
    was accounted for' - observationally similar, different methods-section sentences."""
    if bg_fixed is not None:
        return _fit_one_acquisition_impl(a, taus_ch, x0, False, scatter, float(bg_fixed))
    return _fit_one_acquisition_impl(a, taus_ch, x0, background, scatter, 0.0)


def _fit_one_acquisition_impl(a: Acquisition, taus_ch, x0=None, background=False, scatter=True,
                              bg_const=0.0):
    """Inner problem: amplitudes, scatter fraction and colour shift for fixed lifetimes.

    Parameterisation: log B_j, log s (and log bg) for positivity, shift free.
    `scatter=False` drops the s*IRF term entirely, which is the like-with-like setting
    for comparison against the DAS6 per-file fits (those carry no scatter term).
    """
    k = len(taus_ch)
    y = a.decay[a.slc]
    n_s = 1 if scatter else 0

    def unpack(x):
        B = np.exp(x[:k])
        s = np.exp(x[k]) if scatter else 0.0
        shift = x[k + n_s]
        bg = np.exp(x[k + n_s + 1]) if background else bg_const
        return B, s, shift, bg

    # Evaluate the model ONLY over the fit window. The causal recursion in conv_base still
    # needs the full record (and is cached per lifetime), but the sub-channel shift is an
    # interpolation that only has to be evaluated at the ~1400 channels being fitted
    # rather than all 4096, which is where the inner-loop time actually goes.
    idx_full = np.arange(a.decay.size, dtype=float)
    idx_win = idx_full[a.slc]

    def obj(x):
        B, s, shift, bg = unpack(x)
        m = s * np.interp(idx_win - shift, idx_full, a.prompt, left=0.0, right=0.0) + bg
        for tau, b in zip(taus_ch, B):
            base = conv_base(a.prompt, tau, a.pkey)
            m = m + b * np.interp(idx_win - shift, idx_full, base, left=0.0, right=0.0)
        return poisson_deviance(y, m)

    if x0 is None:
        # Deterministic start, derived only from the data and the trial lifetimes.
        # It must NOT be warm-started from a previous outer iteration: that makes the
        # outer objective path-dependent, and a derivative-free outer optimiser
        # (Nelder-Mead) then stalls with components frozen at their initial values.
        # The inner solve is cheap (~10 ms), so there is nothing to gain from warm starts.
        peak = max(a.decay.max(), 1.0)
        # each component convolved with the IRF peaks at roughly peak/k of the data
        amp0 = [peak / (k * max(conv_base(a.prompt, t, a.pkey).max(), 1e-9)) for t in taus_ch]
        x0 = np.log(np.clip(amp0, 1e-12, None)).tolist()
        if scatter:
            x0 += [np.log(max(peak / max(a.prompt.max(), 1.0) * 1e-2, 1e-12))]
        x0 += [0.0]
        if background:
            x0 += [np.log(max(a.decay[-300:].mean(), 1e-3))]
        x0 = np.array(x0)
    bounds = [(-30, 30)] * k + ([(-30, 30)] if scatter else []) + [(-20, 20)] \
             + ([(-30, 5)] if background else [])
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=600, ftol=1e-12, gtol=1e-10))
    B, s, shift, bg = unpack(res.x)
    return dict(B=B, s=s, shift=shift, bg=bg, deviance=float(res.fun), x=res.x,
                n_channels=int(y.size))


@dataclass
class GlobalFit:
    taus_ns: np.ndarray
    per_acq: list
    deviance: float
    n_params: int
    n_channels: int
    background: bool = False
    scatter: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def dof(self):
        return self.n_channels - self.n_params

    @property
    def dev_per_dof(self):
        return self.deviance / self.dof


def fit_global(acqs, k=2, background=False, scatter=True, taus_ns_init=None, verbose=False,
               bg_fixed=None):
    """Lifetimes linked across every acquisition in `acqs`, amplitudes free per acquisition.

    `acqs` is normally one fluorophore series (all TO1-B constructs, or all SALAD1).
    Returns a GlobalFit. Lifetimes are reported in ns; internally they are in channels.
    """
    if taus_ns_init is None:
        taus_ns_init = {2: [2.0, 4.8], 3: [1.0, 3.5, 10.0]}[k]
    taus_ns_init = np.asarray(taus_ns_init, float)
    state = {"n": 0}

    def total_deviance(log_tau_ch):
        taus_ch = np.exp(log_tau_ch)
        if np.any(np.diff(taus_ch) <= 0):          # keep components ordered, avoid swaps
            return 1e18
        tot, fits = 0.0, []
        for a in acqs:
            bgf = None
            if bg_fixed is not None:
                # per-acquisition dark level, taken from the far tail; not a free parameter
                bgf = a.decay[-300:].mean() if bg_fixed == "tail" else bg_fixed
            f = _fit_one_acquisition(a, taus_ch, x0=None, background=background,
                                     scatter=scatter, bg_fixed=bgf)
            tot += f["deviance"]; fits.append(f)
        state["fits"] = fits
        state["n"] += 1
        if verbose and state["n"] % 10 == 0:
            print(f"  outer {state['n']:3d}  taus_ns={np.round(taus_ch*NS_PER_CHANNEL,4)}  D={tot:.1f}")
        return tot

    # Multi-start over the shared lifetimes: the outer surface for k=3 has flat
    # directions, so a single Nelder-Mead run can terminate with a component still at
    # its starting value. Perturbed restarts make the reported optimum reproducible.
    # k=2 has a well-behaved outer surface and a single start reproduces the multi-start
    # optimum exactly; k>=3 has flat directions and needs the restarts.
    starts = [taus_ns_init]
    if k >= 3:
        for fac in (0.7, 1.4):
            starts.append(taus_ns_init * np.linspace(1.0, fac, len(taus_ns_init)))
    best = None
    for s0 in starts:
        r = minimize(total_deviance, np.log(np.sort(s0) / NS_PER_CHANNEL),
                     method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-4, maxiter=2000, maxfev=4000))
        if best is None or r.fun < best.fun:
            best = r
    res = best
    taus_ch = np.exp(res.x)
    total_deviance(res.x)                          # ensure state["fits"] matches the optimum
    n_per = k + 1 + (1 if scatter else 0) + (1 if background else 0)   # B_j, shift, s, bg
    n_ch = int(sum(f["n_channels"] for f in state["fits"]))
    return GlobalFit(taus_ns=taus_ch * NS_PER_CHANNEL, per_acq=state["fits"],
                     deviance=float(res.fun), n_params=k + n_per * len(acqs),
                     n_channels=n_ch, background=background, scatter=scatter,
                     meta=dict(acqs=[a.label for a in acqs], outer_iterations=state["n"],
                               converged=bool(res.success)))


# ----------------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------------
def mean_lifetimes(taus_ns, B):
    """Amplitude- and intensity-weighted means over the EXPONENTIAL components only.

    The zero-lifetime scatter term is instrumental, not a population, and is excluded -
    as it must be, since including a tau=0 component would drag tau_amp towards zero.
    """
    B = np.asarray(B, float); t = np.asarray(taus_ns, float)
    tau_amp = float((B * t).sum() / B.sum())
    tau_int = float((B * t**2).sum() / (B * t).sum())
    return tau_amp, tau_int


def amplitude_fractions_pct(B):
    B = np.asarray(B, float)
    return 100.0 * B / B.sum()


def scatter_fraction_pct(a: Acquisition, taus_ns, B, s, shift, bg=0.0):
    """Scatter term as a percentage of MODEL COUNTS inside the fit window.

    Reported as a photon fraction, which is how the ~2.1% figures in the handover read.
    """
    taus_ch = np.asarray(taus_ns) / NS_PER_CHANNEL
    p = shift_irf(a.prompt, shift)
    scat = (s * p)[a.slc].sum()
    total = _model(a.prompt, taus_ch, B, s, shift, bg)[a.slc].sum()
    return 100.0 * scat / total


def durbin_watson(y, m):
    """DW on Pearson residuals. ~2 means uncorrelated; <2 means positive correlation,
    i.e. the model is missing structure."""
    r = (y - m) / np.sqrt(np.clip(m, TINY, None))
    return float(np.sum(np.diff(r) ** 2) / np.sum(r**2))


def weighted_residuals(a: Acquisition, taus_ns, B, s, shift, bg=0.0):
    taus_ch = np.asarray(taus_ns) / NS_PER_CHANNEL
    m = _model(a.prompt, taus_ch, B, s, shift, bg)
    y = a.decay
    r = np.zeros_like(y)
    sl = a.slc
    r[sl] = (y[sl] - m[sl]) / np.sqrt(np.clip(m[sl], TINY, None))
    return m, r


# ----------------------------------------------------------------------------------
# fit-window rules  (section 1 of the reanalysis brief)
# ----------------------------------------------------------------------------------
def window_start(prompt: np.ndarray, frac=0.01, margin=10) -> int:
    """Channel where the IRF has risen to `frac` of its peak, minus `margin` channels.

    Rationale: before and during the rise the record is dominated by the instrument
    response, and any timing or shape mismatch between prompt and decay lands in
    exactly those channels, where the counts - and therefore the Poisson weights -
    are largest. Applied identically to every decay.
    """
    pk = int(np.argmax(prompt))
    thresh = frac * prompt[pk]
    rising = np.nonzero(prompt[:pk + 1] >= thresh)[0]
    first = int(rising[0]) if rising.size else pk
    return max(first - margin, 0)


def window_end(decay: np.ndarray, n_sd=3.0, smooth=15, bg_region=slice(-300, None)) -> int:
    """First channel after the peak at which the smoothed decay falls below
    background + `n_sd` * SD(background).

    Rationale: once the decay has reached background the channels carry no information
    about the sample but a great deal about the baseline, letting a long low-amplitude
    component trade against it. Returned value is EXCLUSIVE.
    """
    bg = decay[bg_region]
    level = bg.mean() + n_sd * bg.std(ddof=1)
    kern = np.ones(smooth) / smooth
    sm = np.convolve(decay, kern, mode="same")
    pk = int(np.argmax(decay))
    below = np.nonzero(sm[pk:] < level)[0]
    return int(pk + below[0]) if below.size else decay.size


def apply_window(a: Acquisition, frac=0.01, margin=10, n_sd=3.0, smooth=15):
    a.lo = window_start(a.prompt, frac=frac, margin=margin)
    a.hi = window_end(a.decay, n_sd=n_sd, smooth=smooth)
    return a
