#!/usr/bin/env python3
"""
das_fit.py -- Reconvolution multi-exponential fitting of HORIBA .das TCSPC decays.

Reads .das files DIRECTLY (via das_reader). CSV export is not used or needed here.

Method (the parts that decide whether lifetimes are correct):
  * RECONVOLUTION: model = IRF (x) sum_i a_i exp(-t/tau_i) + bg  (tau ~ IRF width).
  * Fitted COLOUR SHIFT (IRF<->decay timing) and constant background.
  * POISSON / MLE fit (minimises Poisson deviance); Pearson chi2_red also reported.
  * MODEL SELECTION: reduced chi-square (->1), nested F-test, Durbin-Watson on
    residuals (->2), MULTI-START stability, and asymptotic parameter errors.
    A tau that moves between random starts, or whose error is a large fraction of
    its value, is NOT determined by the data -- reported as unstable / ill-defined.

Time calibration (dt, ns/channel):
  Single-decay files store it in /Information/TCal. Sequential/batch files store 0.
  dt is resolved AUTOMATICALLY: the value from any TCal-bearing file in the run is
  applied to the TCal=0 files (same instrument/TAC = same dt). Override with --dt.

Deps: numpy, scipy, das_reader.py (same folder). No lmfit.

CLI:
  python das_fit.py *.das               # auto dt, writes fit_summary.csv
  python das_fit.py --dt 0.055176 *.das # force a dt
  python das_fit.py --starts 16 *.das   # more multi-start restarts (default 10)
"""
import sys, os, glob, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.stats import f as fdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from das_reader import read_das

# ---------- automatic dt resolution across the whole file set ----------
def auto_window(dec, irf, dt):
    """Mathematically determined fit window (no eyeballing).
    lo: a few channels before the IRF (excitation) leaves its baseline -> captures
        the full rise for reconvolution without including a long noise-only pre-region.
    hi: last channel where the smoothed decay is still above background+3sigma, i.e.
        where signal becomes statistically indistinguishable from the constant background.
    Returns (lo, hi, bg_estimate).
    """
    n = len(dec); pk = int(np.argmax(dec))
    bg = float(np.mean(dec[int(0.85*n):]))              # background from far tail (mean: robust for sparse low counts)
    # --- lo from IRF onset ---
    irf_base = float(np.median(irf[:max(pk-50, 10)]))
    thr_irf = irf_base + 0.005*(float(irf.max()) - irf_base)   # 0.5% of IRF peak
    crossings = np.where(irf > thr_irf)[0]
    rise = int(crossings[0]) if len(crossings) else max(0, pk-40)
    lo = max(0, rise - 10)
    # --- hi where signal meets background ---
    w = 15; sm = np.convolve(dec, np.ones(w)/w, mode='same')
    thr_tail = bg + 3*np.sqrt(max(bg, 1.0))
    above = np.where(sm[pk:] > thr_tail)[0]
    hi = min(n, pk + (int(above[-1]) if len(above) else int(90.0/dt)) + w)
    return int(lo), int(hi), bg

def resolve_dt(files, override):
    if override:
        return override, "forced --dt %.6f" % override
    seen = {}
    for f in files:
        try: tc = read_das(f)['tcal_ns']
        except Exception: tc = None
        if tc and tc > 0:
            seen.setdefault(round(tc, 6), []).append(os.path.basename(f))
    if not seen:
        return None, "no file carries a TCal -- pass --dt <ns_per_channel>"
    dt = max(seen, key=lambda k: len(seen[k]))          # most common value
    note = "dt=%.6f ns/ch auto-detected from %d file(s)" % (dt, sum(len(v) for v in seen.values()))
    if len(seen) > 1:
        note += "  WARNING: files report different TCal values %s -- check TAC settings" % sorted(seen)
    return dt, note

# ---------- model + Poisson-MLE fit with multi-start + errors ----------
def _make(dec, irf, dt, lo, hi):
    N = hi                                    # nothing past hi affects model[lo:hi] (causal conv)
    t = np.arange(N)*dt; idx = np.arange(N)
    irf_n = irf[:N]/irf[:N].sum(); sl = slice(lo, hi); d = dec[sl].astype(float)
    logd = np.where(d > 0, np.log(np.where(d > 0, d, 1.0)), 0.0)
    def model(p, k):
        a = p[:k]; tau = p[k:2*k]; shift = p[2*k]; bg = p[2*k+1]
        irf_s = np.interp(idx - shift, idx, irf_n, left=0.0, right=0.0)
        kern = np.zeros(N)
        for ai, ti in zip(a, tau): kern += ai*np.exp(-t/ti)
        return fftconvolve(irf_s, kern)[:N] + bg
    def dev_resid(p, k):
        m = np.clip(model(p, k)[sl], 1e-9, None)
        term = m - d + d*(logd - np.log(m))
        return np.sign(m - d)*np.sqrt(2*np.clip(term, 0, None))
    return model, dev_resid, sl, d

def _one_fit(dev_resid, p0, k, lob, hib):
    r = least_squares(dev_resid, p0, args=(k,), bounds=(lob, hib),
                      method='trf', max_nfev=4000, xtol=1e-10, ftol=1e-10)
    return r

def fit_decay(dec, irf, dt, k, lo, hi, starts=8, rng=None):
    rng = rng or np.random.default_rng(0)          # fixed seed => reproducible
    model, dev_resid, sl, d = _make(dec, irf, dt, lo, hi)
    pk = float(dec.max()); bg0 = max(float(dec[-500:].mean()), 0.1)
    lob = [0]*k + [0.05]*k + [-5, 0]
    hib = [np.inf]*k + [50]*k + [5, np.inf]
    results = []
    # start 0 = deterministic sensible guess; the rest = random log-spread taus
    guesses = [ {2:[1.0,3.5],3:[0.4,2.0,4.5]}[k] ]
    for _ in range(starts-1):
        guesses.append(sorted(rng.uniform(np.log(0.15), np.log(8), k).tolist()))
        guesses[-1] = list(np.exp(guesses[-1]))
    for g in guesses:
        p0 = [pk/k]*k + list(g) + [0.0, bg0]
        try: results.append(_one_fit(dev_resid, p0, k, lob, hib))
        except Exception: pass
    results.sort(key=lambda r: np.sum(r.fun**2))
    best = results[0]
    D = float(np.sum(best.fun**2)); dof = len(d) - len(best.x)
    m = model(best.x, k)[sl]
    chi2_red = float(np.sum((m - d)**2/np.clip(m, 1, None))/dof)
    dw = float(np.sum(np.diff(best.fun)**2)/np.sum(best.fun**2))
    # asymptotic parameter errors from the deviance Jacobian
    try:
        JtJ = best.jac.T @ best.jac
        cov = np.linalg.pinv(JtJ)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(len(best.x), np.nan)
    a = np.array(best.x[:k]); tau = np.array(best.x[k:2*k]); tau_se = se[k:2*k]
    o = np.argsort(tau); a, tau, tau_se = a[o], tau[o], tau_se[o]; frac = a/a.sum()
    ifrac = (a*tau)/np.sum(a*tau)          # intensity fraction = DAS6 "Rel.Ampl"
    # multi-start stability: spread of each tau among starts near the best deviance
    near = [r for r in results if np.sum(r.fun**2) <= D*1.05 + 1e-9]
    taus_near = np.array([np.sort(r.x[k:2*k]) for r in near])
    tau_spread = (taus_near.max(0) - taus_near.min(0))/np.clip(tau, 1e-9, None) if len(taus_near) > 1 else np.zeros(k)
    return dict(k=k, tau=tau, tau_se=tau_se, tau_spread=tau_spread, frac=frac, ifrac=ifrac,
                shift=float(best.x[2*k]), bg=float(best.x[2*k+1]),
                D=D, dof=dof, npar=len(best.x), chi2_red=chi2_red, dw=dw,
                tau_amp=float(np.sum(frac*tau)),
                tau_int=float(np.sum(frac*tau**2)/np.sum(frac*tau)),
                n_starts=len(results), n_near=len(near))

def compare(r2, r3, dt):
    F = ((r2['D']-r3['D'])/(r3['npar']-r2['npar']))/(r3['D']/r3['dof'])
    p = float(1 - fdist.cdf(F, r3['npar']-r2['npar'], r3['dof']))
    res_lim = 20*dt
    flags = []
    if r3['tau'][0] < res_lim:               flags.append("tri-tau1<res")
    if r3['frac'].min() < 0.02:              flags.append("tri-amp<2%")
    if np.min(r3['tau'][1:]/r3['tau'][:-1]) < 1.4: flags.append("tri-tau-overlap")
    # NEW: identifiability from multi-start spread and relative error
    if np.max(r3['tau_spread']) > 0.10:      flags.append("tri-unstable(starts)")
    rel_se3 = r3['tau_se']/np.clip(r3['tau'], 1e-9, None)
    if np.nanmax(rel_se3) > 0.30:            flags.append("tri-tau-err>30%")
    bi_unstable = (np.max(r2['tau_spread']) > 0.10) or (np.nanmax(r2['tau_se']/np.clip(r2['tau'],1e-9,None)) > 0.30)
    if p < 0.01 and not flags:  rec = "tri"
    elif p < 0.01:              rec = "tri? (flags)"
    else:                       rec = "bi"
    if bi_unstable: flags.append("bi-unstable")
    return dict(F=float(F), p=p, rec=rec, flags=";".join(flags) or "-")

def process(path, dt, starts):
    d = read_das(path); prompts = d['prompts']; rows = []
    for i, dd in enumerate(d['decays']):
        dec = np.array(dd['counts'], float)
        irf = np.array(prompts[min(i, len(prompts)-1)]['counts'], float) if prompts else None
        if irf is None or irf.sum() == 0:
            print("  ! %s decay %d: no IRF -- skipped" % (os.path.basename(path), i+1)); continue
        lo, hi, _bg = auto_window(dec, irf, dt)
        r2 = fit_decay(dec, irf, dt, 2, lo, hi, starts)
        r3 = fit_decay(dec, irf, dt, 3, lo, hi, starts)
        rows.append((os.path.basename(path), i+1, r2, r3, compare(r2, r3, dt)))
    return rows

def _tau_str(r):
    return "/".join("%.2f\u00b1%.2f" % (t, s) for t, s in zip(r['tau'], r['tau_se']))

def main(argv):
    override = None; starts = 8; excl = []; pats = []
    it = iter(argv)
    for x in it:
        if x == '--dt': override = float(next(it))
        elif x == '--starts': starts = int(next(it))
        elif x == '--exclude': excl += [w.strip() for w in next(it).split(',') if w.strip()]
        else: pats += glob.glob(x)
    files = sorted(f for f in pats if f.lower().endswith('.das'))
    if excl:
        def _hit(f, e):
            b = os.path.basename(f)
            # exact filename match (with or without .das) to avoid '6bpMII_1' matching '16bpMII_1'
            return b == e or b == e + '.das' or b[:-4] == e
        dropped = [f for f in files if any(_hit(f, e) for e in excl)]
        files = [f for f in files if f not in dropped]
        for f in dropped: print('EXCLUDED: %s' % os.path.basename(f))
    if not files: print(__doc__); return
    dt, note = resolve_dt(files, override); print(note)
    if dt is None: return
    out = os.path.join(os.path.dirname(os.path.abspath(files[0])) or ".", "fit_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file","decay","model","chi2_red","dev/DoF","durbin_watson",
                    "tau_ns(+/-SE)","amp_frac_preexp","amp_frac_intensity(DAS6)","tau_max_start_spread","tau_amp_ns","tau_int_ns",
                    "shift_ch","bg","F","p_value","recommend","flags"])
        for path in files:
            for fname, di, r2, r3, c in process(path, dt, starts):
                for r in (r2, r3):
                    w.writerow([fname, di, "%d-exp"%r['k'], "%.3f"%r['chi2_red'],
                                "%.3f"%(r['D']/r['dof']), "%.3f"%r['dw'], _tau_str(r),
                                "/".join("%d%%"%round(x*100) for x in r["frac"]),
                                "/".join("%d%%"%round(x*100) for x in r["ifrac"]),
                                "%.1f%%"%(100*np.max(r['tau_spread'])),
                                "%.3f"%r['tau_amp'], "%.3f"%r['tau_int'],
                                "%+.2f"%r['shift'], "%.2f"%r['bg'],
                                "%.1f"%c['F'] if r is r3 else "", "%.3g"%c['p'] if r is r3 else "",
                                c['rec'] if r is r3 else "", c['flags'] if r is r3 else ""])
                print("%-20s d%d | bi chi2r=%.2f DW=%.2f tau=%s spread<%.0f%% | tri chi2r=%.2f DW=%.2f tau=%s spread<%.0f%% | -> %s %s"
                      % (fname, di, r2['chi2_red'], r2['dw'], _tau_str(r2), 100*np.max(r2['tau_spread']),
                         r3['chi2_red'], r3['dw'], _tau_str(r3), 100*np.max(r3['tau_spread']),
                         c['rec'], "["+c['flags']+"]" if c['flags']!="-" else ""))
    print("\nwrote", out)

if __name__ == "__main__":
    main(sys.argv[1:])
