#!/usr/bin/env python3
"""
das_global.py -- Global (linked) reconvolution fitting of HORIBA .das TCSPC decays.

Why global fitting:
  If the same fluorophore reports SIMILAR lifetimes across samples and only the
  POPULATIONS (amplitudes) differ between constructs, then tau should be SHARED and
  only the amplitudes should float. Linking many decays to a common tau set:
    * over-constrains tau, which kills the unstable extra component that per-file
      fitting (and DAS6) produce when a 3rd exponential is not really supported;
    * turns "do lifetimes actually differ between constructs?" into a FALSIFIABLE test:
      if the shared-tau fit stays good (deviance/DoF ~ the per-file value, residuals
      structureless) the "same lifetimes, different populations" hypothesis holds; if
      it blows up, the lifetimes genuinely differ between constructs.

Modes:
  (default)  group by sample: 'NAME_1.das','NAME_2.das',... are grouped by NAME
             (trailing _<rep> stripped). One global fit per sample links its
             replicates: shared tau, per-replicate amplitude/shift/background.
  --all      ONE global fit across every decay of every file (tau shared across all
             constructs); the direct test of the hypothesis above.

Window is chosen mathematically per dataset via das_fit.auto_window (no eyeballing).
Objective is the Poisson deviance (maximum likelihood).

Deps: numpy, scipy, das_reader.py, das_fit.py (same folder).

CLI:
  python das_global.py *.das            # per-sample global fits (link replicates)
  python das_global.py --k 3 *.das      # use 3 shared lifetimes instead of 2
  python das_global.py --all *.das      # grand global across all constructs
  python das_global.py --dt 0.055176 *.das   # force ns/channel (batch/Sequential files)
"""
import sys, os, re, glob, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.stats import f as fdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from das_reader import read_das
from das_fit import auto_window, resolve_dt

def sample_name(fname):
    b = re.sub(r'^\d+_', '', os.path.basename(fname)).replace('.das', '')
    return re.sub(r'_\d+$', '', b)          # strip trailing _<rep>

def _bp(sample):
    m = re.match(r'(\d+)\s*bp', sample, flags=re.I)
    return int(m.group(1)) if m else 9999

def series_name(sample):
    """Fluorophore/series a construct belongs to: '12bpSALAD' -> 'SALAD', '8bpMII' -> 'MII'.
    Different fluorophores have no reason to share lifetimes, so tau is only ever
    linked WITHIN a series."""
    s = re.sub(r'^\d+\s*bp', '', sample, flags=re.I)
    return s.strip('_- ') or sample

def load_decays(files):
    """Flatten to a list of (sample, dec, irf, srcfile, decay_index)."""
    out = []
    for f in files:
        d = read_das(f); prompts = d['prompts']
        for i, dd in enumerate(d['decays']):
            dec = np.array(dd['counts'], float)
            irf = np.array(prompts[min(i, len(prompts)-1)]['counts'], float) if prompts else None
            if irf is None or irf.sum() == 0:
                print("  ! %s decay %d: no IRF -- skipped" % (os.path.basename(f), i+1)); continue
            out.append((sample_name(f), dec, irf, os.path.basename(f), i+1))
    return out

def global_fit(datasets, dt, k, amp_groups=None, scatter=False):
    """datasets: list of (dec, irf). Shared tau[k] across all datasets.
    amp_groups: list of group ids (one per dataset); datasets sharing an id share ONE
    amplitude vector. None -> every dataset gets its own amplitudes.
    shift and background are always free per dataset (instrumental, per-acquisition)."""
    prep = []
    for dec, irf in datasets:
        lo, hi, bg0 = auto_window(dec, irf, dt); N = hi
        d = dec[lo:hi].astype(float)
        prep.append(dict(lo=lo, hi=hi, N=N, t=np.arange(N)*dt, idx=np.arange(N),
                         irf_n=irf[:N]/irf[:N].sum(), d=d,
                         logd=np.where(d > 0, np.log(np.clip(d, 1, None)), 0.0),
                         bg0=bg0, peak=float(dec.max())))
    Dn = len(datasets)
    if amp_groups is None:
        amp_groups = list(range(Dn))
    uniq = sorted(set(amp_groups)); gidx = {g: i for i, g in enumerate(uniq)}
    ag = [gidx[g] for g in amp_groups]; nA = len(uniq)
    # layout: [tau(k)] [amps: nA*k] [per-dataset (scatter?) shift,bg]
    ns_ = 1 if scatter else 0
    aoff = k; soff = k + nA*k          # per-dataset block: [scatter?] shift bg
    nper = ns_ + 2
    def resid(p):
        tau = p[:k]; out = []
        for di, pr in enumerate(prep):
            a = p[aoff + ag[di]*k: aoff + ag[di]*k + k]
            b = soff + di*nper
            sc = p[b] if scatter else 0.0
            sh = p[b + ns_]; bg = p[b + ns_ + 1]
            irf_s = np.interp(pr['idx']-sh, pr['idx'], pr['irf_n'], left=0, right=0)
            kern = np.zeros(pr['N'])
            for ai, ti in zip(a, tau):
                kern += ai*np.exp(-pr['t']/ti)
            mm = fftconvolve(irf_s, kern)[:pr['N']]
            if scatter:
                mm = mm + sc*irf_s          # zero-lifetime (scatter / IRF-mismatch) term
            m = np.clip(mm[pr['lo']:pr['hi']] + bg, 1e-9, None)
            term = m - pr['d'] + pr['d']*(pr['logd'] - np.log(m))
            out.append(np.sign(m - pr['d'])*np.sqrt(2*np.clip(term, 0, None)))
        return np.concatenate(out)
    tau0 = [1.0, 3.5] if k == 2 else [0.4, 2.0, 4.5]
    p0 = list(tau0); lob = [0.05]*k; hib = [50.0]*k
    meanpeak = float(np.mean([pr['peak'] for pr in prep]))
    for _ in range(nA):
        p0 += [meanpeak/k]*k; lob += [0]*k; hib += [np.inf]*k
    for pr in prep:
        if scatter:
            p0 += [0.01*pr['peak']]; lob += [0]; hib += [np.inf]
        p0 += [0.0, pr['bg0']]; lob += [-5, 0]; hib += [5, np.inf]
    r = least_squares(resid, p0, bounds=(lob, hib), method='trf', max_nfev=40000, xtol=1e-10, ftol=1e-10)
    tau = np.array(r.x[:k])
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.pinv(r.jac.T @ r.jac)), 0, None))
    except Exception:
        se = np.full(len(r.x), np.nan)
    o = np.argsort(tau); tau, tau_se = tau[o], se[:k][o]
    Ddev = float(np.sum(r.fun**2)); npar = len(r.x); dof = len(r.fun) - npar
    per = []
    for di in range(Dn):
        a = np.array(r.x[aoff + ag[di]*k: aoff + ag[di]*k + k])[o]
        b = soff + di*nper
        sc = float(r.x[b]) if scatter else 0.0
        Iexp = a*tau                      # exponential intensity (a_i * tau_i)
        tot = Iexp.sum() + sc*dt          # scatter intensity in the same units
        per.append(dict(a=a, frac=a/a.sum(), ifrac=Iexp/Iexp.sum(),
                        scat_pct=100*sc*dt/tot if tot > 0 else 0.0,
                        shift=float(r.x[b + ns_]), bg=float(r.x[b + ns_ + 1])))
    return dict(tau=tau, tau_se=tau_se, per=per, dev=Ddev, npar=npar, dof=dof,
                dev_dof=Ddev/dof)

def _tau_str(res):
    return "/".join("%.3f\u00b1%.3f" % (t, s) for t, s in zip(res['tau'], res['tau_se']))

def main(argv):
    override = None; k = 2; mode = 'series'; scatter = False; excl = []; pats = []
    it = iter(argv)
    for x in it:
        if x == '--dt': override = float(next(it))
        elif x == '--k': k = int(next(it))
        elif x == '--all': mode = 'all'
        elif x == '--by-sample': mode = 'sample'
        elif x == '--by-series': mode = 'series'
        elif x == '--scatter': scatter = True
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
    if not files:
        print(__doc__); return
    dt, note = resolve_dt(files, override); print(note)
    if dt is None: return
    items = load_decays(files)

    groups = {}
    for it_ in items:
        key = {'series': series_name(it_[0]), 'sample': it_[0], 'all': 'ALL'}[mode]
        groups.setdefault(key, []).append(it_)
    print("grouping mode: %s  -> tau linked within: %s" % (mode, ", ".join(groups)))

    out = os.path.join(os.path.dirname(os.path.abspath(files[0])) or ".", "global_summary.csv")
    rows = []
    for gname, members in groups.items():
        datasets = [(m[1], m[2]) for m in members]
        samples = [m[0] for m in members]
        uniq = sorted(set(samples), key=lambda s: (_bp(s), s))
        # tau shared across the series; amplitudes free for EVERY replicate, so that
        # replicate scatter can serve as the experimental noise floor.
        res = global_fit(datasets, dt, k, amp_groups=list(range(len(datasets))), scatter=scatter)
        print("\n== series %s ==  shared tau = %s ns   deviance/DoF = %.4f   (%d decays, %d constructs)%s"
              % (gname, _tau_str(res), res['dev_dof'], len(members), len(uniq),
                 "   scatter=%.2f%%" % np.mean([q['scat_pct'] for q in res['per']]) if scatter else ""))

        pre = {s: [] for s in uniq}
        for m, pr in zip(members, res['per']):
            pre[m[0]].append(pr['frac']*100)
        print("   pre-exponential population %, mean +/- sd over replicates:")
        for s in uniq:
            A = np.array(pre[s])
            mu = A.mean(0); sd = A.std(0, ddof=1) if len(A) > 1 else np.zeros(A.shape[1])
            print("     %-12s %s   (n=%d)" % (s, "  ".join("%5.2f+/-%.2f" % (m_, s_) for m_, s_ in zip(mu, sd)), len(A)))

        # one-way ANOVA per component: between-construct vs within-construct (replicate) scatter
        if len(uniq) > 1 and all(len(pre[s]) > 1 for s in uniq):
            print("   do constructs differ by more than replicate reproducibility?")
            for c in range(k):
                gvals = [np.array([r[c] for r in pre[s]]) for s in uniq]
                gm = np.array([g.mean() for g in gvals]); ns = np.array([len(g) for g in gvals])
                grand = np.concatenate(gvals).mean()
                ssb = float(np.sum(ns*(gm-grand)**2)); dfb = len(uniq)-1
                ssw = float(np.sum([np.sum((g-g.mean())**2) for g in gvals])); dfw = int(ns.sum())-len(uniq)
                F = (ssb/dfb)/(ssw/dfw) if ssw > 0 and dfw > 0 else float('inf')
                p = float(1 - fdist.cdf(F, dfb, dfw)) if np.isfinite(F) else 0.0
                spread = gm.max()-gm.min()
                verdict = "DIFFERS" if p < 0.05 else "within replicate noise"
                print("     tau%d (%.2f ns): range %.2f%% across constructs, F(%d,%d)=%.1f p=%.3g  -> %s"
                      % (c+1, res['tau'][c], spread, dfb, dfw, F, p, verdict))

        for m, pr in zip(members, res['per']):
            rows.append([gname, m[0], m[3], m[4], k,
                         "/".join("%.4f" % t for t in res['tau']),
                         "/".join("%.4f" % s for s in res['tau_se']),
                         "/".join("%.2f%%" % (x*100) for x in pr['frac']),
                         "/".join("%.2f%%" % (x*100) for x in pr['ifrac']),
                         "%+.3f" % pr['shift'], "%.3f" % pr['bg'],
                         "%.4f" % res['dev_dof']])

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "sample", "file", "decay", "k", "shared_tau_ns", "shared_tau_SE",
                    "amp_frac_preexp", "amp_frac_intensity", "shift_ch", "bg", "group_dev/DoF"])
        w.writerows(rows)
    print("\nwrote", out)

if __name__ == "__main__":
    main(sys.argv[1:])
