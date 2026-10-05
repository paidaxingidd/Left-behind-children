"""F. Latent-variable versions of the main mediation model.
   F1 single-indicator SEM with reliability correction (error variance fixed at (1-alpha)*var) -- analytic, bootstrap CI
   F2 full latent SEM: FF second-order factor over five item-level FAD factors (29 items), LS latent (5 items),
      LE latent (3 parcels vi/de/ab); covariates girl, grandparent, divorced predict LS and LE and covary with FF.
      Fit on the pooled sample covariance of each imputed dataset; standardized a, b, c', a*b pooled over imputations;
      bootstrap CI with 1,000 resamples (50 per imputation)."""
import numpy as np, pandas as pd, pickle, time, json
from lbc_core import *
from lbc_core import _z, _ols
from sem import SEM
t0 = time.time()
imps = pickle.load(open(PREFIX + 'imps_lbc.pkl', 'rb'))
covs = ['girl', 'grandparent', 'divorced']
fad29 = [c for c in FAD if c in imps[0].columns]   # 30 items in final mode, 29 in interim

def alpha(X):
    X = np.asarray(X, float); k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))

# ---------- F1: single-indicator reliability correction (disattenuation) ----------
def f1_stat(imp):
    rel = {'fad': alpha(imp[fad29]), 'ls': alpha(imp[SWLS]), 'le': alpha(imp[UWES])}
    V = ['fad', 'ls', 'le'] + covs
    Z = np.column_stack([_z(imp[v].to_numpy(float)) for v in V])
    R = np.corrcoef(Z, rowvar=False)
    # disattenuate correlations among the three focal variables and between focal and covariates
    D = np.diag([np.sqrt(rel['fad']), np.sqrt(rel['ls']), np.sqrt(rel['le'])] + [1.0] * len(covs))
    Rc = np.linalg.inv(D) @ R @ np.linalg.inv(D)
    np.fill_diagonal(Rc, 1.0)
    # path coefficients from the corrected correlation matrix
    def reg(y, xs):
        idx = [V.index(v) for v in xs]; return np.linalg.solve(Rc[np.ix_(idx, idx)], Rc[idx, V.index(y)])
    a = reg('ls', ['fad'] + covs)[0]
    cy = reg('le', ['fad', 'ls'] + covs); cp, b = cy[0], cy[1]
    c = reg('le', ['fad'] + covs)[0]
    raw = mediation(imp['fad'].to_numpy(float), imp['ls'].to_numpy(float), imp['le'].to_numpy(float), imp[covs].to_numpy(float))
    return np.array([a, b, cp, c, a * b, rel['fad'], rel['ls'], rel['le'], raw[4]])
import sys
F2ONLY = '--f2only' in sys.argv
if not F2ONLY:
    pt, bt = boot_over_imputations(imps, f1_stat, B=5000, seed=51)
if not F2ONLY:
    lo, hi = ci(bt)
    f1 = pd.DataFrame(dict(param=['a', 'b', 'cprime', 'c', 'indirect', 'alpha_fad', 'alpha_ls', 'alpha_le', 'indirect_observed'], est=pt, lo=lo, hi=hi, p=boot_p(bt)))
    print('F1 reliability-corrected (single-indicator):'); print(f1.round(3).to_string(index=False))
    f1.to_csv(PREFIX + 'res_F1_disattenuated.csv', index=False)

# ---------- F2: full latent SEM ----------
fac = {'EM': [f'b{i}' for i in range(1, 9)], 'PC': [c for c in ['b9', 'b10', 'b11', 'b12', 'b13'] if c in fad29], 'ES': [f'b{i}' for i in range(14, 20)],
       'PS': [f'b{i}' for i in range(20, 26)], 'FR': [f'b{i}' for i in range(26, 31)]}
obs = fad29 + SWLS + ['vi', 'de', 'ab'] + covs
lat = list(fac) + ['FF', 'LS', 'LE']
free, fixed = [], {}
for f, its in fac.items():                       # first-order FAD factors: first loading fixed to 1, disturbance free
    fixed[('A', its[0], f)] = 1.0
    free += [('A', it, f) for it in its[1:]] + [('S', f, f)]
free += [('A', f, 'FF') for f in fac]            # second-order loadings (FF variance fixed to 1)
fixed[('S', 'FF', 'FF')] = 1.0
fixed[('A', 'c1', 'LS')] = 1.0; free += [('A', it, 'LS') for it in SWLS[1:]] + [('S', 'LS', 'LS')]
fixed[('A', 'vi', 'LE')] = 1.0; free += [('A', it, 'LE') for it in ['de', 'ab']] + [('S', 'LE', 'LE')]
free += [('S', it, it) for it in fad29 + SWLS + ['vi', 'de', 'ab']]
free += [('A', 'LS', 'FF'), ('A', 'LE', 'FF'), ('A', 'LE', 'LS')]
free += [('A', 'LS', cv) for cv in covs] + [('A', 'LE', cv) for cv in covs]
free += [('S', cv, cv) for cv in covs] + [('S', covs[i], covs[j]) for i in range(3) for j in range(i + 1, 3)] + [('S', 'FF', cv) for cv in covs]
model = SEM(obs, lat, free, fixed)

def fit_once(imp, start=None):
    X = imp[obs].to_numpy(float); S = np.cov(X, rowvar=False, bias=True)
    m = SEM(obs, lat, free, fixed).fit(S, len(X), start=start)
    a = m.get('A', 'LS', 'FF'); b = m.get('A', 'LE', 'LS'); cp = m.get('A', 'LE', 'FF')
    return m, np.array([a, b, cp, a * b + cp, a * b])
m0, est0 = fit_once(imps[0])
print('F2 latent SEM, imputation 1: converged', m0.converged, {k: round(float(v), 3) for k, v in m0.fit_indices().items()})
print('   second-order loadings', {f: round(float(m0.get('A', f, 'FF')), 2) for f in fac}, '| LS loadings', [round(float(m0.get('A', it, 'LS')), 2) for it in SWLS], '| LE loadings', [round(float(m0.get('A', it, 'LE')), 2) for it in ['vi', 'de', 'ab']])
print('   std a=%.3f b=%.3f cp=%.3f c=%.3f ind=%.3f' % tuple(est0), '| time %.1fs' % (time.time() - t0))
start = m0.theta
ests, fits = [], []
for imp in imps:
    m, e = fit_once(imp, start=start); ests.append(e); fits.append(m.fit_indices())
ests = np.array(ests)
print('   pooled over 20 imputations: a=%.3f b=%.3f cp=%.3f c=%.3f ind=%.3f' % tuple(ests.mean(0)))
print('   pooled fit:', {k: round(float(np.mean([f[k] for f in fits])), 3) for k in fits[0]})
# bootstrap (500 resamples: 25 per imputation), warm-started
rng = np.random.default_rng(61); boots = []
tb = time.time()
for imp in imps:
    n = len(imp)
    for _ in range(25):
        s = imp.iloc[rng.choice(n, n, replace=True)]
        try:
            m, e = fit_once(s, start=start)
            if m.converged and np.all(np.abs(e) < 1.5):
                boots.append(e)
        except Exception:
            pass
boots = np.array(boots)
lo, hi = ci(boots)
f2 = pd.DataFrame(dict(param=['a', 'b', 'cprime', 'c', 'indirect'], est=ests.mean(0), lo=lo, hi=hi, p=boot_p(boots)))
print('F2 latent SEM (bootstrap n=%d):' % len(boots)); print(f2.round(3).to_string(index=False))
f2.to_csv(PREFIX + 'res_F2_latent_sem.csv', index=False)
json.dump({'fit_pooled': {k: float(np.mean([f[k] for f in fits])) for k in fits[0]}, 'n_boot': int(len(boots))}, open(PREFIX + 'res_F2_fit.json', 'w'))
print('elapsed %.0fs (bootstrap %.0fs)' % (time.time() - t0, time.time() - tb))
