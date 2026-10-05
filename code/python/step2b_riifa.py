"""A3 (cont.): random-intercept (acquiescence) factor model for the FAD -- Maydeu-Olivares & Coffman (2006).
Loadings on the wording factor are fixed to +1 (positively worded, kept as is) and -1 (negatively worded, reverse-scored);
only its variance is estimated. This is more stable than a free-loading method factor."""
import numpy as np, pickle, json
from lbc_core import *
from sem import SEM
imps = pickle.load(open(PREFIX + 'imps_lbc.pkl', 'rb'))
fad29 = [c for c in FAD if c in imps[0].columns]   # 30 items in final mode, 29 in interim
neg = [f'b{i}' for i in list(range(1, 9)) + list(range(14, 20)) + list(range(26, 31))]
fac = {'EM': [f'b{i}' for i in range(1, 9)], 'PC': [c for c in ['b9', 'b10', 'b11', 'b12', 'b13'] if c in fad29], 'ES': [f'b{i}' for i in range(14, 20)],
       'PS': [f'b{i}' for i in range(20, 26)], 'FR': [f'b{i}' for i in range(26, 31)]}
lat = list(fac) + ['RI']
free = [('A', it, f) for f, its in fac.items() for it in its] + [('S', it, it) for it in fad29]
free += [('S', f1, f2) for i, f1 in enumerate(list(fac)) for f2 in list(fac)[i + 1:]] + [('S', 'RI', 'RI')]
fixed = {('S', f, f): 1.0 for f in fac}
for it in fad29:
    fixed[('A', it, 'RI')] = -1.0 if it in neg else 1.0
out = {}
for k, imp in enumerate(imps[:5]):
    X = imp[fad29].to_numpy(float); S = np.cov(X, rowvar=False, bias=True)
    m = SEM(fad29, lat, free, fixed).fit(S, len(X))
    fi = m.fit_indices(); As, Ss, C = m.std_solution()
    A, Sm = m.matrices(m.theta)
    ri_var = Sm[m.idx['RI'], m.idx['RI']]
    item_var = np.diag(C)[:len(fad29)]
    share = np.mean(ri_var / item_var)
    corr = {f'{a}-{b}': float(Ss[m.idx[a], m.idx[b]]) for i, a in enumerate(list(fac)) for b in list(fac)[i + 1:]}
    load = {f: [float(As[m.idx[it], m.idx[f]]) for it in its] for f, its in fac.items()}
    out[k] = dict(fit=fi, ri_var=float(ri_var), share=float(share), corr=corr, load=load, conv=bool(m.converged))
    if k == 0:
        print('imputation 1:', {a: round(float(b), 3) for a, b in fi.items()}, 'converged', m.converged)
        print('  RI variance %.3f -> mean share of item variance %.3f' % (ri_var, share))
        print('  factor corr', {a: round(b, 2) for a, b in corr.items()})
        print('  loadings', {a: [round(x, 2) for x in b] for a, b in load.items()})
print('fit pooled over 5 imputations:', {a: round(float(np.mean([out[k]['fit'][a] for k in out])), 3) for a in ['chi2', 'df', 'cfi', 'tli', 'rmsea', 'srmr']})
print('corr pooled:', {a: round(float(np.mean([out[k]['corr'][a] for k in out])), 2) for a in out[0]['corr']})
json.dump({str(k): v for k, v in out.items()}, open(PREFIX + 'res_A3_riifa.json', 'w'), indent=1, default=float)
