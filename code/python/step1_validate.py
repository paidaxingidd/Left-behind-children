"""Step 1: rebuild the manuscript's main analyses with b13 fully imputed (interim),
and check the numbers that do not depend on b13 against the manuscript."""
import numpy as np, pandas as pd, pickle, time
from lbc_core import *

t0=time.time()
df = load_default()
ITEMS29 = [c for c in ITEMS if c in df.columns]
lbc = df[df.lbc == 1].reset_index(drop=True)
print('LBC n =', len(lbc), '| non-LBC n =', (df.lbc == 0).sum())
cols = ITEMS29 + ['girl', 'age', 'grandparent', 'divorced', 'both_migrated', 'long_sep', 'ns']
print('missing per column (LBC):', {c: int(lbc[c].isna().sum()) for c in cols if lbc[c].isna().any()})
imps = [add_scores(i) for i in mice_norm(lbc, cols, binary=('grandparent', 'divorced', 'both_migrated', 'long_sep'), m=20, iters=10)]
pickle.dump(imps, open(PREFIX + 'imps_lbc.pkl', 'wb'))
print('MI done %.0fs' % (time.time() - t0))

# ---- Table 2 check (pooled means/SDs/correlations)
vars_ = ['fad', 'em', 'pc', 'es', 'ps', 'fr', 'ls', 'le']
M = np.mean([i[vars_].mean() for i in imps], axis=0); SD = np.mean([i[vars_].std(ddof=1) for i in imps], axis=0)
R = np.mean([i[vars_].corr().to_numpy() for i in imps], axis=0)
print('\nTable 2 (pooled):')
for k, v in enumerate(vars_):
    print(f'{v:4s} M={M[k]:6.2f} SD={SD[k]:5.2f} ', ' '.join(f'{R[k, j]:5.2f}' for j in range(k)))

# ---- Table 3: mediation models
covs = ['girl', 'grandparent', 'divorced']
def stat_factory(xname):
    def s(imp):
        C = imp[covs].to_numpy(float)
        return mediation(imp[xname].to_numpy(float), imp['ls'].to_numpy(float), imp['le'].to_numpy(float), C)
    return s
rows = []
for x in ['fad', 'em', 'pc', 'es', 'ps', 'fr']:
    pt, bt = boot_over_imputations(imps, stat_factory(x), B=5000, seed=11)
    lo, hi = ci(bt)
    p = boot_p(bt[:, 4]); pp = boot_p(bt)                     # pp: bootstrap p for a, b, c', c, indirect (stars in Table 3)
    rows.append(dict(predictor=x, a=pt[0], b=pt[1], cprime=pt[2], c=pt[3], ind=pt[4], lo=lo[4], hi=hi[4], p=p,
                     pct=100 * pt[4] / pt[3] if abs(pt[3]) > .05 else np.nan,
                     p_a=float(pp[0]), p_b=float(pp[1]), p_cprime=float(pp[2]), p_c=float(pp[3])))
T3 = pd.DataFrame(rows)
T3['p_holm'] = np.r_[T3.p.iloc[0], holm(T3.p.iloc[1:])]
print('\nTable 3 (%s):' % MODE)
print(T3.drop(columns=['p_a', 'p_b', 'p_cprime', 'p_c']).round(3).to_string(index=False))
print('bootstrap p for the a, b, c\' and c paths (stars in Table 3):'); print(T3[['predictor', 'p_a', 'p_b', 'p_cprime', 'p_c']].round(4).to_string(index=False))
T3.to_csv(PREFIX + 'table3.csv', index=False)
print('elapsed %.0fs' % (time.time() - t0))
