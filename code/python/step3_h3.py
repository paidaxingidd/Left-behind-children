"""B. Hypothesis 3 re-examined: (B1) five dimensions in one joint mediation model, (B2) affective vs organisational
composites in one model, pairwise differences of indirect associations with bootstrap CIs (90% CI = TOST at alpha=.05),
(B3) differences between the separately estimated indirect associations (same resamples)."""
import numpy as np, pandas as pd, pickle, json, time
from lbc_core import *
from lbc_core import _z, _ols
t0 = time.time()
imps = pickle.load(open(PREFIX + 'imps_lbc.pkl', 'rb'))
covs = ['girl', 'grandparent', 'divorced']
dims = ['em', 'pc', 'es', 'ps', 'fr']

def joint_stat(imp):
    C = imp[covs].to_numpy(float); one = np.ones(len(imp))
    X = np.column_stack([_z(imp[d].to_numpy(float)) for d in dims]); m = _z(imp['ls'].to_numpy(float)); y = _z(imp['le'].to_numpy(float))
    a = _ols(np.column_stack([one, X, C]), m)[1:6]
    cf = _ols(np.column_stack([one, X, m, C]), y); b = cf[6]; cp = cf[1:6]
    ind = a * b
    # composites
    A2 = np.column_stack([_z(imp['aff'].to_numpy(float)), _z(imp['org'].to_numpy(float))])
    a2 = _ols(np.column_stack([one, A2, C]), m)[1:3]
    cf2 = _ols(np.column_stack([one, A2, m, C]), y); b2 = cf2[3]; cp2 = cf2[1:3]
    ind2 = a2 * b2
    # separately estimated (as in Table 3) from the same resample
    sep = np.array([mediation(imp[d].to_numpy(float), imp['ls'].to_numpy(float), imp['le'].to_numpy(float), C)[4] for d in dims])
    sep_aff = sep[:3].mean(); sep_org = sep[3:].mean()
    return np.concatenate([a, [b], cp, ind, a2, [b2], cp2, ind2, [ind2[0] - ind2[1]], sep, [sep_aff - sep_org],
                           [sep[i] - sep[j] for i in range(5) for j in range(i + 1, 5)],
                           [ind[i] - ind[j] for i in range(5) for j in range(i + 1, 5)]])
names = ([f'a_{d}' for d in dims] + ['b'] + [f'cp_{d}' for d in dims] + [f'ind_{d}' for d in dims] +
         ['a_aff', 'a_org', 'b2', 'cp_aff', 'cp_org', 'ind_aff', 'ind_org', 'diff_aff_org_joint'] +
         [f'sep_{d}' for d in dims] + ['diff_aff_org_sep'] +
         [f'sepdiff_{dims[i]}_{dims[j]}' for i in range(5) for j in range(i + 1, 5)] +
         [f'jointdiff_{dims[i]}_{dims[j]}' for i in range(5) for j in range(i + 1, 5)])
pt, bt = boot_over_imputations(imps, joint_stat, B=5000, seed=31)
lo95, hi95 = ci(bt, .95); lo90, hi90 = ci(bt, .90)
res = pd.DataFrame(dict(param=names, est=pt, lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, p=boot_p(bt)))
res.to_csv(PREFIX + 'res_B_h3.csv', index=False)
pd.set_option('display.width', 200)
print(res.round(3).to_string(index=False))
# equivalence: largest |90% CI bound| for the differences = smallest delta for which equivalence holds
for nm in ['diff_aff_org_joint', 'diff_aff_org_sep']:
    r = res[res.param == nm].iloc[0]
    print(f'{nm}: est={r.est:.3f} 90% CI [{r.lo90:.3f}, {r.hi90:.3f}] -> equivalence supported for bounds beyond +/-{max(abs(r.lo90), abs(r.hi90)):.3f}')
sd = res[res.param.str.startswith('sepdiff')]
print('max |90%% bound| over 10 pairwise separate-model differences: %.3f; over joint-model differences: %.3f' % (
    max(sd.lo90.abs().max(), sd.hi90.abs().max()), max(res[res.param.str.startswith('jointdiff')].lo90.abs().max(), res[res.param.str.startswith('jointdiff')].hi90.abs().max())))
print('elapsed %.0fs' % (time.time() - t0))
