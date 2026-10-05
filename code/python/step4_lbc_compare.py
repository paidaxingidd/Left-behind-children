"""C. LBC vs non-LBC comparison (secondary aim) re-examined:
   C0 replicate manuscript model (LBC moderates a, b, c'; covariates girl, age, divorced)
   C1 + age x LS and age x FF interactions (and age x LBC)
   C2 grade fixed effects (+ grade x LS)
   C3 Grade 6 only (grade distributions overlap)
   C4 LBC sample post-stratified (weighted) to the non-LBC grade distribution
   C5 within-group standardized b path (rules out scale differences)"""
import numpy as np, pandas as pd, pickle, time
from lbc_core import *
from lbc_core import _z, _ols
t0 = time.time()
df = load_default()
ITEMS29 = [c for c in ITEMS if c in df.columns]
cols = ITEMS29 + ['girl', 'age', 'divorced', 'ns']
print('missing (all 535):', {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()})
imps = [add_scores(i) for i in mice_norm(df, cols, binary=('divorced',), m=20, iters=10, seed=77)]
for i in imps:
    for g in (4, 5, 6):
        i[f'g{g}'] = (i.grand == g).astype(float)
pickle.dump(imps, open(PREFIX + 'imps_all535.pkl', 'wb'))

def modmed(imp, extra_m=(), extra_y=(), weights=None):
    """returns a_L, a_N, a_diff, b_L, b_N, b_diff, cp_L, cp_N, cp_diff, ind_L, ind_N, ind_diff"""
    x = _z(imp['fad'].to_numpy(float)); m = _z(imp['ls'].to_numpy(float)); y = _z(imp['le'].to_numpy(float))
    L = imp['lbc'].to_numpy(float); one = np.ones(len(imp))
    C = imp[['girl', 'age', 'divorced']].to_numpy(float)
    Em = np.column_stack([imp[c].to_numpy(float) for c in extra_m]) if extra_m else np.empty((len(imp), 0))
    Ey = np.column_stack([imp[c].to_numpy(float) for c in extra_y]) if extra_y else np.empty((len(imp), 0))
    W = np.sqrt(weights) if weights is not None else one
    D1 = np.column_stack([one, x, L, x * L, C, Em]) * W[:, None]
    cm = _ols(D1, m * W)
    D2 = np.column_stack([one, x, m, L, x * L, m * L, C, Ey]) * W[:, None]
    cy = _ols(D2, y * W)
    aN, aL = cm[1], cm[1] + cm[3]
    cpN, cpL = cy[1], cy[1] + cy[4]
    bN, bL = cy[2], cy[2] + cy[5]
    return np.array([aL, aN, aL - aN, bL, bN, bL - bN, cpL, cpN, cpL - cpN, aL * bL, aN * bN, aL * bL - aN * bN])
names = ['a_LBC', 'a_non', 'a_diff', 'b_LBC', 'b_non', 'b_diff', 'cp_LBC', 'cp_non', 'cp_diff', 'ind_LBC', 'ind_non', 'ind_diff']

def run(label, imps_, fn, strata='lbc', B=5000, seed=41):
    pt, bt = boot_over_imputations(imps_, fn, B=B, seed=seed, strata=strata)
    lo, hi = ci(bt); p = boot_p(bt)
    row = dict(label=label, n=len(imps_[0]), n_lbc=int(imps_[0].lbc.sum()))
    for k, nm in enumerate(names):
        row[nm] = pt[k]; row[nm + '_lo'] = lo[k]; row[nm + '_hi'] = hi[k]; row[nm + '_p'] = p[k]
    print(f"{label:48s} n={row['n']} | b: LBC {row['b_LBC']:.2f} non {row['b_non']:.2f} diff {row['b_diff']:.2f} [{row['b_diff_lo']:.2f}, {row['b_diff_hi']:.2f}] p={row['b_diff_p']:.3f} | a diff {row['a_diff']:.2f} [{row['a_diff_lo']:.2f}, {row['a_diff_hi']:.2f}] | ind diff {row['ind_diff']:.2f} [{row['ind_diff_lo']:.2f}, {row['ind_diff_hi']:.2f}]")
    return row

rows = []
rows.append(run('C0 Manuscript model (age as covariate)', imps, lambda i: modmed(i)))
for i in imps:
    i['agec'] = i.age - i.age.mean()
    i['zls'] = _z(i['ls']); i['zfad'] = _z(i['fad'])
    i['age_x_ls'] = i.agec * i.zls; i['age_x_fad'] = i.agec * i.zfad; i['age_x_lbc'] = i.agec * i.lbc
rows.append(run('C1 + age x LS, age x FF, age x LBC', imps, lambda i: modmed(i, extra_m=('age_x_lbc', 'age_x_fad'), extra_y=('age_x_ls', 'age_x_fad', 'age_x_lbc'))))
for i in imps:
    for g in (4, 5, 6):
        i[f'g{g}_x_ls'] = i[f'g{g}'] * i.zls
rows.append(run('C2 Grade fixed effects', imps, lambda i: modmed(i, extra_m=('g4', 'g5', 'g6'), extra_y=('g4', 'g5', 'g6'))))
rows.append(run('C2b Grade FE + grade x LS', imps, lambda i: modmed(i, extra_m=('g4', 'g5', 'g6'), extra_y=('g4', 'g5', 'g6', 'g4_x_ls', 'g5_x_ls', 'g6_x_ls'))))
imps_g6 = [i[i.grand == 6].reset_index(drop=True) for i in imps]
rows.append(run('C3 Grade 6 only', imps_g6, lambda i: modmed(i)))
imps_g56 = [i[i.grand >= 5].reset_index(drop=True) for i in imps]
rows.append(run('C3b Grades 5-6 only', imps_g56, lambda i: modmed(i)))
# C4 post-stratification: weight LBC by grade to match non-LBC grade distribution
for i in imps:
    w = np.ones(len(i))
    pn = i[i.lbc == 0].grand.value_counts(normalize=True); pl = i[i.lbc == 1].grand.value_counts(normalize=True)
    for g in (3, 4, 5, 6):
        w[(i.lbc == 1) & (i.grand == g)] = pn.get(g, 0) / pl.get(g, 1)
    i['w_grade'] = w
rows.append(run('C4 LBC weighted to non-LBC grade distribution', imps, lambda i: modmed(i, weights=i['w_grade'].to_numpy(float))))
# C5 within-group standardization (separate regressions in each group)
def within(imp):
    out = []
    for g in (1, 0):
        s = imp[imp.lbc == g]
        r = mediation(s['fad'].to_numpy(float), s['ls'].to_numpy(float), s['le'].to_numpy(float), s[['girl', 'age', 'divorced']].to_numpy(float))
        out.append(r)
    L, N = out
    return np.array([L[0], N[0], L[0] - N[0], L[1], N[1], L[1] - N[1], L[2], N[2], L[2] - N[2], L[4], N[4], L[4] - N[4]])
rows.append(run('C5 Within-group standardized (separate models)', imps, within))
# descriptive: SD of LS and LE by group; correlation LS-LE by group and by grade
i = imps[0]
for g, lab in ((1, 'LBC'), (0, 'non-LBC')):
    s = i[i.lbc == g]; print(lab, 'SD ls %.2f le %.2f r(ls,le)=%.2f' % (s['ls'].std(), s['le'].std(), s[['ls', 'le']].corr().iloc[0, 1]), 'grade dist', s.grand.value_counts().sort_index().to_dict())
for g in (3, 4, 5, 6):
    s = i[i.grand == g]; sl = s[s.lbc == 1]; sn = s[s.lbc == 0]
    print(f'grade {g}: r(ls,le) LBC={sl[["ls","le"]].corr().iloc[0,1]:.2f} (n={len(sl)})  non-LBC={sn[["ls","le"]].corr().iloc[0,1] if len(sn)>3 else np.nan:.2f} (n={len(sn)})')
pd.DataFrame(rows).to_csv(PREFIX + 'res_C_lbc_compare.csv', index=False)
print('elapsed %.0fs' % (time.time() - t0))
