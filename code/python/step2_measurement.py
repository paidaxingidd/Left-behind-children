"""A. Measurement sensitivity: (A1) exclude Grade 3, (A2) 4-item SWLS, (A3) FAD CFA with a wording method factor,
(A3b) one-factor and two-wording-factor FAD models. Also D/E: covariate sensitivity (grade fixed effects, siblings)."""
import numpy as np, pandas as pd, pickle, json, time
from lbc_core import *
from sem import SEM, cfa_one_factor

t0 = time.time()
imps = pickle.load(open(PREFIX + 'imps_lbc.pkl', 'rb'))
covs = ['girl', 'grandparent', 'divorced']
res = {}

def med_stat(x='fad', mvar='ls', y='le', cov=covs):
    def s(imp):
        return mediation(imp[x].to_numpy(float), imp[mvar].to_numpy(float), imp[y].to_numpy(float), imp[cov].to_numpy(float))
    return s

def run(label, imps_, stat, B=5000, seed=21):
    pt, bt = boot_over_imputations(imps_, stat, B=B, seed=seed)
    lo, hi = ci(bt)
    out = dict(label=label, n=len(imps_[0]), a=pt[0], b=pt[1], cprime=pt[2], c=pt[3], ind=pt[4], ind_lo=lo[4], ind_hi=hi[4],
               p=float(boot_p(bt[:, 4])), a_lo=lo[0], a_hi=hi[0], b_lo=lo[1], b_hi=hi[1])
    print(f"{label:55s} n={out['n']:3d} a={out['a']:.3f} b={out['b']:.3f} c'={out['cprime']:.3f} ind={out['ind']:.3f} [{out['ind_lo']:.3f}, {out['ind_hi']:.3f}] p={out['p']:.3f}")
    return out

rows = []
rows.append(run('Main model (manuscript specification)', imps, med_stat()))
# A1: exclude grade 3
imps_g46 = [i[i.grand >= 4].reset_index(drop=True) for i in imps]
rows.append(run('A1 Excluding Grade 3', imps_g46, med_stat()))
imps_age10 = [i[i.age >= 10].reset_index(drop=True) for i in imps]
rows.append(run('A1b Age >= 10 only', imps_age10, med_stat()))
# A2: SWLS 4 items
rows.append(run('A2 SWLS items 1-4 as mediator', imps, med_stat(mvar='ls4')))
rows.append(run('A1+A2 Grade>=4 and SWLS 1-4', imps_g46, med_stat(mvar='ls4')))
# grade fixed effects (partial answer to clustering)
for i in imps:
    for g in (4, 5, 6):
        i[f'g{g}'] = (i.grand == g).astype(float)
rows.append(run('D Grade fixed effects added as covariates', imps, med_stat(cov=covs + ['g4', 'g5', 'g6'])))
# E: siblings as SES proxy
rows.append(run('E Number of siblings added as covariate', imps, med_stat(cov=covs + ['ns'])))
rows.append(run('E2 Siblings + age + grade FE', imps, med_stat(cov=covs + ['ns', 'age', 'g4', 'g5', 'g6'])))
pd.DataFrame(rows).to_csv(PREFIX + 'res_A_sensitivity.csv', index=False)

# reliability of ls4 / ls
def alpha(X):
    X = np.asarray(X); k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))
print('alpha SWLS5 = %.3f, SWLS4 = %.3f' % (np.mean([alpha(i[SWLS]) for i in imps]), np.mean([alpha(i[['c1','c2','c3','c4']]) for i in imps])))
print('alpha SWLS5 grade>=4 = %.3f' % np.mean([alpha(i[SWLS]) for i in imps_g46]))
# grade-level ICC for outcomes (how much variance is between grades)
def icc(imp, v):
    g = imp.groupby('grand')[v]
    grand = imp[v].mean(); k = g.size().to_numpy(); m = g.mean().to_numpy()
    ssb = np.sum(k * (m - grand) ** 2); msb = ssb / (len(k) - 1)
    ssw = np.sum([((imp.loc[imp.grand == gg, v] - m[j]) ** 2).sum() for j, gg in enumerate(sorted(imp.grand.unique()))])
    msw = ssw / (len(imp) - len(k)); n0 = (len(imp) - np.sum(k ** 2) / len(imp)) / (len(k) - 1)
    return max((msb - msw) / (msb + (n0 - 1) * msw), 0)
print('grade ICC: fad %.3f ls %.3f le %.3f' % tuple(np.mean([icc(i, v) for i in imps]) for v in ['fad', 'ls', 'le']))

# ---- A3: FAD CFA with wording method factor (29 items, complete cases on first imputation vs pooled)
fad29 = [c for c in FAD if c in imps[0].columns]   # 30 items in final mode, 29 in interim
neg = [f'b{i}' for i in list(range(1, 9)) + list(range(14, 20)) + list(range(26, 31))]
pos = [c for c in fad29 if c not in neg]
fac = {'EM': [f'b{i}' for i in range(1, 9)], 'PC': [c for c in ['b9', 'b10', 'b11', 'b12', 'b13'] if c in fad29], 'ES': [f'b{i}' for i in range(14, 20)],
       'PS': [f'b{i}' for i in range(20, 26)], 'FR': [f'b{i}' for i in range(26, 31)]}
def fad_model(method):
    lat = list(fac)
    free = [('A', it, f) for f, its in fac.items() for it in its] + [('S', it, it) for it in fad29]
    free += [('S', f1, f2) for i, f1 in enumerate(lat) for f2 in lat[i + 1:]]
    fixed = {('S', f, f): 1.0 for f in lat}
    if method == 'neg':
        lat = lat + ['NEG']; free += [('A', it, 'NEG') for it in neg]; fixed[('S', 'NEG', 'NEG')] = 1.0
    elif method == 'both':
        lat = lat + ['NEG', 'POS']; free += [('A', it, 'NEG') for it in neg] + [('A', it, 'POS') for it in pos]
        fixed[('S', 'NEG', 'NEG')] = 1.0; fixed[('S', 'POS', 'POS')] = 1.0
    return SEM(fad29, lat, free, fixed)
X = imps[0][fad29].to_numpy(float); S = np.cov(X, rowvar=False, bias=True); n = len(X)
cfa = {}
for method in ['none', 'neg', 'both']:
    m = fad_model(method).fit(S, n)
    fi = m.fit_indices()
    As, Ss, C = m.std_solution()
    lat = list(fac)
    corr = {f'{a}-{b}': round(float(Ss[m.idx[a], m.idx[b]]), 2) for i, a in enumerate(lat) for b in lat[i + 1:]}
    load = {f: [round(float(As[m.idx[it], m.idx[f]]), 2) for it in its] for f, its in fac.items()}
    extra = {}
    if method != 'none':
        extra['neg_method_loadings'] = [round(float(As[m.idx[it], m.idx['NEG']]), 2) for it in neg]
        extra['var_explained_method'] = round(float(np.mean(np.array(extra['neg_method_loadings']) ** 2)), 3)
    if method == 'both':
        extra['pos_method_loadings'] = [round(float(As[m.idx[it], m.idx['POS']]), 2) for it in pos]
    cfa[method] = dict(fit={k: round(float(v), 3) for k, v in fi.items()}, converged=bool(m.converged), factor_corr=corr, loadings=load, **extra)
    print(method, cfa[method]['fit'], corr)
json.dump(cfa, open(PREFIX + 'res_A3_fad_cfa.json', 'w'), indent=1)
print('elapsed %.0fs' % (time.time() - t0))

# ---- A3b: one-factor and two-wording-factor models for the 30 FAD items (Table S9, first two rows; first imputed dataset)
X30 = imps[0][FAD].to_numpy(float); S30 = np.cov(X30, rowvar=False, bias=True); n30 = len(X30)
m_one = cfa_one_factor(FAD, 'G').fit(S30, n30)
NEG = [f'b{i}' for i in list(range(1, 9)) + list(range(14, 20)) + list(range(26, 31))]   # negatively worded (reverse-scored) items
POS = [f'b{i}' for i in list(range(9, 14)) + list(range(20, 26))]                        # positively worded items
free_w = [('A', it, 'NEG') for it in NEG] + [('A', it, 'POS') for it in POS] + [('S', it, it) for it in FAD] + [('S', 'NEG', 'POS')]
m_two = SEM(FAD, ['NEG', 'POS'], free_w, {('S', 'NEG', 'NEG'): 1.0, ('S', 'POS', 'POS'): 1.0}).fit(S30, n30)
fi_ = lambda m: {k: round(float(v), 3) for k, v in m.fit_indices().items()}
wording = dict(one_factor=fi_(m_one), two_wording_factors=fi_(m_two), r_neg_pos=round(float(m_two.get('S', 'NEG', 'POS')), 2))
json.dump(wording, open(PREFIX + 'res_A3_wording_models.json', 'w'), indent=1)
print('A3b one-factor vs two wording factors:', wording)
