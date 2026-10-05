"""Regenerate every number that appears in the article's tables (one run, fixed seeds), after step1 ... step5 have been run:
Table 1 (counts), Table 2 (pooled descriptives/correlations), Table 3 (from step1), Table 4 (group comparison + moderated
mediation with CIs), S1 (component-level models), S2 (six moderators), S3 (sensitivity), S4 (CFA), S5 (CMV), reliabilities,
grade ICCs, and the total-model statistics quoted in the text. Output: <prefix>manuscript_tables.json.
Public data (LBC_MODE=public, the default): the de-identified file carries only the dichotomised migration variables and a
top-coded number of siblings, so (i) the migration-type and separation-duration rows of Table 1 are reported in their
dichotomised form, (ii) the '<= 6 months' sensitivity row of Table S3 cannot be reproduced, and (iii) because the number of
siblings enters the imputation model, imputed values differ slightly from the authors' run; all estimates agree with the
article to within +/-0.005."""
import numpy as np, pandas as pd, pickle, json, time
from scipy import stats
from lbc_core import *
from lbc_core import _z, _ols
from sem import SEM, cfa_one_factor, omega_from_fit
t0 = time.time()
PUBLIC = 'custody' not in load_default().columns
imps = pickle.load(open(PREFIX + 'imps_lbc.pkl', 'rb'))
imps535 = pickle.load(open(PREFIX + 'imps_all535.pkl', 'rb'))
raw = load_default()                                   # un-imputed frame (for Table 1 counts and the <=6 months exclusion)
out = {}
covs = ['girl', 'grandparent', 'divorced']
dims = ['em', 'pc', 'es', 'ps', 'fr']
comps = ['vi', 'de', 'ab']

# ---------------- Table 1
L = raw[raw.lbc == 1]; N = raw[raw.lbc == 0]
def cnt(s, v): return int((s == v).sum())
if not PUBLIC:
    out['table1'] = {
        'lbc': dict(n=len(L), boys=cnt(L.gender, 1), girls=cnt(L.gender, 2), age_m=L.age.mean(), age_sd=L.age.std(ddof=1),
                    grades=[cnt(L.grand, g) for g in (3, 4, 5, 6)], sib_m=L.ns.mean(), sib_sd=L.ns.std(ddof=1), divorced=cnt(L.divorce, 1),
                    mig=[cnt(L.aps, 1), cnt(L.aps, 2), cnt(L.aps, 3)], dur=[cnt(L.apt, 1), cnt(L.apt, 2), cnt(L.apt, 3)],
                    care=[cnt(L.custody, 3), cnt(L.custody, 2), cnt(L.custody, 1)], care_unclass=int((~L.custody.isin([1, 2, 3])).sum())),
        'non': dict(n=len(N), boys=cnt(N.gender, 1), girls=cnt(N.gender, 2), age_m=N.age.mean(), age_sd=N.age.std(ddof=1),
                    grades=[cnt(N.grand, g) for g in (3, 4, 5, 6)], sib_m=N.ns.mean(), sib_sd=N.ns.std(ddof=1), divorced=cnt(N.divorce, 1))}
else:   # public file: dichotomised migration variables, top-coded siblings
    out['table1'] = {
        'lbc': dict(n=len(L), boys=cnt(L.gender, 1), girls=cnt(L.gender, 2), age_m=L.age.mean(), age_sd=L.age.std(ddof=1),
                    grades=[cnt(L.grand, g) for g in (3, 4, 5, 6)], sib_m=L.ns.mean(), sib_sd=L.ns.std(ddof=1), divorced=cnt(L.divorced, 1),
                    mig_one_both=[cnt(L.both_migrated, 0), cnt(L.both_migrated, 1)], dur_le5_gt5=[cnt(L.long_sep, 0), cnt(L.long_sep, 1)],
                    care_grandparent_parent=[cnt(L.grandparent, 1), cnt(L.grandparent, 0)], care_unclass=int(L.grandparent.isna().sum()),
                    note='public file: siblings top-coded at 5, migration type/duration dichotomised'),
        'non': dict(n=len(N), boys=cnt(N.gender, 1), girls=cnt(N.gender, 2), age_m=N.age.mean(), age_sd=N.age.std(ddof=1),
                    grades=[cnt(N.grand, g) for g in (3, 4, 5, 6)], sib_m=N.ns.mean(), sib_sd=N.ns.std(ddof=1), divorced=cnt(N.divorced, 1))}
print('Table 1', json.dumps(out['table1'], default=float)[:400])
ITEMS_LBC = ITEMS
miss_cells = int(L[[c for c in ITEMS if c != 'b13']].isna().sum().sum())
out['missing'] = dict(cells_excl_b13=miss_cells, total_excl_b13=len(L) * 51, b13_missing=int(L.b13.isna().sum()))

# ---------------- Table 2
v2 = ['fad', 'em', 'pc', 'es', 'ps', 'fr', 'ls', 'le']
out['table2'] = dict(vars=v2, M=np.mean([i[v2].mean().to_numpy() for i in imps], 0).tolist(), SD=np.mean([i[v2].std(ddof=1).to_numpy() for i in imps], 0).tolist(),
                     R=np.mean([i[v2].corr().to_numpy() for i in imps], 0).tolist(),
                     comp_r=np.mean([i[['vi', 'de', 'ab', 'le', 'fad', 'ls']].corr().to_numpy() for i in imps], 0).tolist())

# ---------------- Table 3 (from step1)
T3 = pd.read_csv(PREFIX + 'table3.csv')
out['table3'] = T3.drop(columns=[c for c in ['p_a', 'p_b', 'p_cprime', 'p_c'] if c in T3.columns]).to_dict('records')
if 'p_a' in T3.columns:   # bootstrap p-values of the a, b, c', c paths (significance stars in Table 3)
    out['table3_path_p'] = {r.predictor: dict(a=float(r.p_a), b=float(r.p_b), cp=float(r.p_cprime), c=float(r.p_c), ind=float(r.p)) for r in T3.itertuples()}

# ---------------- S1: component-level models (15) + Holm/Bonferroni
rows = []
for d in dims:
    for c in comps:
        def st(imp, d=d, c=c):
            return mediation(imp[d].to_numpy(float), imp['ls'].to_numpy(float), imp[c].to_numpy(float), imp[covs].to_numpy(float))
        pt, bt = boot_over_imputations(imps, st, B=5000, seed=101)
        lo, hi = ci(bt); rows.append(dict(dim=d, comp=c, a=pt[0], b=pt[1], cp=pt[2], ind=pt[4], lo=lo[4], hi=hi[4], p=float(boot_p(bt[:, 4]))))
S1 = pd.DataFrame(rows); S1['p_holm'] = holm(S1.p); S1['p_bonf'] = np.minimum(S1.p * 15, 1)
out['S1'] = S1.to_dict('records'); print('S1 done %.0fs' % (time.time() - t0))

# ---------------- S2: exploratory moderation (six moderators), stratified bootstrap of the difference
for i in imps:
    i['age12'] = (i.age >= 12).astype(float)
mods = [('Gender', 'girl', 'Girls', 'Boys'), ('Guardianship', 'grandparent', 'Grandparent', 'Co-resident parent'),
        ('Parental marital status', 'divorced', 'Divorced', 'Not divorced'), ('Type of migration', 'both_migrated', 'Both parents', 'One parent'),
        ('Duration of separation', 'long_sep', '> 5 years', '≤ 5 years'), ('Age', 'age12', '≥ 12 years', '< 12 years')]
rows = []
for label, var, l1, l0 in mods:
    others = [c for c in covs if c != var]
    def st(imp, var=var, others=others):
        res = []
        for g in (1, 0):
            s = imp[imp[var] == g]
            res.append(mediation(s['fad'].to_numpy(float), s['ls'].to_numpy(float), s['le'].to_numpy(float), s[others].to_numpy(float))[4])
        return np.array([res[0], res[1], res[0] - res[1]])
    imps_v = [i[i[var].notna()].reset_index(drop=True) for i in imps]
    pt, bt = boot_over_imputations(imps_v, st, B=5000, seed=111, strata=var)
    lo, hi = ci(bt)
    n1 = int((imps_v[0][var] == 1).sum()); n0 = int((imps_v[0][var] == 0).sum())
    rows.append(dict(moderator=label, g1=l1, n1=n1, g0=l0, n0=n0, ind1=pt[0], ind0=pt[1], diff=pt[2], lo=lo[2], hi=hi[2], p=float(boot_p(bt[:, 2]))))
    print(label, n1, n0, round(pt[2], 3), round(lo[2], 3), round(hi[2], 3))
out['S2'] = rows

# ---------------- S3: sensitivity analyses (unstandardised B, 2,000 resamples as in the manuscript)
def unstd(X, M, Y, C):
    one = np.ones(len(X)); a = _ols(np.column_stack([one, X, C]), M)[1]; b = _ols(np.column_stack([one, X, M, C]), Y)[2]; return np.array([a * b])
def run_unstd(imps_, x='fad', m='ls', y='le', cov=covs, B=2000, seed=121):
    pt, bt = boot_over_imputations(imps_, lambda i: unstd(i[x].to_numpy(float), i[m].to_numpy(float), i[y].to_numpy(float), i[cov].to_numpy(float)), B=B, seed=seed)
    lo, hi = ci(bt); return dict(B=float(pt[0]), lo=float(lo[0]), hi=float(hi[0]), p=float(boot_p(bt[:, 0])), n=len(imps_[0]))
S3 = []
S3.append(dict(label='Main analysis (multiple imputation)', **run_unstd(imps)))
imps_w = []
for i in imps:
    j = i.copy(); lo_ = j.fad.mean() - 3 * j.fad.std(ddof=1); j.loc[j.fad < lo_, 'fad'] = lo_; imps_w.append(j)
S3.append(dict(label='Extreme family functioning score winsorised', **run_unstd(imps_w)))
if not PUBLIC:
    short = set(raw.num[(raw.lbc == 1) & (raw.apt == 1)])
    S3.append(dict(label='Excluding separations of ≤ 6 months', **run_unstd([i[~i.num.isin(short)].reset_index(drop=True) for i in imps])))
else:
    S3.append(dict(label='Excluding separations of ≤ 6 months', note='not reproducible from the public file (duration dichotomised); the two children concerned are among the 5-years-or-less group'))
S3.append(dict(label='Age added as covariate', **run_unstd(imps, cov=covs + ['age'])))
cc = raw[(raw.lbc == 1)].dropna(subset=ITEMS + covs)
S3.append(dict(label='Complete cases only', **run_unstd([add_scores(cc.reset_index(drop=True))])))
for c, nm in zip(comps, ['vigour', 'dedication', 'absorption']):
    S3.append(dict(label=f'Four-item positive communication → {nm}', **run_unstd(imps, x='pc4', y=c)))
out['S3'] = S3; print('S3 done %.0fs' % (time.time() - t0))
out['extreme_case'] = dict(n_below_3sd=int(np.mean([(i.fad < i.fad.mean() - 3 * i.fad.std(ddof=1)).sum() for i in imps])))

# ---------------- S4: CFA (first imputed dataset for FAD; complete cases for SWLS and UWES-S, as in the manuscript)
fac = {'EM': [f'b{i}' for i in range(1, 9)], 'PC': [f'b{i}' for i in range(9, 14)], 'ES': [f'b{i}' for i in range(14, 20)],
       'PS': [f'b{i}' for i in range(20, 26)], 'FR': [f'b{i}' for i in range(26, 31)]}
def fit_fad(X, one_factor=False):
    n = len(X); S = np.cov(X, rowvar=False, bias=True)
    if one_factor:
        m = cfa_one_factor(FAD, 'G').fit(S, n); As, Ss, C = m.std_solution(); return m, {}, {}
    lat = list(fac); free = [('A', it, f) for f, its in fac.items() for it in its] + [('S', it, it) for it in FAD] + [('S', a, b) for i, a in enumerate(lat) for b in lat[i + 1:]]
    m = SEM(FAD, lat, free, {('S', f, f): 1.0 for f in lat}).fit(S, n); As, Ss, C = m.std_solution()
    load = {f: (min(As[m.idx[it], m.idx[f]] for it in its), max(As[m.idx[it], m.idx[f]] for it in its)) for f, its in fac.items()}
    corr = {f'{a}-{b}': float(Ss[m.idx[a], m.idx[b]]) for i, a in enumerate(lat) for b in lat[i + 1:]}
    return m, load, corr
X = imps[0][FAD].to_numpy(float)
m5, load5, corr5 = fit_fad(X); m1, _, _ = fit_fad(X, True)
Xs = raw[raw.lbc == 1][SWLS].dropna().to_numpy(float); ms = cfa_one_factor(SWLS, 'LS').fit(np.cov(Xs, rowvar=False, bias=True), len(Xs))
Xu = raw[raw.lbc == 1][UWES].dropna().to_numpy(float)
ufree = [('A', f'f{i}', 'VI') for i in range(1, 7)] + [('A', f'f{i}', 'DE') for i in range(7, 12)] + [('A', f'f{i}', 'AB') for i in range(12, 18)] + [('S', x, x) for x in UWES] + [('S', 'VI', 'DE'), ('S', 'VI', 'AB'), ('S', 'DE', 'AB')]
mu = SEM(UWES, ['VI', 'DE', 'AB'], ufree, {('S', 'VI', 'VI'): 1, ('S', 'DE', 'DE'): 1, ('S', 'AB', 'AB'): 1}).fit(np.cov(Xu, rowvar=False, bias=True), len(Xu))
def fi(m): return {k: float(v) for k, v in m.fit_indices().items()}
out['S4'] = dict(fad5=fi(m5), fad1=fi(m1), swls=fi(ms), uwes=fi(mu), n_swls=len(Xs), n_uwes=len(Xu),
                 fad_loadings={k: [float(a), float(b)] for k, (a, b) in load5.items()}, fad_corr=corr5,
                 swls_loadings=[float(ms.get('A', it, 'LS')) for it in SWLS],
                 uwes_loadings={f: [float(min(mu.get('A', f'f{i}', f) for i in r)), float(max(mu.get('A', f'f{i}', f) for i in r))] for f, r in [('VI', range(1, 7)), ('DE', range(7, 12)), ('AB', range(12, 18))]},
                 uwes_corr=[float(mu.get('S', 'VI', 'DE')), float(mu.get('S', 'VI', 'AB')), float(mu.get('S', 'DE', 'AB'))],
                 swls_alpha_omega=[None, float(omega_from_fit(ms, SWLS, 'LS'))])
print('S4', {k: round(v, 3) for k, v in out['S4']['fad5'].items()}, {k: round(v, 3) for k, v in out['S4']['swls'].items()}, {k: round(v, 3) for k, v in out['S4']['uwes'].items()})

# ---------------- reliabilities (pooled over imputations): alpha and omega for every scale
def alpha(Xm):
    Xm = np.asarray(Xm, float); k = Xm.shape[1]; return k / (k - 1) * (1 - Xm.var(0, ddof=1).sum() / Xm.sum(1).var(ddof=1))
def omega(Xm, name):
    S = np.cov(Xm, rowvar=False, bias=True); items = [f'x{i}' for i in range(Xm.shape[1])]
    m = cfa_one_factor(items, name).fit(S, len(Xm)); return omega_from_fit(m, items, name)
scales = {'fad': FAD, 'em': fac['EM'], 'pc': fac['PC'], 'es': fac['ES'], 'ps': fac['PS'], 'fr': fac['FR'], 'ls': SWLS, 'le': UWES, 'vi': UWES[:6], 'de': UWES[6:11], 'ab': UWES[11:]}
out['reliability'] = {k: dict(alpha=float(np.mean([alpha(i[its]) for i in imps])), omega=float(np.mean([omega(i[its].to_numpy(float), k) for i in imps[:5]]))) for k, its in scales.items()}
print('reliability', {k: (round(v['alpha'], 2), round(v['omega'], 2)) for k, v in out['reliability'].items()})

# ---------------- S5: common method variance (52 items, first imputed dataset)
X52 = imps[0][ITEMS].to_numpy(float); S52 = np.cov(X52, rowvar=False, bias=True); n = len(X52)
ev = np.linalg.eigvalsh(np.corrcoef(X52, rowvar=False))[::-1]
harman = float(ev[0] / ev.sum()); n_eig = int((ev > 1).sum())
fac9 = dict(fac); fac9.update({'LS': SWLS, 'VI': UWES[:6], 'DE': UWES[6:11], 'AB': UWES[11:]})
lat9 = list(fac9)
free9 = [('A', it, f) for f, its in fac9.items() for it in its] + [('S', it, it) for it in ITEMS] + [('S', a, b) for i, a in enumerate(lat9) for b in lat9[i + 1:]]
fixed9 = {('S', f, f): 1.0 for f in lat9}
m9 = SEM(ITEMS, lat9, free9, fixed9).fit(S52, n)
mth = SEM(ITEMS, lat9 + ['METHOD'], free9 + [('A', it, 'METHOD') for it in ITEMS], {**fixed9, ('S', 'METHOD', 'METHOD'): 1.0}).fit(S52, n, start=np.r_[m9.theta, np.full(len(ITEMS), 0.1)])
m1all = cfa_one_factor(ITEMS, 'G').fit(S52, n)
As, Ss, C = mth.std_solution()
meth_share = float(np.mean([As[mth.idx[it], mth.idx['METHOD']] ** 2 for it in ITEMS]))
subst_share = float(np.mean([sum(As[mth.idx[it], mth.idx[f]] ** 2 for f in lat9) for it in ITEMS]))
out['S5'] = dict(harman_first=harman, n_eig_gt1=n_eig, one=fi(m1all), nine=fi(m9), nine_method=fi(mth), method_share=meth_share, subst_share=subst_share)
print('S5', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out['S5'].items() if not isinstance(v, dict)}, {k: round(v, 3) for k, v in out['S5']['nine_method'].items()})

# ---------------- Table 4: group comparison (Welch t on pooled imputations) + moderated mediation with CIs
vars4 = ['fad', 'em', 'pc', 'es', 'ps', 'fr', 'ls', 'le']
rows = []
for v in vars4:
    stats_ = []
    for i in imps535:
        a_ = i[i.lbc == 1][v]; b_ = i[i.lbc == 0][v]
        t, p = stats.ttest_ind(a_, b_, equal_var=False)
        sp = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1)) / (len(a_) + len(b_) - 2))
        stats_.append([a_.mean(), a_.std(ddof=1), b_.mean(), b_.std(ddof=1), (a_.mean() - b_.mean()) / sp, p])
    s = np.mean(stats_, 0); rows.append(dict(var=v, m1=s[0], sd1=s[1], m0=s[2], sd0=s[3], d=s[4], p=s[5]))
out['table4_desc'] = rows
def modmed(imp):
    x = _z(imp['fad'].to_numpy(float)); m = _z(imp['ls'].to_numpy(float)); y = _z(imp['le'].to_numpy(float)); Lb = imp['lbc'].to_numpy(float)
    one = np.ones(len(imp)); C = imp[['girl', 'age', 'divorced']].to_numpy(float)
    cm = _ols(np.column_stack([one, x, Lb, x * Lb, C]), m); cy = _ols(np.column_stack([one, x, m, Lb, x * Lb, m * Lb, C]), y)
    aN, aL = cm[1], cm[1] + cm[3]; cpN, cpL = cy[1], cy[1] + cy[4]; bN, bL = cy[2], cy[2] + cy[5]
    return np.array([aL, aN, aL - aN, bL, bN, bL - bN, cpL, cpN, cpL - cpN, aL * bL, aN * bN, aL * bL - aN * bN])
pt, bt = boot_over_imputations(imps535, modmed, B=5000, seed=41, strata='lbc'); lo, hi = ci(bt); p = boot_p(bt)
names = ['a_L', 'a_N', 'a_d', 'b_L', 'b_N', 'b_d', 'cp_L', 'cp_N', 'cp_d', 'ind_L', 'ind_N', 'ind_d']
out['table4_mm'] = {nm: dict(est=float(pt[k]), lo=float(lo[k]), hi=float(hi[k]), p=float(p[k])) for k, nm in enumerate(names)}
print('Table 4 mm:', {nm: (round(v['est'], 2), round(v['lo'], 2), round(v['hi'], 2)) for nm, v in out['table4_mm'].items() if nm.endswith('_d')})

# ---------------- S6-S8 from the supplementary runs (already computed with the same pipeline)
out['S6_A'] = pd.read_csv(PREFIX + 'res_A_sensitivity.csv').to_dict('records')
out['S6_F1'] = pd.read_csv(PREFIX + 'res_F1_disattenuated.csv').to_dict('records')
out['S6_F2'] = pd.read_csv(PREFIX + 'res_F2_latent_sem.csv').to_dict('records'); out['S6_F2_fit'] = json.load(open(PREFIX + 'res_F2_fit.json'))
out['S7'] = pd.read_csv(PREFIX + 'res_B_h3.csv').to_dict('records')
out['S8'] = pd.read_csv(PREFIX + 'res_C_lbc_compare.csv').to_dict('records')
out['A3'] = json.load(open(PREFIX + 'res_A3_fad_cfa.json')); out['A3_wording'] = json.load(open(PREFIX + 'res_A3_wording_models.json'))
ri = json.load(open(PREFIX + 'res_A3_riifa.json')); out['A3_riifa_fit'] = {k: float(np.mean([ri[j]['fit'][k] for j in ri])) for k in ['chi2', 'df', 'cfi', 'tli', 'rmsea', 'srmr']}
out['A3_riifa_maxcorr'] = float(max(abs(v) for j in ri for v in ri[j]['corr'].values()))
gicc = {}
def icc(imp, v):
    g = imp.groupby('grand')[v]; grand = imp[v].mean(); k = g.size().to_numpy(); mm = g.mean().to_numpy()
    msb = np.sum(k * (mm - grand) ** 2) / (len(k) - 1)
    ssw = sum(((imp.loc[imp.grand == gg, v] - mm[j]) ** 2).sum() for j, gg in enumerate(sorted(imp.grand.unique())))
    msw = ssw / (len(imp) - len(k)); n0 = (len(imp) - np.sum(k ** 2) / len(imp)) / (len(k) - 1)
    return max((msb - msw) / (msb + (n0 - 1) * msw), 0)
out['grade_icc'] = {v: float(np.mean([icc(i, v) for i in imps])) for v in ['fad', 'ls', 'le']}

# ---------------- total-model statistics quoted in the text, guardianship contrast as in S2, omega_total for the FAD
# (1) total model: bootstrap p for a, b, c', c ; R^2 of the two equations
def st(imp):
    r = mediation(imp['fad'].to_numpy(float), imp['ls'].to_numpy(float), imp['le'].to_numpy(float), imp[covs].to_numpy(float))
    x=_z(imp['fad'].to_numpy(float)); m=_z(imp['ls'].to_numpy(float)); y=_z(imp['le'].to_numpy(float)); one=np.ones(len(imp)); C=imp[covs].to_numpy(float)
    D1=np.column_stack([one,x,C]); D2=np.column_stack([one,x,m,C])
    r2m = 1 - np.var(m - D1 @ _ols(D1, m)) / np.var(m); r2y = 1 - np.var(y - D2 @ _ols(D2, y)) / np.var(y)
    se_c = None
    return np.r_[r, r2m, r2y]
pt, bt = boot_over_imputations(imps, st, B=5000, seed=11)
lo, hi = ci(bt); p = boot_p(bt)
out['total_model'] = {k: dict(est=float(pt[i]), lo=float(lo[i]), hi=float(hi[i]), p=float(p[i])) for i, k in enumerate(['a', 'b', 'cprime', 'c', 'ind', 'r2_m', 'r2_y'])}
# classical SE for c (manuscript reports SE = .05)
se_c = np.mean([np.sqrt(np.sum((_z(i['le'].to_numpy(float)) - np.column_stack([np.ones(len(i)), _z(i['fad'].to_numpy(float)), i[covs].to_numpy(float)]) @ _ols(np.column_stack([np.ones(len(i)), _z(i['fad'].to_numpy(float)), i[covs].to_numpy(float)]), _z(i['le'].to_numpy(float))))**2) / (len(i) - 5) * np.linalg.inv(np.column_stack([np.ones(len(i)), _z(i['fad'].to_numpy(float)), i[covs].to_numpy(float)]).T @ np.column_stack([np.ones(len(i)), _z(i['fad'].to_numpy(float)), i[covs].to_numpy(float)]))[1, 1]) for i in imps])
out['total_model']['se_c'] = float(se_c)
print('total model', {k: (round(v['est'], 3), round(v['p'], 3)) for k, v in out['total_model'].items() if isinstance(v, dict)}, 'se_c %.3f' % se_c)
# (2) guardianship moderation: exclude unclassifiable, group 1 = co-resident parent (as in the manuscript's S2)
uncl = set(raw.num[(raw.lbc == 1) & (raw.grandparent.isna() if PUBLIC else ~raw.custody.isin([1, 2, 3]))])
imps_v = [i[~i.num.isin(uncl)].reset_index(drop=True) for i in imps]
for i in imps_v: i['parent_care'] = 1 - i['grandparent']
def stg(imp):
    res = []
    for g in (1, 0):
        s = imp[imp.parent_care == g]
        res.append(mediation(s['fad'].to_numpy(float), s['ls'].to_numpy(float), s['le'].to_numpy(float), s[['girl', 'divorced']].to_numpy(float))[4])
    return np.array([res[0], res[1], res[0] - res[1]])
pt, bt = boot_over_imputations(imps_v, stg, B=5000, seed=111, strata='parent_care'); lo, hi = ci(bt)
n1 = int((imps_v[0].parent_care == 1).sum()); n0 = int((imps_v[0].parent_care == 0).sum())
row = dict(moderator='Guardianship', g1='Co-resident parent', n1=n1, g0='Grandparent', n0=n0, ind1=float(pt[0]), ind0=float(pt[1]), diff=float(pt[2]), lo=float(lo[2]), hi=float(hi[2]), p=float(boot_p(bt[:, 2])))
out['S2'] = [row if r['moderator'] == 'Guardianship' else r for r in out['S2']]
print('guardianship', row)
# (3) omega_total for the multidimensional FAD from the five-factor model (first imputed dataset)
fac = {'EM': [f'b{i}' for i in range(1, 9)], 'PC': [f'b{i}' for i in range(9, 14)], 'ES': [f'b{i}' for i in range(14, 20)], 'PS': [f'b{i}' for i in range(20, 26)], 'FR': [f'b{i}' for i in range(26, 31)]}
lat = list(fac); free = [('A', it, f) for f, its in fac.items() for it in its] + [('S', it, it) for it in FAD] + [('S', a, b) for i, a in enumerate(lat) for b in lat[i + 1:]]
om = []
for imp in imps[:5]:
    X = imp[FAD].to_numpy(float); S = np.cov(X, rowvar=False, bias=True)
    m = SEM(FAD, lat, free, {('S', f, f): 1.0 for f in lat}).fit(S, len(X)); A, Sm = m.matrices(m.theta)
    lam = np.array([[A[m.idx[it], m.idx[f]] for f in lat] for it in FAD]); Phi = np.array([[Sm[m.idx[a], m.idx[b]] for b in lat] for a in lat]); th = np.array([Sm[m.idx[it], m.idx[it]] for it in FAD])
    L = lam.sum(0); om.append(L @ Phi @ L / (L @ Phi @ L + th.sum()))
out['reliability']['fad']['omega_total_5factor'] = float(np.mean(om)); print('omega_total (5-factor) %.3f' % np.mean(om))
json.dump(out, open(PREFIX + 'manuscript_tables.json', 'w'), indent=1, default=float)
print('saved %smanuscript_tables.json; elapsed %.0fs' % (PREFIX, time.time() - t0))
