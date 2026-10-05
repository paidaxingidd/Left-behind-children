"""Core data preparation, multiple imputation and mediation routines for the LBC study.

Conventions follow the manuscript's Data analysis section:
  * out-of-range item values -> missing; invalid demographic codes -> missing
  * MICE, Bayesian linear regression with posterior draws (mice 'norm'), m = 20
  * scale scores computed within each imputed dataset
  * OLS on standardized X, M, Y; indirect = a*b
  * 5,000 nonparametric bootstrap resamples spread evenly over the 20 imputations
    (250 each); point estimates = mean over imputations; percentile CIs
"""
import numpy as np
import pandas as pd
try:
    from savreader import read_sav   # only needed for the authors' internal .sav files
except ImportError:
    read_sav = None

FAD = [f'b{i}' for i in range(1, 31)]
SWLS = [f'c{i}' for i in range(1, 6)]
UWES = [f'f{i}' for i in range(1, 18)]
ITEMS = FAD + SWLS + UWES
SUB = {'em': range(1, 9), 'pc': range(9, 14), 'es': range(14, 20), 'ps': range(20, 26), 'fr': range(26, 31)}
UWSUB = {'vi': range(1, 7), 'de': range(7, 12), 'ab': range(12, 18)}
DIM_LABEL = {'fad': 'Family functioning (total)', 'em': 'Affective interaction', 'pc': 'Positive communication',
             'es': 'Self-centredness (reversed)', 'ps': 'Problem solving', 'fr': 'Family rules'}


import os, json
MODE = os.environ.get('LBC_MODE', 'public')          # 'final' = manuscript specification (b13 from 2024 file); 'interim' = b13 dropped
PREFIX = {'final': '', 'public': 'public_', 'interim': 'interim_'}[MODE]


def load_public(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'analysis_data.csv')
    """Load the public analysis-ready file and return the same frame structure as load()."""
    a = pd.read_csv(path)
    df = pd.DataFrame({'num': a.id, 'gender': a.girl + 1, 'age': a.age, 'grand': a.grade, 'ap': 2 - a.lbc,
                       'ns': a.n_siblings, 'order': a.birth_order, 'girl': a.girl, 'lbc': a.lbc, 'divorced': a.divorced,
                       'grandparent': a.grandparent, 'both_migrated': a.both_parents_migrated, 'long_sep': a.separation_over_5y}).astype(float)
    for c in ITEMS:
        df[c] = a[c].astype(float)
    return df.sort_values('num').reset_index(drop=True)


def load_default():
    if MODE == 'public':
        return load_public()
    if MODE == 'final':
        b13 = {int(k): v for k, v in json.load(open('b13_2024.json')).items()}
        return load('data_xinzeng.sav', b13_mode='file2024', b13_2024=b13)
    return load('data_xinzeng.sav', b13_mode='drop')


def load(path, b13_mode='impute_all', b13_2024=None):
    """b13_mode: 'impute_all'  -> b13 missing for everyone (interim)
                 'file2024'    -> b13 from the 2024 file (dict num -> value) for those cases, missing otherwise
                 'as_is'       -> keep the (problematic) b13 values in the file"""
    d, meta = read_sav(path)
    keep = ['num', 'gender', 'age', 'grand', 'ap', 'apt', 'aps', 'custody', 'ns', 'order', 'divorce'] + ITEMS
    df = pd.DataFrame({k: np.asarray(d[k], float) for k in keep})
    for c in FAD:
        df.loc[(df[c] < 1) | (df[c] > 4), c] = np.nan
    if b13_mode == 'drop':
        df['b13'] = np.nan
    for c in SWLS + UWES:
        df.loc[(df[c] < 1) | (df[c] > 7), c] = np.nan
    # data-entry corrections established by comparing with the 2024 file (same questionnaires, earlier entry):
    # cases 165 and 189 carry the invalid code 3 for 'divorce' in the 2025 file but 2 (= not divorced) in the 2024 file
    df.loc[df.num.isin([165, 189]) & (df.divorce == 3), 'divorce'] = 2
    df['girl'] = (df.gender == 2).astype(float)
    df['lbc'] = (df.ap == 1).astype(float)
    df['divorced'] = np.select([df.divorce == 1, df.divorce == 2], [1.0, 0.0], np.nan)
    lbc = df.lbc == 1
    df['grandparent'] = np.select([df.custody == 3, df.custody.isin([1, 2])], [1.0, 0.0], np.nan)
    df['both_migrated'] = np.select([df.aps == 3, df.aps.isin([1, 2])], [1.0, 0.0], np.nan)
    df['long_sep'] = np.select([df.apt == 3, df.apt.isin([1, 2])], [1.0, 0.0], np.nan)
    for c in ['grandparent', 'both_migrated', 'long_sep']:
        df.loc[~lbc, c] = np.nan          # placeholder values entered for non-LBC are not usable
    if b13_mode == 'drop':
        # interim: b13 unusable and no 2024 values available -> 4-item positive communication, 29-item FAD
        df = df.drop(columns=['b13'])
    elif b13_mode == 'file2024':
        df['b13'] = df.num.map(b13_2024).astype(float)
    return df.sort_values('num').reset_index(drop=True)


def mice_norm(df, cols, binary=(), m=20, iters=10, seed=20260924, ridge=1e-5):
    """Chained equations with Bayesian linear regression (mice::mice.impute.norm).
    Binary columns are imputed with norm and rounded to {0,1}."""
    rng = np.random.default_rng(seed)
    X = df[cols].to_numpy(float)
    n, p = X.shape
    miss = np.isnan(X)
    todo = [j for j in range(p) if miss[:, j].any()]
    bin_idx = {cols.index(c) for c in binary}
    out = []
    for k in range(m):
        Z = X.copy()
        for j in todo:
            obs = X[~miss[:, j], j]
            Z[miss[:, j], j] = rng.choice(obs, miss[:, j].sum())
        for _ in range(iters):
            for j in todo:
                o = ~miss[:, j]
                P = np.column_stack([np.ones(n), np.delete(Z, j, axis=1)])
                Po, yo = P[o], Z[o, j]
                xtx = Po.T @ Po
                v = np.linalg.inv(xtx + np.diag(ridge * np.diag(xtx)))
                coef = v @ Po.T @ yo
                res = yo - Po @ coef
                dfree = max(o.sum() - P.shape[1], 1)
                sigma = np.sqrt(res @ res / rng.chisquare(dfree))
                beta = coef + np.linalg.cholesky((v + v.T) / 2) @ rng.standard_normal(P.shape[1]) * sigma
                draw = P[~o] @ beta + rng.standard_normal((~o).sum()) * sigma
                if j in bin_idx:
                    draw = np.clip(np.round(draw), 0, 1)
                Z[~o, j] = draw
        imp = df.copy()
        imp[cols] = Z
        out.append(imp)
    return out


def add_scores(imp):
    imp = imp.copy()
    has13 = 'b13' in imp.columns
    imp['pc4'] = imp[['b9', 'b10', 'b11', 'b12']].sum(1)
    imp['fad29'] = imp[[c for c in FAD if c != 'b13']].sum(1)
    for k, r in SUB.items():
        its = [f'b{i}' for i in r if has13 or i != 13]
        imp[k] = imp[its].sum(1)
    imp['fad'] = imp[[c for c in FAD if has13 or c != 'b13']].sum(1)
    imp['ls'] = imp[SWLS].sum(1)
    imp['ls4'] = imp[['c1', 'c2', 'c3', 'c4']].sum(1)
    imp['le'] = imp[UWES].sum(1)
    for k, r in UWSUB.items():
        imp[k] = imp[[f'f{i}' for i in r]].sum(1)
    # affective / organisational composites: mean of z-scored dimensions
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    imp['aff'] = (z(imp.em) + z(imp.pc) + z(imp.es)) / 3
    imp['org'] = (z(imp.ps) + z(imp.fr)) / 2
    return imp


def _z(v):
    return (v - v.mean()) / v.std(ddof=1)


def _ols(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


def mediation(X, M, Y, C):
    """Standardized single-mediator model. X, M, Y 1-D arrays, C (n x k) covariates.
    Returns a, b, c', c, indirect."""
    x, m, y = _z(X), _z(M), _z(Y)
    one = np.ones(len(x))
    D1 = np.column_stack([one, x, C])
    a = _ols(D1, m)[1]
    D2 = np.column_stack([one, x, m, C])
    cf = _ols(D2, y)
    c = _ols(D1, y)[1]
    return np.array([a, cf[2], cf[1], c, a * cf[2]])


def boot_over_imputations(imps, stat, B=5000, seed=1, strata=None):
    """stat(imp_df) -> 1-D array. Returns point (mean over imputations), bootstrap matrix (B x k).
    strata: column name for stratified resampling (resample within groups)."""
    rng = np.random.default_rng(seed)
    m = len(imps)
    per = B // m
    pts, boots = [], []
    for imp in imps:
        pts.append(stat(imp))
        n = len(imp)
        if strata is None:
            groups = [np.arange(n)]
        else:
            g = imp[strata].to_numpy()
            groups = [np.where(g == u)[0] for u in np.unique(g)]
        for _ in range(per):
            idx = np.concatenate([rng.choice(gi, len(gi), replace=True) for gi in groups])
            boots.append(stat(imp.iloc[idx]))
    return np.mean(pts, axis=0), np.array(boots)


def ci(b, level=0.95):
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return np.quantile(b, lo, axis=0), np.quantile(b, hi, axis=0)


def boot_p(b):
    """two-sided bootstrap p for H0: parameter = 0"""
    b = np.asarray(b)
    p = 2 * np.minimum((b <= 0).mean(axis=0), (b >= 0).mean(axis=0))
    return np.minimum(p, 1.0)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    k = len(ps)
    adj = np.empty(k)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (k - rank) * ps[i]))
        adj[i] = running
    return adj
