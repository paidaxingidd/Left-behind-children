"""Small ML structural equation engine (RAM notation, covariance structure only).

Model is specified as a list of parameter tuples:
    ('A', to, from)   directed path  from -> to  (loading or regression)
    ('S', v1, v2)     (co)variance
Fixed values are given in `fixed` as {('A', to, from): value, ('S', v, v): value}.
Observed variable order = columns of the covariance matrix passed to fit().
"""
import numpy as np
from scipy.optimize import minimize


class SEM:
    def __init__(self, observed, latent, free, fixed=None):
        self.obs = list(observed)
        self.lat = list(latent)
        self.vars = self.obs + self.lat
        self.idx = {v: i for i, v in enumerate(self.vars)}
        self.p = len(self.obs)
        self.v = len(self.vars)
        self.free = list(free)
        self.fixed = dict(fixed or {})
        self.F = np.zeros((self.p, self.v))
        self.F[np.arange(self.p), np.arange(self.p)] = 1.0

    def matrices(self, theta):
        A = np.zeros((self.v, self.v))
        S = np.zeros((self.v, self.v))
        for (m, a, b), val in list(self.fixed.items()):
            i, j = self.idx[a], self.idx[b]
            if m == 'A':
                A[i, j] = val
            else:
                S[i, j] = S[j, i] = val
        for k, (m, a, b) in enumerate(self.free):
            i, j = self.idx[a], self.idx[b]
            if m == 'A':
                A[i, j] = theta[k]
            else:
                S[i, j] = S[j, i] = theta[k]
        return A, S

    def implied(self, theta):
        A, S = self.matrices(theta)
        B = np.linalg.inv(np.eye(self.v) - A)
        G = self.F @ B
        return G @ S @ G.T, A, S, B

    def _obj(self, theta, Sobs, logdetS):
        Sig, A, S, B = self.implied(theta)
        sign, logdet = np.linalg.slogdet(Sig)
        if sign <= 0:
            return 1e10, np.zeros_like(theta)
        Si = np.linalg.inv(Sig)
        f = logdet + np.trace(Sobs @ Si) - logdetS - self.p
        M = Si - Si @ Sobs @ Si
        G = self.F @ B                      # p x v
        D = G.T @ M @ G                     # v x v  (dF/dS)
        DA = 2.0 * D @ S @ B.T              # dF/dA = 2 G'MG Omega B'
        grad = np.empty_like(theta)
        for k, (m, a, b) in enumerate(self.free):
            i, j = self.idx[a], self.idx[b]
            if m == 'A':
                grad[k] = DA[i, j]
            else:
                grad[k] = D[i, j] if i == j else 2.0 * D[i, j]
        return f, grad

    def start_values(self, Sobs):
        th = []
        for (m, a, b) in self.free:
            if m == 'S' and a == b:
                th.append(0.5 * Sobs[self.idx[a], self.idx[a]] if a in self.obs else 0.5)
            elif m == 'S':
                th.append(0.0)
            else:
                th.append(0.5 if b in self.lat else 0.1)
        return np.array(th)

    def fit(self, Sobs, n, start=None):
        Sobs = np.asarray(Sobs, float)
        logdetS = np.linalg.slogdet(Sobs)[1]
        th0 = self.start_values(Sobs) if start is None else start
        bounds = [(1e-6, None) if (m == 'S' and a == b) else (None, None) for (m, a, b) in self.free]
        res = minimize(self._obj, th0, args=(Sobs, logdetS), jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        self.theta = res.x
        self.fmin = res.fun
        self.n = n
        self.Sobs = Sobs
        self.converged = res.success
        return self

    # ---- fit statistics (lavaan conventions: chi2 = N * F_ML, sample cov with divisor N)
    def fit_indices(self):
        p = self.p
        chi2 = (self.n - 1) * self.fmin   # (N-1) convention, as in the manuscript's earlier analyses
        df = p * (p + 1) // 2 - len(self.free)
        S = self.Sobs
        f0 = np.sum(np.log(np.diag(S))) - np.linalg.slogdet(S)[1]
        chi2_0 = (self.n - 1) * f0
        df0 = p * (p - 1) // 2
        cfi = 1 - max(chi2 - df, 0) / max(chi2_0 - df0, chi2 - df, 1e-12)
        tli = ((chi2_0 / df0) - (chi2 / df)) / ((chi2_0 / df0) - 1) if df > 0 else np.nan
        rmsea = np.sqrt(max(chi2 - df, 0) / (df * (self.n - 1))) if df > 0 else 0.0
        Sig = self.implied(self.theta)[0]
        sd = np.sqrt(np.diag(S))
        R = (S - Sig) / np.outer(sd, sd)
        il = np.tril_indices(p)
        srmr = np.sqrt(np.mean(R[il] ** 2))
        return {'chi2': chi2, 'df': df, 'cfi': cfi, 'tli': tli, 'rmsea': rmsea, 'srmr': srmr}

    def std_solution(self):
        """Fully standardized A (paths) and S (covariances)."""
        A, S = self.matrices(self.theta)
        B = np.linalg.inv(np.eye(self.v) - A)
        C = B @ S @ B.T                      # model-implied covariance of all variables
        sd = np.sqrt(np.diag(C))
        As = A * sd[None, :] / sd[:, None]
        Ss = S / np.outer(sd, sd)
        return As, Ss, C

    def get(self, m, a, b, standardized=True):
        As, Ss, C = self.std_solution()
        i, j = self.idx[a], self.idx[b]
        if standardized:
            return As[i, j] if m == 'A' else Ss[i, j]
        A, S = self.matrices(self.theta)
        return A[i, j] if m == 'A' else S[i, j]


def cfa_one_factor(items, name='F'):
    free = [('A', it, name) for it in items] + [('S', it, it) for it in items]
    return SEM(items, [name], free, fixed={('S', name, name): 1.0})


def omega_from_fit(model, items, factor):
    """omega computed from standardized loadings (correlation metric)"""
    lam = np.array([model.get('A', it, factor, standardized=True) for it in items])
    return lam.sum() ** 2 / (lam.sum() ** 2 + np.sum(1 - lam ** 2))
