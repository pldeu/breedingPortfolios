"""
stochastic_core.py: stochastic extension of the two-environment yield model.

Farm-level model (the decision maker is a representative farm with area shares w):

    y_i = beta_e' g_i + eps_i,   e ~ {1: p, 2: 1-p},   eps_i ~ N(0, sigma^2) iid over varieties
    g_c = t + eta,               eta ~ N(0, s^2 G)    (breeding outcome around the target t)

sigma is the farm-level variety-specific shock (variety x farm x year interaction plus
field error when varieties sit on different fields). Shocks common to all varieties on
the farm do not change any decision and are left out.

Timing of the breeding outcome:
    'known'   : the farm observes g_c before choosing shares (recourse). Value E_eta[V(t + eta)].
    'unknown' : the farm adopts on the target t. The candidate carries extra variance
                tau^2 = p b1' S b1 + (1-p) b2' S b2,  S = s^2 G  (closed form, no Monte Carlo).

Clairvoyance sees everything: the environment, every eps_i and eta. It picks the target
after seeing eta and state-contingent shares for every (environment, eps) state.
"""

from dataclasses import dataclass, asdict
import itertools

import numpy as np

TIMINGS = ('known', 'unknown')


@dataclass
class StochasticSpec:
    sigma: float = 0.0          # farm-level idiosyncratic yield sd
    breeding_sd: float = 0.0    # s, with eta ~ N(0, s^2 G)
    timing: str = 'known'       # 'known' or 'unknown' breeding outcome at adoption
    n_draws: int = 200          # breeding draws (optimization and evaluation use separate sets)
    n_states: int = 100         # eps draws per environment for Clairvoyance
    n_clair_draws: int = 50     # breeding draws for Clairvoyance
    max_targets: int = 1500     # coarse target subsample for Monte Carlo searches
    seed: int = 12345

    def __post_init__(self):
        if self.timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, got {self.timing!r}")
        if self.sigma < 0 or self.breeding_sd < 0:
            raise ValueError("sigma and breeding_sd must be non-negative")

    @classmethod
    def from_any(cls, obj):
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        return cls(**obj)

    @property
    def active(self):
        return self.sigma > 0 or self.breeding_sd > 0

    @property
    def mc_targets(self):
        """True when strategy targets must be found by Monte Carlo (recourse on eta)."""
        return self.breeding_sd > 0 and self.timing == 'known'

    def to_dict(self):
        return asdict(self)

    def tau2(self, p, c, G):
        """Extra candidate variance when the farm adopts before eta is known."""
        if self.breeding_sd == 0:
            return 0.0
        S = self.breeding_sd ** 2 * np.asarray(G)
        b1, b2 = np.array([1.0, c]), np.array([c, 1.0])
        return float(p * b1 @ S @ b1 + (1 - p) * b2 @ S @ b2)

    def eta_draws(self, G, n, stream):
        """n draws of eta ~ N(0, s^2 G). stream separates optimization / evaluation draws."""
        if self.breeding_sd == 0:
            return np.zeros((n, 2))
        rng = np.random.default_rng([self.seed, stream])
        L = np.linalg.cholesky(np.asarray(G) + 1e-12 * np.eye(2))
        return self.breeding_sd * rng.standard_normal((n, 2)) @ L.T

    def eps_draws(self, n_assets, stream=3):
        """(n_states, n_assets) farm-level variety shocks for Clairvoyance states."""
        if self.sigma == 0:
            return np.zeros((1, n_assets))
        rng = np.random.default_rng([self.seed, stream])
        return self.sigma * rng.standard_normal((self.n_states, n_assets))


# ---------------------------------------------------------------------------
# Yield moments and the farm's (market's) optimal shares
# ---------------------------------------------------------------------------
def env_yields(gs, c):
    """gs (..., 2) -> yields in environment 1 and 2, each (...)."""
    gs = np.asarray(gs, float)
    return gs[..., 0] + c * gs[..., 1], c * gs[..., 0] + gs[..., 1]


def variety_moments(gs, p, c, sig2=0.0, extra_var=None):
    """
    gs (N, n, 2) -> mu (N, n), Sigma (N, n, n).
    Sigma = p(1-p) d d' + sig2 I + diag(extra_var),  d_i = y_i1 - y_i2.
    """
    y1, y2 = env_yields(gs, c)
    mu = p * y1 + (1 - p) * y2
    d = y1 - y2
    Sig = p * (1 - p) * d[:, :, None] * d[:, None, :]
    diag = np.full(gs.shape[1], sig2, float)
    if extra_var is not None:
        diag = diag + np.asarray(extra_var, float)
    idx = np.arange(gs.shape[1])
    Sig[:, idx, idx] += diag
    return mu, Sig


def market_response(mu, Sig, gamma, ridge=1e-10):
    """
    Batched  max_w  w'mu - gamma/2 w'Sig w   s.t. w >= 0, sum w = 1   (exact, by support
    enumeration; intended for n <= 5 varieties). Returns w (N, n) and utility U (N,).
    """
    N, n = mu.shape
    best_U, best_w = np.full(N, -np.inf), np.zeros((N, n))
    for k in range(1, n + 1):
        if gamma == 0 and k > 1:
            continue  # risk neutral: a vertex is optimal
        for S in map(list, itertools.combinations(range(n), k)):
            if k == 1:
                wS = np.ones((N, 1))
            else:  # KKT of the face: [gamma*Sig 1; 1' 0][w; lam] = [mu; 1]
                K = np.zeros((N, k + 1, k + 1))
                K[:, :k, :k] = gamma * (Sig[:, S][:, :, S] + ridge * np.eye(k))
                K[:, :k, k] = K[:, k, :k] = 1.0
                rhs = np.concatenate([mu[:, S], np.ones((N, 1))], axis=1)
                wS = np.linalg.solve(K, rhs[..., None])[:, :k, 0]
            feas = np.all(wS >= -1e-9, axis=1) & np.all(np.isfinite(wS), axis=1)
            w = np.zeros((N, n))
            w[:, S] = np.clip(np.nan_to_num(wS), 0, None)
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
            U = (np.einsum('ni,ni->n', w, mu)
                 - 0.5 * gamma * np.einsum('ni,nij,nj->n', w, Sig, w))
            upd = feas & (U > best_U)
            best_U[upd], best_w[upd] = U[upd], w[upd]
    return best_w, best_U


def portfolio_stats(gs, p, c, gamma, sig2=0.0, extra_var=None):
    """Optimal shares and moments for each configuration in gs (N, n, 2)."""
    mu, Sig = variety_moments(gs, p, c, sig2, extra_var)
    w, U = market_response(mu, Sig, gamma)
    mean = np.einsum('ni,ni->n', w, mu)
    var = np.einsum('ni,nij,nj->n', w, Sig, w)
    return w, U, mean, var


def _chunks(n_rows, per_row, budget=150_000):
    step = max(1, budget // max(per_row, 1))
    for a in range(0, n_rows, step):
        yield slice(a, min(a + step, n_rows))


# ---------------------------------------------------------------------------
# Evaluation of a fixed set of genotypes / targets
# ---------------------------------------------------------------------------
def evaluate_genotypes(genotypes, n_existing, p, c, gamma, spec, G, adopt_tol=1e-3):
    """
    Farm-level value of a strategy whose genotypes are [existing..., bred targets...].

    known timing:   bred varieties realize t + eta, shares re-optimized per draw
                    (evaluation draws, stream 2). Reports E and sd over eta.
    unknown timing: shares chosen ex ante with tau^2 added to each bred variety.
    """
    g = np.asarray(genotypes, float)
    n = len(g)
    n_c = n - n_existing
    sig2 = spec.sigma ** 2
    if n_c > 0 and spec.mc_targets:
        K = spec.n_draws
        eta = spec.eta_draws(G, K * n_c, stream=2).reshape(K, n_c, 2)
        gs = np.repeat(g[None], K, axis=0)
        gs[:, n_existing:] += eta
        extra = None
    else:
        gs = g[None]
        extra = np.r_[np.zeros(n_existing), np.full(n_c, spec.tau2(p, c, G))] \
            if (n_c > 0 and spec.timing == 'unknown') else None
    w, U, mean, var = portfolio_stats(gs, p, c, gamma, sig2, extra)
    wc = w[:, n_existing:].sum(axis=1)
    w_mean = w.mean(axis=0).tolist()
    return {
        "genotypes": [list(map(float, x)) for x in g],
        "weights": {"env_1": w_mean, "env_2": w_mean},
        "mean_variance": float(U.mean()),
        "mean": float(mean.mean()),
        "variance": float(var.mean()),
        "mv_sd": float(U.std()),
        "p_adopt": float(np.mean(wc > adopt_tol)) if n_c > 0 else float('nan'),
        "e_wc": float(wc.mean()) if n_c > 0 else 0.0,
        "stochastic": True,
    }


def expected_value_of_targets(targets, g_existing, p, c, gamma, spec, G, eta):
    """
    E_eta[V(t + eta)] and E_eta[w_c(t + eta)] for each target t (M, 2), known timing.
    eta: (K, 2) common draws for all targets.
    """
    targets = np.atleast_2d(targets)
    g_existing = np.atleast_2d(g_existing)
    M, K, n_e = len(targets), len(eta), len(g_existing)
    EU, EW = np.empty(M), np.empty(M)
    for sl in _chunks(M, K):
        m = sl.stop - sl.start
        gc = (targets[sl, None, :] + eta[None]).reshape(-1, 1, 2)
        gs = np.concatenate([np.broadcast_to(g_existing, (m * K, n_e, 2)), gc], axis=1)
        w, U, _, _ = portfolio_stats(gs, p, c, gamma, spec.sigma ** 2)
        EU[sl] = U.reshape(m, K).mean(axis=1)
        EW[sl] = w[:, -1].reshape(m, K).mean(axis=1)
    return EU, EW


class EllipseProjector:
    """
    Feasible set of the breeding target: the ellipse (g - g_m)' G^-1 (g - g_m) <= R^2
    intersected with the box used by the Gurobi models (_get_ellipse_bounds clips to [-1, 1]).
    """

    def __init__(self, g_mut, G, R, box=(-1.0, 1.0)):
        self.g_mut = np.asarray(g_mut, float)
        self.L = np.linalg.cholesky(np.asarray(G) + 1e-12 * np.eye(2))
        self.Linv = np.linalg.inv(self.L)
        self.R = R
        self.box = box

    def _u(self, g):
        return (np.atleast_2d(g) - self.g_mut) @ self.Linv.T / max(self.R, 1e-12)

    def __call__(self, g):
        u = self._u(g)
        r = np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where(r > 1, u / np.maximum(r, 1e-300), u)
        return np.clip(self.g_mut + self.R * u @ self.L.T, *self.box)

    def feasible(self, g, tol=1e-9):
        g = np.atleast_2d(g)
        inside = np.linalg.norm(self._u(g), axis=1) <= 1 + tol
        return inside & np.all((g >= self.box[0] - tol) & (g <= self.box[1] + tol), axis=1)


def search_targets(pts, objective, max_targets, h, project=None, tol=1e-5):
    """
    Maximize objective over the breeding ellipse:
    1. coarse subsample of at most max_targets feasible grid points,
    2. all grid points within distance h of the coarse optimum,
    3. continuous pattern search (8 directions, step halving) projected onto the ellipse.
    Returns (best_point, best_value).
    """
    pts = np.asarray(pts, float)
    if len(pts) <= max_targets:
        v = objective(pts)
        best, best_v = pts[np.argmax(v)], float(v.max())
    else:
        idx = np.unique(np.linspace(0, len(pts) - 1, max_targets).round().astype(int))
        v = objective(pts[idx])
        best, best_v = pts[idx][np.argmax(v)], float(v.max())
        near = pts[np.linalg.norm(pts - best, axis=1) <= h]
        if len(near):
            v2 = objective(near)
            if v2.max() > best_v:
                best, best_v = near[np.argmax(v2)], float(v2.max())
    if project is not None:
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        dirs = np.c_[np.cos(ang), np.sin(ang)]
        step = h
        while step > tol:
            cand = project(best + step * dirs)
            cand = cand[project.feasible(cand)] if hasattr(project, 'feasible') else cand
            v = objective(cand) if len(cand) else np.array([-np.inf])
            if v.max() > best_v + 1e-12:
                best, best_v = cand[np.argmax(v)], float(v.max())
            else:
                step *= 0.5
    return best, best_v


def mc_target(pts, g_existing, p, c, gamma, spec, G, h, objective='mv', project=None):
    """Target maximizing E[V] ('mv') or E[w_c] ('share') under known timing (draws: stream 1)."""
    eta = spec.eta_draws(G, spec.n_draws, stream=1)

    def f(t):
        EU, EW = expected_value_of_targets(t, g_existing, p, c, gamma, spec, G, eta)
        return EU if objective == 'mv' else EW + 1e-6 * EU  # tie-break on value

    return search_targets(pts, f, spec.max_targets, h, project)[0]


# ---------------------------------------------------------------------------
# Clairvoyance with full information
# ---------------------------------------------------------------------------
def _clairvoyant_from_state_yields(Ys, probs, gamma):
    """
    Ys (M, S, n): yields of n varieties in S states; probs (S,).
    With state-contingent shares the portfolio yield Y_k can be any value in
    [min_i Ys_k, max_i Ys_k]. Maximizing mean - gamma/2 var over this box gives
    Y_k = clip(mu + 1/gamma, lo_k, hi_k) with mu the resulting mean (unique fixed point).
    """
    lo, hi = Ys.min(axis=2), Ys.max(axis=2)
    if gamma == 0:
        Y = hi
    else:
        a, b = lo.min(axis=1), hi.max(axis=1)
        for _ in range(60):
            mid = 0.5 * (a + b)
            f = (np.clip(mid[:, None] + 1.0 / gamma, lo, hi) * probs).sum(axis=1) - mid
            a, b = np.where(f > 0, mid, a), np.where(f > 0, b, mid)
        Y = np.clip(0.5 * (a + b)[:, None] + 1.0 / gamma, lo, hi)
    mean = (Y * probs).sum(axis=1)
    var = (((Y - mean[:, None]) ** 2) * probs).sum(axis=1)
    return mean - 0.5 * gamma * var, mean, var, Y


def _state_weights(Ys, Y):
    """Canonical state-contingent shares reaching portfolio yield Y_k (mix of best and next below)."""
    S, n = Ys.shape
    best = np.argmax(Ys, axis=1)
    below = np.where(Ys <= Y[:, None] + 1e-12, Ys, -np.inf)
    a = np.argmax(below, axis=1)
    yb, ya = Ys[np.arange(S), best], Ys[np.arange(S), a]
    lam = np.where(yb - ya > 1e-12, (Y - ya) / np.maximum(yb - ya, 1e-12), 1.0)
    W = np.zeros((S, n))
    W[np.arange(S), a] += 1 - lam
    W[np.arange(S), best] += lam
    return W


def clairvoyance_full_info(pts, g_existing, p, c, gamma, spec, G, h, project=None):
    """
    Full-information benchmark: for each breeding draw eta, choose the target t (after
    seeing eta) and state-contingent shares for every (environment, eps) state.
    Returns a result dict compatible with experiment_core outputs.
    """
    g_existing = np.atleast_2d(g_existing)
    n_e = len(g_existing)
    n = n_e + 1
    eps = spec.eps_draws(n)                                    # (S0, n)
    S0 = len(eps)
    probs = np.r_[np.full(S0, p / S0), np.full(S0, (1 - p) / S0)]
    y1e, y2e = env_yields(g_existing, c)

    def state_yields(gc):
        y1c, y2c = env_yields(gc, c)                            # (M,)
        Y1 = np.concatenate([np.broadcast_to(y1e, (len(gc), n_e)), y1c[:, None]], axis=1)
        Y2 = np.concatenate([np.broadcast_to(y2e, (len(gc), n_e)), y2c[:, None]], axis=1)
        return np.concatenate([Y1[:, None, :] + eps[None], Y2[:, None, :] + eps[None]], axis=1)

    def objective_for(e):
        def f(t):
            out = np.empty(len(t))
            for sl in _chunks(len(t), 2 * S0 * n, budget=600_000):
                out[sl] = _clairvoyant_from_state_yields(state_yields(t[sl] + e), probs, gamma)[0]
            return out
        return f

    n_eta = spec.n_clair_draws if spec.breeding_sd > 0 else 1
    etas = spec.eta_draws(G, n_eta, stream=4)
    U, M_, V_, T, W1, W2, WC = [], [], [], [], [], [], []
    for e in etas:
        t, _ = search_targets(pts, objective_for(e), spec.max_targets, h, project)
        Ys = state_yields((t + e)[None])
        u, m, v, Y = _clairvoyant_from_state_yields(Ys, probs, gamma)
        W = _state_weights(Ys[0], Y[0])
        U.append(u[0]); M_.append(m[0]); V_.append(v[0]); T.append(t)
        W1.append(W[:S0].mean(axis=0)); W2.append(W[S0:].mean(axis=0))
        WC.append(p * W[:S0, -1].mean() + (1 - p) * W[S0:, -1].mean())
    U = np.array(U)
    return {
        "genotypes": [list(map(float, x)) for x in g_existing] + [list(map(float, np.mean(T, axis=0)))],
        "weights": {"env_1": np.mean(W1, axis=0).tolist(), "env_2": np.mean(W2, axis=0).tolist()},
        "mean_variance": float(U.mean()),
        "mean": float(np.mean(M_)),
        "variance": float(np.mean(V_)),
        "mv_sd": float(U.std()),
        "p_adopt": float('nan'),
        "e_wc": float(np.mean(WC)),
        "stochastic": True,
    }
