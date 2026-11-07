"""
Numerical exploration of a stochastic extension of the two-environment yield model.

Self-contained (numpy only, no Gurobi). It reproduces the deterministic model of
experiment_core.py (market = mean-variance optimal shares over {fixed, mutable, candidate})
and adds two sources of randomness:

  1. Idiosyncratic yield noise   y_ie = beta_e' g_i + eps_ie,  eps ~ N(0, sigma^2), iid over i
  2. Breeding outcome noise      g_c  = t + eta,               eta ~ N(0, s^2 G)

Usage:  python stochastic_yield/prototype.py [--sweep N]
"""

import argparse
import itertools
import numpy as np


class Scenario:
    """Parameters of one experiment (defaults = app.py)."""

    def __init__(self, p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1,
                 g_fix=(0.6, 0.2), g_mut=(-0.1, 0.2)):
        self.p, self.c, self.gamma, self.r_g, self.R = p, c, gamma, r_g, R
        self.g_fix, self.g_mut = np.array(g_fix, float), np.array(g_mut, float)
        self.G = np.array([[1.0, r_g], [r_g, 1.0]])
        self.L = np.linalg.cholesky(self.G)
        self.B1, self.B2 = np.array([1.0, c]), np.array([c, 1.0])
        self.Bbar = p * self.B1 + (1 - p) * self.B2

    def __repr__(self):
        return ("p=%.2f c=%.2f gamma=%.1f r_g=%.2f R=%.2f g_fix=%s g_mut=%s"
                % (self.p, self.c, self.gamma, self.r_g, self.R, self.g_fix, self.g_mut))


# ---------------------------------------------------------------------------
# Yield model and market (portfolio) response
# ---------------------------------------------------------------------------
def moments(sc, gs, sigma=0.0):
    """
    gs: (N, n, 2) genotypes of n varieties at N configurations.
    Returns mu (N, n) and Sigma (N, n, n) of single-variety yields:
    Sigma = p(1-p) d d' + sigma^2 I,  d_i = y_i1 - y_i2  (rank one + idiosyncratic).
    """
    y1, y2 = gs @ sc.B1, gs @ sc.B2
    mu = sc.p * y1 + (1 - sc.p) * y2
    d = y1 - y2
    Sig = sc.p * (1 - sc.p) * d[:, :, None] * d[:, None, :]
    return mu, Sig + sigma ** 2 * np.eye(gs.shape[1])[None]


def solve_simplex_mv(mu, Sig, gamma, ridge=1e-10):
    """
    Batched  max_w  w'mu - gamma/2 w'Sig w   s.t. w >= 0, sum w = 1.
    Enumerates supports (cheap for n <= 4). Returns (w, U).
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
            feas = np.all(wS >= -1e-9, axis=1)
            w = np.zeros((N, n))
            w[:, S] = np.clip(wS, 0, None)
            w /= w.sum(axis=1, keepdims=True)
            U = (np.einsum('ni,ni->n', w, mu)
                 - 0.5 * gamma * np.einsum('ni,nij,nj->n', w, Sig, w))
            upd = feas & (U > best_U)
            best_U[upd], best_w[upd] = U[upd], w[upd]
    return best_w, best_U


def portfolio_value(sc, g_cand, sigma=0.0):
    """V(g_c): market-optimal MV utility over {fixed, mutable, candidate}; g_cand (N, 2)."""
    g_cand = np.atleast_2d(g_cand)
    N = len(g_cand)
    gs = np.stack([np.tile(sc.g_fix, (N, 1)), np.tile(sc.g_mut, (N, 1)), g_cand], axis=1)
    return solve_simplex_mv(*moments(sc, gs, sigma), sc.gamma)


def base_value(sc, sigma=0.0):
    gs = np.stack([sc.g_fix, sc.g_mut])[None]
    return solve_simplex_mv(*moments(sc, gs, sigma), sc.gamma)[1][0]


def ellipse_points(sc, n_r=30, n_th=120):
    """Points filling the Mahalanobis ellipse {(g-g_m)' G^-1 (g-g_m) <= R^2}."""
    r = np.sqrt(np.linspace(0, 1, n_r))[1:] * sc.R
    rr, tt = np.meshgrid(r, np.linspace(0, 2 * np.pi, n_th, endpoint=False))
    u = np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()], 1)
    return np.vstack([sc.g_mut, u @ sc.L.T + sc.g_mut])


# ---------------------------------------------------------------------------
# Checks and experiments
# ---------------------------------------------------------------------------
def check_rank_one_identity(sc, rng):
    """Var(Y) = p(1-p)(1-c)^2 (gbar_0 - gbar_1)^2 depends only on the area-weighted genotype."""
    gs = rng.uniform(-1, 1, size=(1000, 3, 2))
    w = rng.dirichlet(np.ones(3), size=1000)
    _, Sig = moments(sc, gs)
    var = np.einsum('ni,nij,nj->n', w, Sig, w)
    gbar = np.einsum('ni,nik->nk', w, gs)
    var2 = sc.p * (1 - sc.p) * (1 - sc.c) ** 2 * (gbar[:, 0] - gbar[:, 1]) ** 2
    return np.max(np.abs(var - var2))


def check_monte_carlo(sc, rng, sigma=0.1, n_years=400_000):
    """Simulated years vs. analytical moments for one fixed portfolio."""
    gs = np.stack([sc.g_fix, sc.g_mut, np.array([0.0, 0.25])])
    w = np.array([0.5, 0.3, 0.2])
    mu, Sig = moments(sc, gs[None], sigma)
    env1 = rng.random(n_years) < sc.p
    y = np.where(env1[:, None], gs @ sc.B1, gs @ sc.B2) + sigma * rng.standard_normal((n_years, 3))
    Y = y @ w
    return (w @ mu[0], Y.mean()), (w @ Sig[0] @ w, Y.var())


def exp_idiosyncratic(sc, sigma):
    """PoB vs BeatBest when variety yields carry iid noise of sd sigma."""
    pts = ellipse_points(sc)
    U0 = base_value(sc, sigma)
    w, U = portfolio_value(sc, pts, sigma)
    i, j = np.argmax(U), np.argmax(pts @ sc.Bbar)          # PoB target, BeatBest target
    _, U_clone = portfolio_value(sc, sc.g_mut[None], sigma)  # copy of mutable parent
    return dict(U_base=U0, PoB_gain=U[i] - U0, PoB_g=pts[i], PoB_wc=w[i, 2],
                BB_gain=U[j] - U0, BB_wc=w[j, 2], adopt=np.mean(w[:, 2] > 0.01),
                clone_gain=U_clone[0] - U0)


def exp_breeding_noise(sc, scale, sigma=0.0, K=300, seed=1, n_r=30, n_th=120):
    """
    Two-stage view: breeder picks target t, nature draws eta ~ N(0, scale^2 G),
    market re-optimizes shares after seeing g_c = t + eta.
    Compares the deterministic PoB target with argmax_t E[V(t + eta)].
    """
    rng = np.random.default_rng(seed)
    pts = ellipse_points(sc, n_r, n_th)
    eta = (rng.standard_normal((K, 2)) * scale) @ sc.L.T
    U0 = base_value(sc, sigma)
    _, Udet = portfolio_value(sc, pts, sigma)
    w_s, U_s = portfolio_value(sc, (pts[:, None, :] + eta[None]).reshape(-1, 2), sigma)
    U_s, wc_s = U_s.reshape(len(pts), K), w_s[:, 2].reshape(len(pts), K)
    EU = U_s.mean(axis=1)
    out = {}
    for name, idx in [('det_PoB', np.argmax(Udet)), ('stoch_PoB', np.argmax(EU)),
                      ('BeatBest', np.argmax(pts @ sc.Bbar))]:
        out[name] = dict(target=pts[idx], V_det=Udet[idx] - U0, EV=EU[idx] - U0,
                         sd_V=U_s[idx].std(), P_adopt=np.mean(wc_s[idx] > 0.01),
                         E_wc=wc_s[idx].mean(), regret_det=EU.max() - EU[np.argmax(Udet)])
    return out


def random_scenario(rng):
    return Scenario(p=rng.uniform(0.2, 0.8), c=rng.uniform(0.0, 0.6),
                    gamma=rng.choice([5.0, 21.0, 50.0]), r_g=rng.uniform(-0.7, 0.7),
                    R=rng.uniform(0.05, 0.3), g_fix=rng.uniform(-0.2, 1.0, 2),
                    g_mut=rng.uniform(-0.2, 1.0, 2))


def sweep(n, seed=7):
    """Robustness of the headline effects over random scenarios."""
    rng = np.random.default_rng(seed)
    sigmas = (0.0, 0.1, 0.2)
    gap = {s: [] for s in sigmas}       # (PoB - BB) gain, net of nothing
    adopt = {s: [] for s in sigmas}
    regret, rel_regret, V_det, jensen, gap_noise = [], [], [], [], []
    for _ in range(n):
        sc = random_scenario(rng)
        for s in sigmas:
            r = exp_idiosyncratic(sc, s)
            gap[s].append(r['PoB_gain'] - r['BB_gain'])
            adopt[s].append(r['adopt'])
        b = exp_breeding_noise(sc, scale=sc.R, K=150, n_r=10, n_th=40)
        d = b['det_PoB']
        regret.append(d['regret_det'])
        rel_regret.append(d['regret_det'] / max(b['stoch_PoB']['EV'], 1e-9))
        for key in ('det_PoB', 'BeatBest'):
            V_det.append(b[key]['V_det'])
            jensen.append(b[key]['EV'] - b[key]['V_det'])
        gap_noise.append((d['V_det'] - b['BeatBest']['V_det'],
                          b['stoch_PoB']['EV'] - b['BeatBest']['EV']))
    print("\n[5] Sweep over %d random scenarios" % n)
    for s in sigmas:
        g = np.array(gap[s])
        print("  sigma=%.2f  median(PoB-BB gain)=%.5f  median adoption region=%.0f%%"
              % (s, np.median(g), 100 * np.median(adopt[s])))
    g0, g2 = np.array(gap[0.0]), np.array(gap[0.2])
    print("  share of scenarios where PoB-BB gap shrinks from sigma=0 to 0.2: %.0f%%"
          % (100 * np.mean(g2 < g0 - 1e-9)))
    gn = np.array(gap_noise)
    print("  breeding noise s=R: median PoB-BB gap  deterministic %.5f  ->  stochastic %.5f"
          % (np.median(gn[:, 0]), np.median(gn[:, 1])))
    V_det, jensen = np.array(V_det), np.array(jensen)
    for lo, hi in [(-1, 1e-6), (1e-6, 0.02), (0.02, np.inf)]:
        m = (V_det > lo) & (V_det <= hi)
        print("  targets with V(t)-V0 in (%g, %g]: n=%3d, E[V] > V(t) in %3.0f%%, median E[V]-V(t) %+.5f"
              % (max(lo, 0), hi, m.sum(), 100 * np.mean(jensen[m] > 1e-6), np.median(jensen[m])))
    rg, rr = np.array(regret), np.array(rel_regret)
    print("  regret of deterministic PoB target: median %.2e, max %.2e; relative regret > 5%% in %.0f%%"
          % (np.median(rg), rg.max(), 100 * np.mean(rr > 0.05)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', type=int, default=0, help='number of random scenarios')
    args = ap.parse_args()

    sc = Scenario()
    rng = np.random.default_rng(0)
    print("Scenario:", sc)
    print("\n[1] Rank-one identity, max abs error: %.2e" % check_rank_one_identity(sc, rng))
    (m_an, m_mc), (v_an, v_mc) = check_monte_carlo(sc, rng)
    print("[2] Monte Carlo check (sigma=0.1): mean %.4f vs %.4f, var %.5f vs %.5f"
          % (m_an, m_mc, v_an, v_mc))

    print("\n[3] Idiosyncratic yield noise sigma (gains relative to base portfolio)")
    print("  sigma  U_base   PoB_gain  PoB_wc  PoB_g             BB_gain   BB_wc  adopt%  clone_gain")
    for s in (0.0, 0.05, 0.1, 0.2):
        r = exp_idiosyncratic(sc, s)
        print("  %.2f  %7.4f  %8.5f  %6.3f  (%6.3f,%6.3f)  %8.5f  %6.3f  %5.1f  %9.5f"
              % (s, r['U_base'], r['PoB_gain'], r['PoB_wc'], *r['PoB_g'],
                 r['BB_gain'], r['BB_wc'], 100 * r['adopt'], r['clone_gain']))

    for scale in (0.05, 0.1):
        print("\n[4] Breeding outcome noise eta ~ N(0, s^2 G), s=%.2f (R=%.2f), sigma=0" % (scale, sc.R))
        print("  strategy    target             V(t)-V0   E[V]-V0   sd[V]    P(adopt)  E[w_c]")
        for name, r in exp_breeding_noise(sc, scale).items():
            print("  %-10s (%6.3f,%6.3f)  %8.5f  %8.5f  %7.5f  %6.2f    %6.3f"
                  % (name, *r['target'], r['V_det'], r['EV'], r['sd_V'], r['P_adopt'], r['E_wc']))

    if args.sweep:
        sweep(args.sweep)


if __name__ == "__main__":
    main()
