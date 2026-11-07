"""
Consistency tests for the stochastic extension. Run from the repository root:

    python stochastic_yield/test_stochastic.py      (or: pytest stochastic_yield/test_stochastic.py)

Each test compares two independent computations of the same quantity.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import stochastic_core as stc  # noqa: E402
from experiment_core import ExperimentRunner  # noqa: E402

RNG = np.random.default_rng(2026)


def make_runner(spec=None, p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1,
                g_fixed=(0.6, 0.2), g_mutable=(-0.1, 0.2), n=201):
    sc = [{"label": "t", "g_fixed": np.array(g_fixed), "g_mutable": np.array(g_mutable)}]
    return ExperimentRunner(p=p, c=c, gamma=gamma, r_g=r_g, R=R, scenario_pairs=sc, n=n, stochastic=spec)


def random_params():
    return dict(p=RNG.uniform(0.2, 0.8), c=RNG.uniform(0, 0.6), gamma=float(RNG.choice([5.0, 21.0, 50.0])),
                r_g=RNG.uniform(-0.6, 0.6), R=RNG.uniform(0.05, 0.25),
                g_fixed=tuple(RNG.uniform(-0.2, 1.0, 2)), g_mutable=tuple(RNG.uniform(-0.2, 1.0, 2)))


def grid_of(r):
    """The runner's grid dict, as built in compute()."""
    sc = r.scenario_pairs[0]
    gf, gm = sc['g_fixed'], sc['g_mutable']
    u1, u2 = r.X - gm[0], r.Y - gm[1]
    d2 = r.Ginv[0, 0] * u1 ** 2 + 2 * r.Ginv[0, 1] * u1 * u2 + r.Ginv[1, 1] * u2 ** 2
    return {'X': r.X, 'Y': r.Y, 'feasible_mask': d2 <= r.R ** 2, 'g_fixed': gf, 'g_mutable': gm,
            'p': r.p, 'c': r.c, 'gamma': r.gamma, 'replace': r.replace,
            'sigma2': r.sigma2, 'tau2_c': r.tau2_c}


def test_standard_output_matches_enumeration():
    """Gurobi QP with sigma^2 I equals the batched enumeration solver."""
    for _ in range(20):
        sig = RNG.uniform(0, 0.2)
        pr = random_params()
        r = make_runner(dict(sigma=sig), **pr)
        gs = RNG.uniform(-0.3, 1.1, size=(3, 2))
        out = r._get_standard_output(list(gs))
        _, U, _, _ = stc.portfolio_stats(gs[None], r.p, r.c, r.gamma, sig ** 2)
        assert abs(out['mean_variance'] - U[0]) < 1e-6, (out['mean_variance'], U[0])


def test_grid_stats_match_enumeration():
    """Closed-form grid stats (incl. the 3-asset interior branch) equal the enumeration solver."""
    for timing in ('known', 'unknown'):
        for _ in range(5):
            pr = random_params()
            r = make_runner(dict(sigma=RNG.uniform(0.02, 0.2), breeding_sd=RNG.uniform(0, 0.1),
                                 timing=timing), **pr)
            grid = grid_of(r)
            st = r.get_exact_grid_stats(grid)
            idx = RNG.choice(r.X.size, 300, replace=False)
            gc = np.c_[r.X.ravel()[idx], r.Y.ravel()[idx]]
            gs = np.stack([np.tile(grid['g_fixed'], (300, 1)), np.tile(grid['g_mutable'], (300, 1)), gc], 1)
            _, U, _, _ = stc.portfolio_stats(gs, r.p, r.c, r.gamma, r.sigma2, [0, 0, r.tau2_c])
            err = np.max(np.abs(st['v_port'].ravel()[idx] - U))
            assert err < 1e-8, err


def test_clairvoyance_box_matches_gurobi():
    """Deterministic limit: full-information Clairvoyance equals the Gurobi hindsight model."""
    for _ in range(6):
        pr = random_params()
        r = make_runner(None, **pr)
        grid = grid_of(r)
        from GurobiPortfolioOptimizer import GurobiPortfolioOptimizer
        grid.update({'R': r.R, 'Ginv': r.Ginv})
        gurobi = GurobiPortfolioOptimizer(grid).strat_hindsight()
        args = list(r._stochastic_search_args(grid))
        box = stc.clairvoyance_full_info(*args, project=r._projector(grid))
        assert box['mean_variance'] >= gurobi['mean_variance'] - 1e-4, (box['mean_variance'], gurobi['mean_variance'])
        assert abs(box['mean_variance'] - gurobi['mean_variance']) < 1e-4, (box['mean_variance'], gurobi['mean_variance'])


def test_pob_gurobi_matches_search_with_sigma():
    """Gurobi PoB with sigma equals the ellipse search on the exact market response."""
    for _ in range(5):
        pr = random_params()
        sig = RNG.uniform(0.02, 0.2)
        r = make_runner(dict(sigma=sig), **pr)
        txt, data = r.compute(strategy_keys=['PoB'])
        pob = data[0]['details_per_strategy']['PoB']['out_dict']['mean_variance']
        grid = data[0]['grid_dict']
        t = stc.mc_target(*r._stochastic_search_args(grid), objective='mv', project=r._projector(grid))
        _, U, _, _ = stc.portfolio_stats(np.array([[grid['g_fixed'], grid['g_mutable'], t]]),
                                         r.p, r.c, r.gamma, r.sigma2)
        assert abs(pob - U[0]) < 2e-4, (pob, U[0])


def test_maxmarket_kkt_with_sigma_and_tau():
    """MaxMarket (Gurobi MPEC) shares equal the true market response at its genotype."""
    for timing in ('known', 'unknown'):
        pr = dict(p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1, g_fixed=(0.6, 0.2), g_mutable=(-0.1, 0.2))
        spec = dict(sigma=0.08) if timing == 'known' else dict(sigma=0.08, breeding_sd=0.05, timing='unknown')
        r = make_runner(spec, **pr)
        grid = grid_of(r)
        from GurobiPortfolioOptimizer import GurobiPortfolioOptimizer
        grid.update({'R': r.R, 'Ginv': r.Ginv})
        res = GurobiPortfolioOptimizer(grid).strat_max_investability(min_improvement=1e-4)
        gs = np.array(res['genotypes'])[None]
        w, _, _, _ = stc.portfolio_stats(gs, r.p, r.c, r.gamma, r.sigma2, [0, 0, r.tau2_c])
        assert abs(w[0, -1] - res['weights']['env_1'][-1]) < 1e-3, (w[0], res['weights'])


def test_unknown_timing_variance_by_simulation():
    """Ex-ante variance with tau^2 matches simulated environment, farm shocks and breeding outcome."""
    p, c, s, sig = 0.4, 0.2, 0.08, 0.1
    G = np.array([[1.0, -0.3], [-0.3, 1.0]])
    spec = stc.StochasticSpec(sigma=sig, breeding_sd=s, timing='unknown')
    gs = np.array([[0.6, 0.2], [-0.1, 0.2], [0.05, 0.3]])
    w = np.array([0.5, 0.2, 0.3])
    mu, Sig = stc.variety_moments(gs[None], p, c, sig ** 2, [0, 0, spec.tau2(p, c, G)])
    n = 1_000_000
    rng = np.random.default_rng(1)
    env1 = rng.random(n) < p
    eta = s * rng.standard_normal((n, 2)) @ np.linalg.cholesky(G).T
    g_c = gs[2] + eta
    y = np.empty((n, 3))
    y1, y2 = stc.env_yields(gs, c)
    y[:, :2] = np.where(env1[:, None], y1[:2], y2[:2])
    y[:, 2] = np.where(env1, g_c[:, 0] + c * g_c[:, 1], c * g_c[:, 0] + g_c[:, 1])
    y += sig * rng.standard_normal((n, 3))
    Y = y @ w
    assert abs(Y.mean() - w @ mu[0]) < 1e-3
    assert abs(Y.var() / (w @ Sig[0] @ w) - 1) < 0.01, (Y.var(), w @ Sig[0] @ w)


def test_value_of_information_nonnegative():
    """Knowing the breeding outcome before adoption is worth at least as much as not knowing it."""
    pr = dict(p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1, g_fixed=(0.6, 0.2), g_mutable=(-0.1, 0.2))
    G = np.array([[1.0, -0.4], [-0.4, 1.0]])
    for t in RNG.uniform(-0.2, 0.4, size=(10, 2)):
        gl = [pr['g_fixed'], pr['g_mutable'], tuple(t)]
        known = stc.evaluate_genotypes(gl, 2, 0.5, 0.1, 21.0,
                                       stc.StochasticSpec(sigma=0.05, breeding_sd=0.05, n_draws=4000), G)
        unknown = stc.evaluate_genotypes(gl, 2, 0.5, 0.1, 21.0,
                                         stc.StochasticSpec(sigma=0.05, breeding_sd=0.05, timing='unknown'), G)
        assert known['mean_variance'] >= unknown['mean_variance'] - 5e-4, (known, unknown)


def test_inactive_spec_is_deterministic():
    """sigma = breeding_sd = 0 gives exactly the deterministic results."""
    keys = ['Base', 'BeatBest', 'PoB', 'BeatBestS', 'Clairvoyance']
    a = make_runner(None).compute(strategy_keys=keys)[0]
    b = make_runner(dict(sigma=0.0, breeding_sd=0.0, timing='unknown')).compute(strategy_keys=keys)[0]
    assert a == b


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in tests:
        f()
        print("passed", f.__name__)
