"""
experiment_core.py — Single shared ExperimentRunner class.
Both the CLI (Jacob_experiment_base4.py) and the Streamlit app (stream.py) import from here.
"""

import gurobipy as gp
import io
import math
from contextlib import redirect_stdout
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from gurobipy import GRB
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from GurobiPortfolioOptimizer import GurobiPortfolioOptimizer

# ---------------------------------------------------------------------------
# Strategy registry (metadata only — callables resolved in _build_strategy_map)
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY = {
    'Base':         {'label': 'Baseline',                 'group': 'benchmark'},
    'BeatBest':     {'label': 'BeatBest',                 'group': 'primary'},
    'PoB':          {'label': 'PoB',                      'group': 'primary'},
    'BeatBestS':    {'label': 'BeatBestS',                'group': 'primary'},
    'MaxMarket':  {'label': 'MaxMarket',             'group': 'primary'},
    'Clairvoyance': {'label': 'Hindsight (Clairvoyance)', 'group': 'benchmark'}
}

DEFAULT_STRATEGY_KEYS = ['Base', 'BeatBest', 'PoB', 'BeatBestS', 'MaxAdoption', 'Clairvoyance']


# ---------------------------------------------------------------------------
class ExperimentRunner:
# ---------------------------------------------------------------------------

    def __init__(self, p, c, gamma, r_g, R, scenario_pairs, replace=False, n=401):
        self.p = p
        self.c = c
        self.beta1 = np.array([1.0, c])
        self.beta2 = np.array([c, 1.0])
        self.beta = p * self.beta1 + (1 - p) * self.beta2
        self.gamma = gamma
        self.r_g = r_g
        self.R = R
        self.G = np.array([[1.0, r_g], [r_g, 1.0]])
        self.eps = 1e-12
        self.Ginv = np.linalg.inv(self.G + self.eps * np.eye(2))
        self.lim = 1.2
        self.n = n
        self.scenario_pairs = scenario_pairs
        self.x = np.linspace(-1, self.lim, self.n)
        self.y = np.linspace(-1, self.lim, self.n)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.replace = replace
        self.grid_dict = None

    # ------------------------------------------------------------------
    # Core grid / portfolio math  (unchanged from GUI version)
    # ------------------------------------------------------------------

    def get_exact_grid_stats(self, grid):
        """
        Calculates the EXACT constrained optimal portfolio.
        Fixes the 'Zero Variance' bug by explicitly checking Corners
        and handling singular matrices.
        """
        p, c, gamma = grid['p'], grid['c'], grid['gamma']
        X, Y = grid['X'], grid['Y']
        orig_shape = X.shape
        n_points = X.size

        yf = np.array([grid['g_fixed'][0] + c * grid['g_fixed'][1],
                       c * grid['g_fixed'][0] + grid['g_fixed'][1]])
        mu_f = p * yf[0] + (1 - p) * yf[1]

        has_mutable = not grid['replace']
        if has_mutable:
            ym = np.array([grid['g_mutable'][0] + c * grid['g_mutable'][1],
                           c * grid['g_mutable'][0] + grid['g_mutable'][1]])
            mu_m = p * ym[0] + (1 - p) * ym[1]

        X_flat, Y_flat = X.ravel(), Y.ravel()
        yc_0 = X_flat + c * Y_flat
        yc_1 = c * X_flat + Y_flat
        mu_c = p * yc_0 + (1 - p) * yc_1

        v_ff = (p * yf[0] ** 2 + (1 - p) * yf[1] ** 2) - mu_f ** 2
        v_cc = (p * yc_0 ** 2 + (1 - p) * yc_1 ** 2) - mu_c ** 2
        v_fc = (p * yf[0] * yc_0 + (1 - p) * yf[1] * yc_1) - mu_f * mu_c

        util_pure_c = mu_c - 0.5 * gamma * v_cc
        util_pure_f = mu_f - 0.5 * gamma * v_ff
        if has_mutable:
            v_mm = (p * ym[0] ** 2 + (1 - p) * ym[1] ** 2) - mu_m ** 2
            util_pure_m = mu_m - 0.5 * gamma * v_mm

        def solve_2_asset(m1, m2, v11, v22, v12):
            denom = gamma * (v11 + v22 - 2 * v12)
            num = (m1 - m2) + gamma * (v22 - v12)
            w1 = np.divide(num, denom, out=np.zeros_like(num), where=np.abs(denom) > 1e-9)
            mask_sing = np.abs(denom) <= 1e-9
            if np.any(mask_sing):
                w1_sing = np.where(m1 > m2, 1.0, 0.0)
                w1[mask_sing] = w1_sing[mask_sing]
            w1 = np.clip(w1, 0.0, 1.0)
            w2 = 1.0 - w1
            mean = w1 * m1 + w2 * m2
            var = w1 ** 2 * v11 + w2 ** 2 * v22 + 2 * w1 * w2 * v12
            util = mean - 0.5 * gamma * var
            return w1, w2, util, mean, var

        if grid['replace']:
            wc, wf, v_opt, m_opt, var_opt = solve_2_asset(mu_c, mu_f, v_cc, v_ff, v_fc)
            return {
                'w_c': wc.reshape(orig_shape),
                'w_f': wf.reshape(orig_shape),
                'w_m': np.zeros(orig_shape),
                'v_port': v_opt.reshape(orig_shape),
                'mean_port': m_opt.reshape(orig_shape),
                'var_port': var_opt.reshape(orig_shape)
            }

        v_fm = (p * yf[0] * ym[0] + (1 - p) * yf[1] * ym[1]) - mu_f * mu_m
        v_mc = (p * ym[0] * yc_0 + (1 - p) * ym[1] * yc_1) - mu_m * mu_c

        N = n_points
        V_ff, V_mm, V_fm = np.full(N, v_ff), np.full(N, v_mm), np.full(N, v_fm)
        Mu_f, Mu_m = np.full(N, mu_f), np.full(N, mu_m)

        A, B, C = V_ff, V_fm, v_fc
        D, E, F = V_mm, v_mc, v_cc
        det = A * (D * F - E ** 2) - B * (B * F - C * E) + C * (B * E - C * D)

        i11, i12, i13 = (D * F - E ** 2), (C * E - B * F), (B * E - C * D)
        i22, i23 = (A * F - C ** 2), (B * C - A * E)
        i33 = (A * D - B ** 2)
        i21, i31, i32 = i12, i13, i23

        s1, s2, s3 = (i11 + i12 + i13), (i21 + i22 + i23), (i31 + i32 + i33)
        sum_S_inv = s1 + s2 + s3
        m1 = i11 * Mu_f + i12 * Mu_m + i13 * mu_c
        m2 = i21 * Mu_f + i22 * Mu_m + i23 * mu_c
        m3 = i31 * Mu_f + i32 * Mu_m + i33 * mu_c
        sum_S_inv_mu = m1 + m2 + m3

        valid_det = np.abs(det) > 1e-12
        v_int = np.full(N, -np.inf)
        w_int_c = np.zeros(N)
        w_int_f = np.zeros(N)
        w_int_m = np.zeros(N)
        mean_int = np.full(N, -np.inf)
        var_int = np.full(N, np.inf)

        if np.any(valid_det):
            lam = (sum_S_inv_mu[valid_det] - gamma * det[valid_det]) / sum_S_inv[valid_det]
            denom_w = gamma * det[valid_det]
            w1 = (m1[valid_det] - lam * s1[valid_det]) / denom_w
            w2 = (m2[valid_det] - lam * s2[valid_det]) / denom_w
            w3 = (m3[valid_det] - lam * s3[valid_det]) / denom_w
            feasible = (w1 >= -1e-7) & (w2 >= -1e-7) & (w3 >= -1e-7)
            mean_temp = w1 * Mu_f[valid_det] + w2 * Mu_m[valid_det] + w3 * mu_c[valid_det]
            var_temp = (w1 ** 2 * A[valid_det] + w2 ** 2 * D[valid_det] + w3 ** 2 * F[valid_det] +
                        2 * w1 * w2 * B[valid_det] + 2 * w1 * w3 * C[valid_det] + 2 * w2 * w3 * E[valid_det])
            v_calc = mean_temp - 0.5 * gamma * var_temp
            idx_feas = np.where(valid_det)[0][feasible]
            if len(idx_feas) > 0:
                v_int[idx_feas] = v_calc[feasible]
                w_int_c[idx_feas] = w3[feasible]
                w_int_f[idx_feas] = w1[feasible]
                w_int_m[idx_feas] = w2[feasible]
                mean_int[idx_feas] = mean_temp[feasible]
                var_int[idx_feas] = var_temp[feasible]

        w_fm_f, w_fm_m, v_fm, m_fm, var_fm = solve_2_asset(Mu_f, Mu_m, V_ff, V_mm, V_fm)
        w_fc_c, w_fc_f, v_fc, m_fc, var_fc = solve_2_asset(mu_c, Mu_f, v_cc, V_ff, v_fc)
        w_mc_c, w_mc_m, v_mc, m_mc, var_mc = solve_2_asset(mu_c, Mu_m, v_cc, V_mm, v_mc)

        V_pure_c = util_pure_c
        V_pure_f = np.full(N, util_pure_f)
        V_pure_m = np.full(N, util_pure_m)

        all_v = np.vstack([v_int, v_fm, v_fc, v_mc, V_pure_c, V_pure_f, V_pure_m])
        best_idx = np.argmax(all_v, axis=0)

        w_c_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3,
             best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_c, np.zeros(N), w_fc_c, w_mc_c, np.ones(N), np.zeros(N), np.zeros(N)]
        )
        w_f_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3,
             best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_f, w_fm_f, w_fc_f, np.zeros(N), np.zeros(N), np.ones(N), np.zeros(N)]
        )
        w_m_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3,
             best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_m, w_fm_m, np.zeros(N), w_mc_m, np.zeros(N), np.zeros(N), np.ones(N)]
        )
        mean_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3,
             best_idx == 4, best_idx == 5, best_idx == 6],
            [mean_int, m_fm, m_fc, m_mc, mu_c, Mu_f, Mu_m]
        )
        var_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3,
             best_idx == 4, best_idx == 5, best_idx == 6],
            [var_int, var_fm, var_fc, var_mc, v_cc, V_ff, V_mm]
        )
        v_final = np.max(all_v, axis=0)

        return {
            'w_c': w_c_final.reshape(orig_shape),
            'w_f': w_f_final.reshape(orig_shape),
            'w_m': w_m_final.reshape(orig_shape),
            'v_port': v_final.reshape(orig_shape),
            'mean_port': mean_final.reshape(orig_shape),
            'var_port': var_final.reshape(orig_shape)
        }

    def get_metrics(self, x_sol):
        variance = x_sol @ self.G @ x_sol
        mean = self.beta @ x_sol
        return variance, mean

    def _compute_grid(self):
        for scen in self.scenario_pairs:
            g_fixed = scen["g_fixed"]
            g_mutable = scen["g_mutable"]

            U_fix = self.get_mean_yield(g_fixed)
            Var_fix = self.get_variance(g_fixed)

            U_grid = self.beta[0] * self.X + self.beta[1] * self.Y
            Var_grid = self.var_y_grid()
            Cov_grid_fix = self.cov_y_grid_with_fixed(g_fixed)

            Delta_mu = U_fix - U_grid
            Var_diff = Var_fix + Var_grid - 2 * Cov_grid_fix
            denom_safe = np.where(Var_diff < 1e-9, 1e-9, Var_diff)

            w_fix_grid = (Delta_mu / self.gamma + (Var_grid - Cov_grid_fix)) / denom_safe
            w_fix_grid = np.clip(w_fix_grid, 0.0, 1.0)
            w_new_grid = 1.0 - w_fix_grid

            u1 = self.X - g_mutable[0]
            u2 = self.Y - g_mutable[1]
            d2 = self.Ginv[0, 0] * u1 ** 2 + 2 * self.Ginv[0, 1] * u1 * u2 + self.Ginv[1, 1] * u2 ** 2
            feasible_mask = (d2 <= self.R ** 2)

            grid_dict = {
                'X': self.X, 'Y': self.Y, 'U_grid': U_grid, 'Var_grid': Var_grid,
                'feasible_mask': feasible_mask, 'g_fixed': g_fixed, 'g_mutable': g_mutable,
                'U_fix': U_fix, 'Cov_grid_fix': Cov_grid_fix,
                'w_fix_grid': w_fix_grid, 'w_new_grid': w_new_grid,
                'p': self.p, 'beta': self.beta, 'r_g': self.r_g,
                'gamma': self.gamma, 'c': self.c, 'Ginv': self.Ginv,
                'R': self.R, 'replace': self.replace
            }
            grid_dict['stats'] = self.get_exact_grid_stats(grid_dict)
            grid_dict['optimizer'] = GurobiPortfolioOptimizer(grid_dict)
            return grid_dict

    def find_optimal_action_constrained(self, grid, gamma_val):
        original_gamma = self.gamma
        self.gamma = gamma_val
        try:
            stats = grid['stats']
            v_scores = np.where(grid['feasible_mask'], stats['v_port'], -np.inf)
            idx = np.unravel_index(np.nanargmax(v_scores), grid['X'].shape)
            opt_mean = stats['mean_port'][idx]
            opt_utility = stats['v_port'][idx]
            if gamma_val > 1e-9:
                opt_var = (opt_mean - opt_utility) * 2 / gamma_val
            else:
                opt_var = 0.0
            x_candidate = np.array([grid['X'][idx], grid['Y'][idx]])
        finally:
            self.gamma = original_gamma
        return x_candidate

    def solve_nise_frontier(self, grid=None, noBreed=False, tol=1e-4):
        """NISE Algorithm to find vertices of the efficient frontier using Gurobi."""
        frontier = []
        if grid is None:
            grid = self.grid_dict
            if grid is None:
                grid = self._compute_grid()

        def solve_point(gamma_v):
            orig_g = self.gamma
            self.gamma = gamma_v
            if noBreed:
                g_fixed = grid["g_fixed"]
                g_mutable = grid["g_mutable"]
                opt_result = self._get_standard_output([g_fixed, g_mutable])
            else:
                opt_result = grid['optimizer'].change_gamma_B4P(gamma_v)
            if opt_result is None:
                self.gamma = orig_g
                return None
            x_cand = opt_result["genotypes"][-1]
            mean_p = opt_result["mean"]
            var_p = opt_result["variance"]
            self.gamma = orig_g
            grid['optimizer'].gamma = orig_g
            return (var_p, mean_p, gamma_v, x_cand)

        p_min_var = solve_point(1e5)
        p_max_ret = solve_point(0.0)
        if p_min_var is None or p_max_ret is None:
            print("Failed to find anchor points for the Pareto frontier.")
            return frontier

        frontier.append(p_min_var)
        frontier.append(p_max_ret)

        def refine(p1, p2):
            if p1 is None or p2 is None:
                return
            v1, m1, _, _ = p1
            v2, m2, _, _ = p2
            if abs(v2 - v1) < 1e-9:
                return
            slope = (m2 - m1) / (v2 - v1)
            gamma_new = 2 * slope
            if gamma_new < 0:
                return
            p_new = solve_point(gamma_new)
            if p_new is None:
                return
            v_new, m_new = p_new[0], p_new[1]
            expected_mean = m1 + slope * (v_new - v1)
            if (m_new - expected_mean) > tol:
                frontier.append(p_new)
                refine(p1, p_new)
                refine(p_new, p2)

        refine(p_min_var, p_max_ret)
        frontier.sort(key=lambda x: x[0])
        return frontier

    def _get_yields_and_stats(self, g0, g1):
        """Central source of truth for Yield, Mean, and Statistical Variance."""
        y_env1 = 1.0 * g0 + self.c * g1
        y_env2 = self.c * g0 + 1.0 * g1
        mean_y = self.p * y_env1 + (1 - self.p) * y_env2
        exp_sq_y = self.p * (y_env1 ** 2) + (1 - self.p) * (y_env2 ** 2)
        var_y = exp_sq_y - mean_y ** 2
        mv_score = mean_y - 0.5 * self.gamma * var_y
        return mean_y, var_y, mv_score, y_env1, y_env2

    def get_mean_yield(self, g_vec):
        mu, _, _, _, _ = self._get_yields_and_stats(g_vec[0], g_vec[1])
        return mu

    def get_variance(self, g_vec):
        _, var, _, _, _ = self._get_yields_and_stats(g_vec[0], g_vec[1])
        return var

    def get_mean_variance(self, g_vec):
        _, _, mv, _, _ = self._get_yields_and_stats(g_vec[0], g_vec[1])
        return mv

    def get_covariance(self, g_a, g_b):
        ya_1 = 1.0 * g_a[0] + self.c * g_a[1]
        ya_2 = self.c * g_a[0] + 1.0 * g_a[1]
        mu_a = self.p * ya_1 + (1 - self.p) * ya_2
        yb_1 = 1.0 * g_b[0] + self.c * g_b[1]
        yb_2 = self.c * g_b[0] + 1.0 * g_b[1]
        mu_b = self.p * yb_1 + (1 - self.p) * yb_2
        expected_product = self.p * (ya_1 * yb_1) + (1 - self.p) * (ya_2 * yb_2)
        return expected_product - (mu_a * mu_b)

    def var_y_grid(self):
        _, var_grid, _, _, _ = self._get_yields_and_stats(self.X, self.Y)
        return var_grid

    def cov_y_grid_with_fixed(self, g_fixed):
        yg_1 = 1.0 * self.X + self.c * self.Y
        yg_2 = self.c * self.X + 1.0 * self.Y
        mu_grid = self.p * yg_1 + (1 - self.p) * yg_2
        yf_1 = 1.0 * g_fixed[0] + self.c * g_fixed[1]
        yf_2 = self.c * g_fixed[0] + 1.0 * g_fixed[1]
        mu_fixed = self.p * yf_1 + (1 - self.p) * yf_2
        expected_product = self.p * (yg_1 * yf_1) + (1 - self.p) * (yg_2 * yf_2)
        return expected_product - (mu_grid * mu_fixed)

    def mahalanobis_ellipse(self, center, n_points=400):
        Ginv = np.linalg.inv(self.G)
        angles = np.linspace(0, 2 * np.pi, n_points)
        ellipse = []
        for th in angles:
            d = np.array([np.cos(th), np.sin(th)])
            denom = d.T @ Ginv @ d
            if denom <= 0:
                k = 0
            else:
                k = self.R / np.sqrt(denom)
            u = k * d
            ellipse.append(center + u)
        ellipse = np.array(ellipse)
        return ellipse[:, 0], ellipse[:, 1]

    def calculate_stats_from_dict(self, data_dict):
        gs = data_dict['genotypes']
        w_env1 = data_dict['weights']['env_1']
        w_env2 = data_dict['weights']['env_2']

        Y1_total = 0.0
        for i, g in enumerate(gs):
            g_arr = np.array(g)
            y1_g = 1.0 * g_arr[0] + self.c * g_arr[1]
            Y1_total += w_env1[i] * y1_g

        Y2_total = 0.0
        for i, g in enumerate(gs):
            g_arr = np.array(g)
            y2_g = self.c * g_arr[0] + 1.0 * g_arr[1]
            Y2_total += w_env2[i] * y2_g

        mean = self.p * Y1_total + (1 - self.p) * Y2_total
        expected_sq = self.p * (Y1_total ** 2) + (1 - self.p) * (Y2_total ** 2)
        variance = expected_sq - mean ** 2
        mean_variance = mean - 0.5 * self.gamma * variance
        return {
            'mean_variance': mean_variance, 'mean': mean, 'variance': variance,
            'weights': data_dict['weights'], 'genotypes': data_dict['genotypes']
        }

    def _get_standard_output(self, g_list):
        """Generalized portfolio optimization for arbitrary genotypes using Gurobi."""
        def get_y_vec(g):
            return np.array([g[0] + self.c * g[1], self.c * g[0] + g[1]])

        Y = np.column_stack([get_y_vec(g) for g in g_list])
        probs = np.array([self.p, 1.0 - self.p])
        y_bar = Y.T @ probs
        Sigma = (Y.T * probs) @ Y - np.outer(y_bar, y_bar)

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam("NonConvex", 2)
        m.setParam("TimeLimit", 60)
        n = len(g_list)
        w = m.addVars(n, lb=0, ub=1)
        m.addConstr(gp.quicksum(w) == 1)
        mu_p = gp.quicksum(w[i] * y_bar[i] for i in range(n))
        var_p = gp.quicksum(w[i] * Sigma[i, j] * w[j] for i in range(n) for j in range(n))
        m.setObjective(mu_p - 0.5 * self.gamma * var_p, GRB.MAXIMIZE)
        m.optimize()
        weights = [w[i].X for i in range(n)]

        if m.Status != GRB.OPTIMAL:
            print('ERROR')

        return {
            "genotypes": g_list,
            "weights": {"env_1": weights, "env_2": weights},
            "mean_variance": m.ObjVal,
            "mean": sum(w[i].X * y_bar[i] for i in range(n)),
            "variance": sum(w[i].X * Sigma[i, j] * w[j].X for i in range(n) for j in range(n))
        }

    def _get_hindsight_output_2_assets(self, g_fixed, g_cand):
        yf = np.array([g_fixed[0] + self.c * g_fixed[1], self.c * g_fixed[0] + g_fixed[1]])
        yc = np.array([g_cand[0] + self.c * g_cand[1], self.c * g_cand[0] + g_cand[1]])
        y_opt = np.maximum(yf, yc)
        mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
        var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2
        w_env1 = [1.0, 0.0] if yf[0] >= yc[0] else [0.0, 1.0]
        w_env2 = [1.0, 0.0] if yf[1] >= yc[1] else [0.0, 1.0]
        return {
            "genotypes": [g_fixed.tolist(), g_cand.tolist()],
            "weights": {"env_1": w_env1, "env_2": w_env2},
            "hindsight_score": mu - 0.5 * self.gamma * var
        }

    def _get_hindsight_output_3_assets(self, g_fixed, g_mut, g_cand):
        yf = np.array([g_fixed[0] + self.c * g_fixed[1], self.c * g_fixed[0] + g_fixed[1]])
        ym = np.array([g_mut[0] + self.c * g_mut[1], self.c * g_mut[0] + g_mut[1]])
        yc = np.array([g_cand[0] + self.c * g_cand[1], self.c * g_cand[0] + g_cand[1]])
        y_opt = np.maximum.reduce([yf, ym, yc])
        mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
        var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2

        if yf[0] >= ym[0] and yf[0] >= yc[0]:
            w_env1 = [1.0, 0.0, 0.0]
        elif ym[0] >= yc[0]:
            w_env1 = [0.0, 1.0, 0.0]
        else:
            w_env1 = [0.0, 0.0, 1.0]

        if yf[1] >= ym[1] and yf[1] >= yc[1]:
            w_env2 = [1.0, 0.0, 0.0]
        elif ym[1] >= yc[1]:
            w_env2 = [0.0, 1.0, 0.0]
        else:
            w_env2 = [0.0, 0.0, 1.0]

        return {
            "genotypes": [g_fixed.tolist(), g_mut.tolist(), g_cand.tolist()],
            "weights": {"env_1": w_env1, "env_2": w_env2},
            "hindsight_score": mu - 0.5 * self.gamma * var
        }

    def _get_final_output(self, grid, g_cand):
        """Builds standard output for a candidate, including existing assets."""
        if self.replace:
            g_list = [grid['g_fixed'], g_cand]
        else:
            g_list = [grid['g_fixed'], grid['g_mutable'], g_cand]
        return self._get_standard_output(g_list)

    # ------------------------------------------------------------------
    # Strategy methods  (moved from closures inside run())
    # ------------------------------------------------------------------

    def _strat_base(self, grid):
        details = self._get_standard_output([grid["g_fixed"], grid["g_mutable"]])
        return None, details

    def _strat_B4P(self, grid):
        """Maximize Portfolio Utility."""
        return None, grid['optimizer'].strat_B4P()

    def _strat_B4M(self, grid):
        """Maximize Portfolio Mean Yield."""
        return None, grid['optimizer'].strat_B4M()

    def _strat_max_share(self, grid):
        """Maximize Candidate Share (Adaptation)."""
        return None, grid['optimizer'].strat_max_share()

    def _strat_max_investability(self, grid):
        return None, grid['optimizer'].strat_max_investability(min_improvement=1e-4)

    def _strat_max_investability3(self, grid):
        return None, grid['optimizer'].strat_max_investability(min_improvement=1e-3)

    def _strat_BB(self, grid):
        """BeatBest: Standalone Max."""
        return None, grid['optimizer'].strat_BB()

    def _get_pareto_frontier(self, grid):
        stats = grid['stats']
        W = stats['w_c']
        U = stats['v_port']
        M = stats['mean_port']
        mask = grid['feasible_mask']

        rows, cols = np.where(mask)
        valid_W = W[mask]
        valid_U = U[mask]
        valid_M = M[mask]

        data = []
        for r, c, w, u, m in zip(rows, cols, valid_W, valid_U, valid_M):
            data.append((w, u, m, r, c))
        data.sort(key=lambda x: x[0], reverse=True)

        pareto_list = []
        current_max_u = -np.inf
        for w, u, m, r, c in data:
            if u > current_max_u:
                res = {
                    'mv_val': u, 'mean_val': m,
                    'var_val': (m - u) * 2 / self.gamma,
                    'w_cand': w,
                    'g1': grid['X'][r, c], 'g2': grid['Y'][r, c],
                    'w_fixed': 1.0 - w,
                }
                pareto_list.append(res)
                current_max_u = u
        return pareto_list

    def _strat_opposite_direction(self, grid):
        """Maximizes projection onto the negative centroid vector."""
        X, Y = grid['X'], grid['Y']
        if self.replace:
            center_x, center_y = grid['g_fixed'][0], grid['g_fixed'][1]
        else:
            center_x = 0.5 * (grid['g_fixed'][0] + grid['g_mutable'][0])
            center_y = 0.5 * (grid['g_fixed'][1] + grid['g_mutable'][1])

        metric = -(X * center_x + Y * center_y)
        metric = np.where(grid['feasible_mask'], metric, -np.inf)
        idx = np.unravel_index(np.nanargmax(metric), grid['X'].shape)
        g_cand = np.array([grid['X'][idx], grid['Y'][idx]])
        return idx, self._get_final_output(grid, g_cand)

    def _strat_max_distance(self, grid):
        """Maximizes minimum Euclidean distance (Maximin) to existing assets."""
        X, Y = grid['X'], grid['Y']
        d2_fixed = (X - grid['g_fixed'][0]) ** 2 + (Y - grid['g_fixed'][1]) ** 2
        if self.replace:
            metric = d2_fixed
        else:
            d2_mutable = (X - grid['g_mutable'][0]) ** 2 + (Y - grid['g_mutable'][1]) ** 2
            metric = np.minimum(d2_fixed, d2_mutable)
        metric = np.where(grid['feasible_mask'], metric, -np.inf)
        idx = np.unravel_index(np.nanargmax(metric), grid['X'].shape)
        g_cand = np.array([grid['X'][idx], grid['Y'][idx]])
        return idx, self._get_final_output(grid, g_cand)

    def _strat_hindsight_base(self, grid):
        """Benchmark: Perfect information performance of EXISTING portfolio."""
        details = self._get_hindsight_output_2_assets(grid["g_fixed"], grid["g_mutable"])
        return None, details

    def _strat_hindsight(self, grid):
        """True Hindsight: Grid search for candidate that maximizes switching utility."""
        yf1 = grid['g_fixed'][0] + self.c * grid['g_fixed'][1]
        yf2 = self.c * grid['g_fixed'][0] + grid['g_fixed'][1]
        if self.replace:
            yc1 = grid['X'] + self.c * grid['Y']
            yc2 = self.c * grid['X'] + grid['Y']
            y_opt1, y_opt2 = np.maximum(yf1, yc1), np.maximum(yf2, yc2)
        else:
            ym1 = grid['g_mutable'][0] + self.c * grid['g_mutable'][1]
            ym2 = self.c * grid['g_mutable'][0] + grid['g_mutable'][1]
            yc1 = grid['X'] + self.c * grid['Y']
            yc2 = self.c * grid['X'] + grid['Y']
            y_opt1 = np.maximum.reduce([yf1, ym1, yc1])
            y_opt2 = np.maximum.reduce([yf2, ym2, yc2])

        Mu = self.p * y_opt1 + (1 - self.p) * y_opt2
        Var = (self.p * y_opt1 ** 2 + (1 - self.p) * y_opt2 ** 2) - Mu ** 2
        Score = Mu - 0.5 * self.gamma * Var
        Score[~grid['feasible_mask']] = -np.inf
        idx = np.unravel_index(np.nanargmax(Score), Score.shape)
        g_cand = np.array([grid['X'][idx], grid['Y'][idx]])

        if self.replace:
            details = self._get_hindsight_output_2_assets(grid["g_fixed"], g_cand)
        else:
            details = self._get_hindsight_output_3_assets(grid["g_fixed"], grid["g_mutable"], g_cand)
        return idx, details

    def _strat_hindsight_optimized(self, grid):
        """Hindsight Portfolio – fully optimized via Gurobi."""
        return None, grid['optimizer'].strat_hindsight()

    def _calculate_split_mv(self, y1, y2):
        """Helper: compute global Mean-Variance from environment-specific yields."""
        mu = self.p * y1 + (1 - self.p) * y2
        exp_sq = self.p * (y1 ** 2) + (1 - self.p) * (y2 ** 2)
        var = exp_sq - mu ** 2
        return mu - 0.5 * self.gamma * var

    def _get_split_strategies(self, grid):
        """Internal: find best of 4 split-environment cases."""
        p, c = self.p, self.c
        g_f = np.asarray(grid["g_fixed"])
        X, Y = np.asarray(grid["X"]), np.asarray(grid["Y"])
        mask = np.asarray(grid["feasible_mask"])

        y1_f = g_f[0] + c * g_f[1]
        y2_f = c * g_f[0] + g_f[1]
        y1_c = X + c * Y
        y2_c = c * X + Y

        scores = {
            "f_c": self._calculate_split_mv(y1_f, y2_c),
            "c_f": self._calculate_split_mv(y1_c, y2_f),
            "f_f": self._calculate_split_mv(y1_f, y2_f),
            "c_c": self._calculate_split_mv(y1_c, y2_c),
        }

        best_val = -np.inf
        best_case = None
        best_idx = (0, 0)
        for case, score_grid in scores.items():
            masked = np.where(mask, score_grid, -np.inf)
            idx_flat = np.nanargmax(masked)
            if masked[np.unravel_index(idx_flat, X.shape)] > best_val:
                best_val = masked[np.unravel_index(idx_flat, X.shape)]
                best_idx = np.unravel_index(idx_flat, X.shape)
                best_case = case

        weight_map = {
            "f_c": {"env_1": [1.0, 0.0], "env_2": [0.0, 1.0]},
            "c_f": {"env_1": [0.0, 1.0], "env_2": [1.0, 0.0]},
            "f_f": {"env_1": [1.0, 0.0], "env_2": [1.0, 0.0]},
            "c_c": {"env_1": [0.0, 1.0], "env_2": [0.0, 1.0]},
        }

        g_cand = np.array([X[best_idx], Y[best_idx]])
        return best_idx, {
            "genotypes": [g_f.tolist(), g_cand.tolist()],
            "weights": weight_map[best_case]
        }

    def _strat_splitEnv_0B(self, grid):
        """Zero Breeding Split: Uses original Fixed and Mutable assets."""
        g_m = grid["g_mutable"]
        mock_grid = grid.copy()
        mock_grid.update({
            'X': np.array([[g_m[0]]]),
            'Y': np.array([[g_m[1]]]),
            'feasible_mask': np.array([[True]])
        })
        _, details = self._get_split_strategies(mock_grid)
        dist = (grid['X'] - g_m[0]) ** 2 + (grid['Y'] - g_m[1]) ** 2
        return np.unravel_index(np.argmin(dist), grid['X'].shape), details

    def _strat_splitEnv_fixed(self, grid):
        """Breeding Split: Finds best g_cand to pair with g_fixed in a split env."""
        return self._get_split_strategies(grid)

    def _strat_splitEnv_freeWeight(self, grid):
        """Optimized Free Weight Split using analytical boundary search."""
        p, c, gamma = self.p, self.c, self.gamma
        g_f = grid["g_fixed"]
        X, Y, mask = grid["X"], grid["Y"], grid["feasible_mask"]

        y1_f, y2_f = g_f[0] + c * g_f[1], c * g_f[0] + g_f[1]
        y1_c, y2_c = X + c * Y, c * X + Y
        K = 0.5 * gamma * p * (1 - p)

        def solve_edge(fixed_w1=None, fixed_w2=None):
            if fixed_w1 is not None:
                Y1 = y1_c + fixed_w1 * (y1_f - y1_c)
                d2 = y2_f - y2_c
                a = -K * (d2 ** 2)
                b = (1 - p) * d2 + 2 * K * d2 * (Y1 - y2_c)
            else:
                Y2 = y2_c + fixed_w2 * (y2_f - y2_c)
                d1 = y1_f - y1_c
                a = -K * (d1 ** 2)
                b = p * d1 - 2 * K * d1 * (y1_c - Y2)

            w_opt = np.zeros_like(X)
            safe = np.abs(a) > 1e-12
            w_opt[safe] = -b[safe] / (2 * a[safe])
            w_opt[~safe] = np.where(b[~safe] > 0, 1.0, 0.0)
            w_opt = np.clip(w_opt, 0.0, 1.0)

            if fixed_w1 is not None:
                res_w1, res_w2 = np.full_like(X, fixed_w1), w_opt
            else:
                res_w1, res_w2 = w_opt, np.full_like(X, fixed_w2)

            yp1 = y1_c + res_w1 * (y1_f - y1_c)
            yp2 = y2_c + res_w2 * (y2_f - y2_c)
            scores = self._calculate_split_mv(yp1, yp2)
            return scores, res_w1, res_w2

        edges = [
            solve_edge(fixed_w1=0.0),
            solve_edge(fixed_w1=1.0),
            solve_edge(fixed_w2=0.0),
            solve_edge(fixed_w2=1.0),
        ]

        all_scores = np.stack([e[0] for e in edges])
        best_edge_idx = np.argmax(all_scores, axis=0)
        best_scores = np.take_along_axis(all_scores, best_edge_idx[None, ...], axis=0)[0]

        masked_scores = np.where(mask, best_scores, -np.inf)
        win_idx_flat = np.nanargmax(masked_scores)
        idx = np.unravel_index(win_idx_flat, X.shape)

        win_edge = best_edge_idx[idx]
        w1_win = edges[win_edge][1][idx]
        w2_win = edges[win_edge][2][idx]

        return idx, {
            "genotypes": [g_f.tolist(), [float(X[idx]), float(Y[idx])]],
            "weights": {
                "env_1": [float(w1_win), 1.0 - float(w1_win)],
                "env_2": [float(w2_win), 1.0 - float(w2_win)],
            }
        }

    def _get_split_effort_output(self, grid, g_c1, g_c2):
        """Constructs portfolio output for two specialist candidates (Split Effort)."""
        yf = np.array([grid['g_fixed'][0] + self.c * grid['g_fixed'][1],
                       self.c * grid['g_fixed'][0] + grid['g_fixed'][1]])
        assets_y = [yf]
        genotypes = [grid['g_fixed'].tolist()]

        if not self.replace:
            gm = grid['g_mutable']
            ym = np.array([gm[0] + self.c * gm[1], self.c * gm[0] + gm[1]])
            assets_y.append(ym)
            genotypes.append(gm.tolist())

        yc1 = np.array([g_c1[0] + self.c * g_c1[1], self.c * g_c1[0] + g_c1[1]])
        yc2 = np.array([g_c2[0] + self.c * g_c2[1], self.c * g_c2[0] + g_c2[1]])
        assets_y.extend([yc1, yc2])
        genotypes.extend([g_c1.tolist(), g_c2.tolist()])

        all_yields = np.vstack(assets_y).T
        y_opt = np.max(all_yields, axis=1)
        win_idx = np.argmax(all_yields, axis=1)

        mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
        var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2
        score = mu - 0.5 * self.gamma * var

        n_assets = len(assets_y)
        w_env1 = [0.0] * n_assets
        w_env2 = [0.0] * n_assets
        w_env1[win_idx[0]] = 1.0
        w_env2[win_idx[1]] = 1.0

        return {
            "genotypes": genotypes,
            "weights": {"env_1": w_env1, "env_2": w_env2},
            "split_effort_score": score,
        }

    def _strat_SplitEffort(self, grid):
        """Independent Specialists: Best for Env1 and Env2 separately."""
        Y1 = grid['X'] + self.c * grid['Y']
        Y2 = self.c * grid['X'] + grid['Y']
        Y1 = Y1.copy()
        Y2 = Y2.copy()
        Y1[~grid['feasible_mask']] = -np.inf
        Y2[~grid['feasible_mask']] = -np.inf

        idx1 = np.unravel_index(np.nanargmax(Y1), Y1.shape)
        g_c1 = np.array([grid['X'][idx1], grid['Y'][idx1]])
        idx2 = np.unravel_index(np.nanargmax(Y2), Y2.shape)
        g_c2 = np.array([grid['X'][idx2], grid['Y'][idx2]])

        details = self._get_split_effort_output(grid, g_c1, g_c2)
        return idx1, details

    def _strat_SplitEffortOptimal(self, grid):
        """Theoretical Specialists: Closest grid points to analytical max-yield directions."""
        b1 = np.array([1.0, self.c])
        b2 = np.array([self.c, 1.0])
        dir1 = self.Ginv @ b1
        dir2 = self.Ginv @ b2

        proj1 = grid['X'] * dir1[0] + grid['Y'] * dir1[1]
        proj2 = grid['X'] * dir2[0] + grid['Y'] * dir2[1]
        proj1 = proj1.copy()
        proj2 = proj2.copy()
        proj1[~grid['feasible_mask']] = -np.inf
        proj2[~grid['feasible_mask']] = -np.inf

        idx1 = np.unravel_index(np.nanargmax(proj1), proj1.shape)
        g_c1 = np.array([grid['X'][idx1], grid['Y'][idx1]])
        idx2 = np.unravel_index(np.nanargmax(proj2), proj2.shape)
        g_c2 = np.array([grid['X'][idx2], grid['Y'][idx2]])

        details = self._get_split_effort_output(grid, g_c1, g_c2)
        return idx1, details

    def _strat_SplitEffortGlobal(self, grid):
        """Joint Optimization: Coordinate descent over two candidates."""
        idx1_curr, details_init = self._strat_SplitEffort(grid)
        g_c1 = np.array(details_init['genotypes'][-2])
        g_c2 = np.array(details_init['genotypes'][-1])

        Y1_grid = (grid['X'] + self.c * grid['Y']).copy()
        Y2_grid = (self.c * grid['X'] + grid['Y']).copy()
        Y1_grid[~grid['feasible_mask']] = -np.inf
        Y2_grid[~grid['feasible_mask']] = -np.inf

        yf = np.array([grid['g_fixed'][0] + self.c * grid['g_fixed'][1],
                       self.c * grid['g_fixed'][0] + grid['g_fixed'][1]])
        if self.replace:
            base_y1 = yf[0]
            base_y2 = yf[1]
        else:
            gm = grid['g_mutable']
            ym = np.array([gm[0] + self.c * gm[1], self.c * gm[0] + gm[1]])
            base_y1 = max(yf[0], ym[0])
            base_y2 = max(yf[1], ym[1])

        for _ in range(10):
            yc2_val = g_c2[0] + self.c * g_c2[1]
            yc2_val_e2 = self.c * g_c2[0] + g_c2[1]
            Y_p2 = max(base_y2, yc2_val_e2)
            Y_p1_grid = np.maximum(base_y1, np.maximum(yc2_val, Y1_grid))
            mu = self.p * Y_p1_grid + (1 - self.p) * Y_p2
            var = (self.p * Y_p1_grid ** 2 + (1 - self.p) * Y_p2 ** 2) - mu ** 2
            score_grid = mu - 0.5 * self.gamma * var
            best_idx1 = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
            g_c1 = np.array([grid['X'][best_idx1], grid['Y'][best_idx1]])

            yc1_val_e1 = g_c1[0] + self.c * g_c1[1]
            yc1_val_e2 = self.c * g_c1[0] + g_c1[1]
            Y_p1 = max(base_y1, yc1_val_e1)
            Y_p2_grid = np.maximum(base_y2, np.maximum(yc1_val_e2, Y2_grid))
            mu = self.p * Y_p1 + (1 - self.p) * Y_p2_grid
            var = (self.p * Y_p1 ** 2 + (1 - self.p) * Y_p2_grid ** 2) - mu ** 2
            score_grid = mu - 0.5 * self.gamma * var
            best_idx2 = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
            g_c2 = np.array([grid['X'][best_idx2], grid['Y'][best_idx2]])

        details = self._get_split_effort_output(grid, g_c1, g_c2)
        return best_idx1, details

    # ------------------------------------------------------------------
    # New API: compute() + build_figure()
    # ------------------------------------------------------------------

    def _build_strategy_map(self, strategy_keys):
        """Returns {name: callable} for the requested strategy keys."""
        all_strats = {
            'Base':         self._strat_base,
            'BeatBest':         self._strat_B4M,
            'PoB':          self._strat_B4P,
            'BeatBestS':     self._strat_BB,
            'MaxMarket':        self._strat_max_investability,
            'Adopt3':       self._strat_max_investability3,
            'Clairvoyance':       self._strat_hindsight_optimized,
            'SplitEnv0B':   self._strat_splitEnv_0B,
            'SplitEnv':     self._strat_splitEnv_fixed,
            'SplitEnvFW':   self._strat_splitEnv_freeWeight,
            'SplitEffort':  self._strat_SplitEffort,
            'SplitEffOpt':  self._strat_SplitEffortOptimal,
            'SplitEffGlob': self._strat_SplitEffortGlobal,
        }
        if strategy_keys is None:
            strategy_keys = DEFAULT_STRATEGY_KEYS
        return {k: all_strats[k] for k in strategy_keys if k in all_strats}

    def compute(self, strategy_keys=None, replace=False, easy_base=False):
        """
        Pure computation: runs all scenarios and evaluates strategies.

        Returns
        -------
        text_report : str
            Captured text output (tables).
        scenario_data_list : list[dict]
            One dict per scenario, containing all data needed by build_figure().
        """
        self.replace = replace
        output_buffer = io.StringIO()
        scenario_data_list = []

        try:
            S = np.linalg.cholesky(self.G)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(self.G)
            vals[vals < 0] = 0.0
            S = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        strategies = self._build_strategy_map(strategy_keys)

        with redirect_stdout(output_buffer):
            for id, scen in enumerate(self.scenario_pairs):
                g_fixed = scen["g_fixed"]
                g_mutable = scen["g_mutable"]
                label = scen.get("label", f"Scenario {id}")

                U_fix = self.get_mean_yield(g_fixed)
                Var_fix = self.get_variance(g_fixed)
                base_line = self._get_standard_output([g_fixed, g_mutable])

                U_grid = self.beta[0] * self.X + self.beta[1] * self.Y
                Var_grid = self.var_y_grid()
                Cov_grid_fix = self.cov_y_grid_with_fixed(g_fixed)

                Delta_mu = U_fix - U_grid
                Var_diff = Var_fix + Var_grid - 2 * Cov_grid_fix
                denom_safe = np.where(Var_diff < 1e-9, 1e-9, Var_diff)
                w_fix_grid = (Delta_mu / self.gamma + (Var_grid - Cov_grid_fix)) / denom_safe
                w_fix_grid = np.clip(w_fix_grid, 0.0, 1.0)
                w_new_grid = 1.0 - w_fix_grid

                u1 = self.X - g_mutable[0]
                u2 = self.Y - g_mutable[1]
                d2 = (self.Ginv[0, 0] * u1 ** 2 + 2 * self.Ginv[0, 1] * u1 * u2
                      + self.Ginv[1, 1] * u2 ** 2)
                feasible_mask = (d2 <= self.R ** 2)

                if feasible_mask.sum() == 0:
                    print(f"Skipping {label}: No feasible points under Mahalanobis constraint.")
                    continue

                grid_dict = {
                    'X': self.X, 'Y': self.Y, 'U_grid': U_grid, 'Var_grid': Var_grid,
                    'feasible_mask': feasible_mask, 'g_fixed': g_fixed, 'g_mutable': g_mutable,
                    'U_fix': U_fix, 'Cov_grid_fix': Cov_grid_fix,
                    'w_fix_grid': w_fix_grid, 'w_new_grid': w_new_grid,
                    'p': self.p, 'beta': self.beta, 'r_g': self.r_g,
                    'gamma': self.gamma, 'c': self.c, 'Ginv': self.Ginv,
                    'R': self.R, 'replace': self.replace,
                }
                grid_dict['stats'] = self.get_exact_grid_stats(grid_dict)
                grid_dict['optimizer'] = GurobiPortfolioOptimizer(grid_dict)
                self.grid_dict = grid_dict

                if not easy_base:
                    best_possible_value = self.calculate_stats_from_dict(
                        self._strat_hindsight_optimized(grid_dict)[1]
                    )['mean_variance']
                else:
                    best_possible_value = None

                details_per_strategy = {}
                for name, strat_func in strategies.items():
                    idx, strat_dict = strat_func(grid_dict)
                    if 'Clairv' not in name:
                        out_dict = self._get_standard_output(strat_dict['genotypes'])
                    else:
                        out_dict = self.calculate_stats_from_dict(strat_dict)
                    gain = out_dict['mean_variance'] - base_line['mean_variance']
                    details_per_strategy[name] = {
                        'name': name,
                        'idx': idx,
                        'out_dict': out_dict,
                        'stats': {
                            'mean_variance': out_dict['mean_variance'],
                            'mean': out_dict['mean'],
                            'variance': out_dict['variance'],
                            'gain': gain,
                        }
                    }

                # Ellipse
                theta = np.linspace(0, 2 * np.pi, 300)
                circle = np.vstack([np.cos(theta), np.sin(theta)]) * self.R
                ellipse_pts = (S @ circle).T + g_mutable.reshape(1, 2)

                # Plot styles
                marker_cycle = ['D', 'o', 's', '^', 'v', 'P', 'X', '*', 'h', 'H']
                color_cycle = ['lime', 'magenta', 'cyan', 'orange', 'purple',
                               'brown', 'teal', 'gold', 'crimson', 'navy']
                strat_names = []
                gains = []
                adaptions1 = []
                adaptions2 = []
                weighted_adaptions = []
                plot_styles = {}

                mk_iter = cycle(marker_cycle)
                col_iter = cycle(color_cycle)

                for name, res in details_per_strategy.items():
                    if name == "Base":
                        continue
                    mv = res['stats']['mean_variance']
                    if easy_base:
                        gain_val = mv - base_line['mean_variance']
                    else:
                        denom = (best_possible_value - base_line['mean_variance'])
                        gain_val = (mv - base_line['mean_variance']) / denom if abs(denom) > 0.0001 else 0
                    gain_val = np.round(gain_val, 4)
                    gains.append(gain_val * 100)
                    if type(res['out_dict']['weights']) == list:
                        adaptions1.append(res['out_dict']['weights'][-1])
                        adaptions2.append(res['out_dict']['weights'][-1])
                        weighted_adaptions.append(res['out_dict']['weights'][-1])
                    else:
                        adaptions1.append(res['out_dict']['weights']['env_1'][-1])
                        adaptions2.append(res['out_dict']['weights']['env_2'][-1])
                        weighted_adaptions.append(self.p * res['out_dict']['weights']['env_1'][-1] + (1-self.p) * res['out_dict']['weights']['env_2'][-1])
                    strat_names.append(name)
                    plot_styles[name] = (next(mk_iter), next(col_iter))

                # --- Pretty text report ---
                NAME_W = 22
                NUM_W = 8
                EXTRA_W = 45
                SEP = " │ "

                def fmt_w(w_list):
                    return ", ".join(f"{x:.2f}" for x in w_list)

                def fmt_weights(weights):
                    if isinstance(weights, dict):
                        w_e1, w_e2 = weights['env_1'], weights['env_2']
                        if w_e1 == w_e2:
                            return f"[{fmt_w(w_e1)}]"
                        return f"E1:[{fmt_w(w_e1)}]  E2:[{fmt_w(w_e2)}]"
                    return f"[{fmt_w(weights)}]"

                def make_row(name, nums, extra=""):
                    num_part = SEP.join(f"{v:>{NUM_W}.4f}" for v in nums)
                    line = f"{name:<{NAME_W}}{SEP}{num_part}"
                    if extra:
                        line += f"{SEP}{extra:<{EXTRA_W}}"
                    return line

                def make_header(name_label, num_labels, extra_label=""):
                    num_part = SEP.join(f"{h:>{NUM_W}}" for h in num_labels)
                    line = f"{name_label:<{NAME_W}}{SEP}{num_part}"
                    if extra_label:
                        line += f"{SEP}{extra_label:<{EXTRA_W}}"
                    return line

                def make_separator(num_labels, has_extra=False):
                    sep_char = "─"
                    num_part = SEP.join(sep_char * NUM_W for _ in num_labels)
                    line = sep_char * NAME_W + SEP + num_part
                    if has_extra:
                        line += SEP + sep_char * EXTRA_W
                    return line

                # --- Table 1: Strategy comparison ---
                header1 = make_header("Strategy", ["MV", "Gain", "Mean", "Var"],
                                    "Weights (Env1 / Env2)")
                print(header1)
                print(make_separator(["MV", "Gain", "Mean", "Var"], has_extra=True))

                print(make_row(
                    "Baseline",
                    [base_line['mean_variance'], 0.0, base_line['mean'], base_line['variance']],
                    fmt_weights(base_line['weights']),
                ))

                for name, res in details_per_strategy.items():
                    s = res['stats']
                    print(make_row(
                        name,
                        [s['mean_variance'], s['gain'], s['mean'], s['variance']],
                        fmt_weights(res['out_dict']['weights']),
                    ))

                print()

                # --- Table 2: Single component ---
                header2 = make_header(
                    "Single Component",
                    ["MV", "Gain", "Mean", "Var", "g1", "g2", "y1", "y2"],
                )
                sep2 = make_separator(["MV", "Gain", "Mean", "Var", "g1", "g2", "y1", "y2"], has_extra=False)
                print(header2)
                print(sep2)

                def print_pure_row(row_name, g_vec, ref_mv):
                    g = np.array(g_vec)
                    mu, var, mv, _, _ = self._get_yields_and_stats(g[0], g[1])
                    gain = mv - ref_mv
                    y1 = self.p * (g[0] + self.c * g[1])
                    y2 = (1 - self.p) * (self.c * g[0] + g[1])
                    print(make_row(row_name, [mv, gain, mu, var, g[0], g[1], y1, y2]))

                print_pure_row("Fixed Asset (Base)", g_fixed, base_line['mean_variance'])
                print_pure_row("Mutable Asset", g_mutable, base_line['mean_variance'])
                for name, res in details_per_strategy.items():
                    gens = res['out_dict']['genotypes']
                    for i, g in enumerate(gens):
                        if i > 1 or ((g[0] != g_fixed[0] or g[1] != g_fixed[1]) and
                                    (g[0] != g_mutable[0] and g[1] != g_mutable[1])):
                            print_pure_row(f"Bred ({name})", g, base_line['mean_variance'])

                print(sep2)

                scenario_data_list.append({
                    'grid_dict': grid_dict,
                    'details_per_strategy': details_per_strategy,
                    'strat_names': strat_names,
                    'gains': gains,
                    'adaptions1': adaptions1,
                    'adaptions2': adaptions2,
                    'weighted_adaptions' : weighted_adaptions,
                    'plot_styles': plot_styles,
                    'ellipse_pts': ellipse_pts,
                    'g_fixed': g_fixed,
                    'g_mutable': g_mutable,
                    'U_grid': U_grid,
                    'Var_grid': Var_grid,
                    'w_new_grid': w_new_grid,
                    'base_line': base_line,
                    'label': label,
                    'easy_base': easy_base,
                    'best_possible_value': best_possible_value,
                    'S': S,
                    # Convenience copies so render functions don't need self
                    'X': self.X,
                    'Y': self.Y,
                    'beta1': self.beta1,
                    'beta2': self.beta2,
                    'p': self.p,
                    'c': self.c,
                })

        text_report = output_buffer.getvalue()
        return text_report, scenario_data_list

    def build_figure(self, scenario_data_list, fig, subplot_ids=None, marker_alpha=0.55):
        """
        Populates an existing matplotlib Figure with the selected subplots.

        Works with both ``plt.subplots()`` and ``matplotlib.figure.Figure()``
        figures since all calls go through ``fig.*`` / ``ax.*`` APIs.

        Parameters
        ----------
        scenario_data_list : list[dict]
            Output of compute().
        fig : matplotlib.figure.Figure
            A Figure that has NOT yet had subplots created on it.
        subplot_ids : list[str] or None
            Keys from SUBPLOT_REGISTRY. None → DEFAULT_SUBPLOT_IDS.
        marker_alpha : float
            Transparency for strategy marker scatter points (0=invisible, 1=opaque).
        """
        from subplot_registry import SUBPLOT_REGISTRY, DEFAULT_SUBPLOT_IDS

        if subplot_ids is None:
            subplot_ids = DEFAULT_SUBPLOT_IDS
        if not scenario_data_list:
            return fig
        

        sd = scenario_data_list[0]
        sd['marker_alpha'] = marker_alpha

        n = len(subplot_ids)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        axs = fig.subplots(nrows, ncols)
        ax_list = axs.flatten() if hasattr(axs, 'flatten') else [axs]

        for i, key in enumerate(subplot_ids):
            if key in SUBPLOT_REGISTRY:
                SUBPLOT_REGISTRY[key].render(ax_list[i], sd)
            else:
                ax_list[i].axis('off')
                ax_list[i].set_title(f"Unknown: {key}")

        for j in range(n, len(ax_list)):
            ax_list[j].axis('off')

        # Show "Genotype dim 1" x-label only on the effective bottom row of each column,
        # and "Genotype dim 2" y-label only on the leftmost column.
        for i in range(n):
            ax = ax_list[i]
            row = i // ncols
            col = i % ncols
            # Leftmost column: keep y-label; all others: clear it
            if col != 0 and ax.get_ylabel() == "Genotype dim 2":
                ax.set_ylabel("")
            # Bottom of this column: keep x-label; all others: clear it
            # A subplot is at the bottom if the slot directly below it is empty or out of range
            below = (row + 1) * ncols + col
            if below < n and ax.get_xlabel() == "Genotype dim 1":
                ax.set_xlabel("")

        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95,
                            hspace=0.4, wspace=0.3)
        return fig

    # ------------------------------------------------------------------
    # Backward-compatible run() wrapper
    # ------------------------------------------------------------------

    def run(self, strategies=None, plot_extensive=False, easy_base=False,
            no_plot=False, replace=False, pareto=False, plot_right=False, dpi=100):
        """
        Backward-compatible entry point.  Returns (text_result, fig) as before.
        Use compute() + build_figure() directly for finer control.
        """
        from subplot_registry import SUBPLOT_REGISTRY, DEFAULT_SUBPLOT_IDS

        if strategies is None:
            strategy_keys = DEFAULT_STRATEGY_KEYS
        else:
            strategy_keys = list(strategies.keys())

        text_result, scenario_data_list = self.compute(
            strategy_keys=strategy_keys, replace=replace, easy_base=easy_base
        )

        if no_plot or not scenario_data_list:
            return text_result, None

        if plot_extensive:
            subplot_ids = list(SUBPLOT_REGISTRY.keys())
        else:
            subplot_ids = DEFAULT_SUBPLOT_IDS

        fig = Figure(figsize=(15.55, 9.6), dpi=dpi)
        self.build_figure(scenario_data_list, fig, subplot_ids=subplot_ids)
        return text_result, fig

    def to_latex(self, scenario_data_list,
             caption_prefix="Scenario", label_prefix="tab:scenario"):
        """
        Generate booktabs LaTeX tables from the output of compute().

        Parameters
        ----------
        scenario_data_list : list[dict]
            Direct output of self.compute().
        caption_prefix : str
            Prefix for table captions.
        label_prefix : str
            Prefix for table \\label{} identifiers.

        Returns
        -------
        str
            A LaTeX string containing one strategy table and one
            single-genotype component table per scenario.
        """

        def fmt_w(w_list):
            return ", ".join([f"{x:.2f}" for x in w_list])

        def weight_str(w):
            if isinstance(w, dict):
                w_e1, w_e2 = w['env_1'], w['env_2']
                if w_e1 == w_e2:
                    return f"[{fmt_w(w_e1)}]"
                return f"E1:[{fmt_w(w_e1)}] E2:[{fmt_w(w_e2)}]"
            return str([round(i, 2) for i in w])

        def component_row(row_name, g_vec, ref_mv):
            g = np.array(g_vec)
            mu, var, mv, _, _ = self._get_yields_and_stats(g[0], g[1])
            gain = mv - ref_mv
            safe_name = row_name.replace("_", r"\_")
            return (
                f"    {safe_name} & {mv:.4f} & {gain:.4f}"
                f" & {mu:.4f} & {var:.4f}"
                f" & {g[0]:.4f} & {g[1]:.4f} \\\\"
            )

        lines = []

        for sd in scenario_data_list:
            label     = sd["label"]
            base      = sd["base_line"]          # keys: mean_variance, mean, variance, weights
            g_fixed   = sd["g_fixed"]
            g_mutable = sd["g_mutable"]
            ref_mv    = base['mean_variance']
            results   = list(sd["details_per_strategy"].values())  # list of strategy dicts

            cap = f"{caption_prefix}: {label.replace('_', ' ')}"
            lbl = f"{label_prefix}_{label.lower().replace(' ', '_')}"

            # ── Strategy table ────────────────────────────────────────────────
            lines += [
                r"\begin{table}[htbp]",
                r"  \centering",
                r"  \small",
                f"  \\caption{{{cap} -- Strategy comparison}}",
                f"  \\label{{{lbl}_strategies}}",
                r"  \begin{tabular}{l r r r r l}",
                r"    \toprule",
                r"    \textbf{Strategy} & \textbf{MV} & \textbf{Gain}"
                r" & \textbf{Mean} & \textbf{Var} & \textbf{Weights} \\",
                r"    \midrule",
                f"    Baseline & {base['mean_variance']:.4f} & {0:.4f}"
                f" & {base['mean']:.4f} & {base['variance']:.4f}"
                f" & {weight_str(base['weights'])} \\\\",
            ]

            for res in results:
                s    = res['stats']
                d    = res['out_dict']
                name = res.get('name', '?').replace("_", r"\_")
                lines.append(
                    f"    {name} & {s['mean_variance']:.4f} & {s['gain']:.4f}"
                    f" & {s['mean']:.4f} & {s['variance']:.4f}"
                    f" & {weight_str(d['weights'])} \\\\"
                )

            lines += [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}", "",
            ]

            # ── Component table ───────────────────────────────────────────────
            lines += [
                r"\begin{table}[htbp]",
                r"  \centering",
                r"  \small",
                f"  \\caption{{{cap} -- Single-genotype components}}",
                f"  \\label{{{lbl}_components}}",
                r"  \begin{tabular}{l r r r r r r}",
                r"    \toprule",
                r"    \textbf{Component} & \textbf{MV} & \textbf{Gain}"
                r" & \textbf{Mean} & \textbf{Var} & $g_1$ & $g_2$ \\",
                r"    \midrule",
                component_row("Fixed Asset (Base)", g_fixed, ref_mv),
                component_row("Mutable Asset",      g_mutable, ref_mv),
            ]

            for res in results:
                gens = res['out_dict']['genotypes']
                name = res.get('name', '?')
                for i, g in enumerate(gens):
                    if i > 1 or (
                        (g[0] != g_fixed[0] or g[1] != g_fixed[1]) and
                        (g[0] != g_mutable[0] and g[1] != g_mutable[1])
                    ):
                        lines.append(component_row(f"Bred ({name})", g, ref_mv))

            lines += [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}", "",
            ]

        return "\n".join(lines)