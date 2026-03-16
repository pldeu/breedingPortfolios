import gurobipy as gp
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from matplotlib.figure import Figure 
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from gurobipy import GRB
from matplotlib.colors import TwoSlopeNorm, Normalize
from GurobiPortfolioOptimizer import GurobiPortfolioOptimizer
import pandas as pd


class ExperimentRunner:

    def __init__(self, p, c, gamma, r_g, R, scenario_pairs, replace=False, n=201):
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
        self.lim = 1.0
        self.n = n
        self.scenario_pairs = scenario_pairs
        self.x = np.linspace(-self.lim, self.lim, self.n)
        self.y = np.linspace(-self.lim, self.lim, self.n)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.replace = replace
        self.grid_dict = None

    def get_exact_grid_stats(self, grid):  # TODO
        """
        Calculates the EXACT constrained optimal portfolio.
        Fixes the 'Zero Variance' bug by explicitly checking Corners 
        and handling singular matrices.
        
        Returns:
            dict: {
                'w_c', 'w_f', 'w_m': Weights for Candidate, Fixed, Mutable
                'v_port': Utility (Mean - 0.5 * gamma * Var)
                'mean_port': Expected Return
                'var_port': Portfolio Variance
            }
        """
        p, c, gamma = grid['p'], grid['c'], grid['gamma']
        X, Y = grid['X'], grid['Y']
        orig_shape = X.shape
        n_points = X.size

        # --- 1. Setup Yields and Means ---
        # Fixed (Scalar)
        yf = np.array([grid['g_fixed'][0] + c * grid['g_fixed'][1], c * grid['g_fixed'][0] + grid['g_fixed'][1]])
        mu_f = p * yf[0] + (1 - p) * yf[1]

        # Mutable (Scalar)
        has_mutable = not grid['replace']
        if has_mutable:
            ym = np.array(
                [grid['g_mutable'][0] + c * grid['g_mutable'][1], c * grid['g_mutable'][0] + grid['g_mutable'][1]])
            mu_m = p * ym[0] + (1 - p) * ym[1]

        # Candidate (Vector)
        X_flat, Y_flat = X.ravel(), Y.ravel()
        yc_0 = X_flat + c * Y_flat
        yc_1 = c * X_flat + Y_flat
        mu_c = p * yc_0 + (1 - p) * yc_1

        # --- 2. Setup Covariances ---
        v_ff = (p * yf[0] ** 2 + (1 - p) * yf[1] ** 2) - mu_f ** 2
        v_cc = (p * yc_0 ** 2 + (1 - p) * yc_1 ** 2) - mu_c ** 2
        v_fc = (p * yf[0] * yc_0 + (1 - p) * yf[1] * yc_1) - mu_f * mu_c

        # Calculate Pure Utility for Corner Checks (Robust fallback)
        util_pure_c = mu_c - 0.5 * gamma * v_cc
        util_pure_f = mu_f - 0.5 * gamma * v_ff
        if has_mutable:
            v_mm = (p * ym[0] ** 2 + (1 - p) * ym[1] ** 2) - mu_m ** 2
            util_pure_m = mu_m - 0.5 * gamma * v_mm

        # --- Helper: Robust 2-Asset Solver ---
        # UPDATED: Returns `var` as the 5th element
        def solve_2_asset(m1, m2, v11, v22, v12):
            denom = gamma * (v11 + v22 - 2 * v12)
            num = (m1 - m2) + gamma * (v22 - v12)

            # Standard solution
            w1 = np.divide(num, denom, out=np.zeros_like(num), where=np.abs(denom) > 1e-9)

            # HANDLING ZERO VARIANCE / SINGULARITY:
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

        # --- CASE A: 2-Asset Mode (Replace=True) ---
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

        # --- CASE B: 3-Asset Mode (Replace=False) ---
        v_fm = (p * yf[0] * ym[0] + (1 - p) * yf[1] * ym[1]) - mu_f * mu_m
        v_mc = (p * ym[0] * yc_0 + (1 - p) * ym[1] * yc_1) - mu_m * mu_c

        # Broadcasters
        N = n_points
        V_ff, V_mm, V_fm = np.full(N, v_ff), np.full(N, v_mm), np.full(N, v_fm)
        Mu_f, Mu_m = np.full(N, mu_f), np.full(N, mu_m)

        # 1. Interior Solution (Lagrange)
        A, B, C = V_ff, V_fm, v_fc
        D, E, F = V_mm, v_mc, v_cc
        det = A * (D * F - E ** 2) - B * (B * F - C * E) + C * (B * E - C * D)

        # Inverse terms
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

        # Only solve Interior where Det is healthy
        valid_det = np.abs(det) > 1e-12

        # Initialize outputs
        v_int = np.full(N, -np.inf)
        w_int_c = np.zeros(N)
        w_int_f = np.zeros(N)  # Added
        w_int_m = np.zeros(N)  # Added
        mean_int = np.full(N, -np.inf)
        var_int = np.full(N, np.inf)  # Added

        # Vectorized Interior Solve (only where valid)
        if np.any(valid_det):
            lam = (sum_S_inv_mu[valid_det] - gamma * det[valid_det]) / sum_S_inv[valid_det]
            denom_w = gamma * det[valid_det]

            # w1 -> F, w2 -> M, w3 -> C
            w1 = (m1[valid_det] - lam * s1[valid_det]) / denom_w
            w2 = (m2[valid_det] - lam * s2[valid_det]) / denom_w
            w3 = (m3[valid_det] - lam * s3[valid_det]) / denom_w

            # Check feasibility (w >= 0)
            feasible = (w1 >= -1e-7) & (w2 >= -1e-7) & (w3 >= -1e-7)

            # Full calculation for valid_det
            mean_temp = w1 * Mu_f[valid_det] + w2 * Mu_m[valid_det] + w3 * mu_c[valid_det]
            var_temp = (w1 ** 2 * A[valid_det] + w2 ** 2 * D[valid_det] + w3 ** 2 * F[valid_det] +
                        2 * w1 * w2 * B[valid_det] + 2 * w1 * w3 * C[valid_det] + 2 * w2 * w3 * E[valid_det])
            v_calc = mean_temp - 0.5 * gamma * var_temp

            # Assign only where feasible
            idx_feas = np.where(valid_det)[0][feasible]
            if len(idx_feas) > 0:
                v_int[idx_feas] = v_calc[feasible]
                w_int_c[idx_feas] = w3[feasible]
                w_int_f[idx_feas] = w1[feasible]  # Store F
                w_int_m[idx_feas] = w2[feasible]  # Store M
                mean_int[idx_feas] = mean_temp[feasible]
                var_int[idx_feas] = var_temp[feasible]  # Store Var

        # 2. Boundary Solutions (Edges) - Unpack Var
        w_fm_f, w_fm_m, v_fm, m_fm, var_fm = solve_2_asset(Mu_f, Mu_m, V_ff, V_mm, V_fm)  # w_c = 0
        w_fc_c, w_fc_f, v_fc, m_fc, var_fc = solve_2_asset(mu_c, Mu_f, v_cc, V_ff, v_fc)  # w_m = 0
        w_mc_c, w_mc_m, v_mc, m_mc, var_mc = solve_2_asset(mu_c, Mu_m, v_cc, V_mm, v_mc)  # w_f = 0

        # 3. Corner Solutions (Explicitly include Pure Assets)
        V_pure_c = util_pure_c
        V_pure_f = np.full(N, util_pure_f)
        V_pure_m = np.full(N, util_pure_m)

        # 4. Final Comparison
        # Stack: [Interior, Edge_FM, Edge_FC, Edge_MC, Pure_C, Pure_F, Pure_M]
        all_v = np.vstack([v_int, v_fm, v_fc, v_mc, V_pure_c, V_pure_f, V_pure_m])
        best_idx = np.argmax(all_v, axis=0)

        # --- Map winner to outputs ---
        # 0: Int 
        # 1: FM (w_c=0)
        # 2: FC (w_m=0)
        # 3: MC (w_f=0)
        # 4: Pure C (w_c=1)
        # 5: Pure F (w_f=1)
        # 6: Pure M (w_m=1)

        # w_c
        w_c_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3, best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_c, np.zeros(N), w_fc_c, w_mc_c, np.ones(N), np.zeros(N), np.zeros(N)]
        )

        # w_f
        w_f_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3, best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_f, w_fm_f, w_fc_f, np.zeros(N), np.zeros(N), np.ones(N), np.zeros(N)]
        )

        # w_m
        w_m_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3, best_idx == 4, best_idx == 5, best_idx == 6],
            [w_int_m, w_fm_m, np.zeros(N), w_mc_m, np.zeros(N), np.zeros(N), np.ones(N)]
        )

        mean_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3, best_idx == 4, best_idx == 5, best_idx == 6],
            [mean_int, m_fm, m_fc, m_mc, mu_c, Mu_f, Mu_m]
        )

        var_final = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2, best_idx == 3, best_idx == 4, best_idx == 5, best_idx == 6],
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
        """Helper to calculate Variance and Mean for a given solution vector x."""
        # Variance = x^T G x
        variance = x_sol @ self.G @ x_sol
        # Mean = beta^T x
        mean = self.beta @ x_sol
        return variance, mean

    def _compute_grid(self):
        for scen in self.scenario_pairs:
            g_fixed = scen["g_fixed"]
            g_mutable = scen["g_mutable"]

            # --- A. Baseline (Before Breeding) ---
            U_fix = self.get_mean_yield(g_fixed)
            Var_fix = self.get_variance(g_fixed)

            # Value of original portfolio

            # --- B. Grid Calculations (Candidates for new Mutable) ---
            U_grid = self.beta[0] * self.X + self.beta[1] * self.Y
            Var_grid = self.var_y_grid()

            Cov_grid_fix = self.cov_y_grid_with_fixed(g_fixed)

            Delta_mu = U_fix - U_grid
            Var_diff = Var_fix + Var_grid - 2 * Cov_grid_fix
            denom_safe = np.where(Var_diff < 1e-9, 1e-9, Var_diff)

            # Weight of FIXED asset
            w_fix_grid = (Delta_mu / self.gamma + (Var_grid - Cov_grid_fix)) / denom_safe
            w_fix_grid = np.clip(w_fix_grid, 0.0, 1.0)
            w_new_grid = 1.0 - w_fix_grid

            # --- C. Mahalanobis Feasibility ---
            u1 = self.X - g_mutable[0]
            u2 = self.Y - g_mutable[1]
            d2 = self.Ginv[0, 0] * u1 ** 2 + 2 * self.Ginv[0, 1] * u1 * u2 + self.Ginv[1, 1] * u2 ** 2
            feasible_mask = (d2 <= self.R ** 2)


            # Prepare a grid_dict passed to each strategy
            grid_dict = {'X': self.X, 'Y': self.Y, 'U_grid': U_grid, 'Var_grid': Var_grid,
                            'feasible_mask': feasible_mask, 'g_fixed': g_fixed, 'g_mutable': g_mutable, 'U_fix': U_fix,
                            'Cov_grid_fix': Cov_grid_fix, 'w_fix_grid': w_fix_grid,
                            'w_new_grid': w_new_grid,
                            'p': self.p, 'beta': self.beta, 'r_g': self.r_g, 'gamma': self.gamma, 'c': self.c,
                            'Ginv': self.Ginv, 'R': self.R, 'replace': self.replace }
            grid_dict['stats'] = self.get_exact_grid_stats(grid_dict)
            grid_dict['optimizer'] = GurobiPortfolioOptimizer(grid_dict)

            return grid_dict
        
    def find_optimal_action_constrained(self, grid, gamma_val):
        """
        Finds the candidate genotype from the grid that maximizes the portfolio
        score for a specific temporary gamma value.
        """
        # 1. Backup original gamma and set temporary gamma
        original_gamma = self.gamma
        self.gamma = gamma_val

        try:
            # 2. Run the exact grid solver
            # This returns dict with keys: 'v_port' (Variance), 'mean_port' (Mean), etc.
            stats = grid['stats']

            # 3. Find the index that maximizes Utility (V = Mean - 0.5 * Gamma * Var)
            # Mask infeasible points
            v_scores = np.where(grid['feasible_mask'], stats['v_port'], -np.inf)
            idx = np.unravel_index(np.nanargmax(v_scores), grid['X'].shape)

            # 4. Extract the optimal Mean and Variance for this point
            opt_mean = stats['mean_port'][idx]

            # Note: The 'v_port' returned by your solver is the Utility (Mean - 0.5*Gamma*Var).
            # We need the actual Variance. 
            # Utility = Mean - 0.5 * Gamma * Var  =>  0.5 * Gamma * Var = Mean - Utility
            # Var = (Mean - Utility) * 2 / Gamma
            # However, simpler is to just recalculate or return variance from _get_exact_grid_stats if possible.
            # Looking at your code, 'v_port' IS the utility V. 
            # We can back-solve for Variance if gamma > 0:
            opt_utility = stats['v_port'][idx]

            if gamma_val > 1e-9:
                opt_var = (opt_mean - opt_utility) * 2 / gamma_val
            else:
                # If Gamma is 0, we simply maximized Mean. 
                # We need to compute variance explicitly for this point.
                # It's safer to just return the genotype and calculate metrics later, 
                # but let's try to extract it if your grid solver supports it.
                # For now, let's just return the vector and calculate metrics outside.
                opt_var = 0.0  # Placeholder

            x_candidate = np.array([grid['X'][idx], grid['Y'][idx]])

        finally:
            # 5. Restore original gamma no matter what
            self.gamma = original_gamma

        return x_candidate


    def solve_nise_frontier(self, grid=None, noBreed=False, tol=1e-4):
        """
        NISE Algorithm to find vertices of the efficient frontier using Gurobi.
        """
        frontier = []
        if grid == None:
            grid = self.grid_dict
            if grid == None:
                grid = self._compute_grid()

        # 1. Helper to solve for a specific Gamma using Gurobi
        def solve_point(gamma_v):
            # Backup original gamma and set the new one for Gurobi
            orig_g = self.gamma
            self.gamma = gamma_v
            if noBreed:
                g_fixed = grid["g_fixed"]
                g_mutable = grid["g_mutable"]
                opt_result = self._get_standard_output([g_fixed, g_mutable])
            else:
                opt_result = grid['optimizer'].change_gamma_B4P(gamma_v)
            
            if opt_result is None:
                # Handle cases where Gurobi fails or times out
                self.gamma = orig_g
                return None

            # Extract the optimal genotype (assuming _build_result returns it under 'g_opt' or similar)
            # You might need to adjust 'g_opt' based on your exact _build_result keys
            x_cand = opt_result["genotypes"][-1]
            mean_p = opt_result["mean"]
            var_p = opt_result["variance"]

            # Restore original gamma
            self.gamma = orig_g
            grid['optimizer'].gamma = orig_g
            return (var_p, mean_p, gamma_v, x_cand)

        # 2. Anchors
        # CAUTION: Gurobi might face numerical instability with an extreme gamma like 1e8. 
        # If the solver struggles or returns infeasible, lower this to 1e5 or 1e6.
        p_min_var = solve_point(1e5)  # High Gamma (Risk-Averse)
        p_max_ret = solve_point(0.0)  # Zero Gamma (Risk-Neutral)

        if p_min_var is None or p_max_ret is None:
            print("Failed to find anchor points for the Pareto frontier.")
            return frontier

        frontier.append(p_min_var)
        frontier.append(p_max_ret)

        # 3. Recursive NISE
        def refine(p1, p2):
            if p1 is None or p2 is None: return
            
            v1, m1, _, _ = p1
            v2, m2, _, _ = p2

            # Slope (Change in Mean / Change in Variance)
            if abs(v2 - v1) < 1e-9: return
            slope = (m2 - m1) / (v2 - v1)
            gamma_new = 2 * slope

            if gamma_new < 0: return  # Concavity error or numerical noise

            p_new = solve_point(gamma_new)
            if p_new is None: return
            
            v_new, m_new = p_new[0], p_new[1]

            # Check improvement (Vertical distance to segment)
            expected_mean = m1 + slope * (v_new - v1)
            if (m_new - expected_mean) > tol:
                frontier.append(p_new)
                refine(p1, p_new)
                refine(p_new, p2)

        refine(p_min_var, p_max_ret)

        # Sort by Variance
        frontier.sort(key=lambda x: x[0])
        return frontier

    def _get_yields_and_stats(self, g0, g1):
        """
        Central source of truth for Yield, Mean, and Statistical Variance.
        Works for scalars, 1D arrays, and 2D grids (broadcasting).
        """
        # 1. Calculate Yields in both environments
        # Env 1: 1.0*g0 + c*g1
        y_env1 = 1.0 * g0 + self.c * g1
        # Env 2: c*g0 + 1.0*g1
        y_env2 = self.c * g0 + 1.0 * g1

        # 2. Mean Yield E[Y]
        mean_y = self.p * y_env1 + (1 - self.p) * y_env2

        # 3. Statistical Variance: E[Y^2] - (E[Y])^2
        exp_sq_y = self.p * (y_env1 ** 2) + (1 - self.p) * (y_env2 ** 2)
        var_y = exp_sq_y - mean_y ** 2

        # 4. Mean-Variance Score
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
        """
        Statistical Covariance: E[Y_a * Y_b] - E[Y_a] * E[Y_b]
        """
        # Get yields for variety A
        ya_1 = 1.0 * g_a[0] + self.c * g_a[1]
        ya_2 = self.c * g_a[0] + 1.0 * g_a[1]
        mu_a = self.p * ya_1 + (1 - self.p) * ya_2

        # Get yields for variety B
        yb_1 = 1.0 * g_b[0] + self.c * g_b[1]
        yb_2 = self.c * g_b[0] + 1.0 * g_b[1]
        mu_b = self.p * yb_1 + (1 - self.p) * yb_2

        # E[Y_a * Y_b]
        expected_product = self.p * (ya_1 * yb_1) + (1 - self.p) * (ya_2 * yb_2)

        return expected_product - (mu_a * mu_b)

    def var_y_grid(self):
        # Uses the helper on the meshgrid X, Y
        _, var_grid, _, _, _ = self._get_yields_and_stats(self.X, self.Y)
        return var_grid

    def cov_y_grid_with_fixed(self, g_fixed):
        # Statistical covariance between the Grid (X,Y) and a Fixed vector

        # Grid Yields
        yg_1 = 1.0 * self.X + self.c * self.Y
        yg_2 = self.c * self.X + 1.0 * self.Y
        mu_grid = self.p * yg_1 + (1 - self.p) * yg_2

        # Fixed Yields
        yf_1 = 1.0 * g_fixed[0] + self.c * g_fixed[1]
        yf_2 = self.c * g_fixed[0] + 1.0 * g_fixed[1]
        mu_fixed = self.p * yf_1 + (1 - self.p) * yf_2

        # E[Y_grid * Y_fixed]
        expected_product = self.p * (yg_1 * yf_1) + (1 - self.p) * (yg_2 * yf_2)

        return expected_product - (mu_grid * mu_fixed)

    def mahalanobis_ellipse(self, center, n_points=400):
        """
        Returns x,y points of the ellipse:
        { g : (g - center)^T G^{-1} (g - center) = R^2 }.
        """
        Ginv = np.linalg.inv(self.G)
        # For plotting we parametrize using direction vectors u
        angles = np.linspace(0, 2 * np.pi, n_points)
        ellipse = []

        for th in angles:
            # direction vector
            d = np.array([np.cos(th), np.sin(th)])
            # find k such that (k d)^T G^{-1} (k d) = R^2  => k = R / sqrt(d^T G^{-1} d)
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
        """
        Takes the standardized output dictionary and returns Mean-Var, Mean, Variance.
        Input dict format:
        {
            "genotypes": [g_1, g_2, ...],
            "weights": {
                "env_1": [w_1, w_2, ...],
                "env_2": [w_1, w_2, ...]
            }
        }
        """
        gs = data_dict['genotypes']
        w_env1 = data_dict['weights']['env_1']
        w_env2 = data_dict['weights']['env_2']

        # 1. Calculate Total Yield in Env 1
        # Y_total_1 = Sum( w_i * Yield_i(Env1) )
        Y1_total = 0.0
        for i, g in enumerate(gs):
            g_arr = np.array(g)  # Ensure numpy
            # Yield of genotype g in Env 1: 1*x + c*y
            y1_g = 1.0 * g_arr[0] + self.c * g_arr[1]
            Y1_total += w_env1[i] * y1_g

        # 2. Calculate Total Yield in Env 2
        # Y_total_2 = Sum( w_i * Yield_i(Env2) )
        Y2_total = 0.0
        for i, g in enumerate(gs):
            g_arr = np.array(g)
            # Yield of genotype g in Env 2: c*x + 1*y
            y2_g = self.c * g_arr[0] + 1.0 * g_arr[1]
            Y2_total += w_env2[i] * y2_g

        # 3. Calculate Global Statistics
        # Mean = p * Y1 + (1-p) * Y2
        mean = self.p * Y1_total + (1 - self.p) * Y2_total

        # Expected Square = p * Y1^2 + (1-p) * Y2^2
        expected_sq = self.p * (Y1_total ** 2) + (1 - self.p) * (Y2_total ** 2)

        # Variance = E[Y^2] - E[Y]^2
        variance = expected_sq - mean ** 2

        # Mean-Variance Score
        mean_variance = mean - 0.5 * self.gamma * variance

        return {'mean_variance': mean_variance, 'mean': mean, 'variance': variance, 'weights': data_dict['weights'], 'genotypes': data_dict['genotypes']}
    

    def _get_standard_output(self, g_list):
        """
        Generalized portfolio optimization for arbitrary number of genotypes using Gurobi.
        Args:
            g_list: List of genotype arrays (each of shape (2,))
        Returns:
            dict: Standardized output with genotypes and optimal weights
        """

        def get_y_vec(g):
            return np.array([g[0] + self.c * g[1], self.c * g[0] + g[1]])

        # 1. Compute yield vectors for all genotypes
        Y = np.column_stack([get_y_vec(g) for g in g_list])
        probs = np.array([self.p, 1.0 - self.p])

        # 2. Compute expected returns and covariance matrix
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
        weights = [w[i].X for i in range(n)] #if m.Status == GRB.OPTIMAL else [1.0] + [0.0] * (n - 1)

        if m.Status != GRB.OPTIMAL:
            print('ERROR')

        # 4. Format output
        return {
            "genotypes": g_list,
            "weights": {
                "env_1": weights,
                "env_2": weights,
            },
            "mean_variance": m.ObjVal,
            "mean": sum(w[i].X * y_bar[i] for i in range(n)),
            "variance":sum(w[i].X * Sigma[i, j] * w[j].X for i in range(n) for j in range(n))
        }

    def _get_hindsight_output_2_assets(self, g_fixed, g_cand):
        """Calculates switching utility and weights for Fixed + Candidate."""
        yf = np.array([g_fixed[0] + self.c * g_fixed[1], self.c * g_fixed[0] + g_fixed[1]])
        yc = np.array([g_cand[0] + self.c * g_cand[1], self.c * g_cand[0] + g_cand[1]])

        # Best yield in each environment
        y_opt = np.maximum(yf, yc)
        mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
        var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2

        # Weights: 1.0 for the winner in each environment
        w_env1 = [1.0, 0.0] if yf[0] >= yc[0] else [0.0, 1.0]
        w_env2 = [1.0, 0.0] if yf[1] >= yc[1] else [0.0, 1.0]

        return {
            "genotypes": [g_fixed.tolist(), g_cand.tolist()],
            "weights": {"env_1": w_env1, "env_2": w_env2},
            "hindsight_score": mu - 0.5 * self.gamma * var
        }

    def _get_hindsight_output_3_assets(self, g_fixed, g_mut, g_cand):
        """Calculates switching utility and weights for Fixed + Mutable + Candidate."""
        yf = np.array([g_fixed[0] + self.c * g_fixed[1], self.c * g_fixed[0] + g_fixed[1]])
        ym = np.array([g_mut[0] + self.c * g_mut[1], self.c * g_mut[0] + g_mut[1]])
        yc = np.array([g_cand[0] + self.c * g_cand[1], self.c * g_cand[0] + g_cand[1]])

        y_opt = np.maximum.reduce([yf, ym, yc])
        mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
        var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2

        # Determine weights for Env 1
        if yf[0] >= ym[0] and yf[0] >= yc[0]:
            w_env1 = [1.0, 0.0, 0.0]
        elif ym[0] >= yc[0]:
            w_env1 = [0.0, 1.0, 0.0]
        else:
            w_env1 = [0.0, 0.0, 1.0]

        # Determine weights for Env 2
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
    
    def run(self, strategies=None, plot_extensive=False, easy_base=False, no_plot=False, replace=False, pareto=False, plot_right=False, dpi=100):
        """
        Run scenarios and evaluate a set of strategies.

        strategies: dict[str, Callable(grid_dict) -> (idx_tuple, info_dict)]
            Where grid_dict contains precomputed arrays:
            The callable should return ((i,j), {'score': float, ...}) where (i,j) is
            an index into the 2D grids (np.unravel_index style).
        """
        self.replace = replace
        full_results = []
         # --- CHANGE 1: Setup Buffer to capture text ---
        output_buffer = io.StringIO()
        fig = None  # Placeholder for the figure

        # --- CHANGE 2: Wrap the execution in redirect_stdout ---
        with redirect_stdout(output_buffer):

            # Precompute Cholesky factor of G for ellipse plotting
            try:
                S = np.linalg.cholesky(self.G)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(self.G)
                vals[vals < 0] = 0.0
                S = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

            def strat_base(grid):
                g_fixed = grid["g_fixed"]
                g_mutable = grid["g_mutable"]

                details = self._get_standard_output([g_fixed, g_mutable])
                return None, details

            def strat_B4P(grid):
                """Maximize Portfolio Utility."""
                return None, grid['optimizer'].strat_B4P()

            def strat_B4M(grid):
                """Maximize Portfolio Mean Yield."""
                return None, grid['optimizer'].strat_B4M()

            def strat_max_share(grid):
                """Maximize Candidate Share (Adaptation)."""
                return None, grid['optimizer'].strat_max_share()

            def strat_max_investability(grid):
                """Maximize Candidate Share (Adaptation)."""
                return None, grid['optimizer'].strat_max_investability(min_improvement=1e-4)

            def strat_max_investability3(grid):
                """Maximize Candidate Share (Adaptation)."""
                return None, grid['optimizer'].strat_max_investability(min_improvement=1e-3)

            def strat_BB(grid):
                """BeatBest: Standalone Max."""
                return None, grid['optimizer'].strat_BB()

            def get_pareto_frontier(grid):
                """
                Computes the Pareto Frontier and returns a list of dictionaries 
                containing all metrics needed for CSV storage.
                """
                stats = grid['stats']

                W = stats['w_c']  # Adoption Share
                U = stats['v_port']  # Portfolio Utility (MV)
                M = stats['mean_port']  # Portfolio Mean
                mask = grid['feasible_mask']

                rows, cols = np.where(mask)
                valid_W = W[mask]
                valid_U = U[mask]
                valid_M = M[mask]

                # 1. Zip data for sorting
                # (Share, Utility, Mean, Row, Col)
                data = []
                for r, c, w, u, m in zip(rows, cols, valid_W, valid_U, valid_M):
                    data.append((w, u, m, r, c))

                # 2. Sort by Share Descending
                data.sort(key=lambda x: x[0], reverse=True)

                pareto_list = []
                current_max_u = -np.inf

                for w, u, m, r, c in data:
                    if u > current_max_u:
                        # 3. Build the dictionary exactly as your CSV logic expects
                        res = {
                            'mv_val': u,
                            'mean_val': m,
                            # Variance = (Mean - Utility) * 2 / Gamma
                            'var_val': (m - u) * 2 / self.gamma,
                            'w_cand': w,
                            'g1': grid['X'][r, c],
                            'g2': grid['Y'][r, c]
                        }

                        # Handle weights for the other assets
                        # We can approximate or just store the candidate weight for the Pareto
                        res['w_fixed'] = 1.0 - w  # Simplified for the CSV

                        pareto_list.append(res)
                        current_max_u = u

                return pareto_list

            def strat_opposite_direction(self, grid):
                """
                Selects the variety that is most 'opposite' to the current portfolio 
                to maximize directional diversification.
                Metric: Maximize projection onto the negative centroid vector.
                """
                X, Y = grid['X'], grid['Y']

                # 1. Determine the 'Center of Mass' of the existing portfolio
                if self.replace:
                    # 2-Assets: Center is just the Fixed asset
                    center_x = grid['g_fixed'][0]
                    center_y = grid['g_fixed'][1]
                else:
                    # 3-Assets: Center is the average of Fixed and Mutable
                    center_x = 0.5 * (grid['g_fixed'][0] + grid['g_mutable'][0])
                    center_y = 0.5 * (grid['g_fixed'][1] + grid['g_mutable'][1])

                # 2. Calculate Projection: Dot Product with the Negative Vector
                # We want to maximize: (g_cand) dot (-center)
                # This is equivalent to minimizing: (g_cand) dot (center)
                metric = -(X * center_x + Y * center_y)

                # 3. Apply Mask and Select
                metric = np.where(grid['feasible_mask'], metric, -np.inf)
                idx = np.unravel_index(np.nanargmax(metric), grid['X'].shape)

                # 4. Return Output
                g_cand = np.array([grid['X'][idx], grid['Y'][idx]])
                return idx, self._get_final_output(grid, g_cand)

            def strat_max_distance(self, grid):
                """
                Selects the variety that is geometrically furthest from the existing assets.
                Logic: Maximize the Minimum Euclidean Distance (Maximin) to ensuring distinctness.
                """
                X, Y = grid['X'], grid['Y']

                # 1. Calculate squared distance to Fixed Asset
                d2_fixed = (X - grid['g_fixed'][0]) ** 2 + (Y - grid['g_fixed'][1]) ** 2

                if self.replace:
                    # 2-Asset: Just maximize distance to Fixed
                    metric = d2_fixed
                else:
                    # 3-Asset: Maximize the distance to the *closest* existing neighbor
                    d2_mutable = (X - grid['g_mutable'][0]) ** 2 + (Y - grid['g_mutable'][1]) ** 2
                    metric = np.minimum(d2_fixed, d2_mutable)

                # 2. Apply Mask and Select
                metric = np.where(grid['feasible_mask'], metric, -np.inf)
                idx = np.unravel_index(np.nanargmax(metric), grid['X'].shape)

                # 3. Return Output
                g_cand = np.array([grid['X'][idx], grid['Y'][idx]])
                return idx, self._get_final_output(grid, g_cand)

            def strat_hindsight_base(grid):
                """
                Benchmark: Perfect information performance of the EXISTING portfolio.
                No grid search required. Returns switching utility of {Fixed, Mutable}.
                """
                g_f = grid["g_fixed"]
                g_m = grid["g_mutable"]

                # Evaluate hindsight switching between Fixed and Mutable
                details = self._get_hindsight_output_2_assets(g_f, g_m)

                # Return None as index because no new genotype was selected from the grid
                return None, details

            def strat_hindsight(grid):
                """
                True Hindsight: Scans the grid for the candidate that maximizes 
                switching utility when paired with existing assets.
                """
                # Precompute yields for fixed/mutable for speed
                yf1 = grid['g_fixed'][0] + self.c * grid['g_fixed'][1]
                yf2 = self.c * grid['g_fixed'][0] + grid['g_fixed'][1]

                if self.replace:
                    # Maximize switching between {Fixed, Candidate}
                    yc1 = grid['X'] + self.c * grid['Y']
                    yc2 = self.c * grid['X'] + grid['Y']
                    y_opt1, y_opt2 = np.maximum(yf1, yc1), np.maximum(yf2, yc2)
                else:
                    # Maximize switching between {Fixed, Mutable, Candidate}
                    ym1 = grid['g_mutable'][0] + self.c * grid['g_mutable'][1]
                    ym2 = self.c * grid['g_mutable'][0] + grid['g_mutable'][1]
                    yc1 = grid['X'] + self.c * grid['Y']
                    yc2 = self.c * grid['X'] + grid['Y']
                    y_opt1 = np.maximum.reduce([yf1, ym1, yc1])
                    y_opt2 = np.maximum.reduce([yf2, ym2, yc2])

                # Calculate Hindsight Score (MV of the switched yields)
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

            def get_hindsight_portfolio_output_2_assets(g_fixed, g_cand):
                """
                Optimizes TWO separate portfolios (w1 for Env1, w2 for Env2) 
                using only Fixed and Candidate assets (Replace Mode).
                """
                import gurobipy as gp
                from gurobipy import GRB

                # 1. Setup Yield Vectors for 2 Assets
                # y_vec_1 = [Yield_Fixed_E1, Yield_Cand_E1]
                y_vec_1 = np.array([
                    g_fixed[0] + self.c * g_fixed[1],
                    g_cand[0] + self.c * g_cand[1]
                ])

                # y_vec_2 = [Yield_Fixed_E2, Yield_Cand_E2]
                y_vec_2 = np.array([
                    self.c * g_fixed[0] + g_fixed[1],
                    self.c * g_cand[0] + g_cand[1]
                ])

                try:
                    m = gp.Model()
                    m.setParam('OutputFlag', 0)
                    m.setParam("TimeLimit", 60)

                    # 2. Decision Variables: Two sets of weights (size 2)
                    w1 = m.addVars(2, lb=0.0, ub=1.0, name="w1")  # Weights for Env 1
                    w2 = m.addVars(2, lb=0.0, ub=1.0, name="w2")  # Weights for Env 2

                    # 3. Constraints: Budget constraint for each environment
                    m.addConstr(gp.quicksum(w1) == 1, "Budget_Env1")
                    m.addConstr(gp.quicksum(w2) == 1, "Budget_Env2")

                    # 4. Define Objective Components
                    # Mean Yield in Env 1 and Env 2
                    mu_1 = gp.quicksum(w1[i] * y_vec_1[i] for i in range(2))
                    mu_2 = gp.quicksum(w2[i] * y_vec_2[i] for i in range(2))

                    # Overall Hindsight Mean and Variance
                    # Mean = p * mu_1 + (1-p) * mu_2
                    h_mean = self.p * mu_1 + (1 - self.p) * mu_2

                    # Variance = p * (1-p) * (mu_1 - mu_2)^2
                    diff = m.addVar(lb=-GRB.INFINITY, name="diff")
                    m.addConstr(diff == mu_1 - mu_2)
                    h_var = self.p * (1 - self.p) * (diff * diff)

                    # Objective Function
                    obj = h_mean - 0.5 * self.gamma * h_var
                    m.setObjective(obj, GRB.MAXIMIZE)

                    # 5. Optimize
                    m.optimize()

                    if m.Status == GRB.OPTIMAL:
                        weights_1 = [w1[i].X for i in range(2)]
                        weights_2 = [w2[i].X for i in range(2)]
                    else:
                        # Fallback to greedy (pure fixed)
                        weights_1 = [1, 0]
                        weights_2 = [1, 0]

                except Exception as e:
                    print(f"Optimization failed: {e}")
                    weights_1 = [1, 0]
                    weights_2 = [1, 0]

                # 6. Format Output
                return {
                    "genotypes": [g_fixed.tolist(), g_cand.tolist()],
                    "weights": {
                        "env_1": weights_1,
                        "env_2": weights_2
                    }
                }

            def get_hindsight_portfolio_output_3_assets(g_fixed, g_mut, g_cand):
                """
                Optimizes TWO separate portfolios (w1 for Env1, w2 for Env2) 
                to maximize the joint Hindsight Utility.
                """
                import gurobipy as gp
                from gurobipy import GRB

                # 1. Setup Yield Vectors
                # y_vec_1 = [Yield_Fixed_E1, Yield_Mut_E1, Yield_Cand_E1]
                y_vec_1 = np.array([
                    g_fixed[0] + self.c * g_fixed[1],
                    g_mut[0] + self.c * g_mut[1],
                    g_cand[0] + self.c * g_cand[1]
                ])

                # y_vec_2 = [Yield_Fixed_E2, Yield_Mut_E2, Yield_Cand_E2]
                y_vec_2 = np.array([
                    self.c * g_fixed[0] + g_fixed[1],
                    self.c * g_mut[0] + g_mut[1],
                    self.c * g_cand[0] + g_cand[1]
                ])

                try:
                    m = gp.Model()
                    m.setParam('OutputFlag', 0)
                    m.setParam("NonConvex", 2)
                    m.setParam("TimeLimit", 60)

                    # 2. Decision Variables: Two sets of weights
                    w1 = m.addVars(3, lb=0.0, ub=1.0, name="w1")  # Weights for Env 1
                    w2 = m.addVars(3, lb=0.0, ub=1.0, name="w2")  # Weights for Env 2

                    # 3. Constraints: Budget constraint for each environment
                    m.addConstr(gp.quicksum(w1) == 1, "Budget_Env1")
                    m.addConstr(gp.quicksum(w2) == 1, "Budget_Env2")

                    # 4. Define Objective Components
                    # Mean Yield in Env 1 and Env 2
                    mu_1 = gp.quicksum(w1[i] * y_vec_1[i] for i in range(3))
                    mu_2 = gp.quicksum(w2[i] * y_vec_2[i] for i in range(3))

                    # Overall Hindsight Mean and Variance
                    # Mean = p * mu_1 + (1-p) * mu_2
                    h_mean = self.p * mu_1 + (1 - self.p) * mu_2

                    # Variance = p * (1-p) * (mu_1 - mu_2)^2
                    # Note: We create a helper variable for the difference to keep it clean
                    diff = m.addVar(lb=-GRB.INFINITY, name="diff")
                    m.addConstr(diff == mu_1 - mu_2)
                    h_var = self.p * (1 - self.p) * (diff * diff)

                    # Objective Function
                    obj = h_mean - 0.5 * self.gamma * h_var
                    m.setObjective(obj, GRB.MAXIMIZE)

                    # 5. Optimize
                    m.optimize()

                    if m.Status == GRB.OPTIMAL:
                        weights_1 = [w1[i].X for i in range(3)]
                        weights_2 = [w2[i].X for i in range(3)]
                    else:
                        # Fallback to greedy (pure fixed)
                        weights_1 = [1, 0, 0]
                        weights_2 = [1, 0, 0]

                except Exception as e:
                    print(f"Optimization failed: {e}")
                    weights_1 = [1, 0, 0]
                    weights_2 = [1, 0, 0]

                # 6. Format Output
                return {
                    "genotypes": [g_fixed.tolist(), g_mut.tolist(), g_cand.tolist()],
                    "weights": {
                        "env_1": weights_1,
                        "env_2": weights_2
                    }
                }

            def strat_hindsight_optimized(grid):
                """
                Strategy C (Hindsight Portfolio - Fully Optimized): 
                Finds the Candidate that maximizes Utility when we are allowed to 
                optimize specific weights (w1, w2) for each environment.
                
                IMPROVEMENT: Instead of searching with 'Pure' assets (w=0 or 1), 
                we estimate the optimal (w1, w2) for every grid point using 
                vectorized coordinate descent. This ensures we don't miss 
                candidates that require mixing to be effective.
                """
                return None, grid['optimizer'].strat_hindsight()
                '''
                p, c, gamma = self.p, self.c, self.gamma
                X, Y = grid['X'], grid['Y']

                # 1. Define Yield Surfaces (Fixed vs Candidate)
                # We treat 'Fixed' as the base asset. If 'Mutable' exists, we 
                # could add it to Fixed, but for the search phase, Fixed vs Cand is sufficient.
                yf_1 = grid['g_fixed'][0] + c * grid['g_fixed'][1]  # Scalar
                yf_2 = c * grid['g_fixed'][0] + grid['g_fixed'][1]  # Scalar

                yc_1 = X + c * Y  # Grid (N, N)
                yc_2 = c * X + Y  # Grid (N, N)

                # Define Delta (Yield Difference: Candidate - Fixed)
                # This represents the gain from shifting weight from Fixed to Candidate
                D1 = yc_1 - yf_1
                D2 = yc_2 - yf_2

                # 2. Vectorized Alternating Optimization
                # We want to find w1, w2 (weight of Candidate) in [0, 1]
                # Initialize weights at 0.5 (Neutral mix)
                w1 = np.full_like(X, 0.5)
                w2 = np.full_like(X, 0.5)

                # Pre-calculate constant penalty factor lambda
                # Variance = p*q*(mu1 - mu2)^2. Utility -0.5*gamma*Var.
                # So penalty term coefficient is 0.5 * gamma * p * (1-p)
                # Let's simplify derivative math:
                # Obj = p*mu1 + q*mu2 - K * (mu1 - mu2)^2
                q = 1.0 - p
                K = 0.5 * gamma * p * q

                # Avoid division by zero in updates
                denom_D1 = np.where(np.abs(D1) < 1e-9, 1e-9, D1)
                denom_D2 = np.where(np.abs(D2) < 1e-9, 1e-9, D2)

                # Iterative Solver (2-3 passes is usually enough for convex QP)
                for _ in range(3):
                    # --- Update w1 (Fix w2) ---
                    # Current Mean of Env 2
                    mu2 = yf_2 + w2 * D2

                    # Analytic solution for w1 where d(Utility)/dw1 = 0
                    # Derived from: p*D1 - 2*K*(yf_1 + w1*D1 - mu2)*D1 = 0
                    # w1_opt = (p/(2K) - yf_1 + mu2) / D1
                    target_val = (p / (2 * K)) - yf_1 + mu2
                    w1_raw = target_val / denom_D1
                    w1 = np.clip(w1_raw, 0.0, 1.0)

                    # --- Update w2 (Fix w1) ---
                    # Current Mean of Env 1
                    mu1 = yf_1 + w1 * D1

                    # Analytic solution for w2 where d(Utility)/dw2 = 0
                    # Derived from: q*D2 + 2*K*(mu1 - (yf_2 + w2*D2))*D2 = 0 (Note sign flip for second term)
                    # w2_opt = (mu1 - yf_2 + q/(2K)) / D2
                    target_val_2 = mu1 - yf_2 + (q / (2 * K))
                    w2_raw = target_val_2 / denom_D2
                    w2 = np.clip(w2_raw, 0.0, 1.0)

                # 3. Calculate Final Utility with Optimal Weights
                opt_mu1 = yf_1 + w1 * D1
                opt_mu2 = yf_2 + w2 * D2

                h_mean = p * opt_mu1 + q * opt_mu2
                h_var = p * q * (opt_mu1 - opt_mu2) ** 2

                final_utility_grid = h_mean - 0.5 * gamma * h_var

                # 4. Select the Winner
                metric = np.where(grid['feasible_mask'], final_utility_grid, -np.inf)
                idx = np.unravel_index(np.nanargmax(metric), grid['X'].shape)
                g_cand = np.array([grid['X'][idx], grid['Y'][idx]])

                # 5. Return Output (Use Gurobi helper for final precision)
                if self.replace:
                    details = get_hindsight_portfolio_output_2_assets(grid["g_fixed"], g_cand)
                else:
                    details = get_hindsight_portfolio_output_3_assets(grid["g_fixed"], grid["g_mutable"], g_cand)

                return idx, details
                '''

            def calculate_split_mv(y1, y2):
                """Helper to compute global Mean-Variance from environment-specific yields."""
                mu = self.p * y1 + (1 - self.p) * y2
                exp_sq = self.p * (y1 ** 2) + (1 - self.p) * (y2 ** 2)
                var = exp_sq - mu ** 2
                return mu - 0.5 * self.gamma * var

            def get_split_strategies(grid):
                """
                Internal logic to find the best of 4 split-environment cases:
                1. Env1: Asset A, Env2: Asset B
                2. Env1: Asset B, Env2: Asset A
                3. Both: Asset A
                4. Both: Asset B
                """
                p, c = self.p, self.c
                g_f = np.asarray(grid["g_fixed"])
                X, Y = np.asarray(grid["X"]), np.asarray(grid["Y"])
                mask = np.asarray(grid["feasible_mask"])

                # Fixed Yields
                y1_f = g_f[0] + c * g_f[1]
                y2_f = c * g_f[0] + g_f[1]

                # Grid/Candidate Yields
                y1_c = X + c * Y
                y2_c = c * X + Y

                # Evaluate the 4 Cases across the whole grid
                scores = {
                    "f_c": calculate_split_mv(y1_f, y2_c),  # Case 1: Fixed/Cand
                    "c_f": calculate_split_mv(y1_c, y2_f),  # Case 2: Cand/Fixed
                    "f_f": calculate_split_mv(y1_f, y2_f),  # Case 3: Fixed/Fixed
                    "c_c": calculate_split_mv(y1_c, y2_c)  # Case 4: Cand/Cand
                }

                # Find the best case and best genotype
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

                # Map case to standard weights [w_fixed, w_cand]
                weight_map = {
                    "f_c": {"env_1": [1.0, 0.0], "env_2": [0.0, 1.0]},
                    "c_f": {"env_1": [0.0, 1.0], "env_2": [1.0, 0.0]},
                    "f_f": {"env_1": [1.0, 0.0], "env_2": [1.0, 0.0]},
                    "c_c": {"env_1": [0.0, 1.0], "env_2": [0.0, 1.0]}
                }

                g_cand = np.array([X[best_idx], Y[best_idx]])
                return best_idx, {
                    "genotypes": [g_f.tolist(), g_cand.tolist()],
                    "weights": weight_map[best_case]
                }

            def strat_splitEnv_0B(grid):
                """Zero Breeding Split: Uses original Fixed and Mutable assets."""
                # Temporarily swap grid X/Y with a single-point 'grid' containing only g_mutable
                g_m = grid["g_mutable"]
                mock_grid = grid.copy()
                mock_grid.update(
                    {'X': np.array([[g_m[0]]]), 'Y': np.array([[g_m[1]]]), 'feasible_mask': np.array([[True]])})
                _, details = get_split_strategies(mock_grid)
                # Find index of g_mutable in actual grid for consistent return
                dist = (grid['X'] - g_m[0]) ** 2 + (grid['Y'] - g_m[1]) ** 2
                return np.unravel_index(np.argmin(dist), grid['X'].shape), details

            def strat_splitEnv_fixed(grid):
                """Breeding Split: Finds best g_cand to pair with g_fixed in a split env."""
                return get_split_strategies(grid)



            def strat_splitEnv_freeWeight(grid):
                """
                Optimized Free Weight Split using analytical boundary search.
                Replaces 10,000 weight combinations with 4 edge evaluations.
                """
                p, c, gamma = self.p, self.c, self.gamma
                g_f = grid["g_fixed"]
                X, Y, mask = grid["X"], grid["Y"], grid["feasible_mask"]

                # Yields
                y1_f, y2_f = g_f[0] + c * g_f[1], c * g_f[0] + g_f[1]
                y1_c, y2_c = X + c * Y, c * X + Y

                # Precompute constants for the MV score: p(1-p)*gamma/2
                K = 0.5 * gamma * p * (1 - p)

                def solve_edge(fixed_w1=None, fixed_w2=None):
                    """Solves for the optimal weight on a specific edge of the [0,1]^2 box."""
                    # Yields in Env 1 and 2 as functions of the free weight 'w'
                    if fixed_w1 is not None:  # Edge where w1 is constant, w2 is free
                        # Y1 is constant: y1_c + fixed_w1 * (y1_f - y1_c)
                        Y1 = y1_c + fixed_w1 * (y1_f - y1_c)
                        # Y2 = y2_c + w2 * (y2_f - y2_c)
                        d2 = y2_f - y2_c
                        # U(w2) = (1-p)*d2*w2 - K*(Y1 - (y2_c + w2*d2))^2 + const
                        # Quadratic: a*w2^2 + b*w2 + c
                        a = -K * (d2 ** 2)
                        b = (1 - p) * d2 + 2 * K * d2 * (Y1 - y2_c)
                    else:  # Edge where w2 is constant, w1 is free
                        Y2 = y2_c + fixed_w2 * (y2_f - y2_c)
                        d1 = y1_f - y1_c
                        a = -K * (d1 ** 2)
                        b = p * d1 - 2 * K * d1 * (y1_c - Y2)

                    # Optimal w = -b / (2a), handle a=0 (linear case)
                    w_opt = np.zeros_like(X)
                    safe = np.abs(a) > 1e-12
                    w_opt[safe] = -b[safe] / (2 * a[safe])
                    # If linear (a=0), pick 0 or 1 based on slope b
                    w_opt[~safe] = np.where(b[~safe] > 0, 1.0, 0.0)

                    w_opt = np.clip(w_opt, 0.0, 1.0)

                    # Calculate scores for this optimal w on this edge
                    if fixed_w1 is not None:
                        res_w1, res_w2 = np.full_like(X, fixed_w1), w_opt
                    else:
                        res_w1, res_w2 = w_opt, np.full_like(X, fixed_w2)

                    yp1 = y1_c + res_w1 * (y1_f - y1_c)
                    yp2 = y2_c + res_w2 * (y2_f - y2_c)
                    scores = calculate_split_mv(yp1, yp2)
                    return scores, res_w1, res_w2

                # Check the 4 boundaries of the [0,1] x [0,1] weight box
                edges = [
                    solve_edge(fixed_w1=0.0),  # w1=0, w2 in [0,1]
                    solve_edge(fixed_w1=1.0),  # w1=1, w2 in [0,1]
                    solve_edge(fixed_w2=0.0),  # w2=0, w1 in [0,1]
                    solve_edge(fixed_w2=1.0)  # w2=1, w1 in [0,1]
                ]

                # Find the best edge for every genotype
                all_scores = np.stack([e[0] for e in edges])
                best_edge_idx = np.argmax(all_scores, axis=0)

                # Extract best scores and weights
                best_scores = np.take_along_axis(all_scores, best_edge_idx[None, ...], axis=0)[0]

                # Selection
                masked_scores = np.where(mask, best_scores, -np.inf)
                win_idx_flat = np.nanargmax(masked_scores)
                idx = np.unravel_index(win_idx_flat, X.shape)

                # Get the winning weights for that specific grid point
                win_edge = best_edge_idx[idx]
                w1_win = edges[win_edge][1][idx]
                w2_win = edges[win_edge][2][idx]

                return idx, {
                    "genotypes": [g_f.tolist(), [float(X[idx]), float(Y[idx])]],
                    "weights": {
                        "env_1": [float(w1_win), 1.0 - float(w1_win)],
                        "env_2": [float(w2_win), 1.0 - float(w2_win)]
                    }
                }

            def get_split_effort_output(grid, g_c1, g_c2):
                """
                Constructs the portfolio output for two specialist candidates (Split Effort).
                Portfolio uses: Max(Fixed, [Mutable], C1, C2) in each environment.
                """
                # 1. Existing Assets Yields
                yf = np.array([grid['g_fixed'][0] + self.c * grid['g_fixed'][1],
                            self.c * grid['g_fixed'][0] + grid['g_fixed'][1]])

                assets_y = [yf]  # List of arrays [y_env1, y_env2]
                genotypes = [grid['g_fixed'].tolist()]

                if not self.replace:
                    gm = grid['g_mutable']
                    ym = np.array([gm[0] + self.c * gm[1], self.c * gm[0] + gm[1]])
                    assets_y.append(ym)
                    genotypes.append(gm.tolist())

                # 2. Candidate Yields
                yc1 = np.array([g_c1[0] + self.c * g_c1[1], self.c * g_c1[0] + g_c1[1]])
                yc2 = np.array([g_c2[0] + self.c * g_c2[1], self.c * g_c2[0] + g_c2[1]])

                assets_y.extend([yc1, yc2])
                genotypes.extend([g_c1.tolist(), g_c2.tolist()])

                # 3. Determine Winner in each Environment (Hindsight Logic)
                # Stack yields: Shape (NumAssets, 2) -> (2, NumAssets) for easy max
                all_yields = np.vstack(assets_y).T

                # Best yield values
                y_opt = np.max(all_yields, axis=1)  # [max_env1, max_env2]

                # Best asset indices (for weights)
                win_idx = np.argmax(all_yields, axis=1)  # [idx_env1, idx_env2]

                # 4. Calculate Portfolio Stats
                mu = self.p * y_opt[0] + (1 - self.p) * y_opt[1]
                var = (self.p * (y_opt[0] ** 2) + (1 - self.p) * (y_opt[1] ** 2)) - mu ** 2
                score = mu - 0.5 * self.gamma * var

                # 5. Construct Weights Dictionary
                # Create one-hot weight vectors
                n_assets = len(assets_y)
                w_env1 = [0.0] * n_assets
                w_env2 = [0.0] * n_assets
                w_env1[win_idx[0]] = 1.0
                w_env2[win_idx[1]] = 1.0

                return {
                    "genotypes": genotypes,
                    "weights": {"env_1": w_env1, "env_2": w_env2},
                    "split_effort_score": score
                }

            def strat_SplitEffort(grid):
                """
                Independent Specialists: Finds the best genotype for Env 1 and 
                the best for Env 2 separately, then combines them.
                """
                # Yield grids
                Y1 = grid['X'] + self.c * grid['Y']
                Y2 = self.c * grid['X'] + grid['Y']

                # Apply mask
                Y1[~grid['feasible_mask']] = -np.inf
                Y2[~grid['feasible_mask']] = -np.inf

                # Select best for E1
                idx1 = np.unravel_index(np.nanargmax(Y1), Y1.shape)
                g_c1 = np.array([grid['X'][idx1], grid['Y'][idx1]])

                # Select best for E2
                idx2 = np.unravel_index(np.nanargmax(Y2), Y2.shape)
                g_c2 = np.array([grid['X'][idx2], grid['Y'][idx2]])

                # Return details (using g_c1 index as the primary index for plotting)
                details = get_split_effort_output(grid, g_c1, g_c2)
                return idx1, details

            def strat_SplitEffortOptimal(grid):
                """
                Theoretical Specialists: Finds grid points closest to the analytical 
                Maximum Yield solution for each environment.
                """
                # Analytical solution for Max Yield subject to g^T G g <= R^2
                # Maximize beta^T g -> g_opt = R * (G^-1 beta) / norm

                # Beta vectors for Env 1 and Env 2
                b1 = np.array([1.0, self.c])
                b2 = np.array([self.c, 1.0])

                # Unconstrained directions (G_inv @ beta)
                dir1 = self.Ginv @ b1
                dir2 = self.Ginv @ b2

                # Scale to radius R (assuming elliptical constraint centered at 0)
                # R is likely stored in self.R or implicit in grid mask. 
                # Using the grid's max feasible norm as proxy if R not explicit, 
                # but usually g_opt is just the direction vector for search.

                # Search Grid for points maximizing the projection onto these optimal directions
                # (This is more robust than scaling if R is complex)

                proj1 = grid['X'] * dir1[0] + grid['Y'] * dir1[1]
                proj2 = grid['X'] * dir2[0] + grid['Y'] * dir2[1]

                proj1[~grid['feasible_mask']] = -np.inf
                proj2[~grid['feasible_mask']] = -np.inf

                idx1 = np.unravel_index(np.nanargmax(proj1), proj1.shape)
                g_c1 = np.array([grid['X'][idx1], grid['Y'][idx1]])

                idx2 = np.unravel_index(np.nanargmax(proj2), proj2.shape)
                g_c2 = np.array([grid['X'][idx2], grid['Y'][idx2]])

                details = get_split_effort_output(grid, g_c1, g_c2)
                return idx1, details

            def strat_SplitEffortGlobal(grid):
                """
                Joint Optimization: Finds the pair (g1, g2) that maximizes the 
                Global Mean-Variance score using Coordinate Descent.
                """
                # 1. Initialize with Independent Specialists (Fast start)
                idx1_curr, details_init = strat_SplitEffort(grid)
                g_c1 = np.array(details_init['genotypes'][-2])  # 2nd to last is c1
                g_c2 = np.array(details_init['genotypes'][-1])  # Last is c2

                # Precompute yield grids
                Y1_grid = grid['X'] + self.c * grid['Y']
                Y2_grid = self.c * grid['X'] + grid['Y']
                Y1_grid[~grid['feasible_mask']] = -np.inf
                Y2_grid[~grid['feasible_mask']] = -np.inf

                # Existing assets yield (Fixed / Mutable)
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

                # 2. Coordinate Descent (Alternating Optimization)
                # Usually converges in 2-3 iterations
                for _ in range(10):
                    # Step A: Fix g_c2, Optimize g_c1
                    # In Env 2, we stick with max(Base, g_c2)
                    yc2_val = g_c2[0] + self.c * g_c2[1]  # Yield of c2 in Env 1 (irrelevant if c1 wins)
                    yc2_val_e2 = self.c * g_c2[0] + g_c2[1]

                    # The portfolio yield in Env 2 is fixed for this step:
                    Y_p2 = max(base_y2, yc2_val_e2)

                    # We need to maximize Score( Y_p1, Y_p2 )
                    # Y_p1 = max(base_y1, Y1_grid, yc2_val) -> usually c2 is weak in E1, so max(base, Y1_grid)
                    # But rigorously:
                    Y_p1_grid = np.maximum(base_y1, np.maximum(yc2_val, Y1_grid))

                    mu = self.p * Y_p1_grid + (1 - self.p) * Y_p2
                    var = (self.p * Y_p1_grid ** 2 + (1 - self.p) * Y_p2 ** 2) - mu ** 2
                    score_grid = mu - 0.5 * self.gamma * var

                    # Update g_c1
                    best_idx1 = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
                    g_c1 = np.array([grid['X'][best_idx1], grid['Y'][best_idx1]])

                    # Step B: Fix g_c1, Optimize g_c2
                    yc1_val_e1 = g_c1[0] + self.c * g_c1[1]
                    yc1_val_e2 = self.c * g_c1[0] + g_c1[1]

                    Y_p1 = max(base_y1, yc1_val_e1)

                    Y_p2_grid = np.maximum(base_y2, np.maximum(yc1_val_e2, Y2_grid))

                    mu = self.p * Y_p1 + (1 - self.p) * Y_p2_grid
                    var = (self.p * Y_p1 ** 2 + (1 - self.p) * Y_p2_grid ** 2) - mu ** 2
                    score_grid = mu - 0.5 * self.gamma * var

                    # Update g_c2
                    best_idx2 = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
                    g_c2 = np.array([grid['X'][best_idx2], grid['Y'][best_idx2]])

                details = get_split_effort_output(grid, g_c1, g_c2)
                return best_idx1, details

            # Build strategy dict
            default_strategies = {
                'Base': strat_base,
                'Mean': strat_B4M,
                'MVP': strat_B4P,
                'BeatBest': strat_BB,
                'Adopt': strat_max_investability,
                'Clairv': strat_hindsight_optimized,
            }
            if strategies is None:
                strategies = default_strategies
            else:
                # merge defaults for any missing convenience items (optional)
                default_strategies = dict(default_strategies)
                for k in strategies:
                    if k in default_strategies.keys():
                        strategies[k] = default_strategies[k]

            for id, scen in enumerate(self.scenario_pairs):
                g_fixed = scen["g_fixed"]
                g_mutable = scen["g_mutable"]
                label = scen.get("label", f"Scenario {id}")

                # --- A. Baseline (Before Breeding) ---
                U_fix = self.get_mean_yield(g_fixed)
                Var_fix = self.get_variance(g_fixed)

                # Value of original portfolio
                base_line = self._get_standard_output([g_fixed, g_mutable])

                # --- B. Grid Calculations (Candidates for new Mutable) ---
                U_grid = self.beta[0] * self.X + self.beta[1] * self.Y
                Var_grid = self.var_y_grid()

                Cov_grid_fix = self.cov_y_grid_with_fixed(g_fixed)

                Delta_mu = U_fix - U_grid
                Var_diff = Var_fix + Var_grid - 2 * Cov_grid_fix
                denom_safe = np.where(Var_diff < 1e-9, 1e-9, Var_diff)

                # Weight of FIXED asset
                w_fix_grid = (Delta_mu / self.gamma + (Var_grid - Cov_grid_fix)) / denom_safe
                w_fix_grid = np.clip(w_fix_grid, 0.0, 1.0)
                w_new_grid = 1.0 - w_fix_grid

                # --- C. Mahalanobis Feasibility ---
                u1 = self.X - g_mutable[0]
                u2 = self.Y - g_mutable[1]
                d2 = self.Ginv[0, 0] * u1 ** 2 + 2 * self.Ginv[0, 1] * u1 * u2 + self.Ginv[1, 1] * u2 ** 2
                feasible_mask = (d2 <= self.R ** 2)

                if feasible_mask.sum() == 0 and not no_plot:
                    print(f"Skipping {label}: No feasible points under Mahalanobis constraint.")
                    continue

                # Prepare a grid_dict passed to each strategy
                grid_dict = {'X': self.X, 'Y': self.Y, 'U_grid': U_grid, 'Var_grid': Var_grid,
                            'feasible_mask': feasible_mask, 'g_fixed': g_fixed, 'g_mutable': g_mutable, 'U_fix': U_fix,
                            'Cov_grid_fix': Cov_grid_fix, 'w_fix_grid': w_fix_grid,
                            'w_new_grid': w_new_grid,
                            'p': self.p, 'beta': self.beta, 'r_g': self.r_g, 'gamma': self.gamma, 'c': self.c,
                            'Ginv': self.Ginv, 'R': self.R, 'replace': self.replace}
                grid_dict['stats'] = self.get_exact_grid_stats(grid_dict)
                grid_dict['optimizer'] = GurobiPortfolioOptimizer(grid_dict)

                self.grid_dict = grid_dict

                if not easy_base:
                    best_possible_value = self.calculate_stats_from_dict(strat_hindsight_optimized(grid_dict)[1])['mean_variance']

                details_per_strategy = {}

                for name, strat_func in strategies.items():
                    # Call strategy
                    idx, strat_dict = strat_func(grid_dict)  # grid_inputs is your 'grid' dict

                    # Calculate Stats using the new helper method
                    if 'Clairv' not in name:
                        out_dict = self._get_standard_output(strat_dict['genotypes'])
                    else:
                        out_dict = self.calculate_stats_from_dict(strat_dict)

                    # Calculate Gain (Improvement over purely Fixed)
                    gain = out_dict['mean_variance'] - base_line['mean_variance']

                    # Store result
                    details_per_strategy[name] = {
                        'name': name,
                        'idx': idx,
                        'out_dict': out_dict,  # The standardized dict
                        'stats': {
                            'mean_variance': out_dict['mean_variance'],
                            'mean': out_dict['mean'],
                            'variance': out_dict['variance'],
                            'gain': gain
                        }
                    }

                # --- E. Visualizations: 6 Subplots ---
                # 1. Prepare Plotting Data (Ellipse)
                theta = np.linspace(0, 2 * np.pi, 300)
                circle = np.vstack([np.cos(theta), np.sin(theta)]) * self.R
                ellipse_pts = (S @ circle).T + g_mutable.reshape(1, 2)

                # 2. Prepare Colors and Markers (Pre-calculate for consistency across subplots)
                from itertools import cycle
                marker_cycle = ['D', 'o', 's', '^', 'v', 'P', 'X', '*', 'h', 'H']
                color_cycle = ['lime', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'teal', 'gold', 'crimson', 'navy']

                strat_names = []
                gains = []
                adaptions1 = []
                adaptions2 = []
                plot_styles = {}  # name -> (marker, color)

                mk_iter = cycle(marker_cycle)
                col_iter = cycle(color_cycle)

                # Extract data for bar chart and assign styles
                for name, res in details_per_strategy.items():
                    if name == "Base":
                        continue
                    mv = res['stats']['mean_variance']
                    # Normalize gain relative to best possible
                    if easy_base:
                        gain_val = mv - base_line['mean_variance']
                    else:
                        gain_val = (mv - base_line['mean_variance']) / (
                                    best_possible_value - base_line['mean_variance']) if (best_possible_value - base_line['mean_variance']) != 0 else 0
                    gain_val = np.round(gain_val, 4)
                    gains.append(gain_val * 100)
                    if type(res['out_dict']['weights']) == list:
                        adaptions1.append(res['out_dict']['weights'][-1])
                        adaptions2.append(res['out_dict']['weights'][-1])
                    else:
                        adaptions1.append(res['out_dict']['weights']['env_1'][-1])
                        adaptions2.append(res['out_dict']['weights']['env_2'][-1])
                    strat_names.append(name)
                    plot_styles[name] = (next(mk_iter), next(col_iter))

                # Header
                # Gain | MV | Mean | Var | Weights (Env1) | Weights (Env2)
                header = f"{'Strategy':<25} |  {'MV':<8} | {'Gain':<8} |{'Mean':<8} | {'Var':<8} | {'Weights (Env1 / Env2)'}"
                if not no_plot:
                    print(header)
                if not no_plot:
                    print("-" * 115)

                def fmt_w(w_list):
                    # Format list of weights to string "0.50, 0.50"
                    return ", ".join([f"{x:.2f}" for x in w_list])
                
                if type(base_line['weights']) == dict:
                    w_e1 = base_line['weights']['env_1']
                    w_e2 = base_line['weights']['env_2']

                    # Check if weights are identical for both envs (for cleaner display)
                    if w_e1 == w_e2:
                        w_str = f"[{fmt_w(w_e1)}]"
                    else:
                        w_str = f"E1:[{fmt_w(w_e1)}] E2:[{fmt_w(w_e2)}]"
                else:
                    w_str = [round(i, 2) for i in base_line['weights']]
                name = 'Baseline'
                row_str = (f"{name:<25} | "
                            f"{base_line['mean_variance']:<8.4f} | "
                            f"{0:<8.4f} | "
                            f"{base_line['mean']:<8.4f} | "
                            f"{base_line['variance']:<8.4f} | "
                            f"{w_str}")
                if not no_plot:
                    print(row_str)

                for name, res in details_per_strategy.items():
                    s = res['stats']
                    d = res['out_dict']

                    if type(d['weights']) == dict:
                        w_e1 = d['weights']['env_1']
                        w_e2 = d['weights']['env_2']

                        # Check if weights are identical for both envs (for cleaner display)
                        if w_e1 == w_e2:
                            w_str = f"[{fmt_w(w_e1)}]"
                        else:
                            w_str = f"E1:[{fmt_w(w_e1)}] E2:[{fmt_w(w_e2)}]"
                    else:
                        w_str = [round(i, 2) for i in d['weights']]

                    row_str = (f"{name:<25} | "
                            f"{s['mean_variance']:<8.4f} | "
                            f"{s['gain']:<8.4f} | "
                            f"{s['mean']:<8.4f} | "
                            f"{s['variance']:<8.4f} | "
                            f"{w_str}")
                    if not no_plot:
                        print(row_str)

                # Build output for this scenario
                scenario_out = {
                    "Scenario": label,
                    "Baseline": base_line,
                    "Results": [details_per_strategy[n] for n in strat_names]
                }
                full_results.append(scenario_out)
                if no_plot:
                    return full_results

                # --- NEW BLOCK: Single Genotype Metrics ---
                print("-" * 115)
                print(
                    f"{'Single Component':<25} | {'MV':<8} | {'Gain':<8} |{'Mean':<8}  | {'Var':<8} | {'g1':<8} | {'g2':<8} | {'y1':<8} | {'y2':<8}  ")

                def print_pure_row(row_name, g_vec, ref_mv):
                    g = np.array(g_vec)  # Ensure numpy array
                    mu, var, mv, _, _ = self._get_yields_and_stats(g[0], g[1])
                    gain = mv - ref_mv
                    # Print row matching the table format
                    print(
                        f"{row_name:<25} | {mv:<8.4f} | {gain:<8.4f} | {mu:<8.4f} | {var:<8.4f} | {g[0]:<8.4f} | {g[1]:<8.4f}| {(self.p * (g[0] + self.c * g[1])):<8.4f} | {((1 - self.p) * (self.c * g[0] + g[1])):<8.4f} ")

                # 2. Print Standard Components
                print_pure_row("Fixed Asset (Base)", g_fixed, base_line['mean_variance'])
                print_pure_row("Mutable Asset", g_mutable, base_line['mean_variance'])

                # 3. Print the 'Bred' Component from each Strategy
                for name, res in details_per_strategy.items():
                    # logic: The 'new' variety is always the last one in the genotypes list
                    # Strat A/B/C: [Fixed, New]
                    # Strat D:     [Fixed, Mutable, New]
                    gens = res['out_dict']['genotypes']
                    for i, g in enumerate(gens):
                        if i > 1 or ((g[0] != g_fixed[0] or g[1] != g_fixed[1]) and (
                                g[0] != g_mutable[0] and g[1] != g_mutable[1])):
                            print_pure_row(f"Bred ({name})", g, base_line['mean_variance'])

                print("-" * 115)

                # --- 1. Calculate Statistics & Covariance Grids (Pre-calculation) ---
                # Fixed Asset stats
                yf_1 = g_fixed[0] + self.c * g_fixed[1]
                yf_2 = self.c * g_fixed[0] + g_fixed[1]
                mu_f = self.p * yf_1 + (1 - self.p) * yf_2

                # Candidate Grid stats
                yc_1 = self.X + self.c * self.Y
                yc_2 = self.c * self.X + self.Y
                mu_c = self.p * yc_1 + (1 - self.p) * yc_2  # Should match U_grid

                # Covariance = E[(F - mu_f)(C - mu_c)]
                # Sum weighted products for Bernoulli env (Env1 or Env2)
                Cov_grid = (self.p * (yf_1 - mu_f) * (yc_1 - mu_c)) + \
                        ((1 - self.p) * (yf_2 - mu_f) * (yc_2 - mu_c))

                fig = Figure(figsize=(15.55, 9.6), dpi=dpi) 
                axs = fig.subplots(4, 4)
                # Flatten axes list for unified 0-11 indexing regardless of grid shape
                # Order matches reading direction (Left->Right, Top->Bottom)
                ax_list = axs.flatten()

                # --- Helper: Overlay Plotter ---
                def add_overlays(ax_target, show_legend=False):
                    # Ellipse
                    ax_target.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], 'k--', label='Breeding Ellipse')
                    # Anchors
                    ax_target.scatter(g_fixed[0], g_fixed[1], c='gold', marker='*', s=300, edgecolors='k',
                                    label='Fixed Anchor', zorder=10)
                    ax_target.scatter(g_mutable[0], g_mutable[1], c='gray', marker='o', s=100, 
                                    label='Original Mutable', zorder=9)
                    # Strategies
                    for s_name, (p_mk, p_col) in plot_styles.items():
                        res = details_per_strategy[s_name]
                        genotypes = res['out_dict']['genotypes']
                        # The 'new' candidate is usually the last one in the list
                        for i, g_cand_plot in enumerate(genotypes):
                            # Filter to avoid overplotting original/fixed if included in list
                            is_diff_mut = (g_cand_plot[0] != g_mutable[0] or g_cand_plot[1] != g_mutable[1])
                            is_diff_fix = (g_cand_plot[0] != g_fixed[0] or g_cand_plot[1] != g_fixed[1])
                            
                            if i > 1 or (is_diff_mut and is_diff_fix):
                                ax_target.scatter(g_cand_plot[0], g_cand_plot[1], marker=p_mk, color=p_col, 
                                                s=120, label=s_name, zorder=11)

                    if show_legend:
                    # Smart legend positioning based on anchor location
                        if g_fixed[0]:
                            loc = 'left'
                        else:
                            loc = 'right'
                        if g_mutable[1]:
                            loc = 'lower ' + loc
                        else:
                            loc = 'upper ' + loc
                        ax_target.legend(loc=loc, fontsize='small', framealpha=0.8)
                def compute_analytical_overlay(grid_dict):
                    """
                    Compute analytical MVP, BB, BBS points and condition regions
                    on the (g_new_1, g_new_2) grid.
                    
                    Returns a dict with everything needed for plotting.
                    """
                    # --- Extract parameters ---
                    g_f = grid_dict['g_fixed']       # (g_f1, g_f2)
                    g_m = grid_dict['g_mutable']     # (g_m1, g_m2)
                    gamma = grid_dict['gamma']
                    p = grid_dict['p']
                    c = grid_dict['c']
                    R = grid_dict['R']
                    r_g = grid_dict['r_g']           # rho_g
                    X = grid_dict['X']               # grid of g_new_1
                    Y = grid_dict['Y']               # grid of g_new_2

                    # Derived constants
                    mu = (1 + c) / 2
                    S = p * (1 - p) * (1 - c)**2
                    A = gamma * S

                    # Sufficient statistics of fixed and mutable
                    s_f = g_f[0] + g_f[1]
                    d_f = g_f[0] - g_f[1]
                    s_m = g_m[0] + g_m[1]
                    d_m = g_m[0] - g_m[1]

                    # Ellipse semi-axes
                    r_s = R * np.sqrt(2 * (1 + r_g))
                    r_d = R * np.sqrt(2 * (1 - r_g))

                    # Gap parameters
                    alpha = s_m - s_f
                    delta = d_m - d_f

                    # Utility of incumbent
                    U_f = mu * s_f - (A / 2) * d_f**2

                    # --- Grid sufficient statistics ---
                    S_grid = X + Y           # s_new for each grid point
                    D_grid = X - Y           # d_new for each grid point

                    # --- V_fn for each grid point ---
                    # P and Q for portfolio (incumbent + new)
                    P_grid = mu * (S_grid - s_f) - A * d_f * (D_grid - d_f)
                    Q_grid = D_grid - d_f
                    Q2_grid = Q_grid**2
                    Q2_safe = np.where(Q2_grid > 1e-20, Q2_grid, 1e-20)

                    w_grid = P_grid / (A * Q2_safe)

                    # Standalone utility of each grid point
                    U_grid = mu * S_grid - (A / 2) * D_grid**2

                    # Full V_fn across regimes
                    V_interior = U_f + P_grid**2 / (2 * A * Q2_safe)
                    V_grid = np.where(w_grid <= 0, U_f,
                            np.where(w_grid >= 1, U_grid, V_interior))

                    # --- BB solution ---
                    # theta_BB = 0: Delta_s = r_s, Delta_d = 0
                    s_BB = s_m + r_s
                    d_BB = d_m
                    g_BB = np.array([(s_BB + d_BB) / 2, (s_BB - d_BB) / 2])
                    U_BB = mu * s_BB - (A / 2) * d_BB**2
                    P_BB = mu * (s_BB - s_f) - A * d_f * (d_BB - d_f)
                    Q_BB = d_BB - d_f
                    w_BB = P_BB / (A * Q_BB**2) if abs(Q_BB) > 1e-12 else np.inf
                    if w_BB <= 0:
                        V_BB = U_f
                    elif w_BB >= 1:
                        V_BB = U_BB
                    else:
                        V_BB = U_f + P_BB**2 / (2 * A * Q_BB**2)

                    # --- BBS solution (numerical on boundary) ---
                    # Evaluate U on ellipse boundary and pick the max
                    theta_fine = np.linspace(0, 2 * np.pi, 10000)
                    ds_fine = r_s * np.cos(theta_fine)
                    dd_fine = r_d * np.sin(theta_fine)
                    s_fine = s_m + ds_fine
                    d_fine = d_m + dd_fine
                    U_fine = mu * s_fine - (A / 2) * d_fine**2
                    idx_BBS = np.argmax(U_fine)
                    theta_BBS = theta_fine[idx_BBS]
                    s_BBS = s_fine[idx_BBS]
                    d_BBS = d_fine[idx_BBS]
                    g_BBS = np.array([(s_BBS + d_BBS) / 2, (s_BBS - d_BBS) / 2])
                    U_BBS = U_fine[idx_BBS]
                    # V_fn at BBS point
                    P_BBS = mu * (s_BBS - s_f) - A * d_f * (d_BBS - d_f)
                    Q_BBS = d_BBS - d_f
                    w_BBS_port = P_BBS / (A * Q_BBS**2) if abs(Q_BBS) > 1e-12 else np.inf
                    if w_BBS_port <= 0:
                        V_BBS = U_f
                    elif w_BBS_port >= 1:
                        V_BBS = U_BBS
                    else:
                        V_BBS = U_f + P_BBS**2 / (2 * A * Q_BBS**2)

                    # --- MVP solution (analytical) ---
                    L_tilde = np.sqrt(alpha**2 * r_d**2 + delta**2 * r_s**2)
                    feasible = L_tilde >= r_s * r_d

                    if feasible and abs(L_tilde) > 1e-12:
                        base_angle = np.arctan2(delta * r_s, alpha * r_d)
                        cos_arg = np.clip(r_s * r_d / L_tilde, -1, 1)
                        delta_angle = np.arccos(cos_arg)

                        theta1 = np.pi + base_angle + delta_angle
                        theta2 = np.pi + base_angle - delta_angle

                        def eval_V(theta):
                            sn = s_m + r_s * np.cos(theta)
                            dn = d_m + r_d * np.sin(theta)
                            U_new = mu * sn - (A / 2) * dn**2
                            P_ = mu * (sn - s_f) - A * d_f * (dn - d_f)
                            Q_ = dn - d_f
                            if abs(Q_) < 1e-12:
                                return U_new, 1.0
                            w_ = P_ / (A * Q_**2)
                            if w_ <= 0:
                                return U_f, w_
                            elif w_ >= 1:
                                return U_new, w_
                            else:
                                return U_f + P_**2 / (2 * A * Q_**2), w_

                        V1, w1 = eval_V(theta1)
                        V2, w2 = eval_V(theta2)

                        if V1 >= V2:
                            theta_MVP = theta1
                            V_MVP_val = V1
                            w_MVP_val = w1
                        else:
                            theta_MVP = theta2
                            V_MVP_val = V2
                            w_MVP_val = w2

                        s_MVP = s_m + r_s * np.cos(theta_MVP)
                        d_MVP = d_m + r_d * np.sin(theta_MVP)
                        g_MVP = np.array([(s_MVP + d_MVP) / 2, (s_MVP - d_MVP) / 2])
                    else:
                        # Infeasible interior: fall back to BBS
                        theta_MVP = theta_BBS
                        V_MVP_val = V_BBS
                        w_MVP_val = w_BBS_port
                        g_MVP = g_BBS

                    # --- Condition grids ---
                    # For each grid point: does V_fn(g_new) > V_BB? (feasible region only)
                    beats_BB = V_grid > V_BB + 1e-10
                    beats_BBS = V_grid > V_BBS + 1e-10

                    # Feasibility mask (on the ellipse)
                    feasible_mask = grid_dict.get('feasible_mask', np.ones_like(X, dtype=bool))

                    return {
                        # Grid data
                        'V_grid': V_grid, 'w_grid': w_grid, 'U_grid': U_grid,
                        'beats_BB': beats_BB, 'beats_BBS': beats_BBS,
                        'feasible_mask': feasible_mask,
                        # Reference values
                        'V_BB': V_BB, 'V_BBS': V_BBS, 'V_MVP': V_MVP_val,
                        'U_BB': U_BB, 'U_BBS': U_BBS,
                        'w_BB': w_BB, 'w_MVP': w_MVP_val,
                        # Optimal points in (g1, g2) space
                        'g_BB': g_BB, 'g_BBS': g_BBS, 'g_MVP': g_MVP,
                        'g_f': g_f, 'g_m': g_m,
                        # Angles
                        'theta_BB': 0.0, 'theta_BBS': theta_BBS, 'theta_MVP': theta_MVP,
                        # Ellipse params
                        'r_s': r_s, 'r_d': r_d, 's_m': s_m, 'd_m': d_m,
                    }


                def plot_mvp_conditions(ax_bb, ax_bbs, grid_dict, overlay_data=None):
                    """
                    Plot two subplots showing where MVP beats BB (left) and BBS (right).
                    
                    Parameters
                    ----------
                    ax_bb : matplotlib axis for the MVP > BB panel
                    ax_bbs : matplotlib axis for the MVP > BBS panel
                    grid_dict : your existing grid_dict
                    overlay_data : output of compute_analytical_overlay (computed if None)
                    """
                    if overlay_data is None:
                        overlay_data = compute_analytical_overlay(grid_dict)

                    X = grid_dict['X']
                    Y = grid_dict['Y']
                    od = overlay_data
                    
                    # --- Left panel: V_fn(g_new) vs V_BB ---
                    # Show V_grid - V_BB as heatmap (red = worse, blue = better)
                    # --- Left panel: V_fn(g_new) vs V_BB ---
                    diff_BB = od['V_grid'] - od['V_BB']
                    vmax_bb = np.max(diff_BB)
                    vmin_bb = np.min(diff_BB)

                    # Plot
                    cf_bb = ax_bb.contourf(X, Y, diff_BB, levels=np.linspace(vmin_bb, vmax_bb, 50), cmap='RdBu', alpha=0.8)
                    contour_bb = ax_bb.contour(X, Y, diff_BB, levels=[0], colors='black', linewidths=1.5, linestyles=':')
                    contour_w0 = ax_bb.contour(X, Y, od['w_grid'], levels=[0], colors='magenta', linewidths=2, linestyles=':')
                    contour_w1 = ax_bb.contour(X, Y, od['w_grid'], levels=[1], colors='orange', linewidths=2, linestyles=':')

                    # Collect legend handles and labels
                    handles = [
                        plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1.5),  # 'Same value as BB'
                        plt.Line2D([0], [0], color='magenta', linestyle=':', linewidth=2),  # '$w^*=0$'
                        plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=2),  # '$w^*=1$'
                    ]
                    labels = [
                        'MVP = BB',
                        '$w(\mathbf{g}_{\\mathrm{new}})=0$',
                        '$w(\mathbf{g}_{\\mathrm{new}})=1$',
                    ]
                    ax_bb.set_ylabel("Genotype dim 2")
                    ax_bb.set_xlabel("Genotype dim 1")

                    # Add overlays without legend
                    add_overlays(ax_bb, show_legend=False)

                    # Add combined legend
                    ax_bb.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)

                    cb_bb = plt.colorbar(cf_bb, ax=ax_bb)
                    cb_bb.set_label('$V_{fn}(\\mathbf{g}_{\\mathrm{new}}) - V_{fn}(\\mathbf{g}^{BB})$')
                    ax_bb.set_title('MVP vs BB')
                    

                    # --- Right panel: V_fn(g_new) vs V_BBS ---
                    diff_BBS = od['V_grid'] - od['V_BBS']
                    vmax_bbs = np.max(diff_BBS)
                    vmin_bbs = np.min(diff_BBS)

                    # Plot
                    cf_bbs = ax_bbs.contourf(X, Y, diff_BBS, levels=np.linspace(vmin_bbs, vmax_bbs, 50), cmap='RdBu', alpha=0.8)
                    contour_bbs = ax_bbs.contour(X, Y, diff_BBS, levels=[0], colors='black', linewidths=1.5, linestyles=':')
                    contour_w0 = ax_bbs.contour(X, Y, od['w_grid'], levels=[0], colors='magenta', linewidths=1, linestyles=':')
                    contour_w1 = ax_bbs.contour(X, Y, od['w_grid'], levels=[1], colors='orange', linewidths=1, linestyles=':')

                    # Collect legend handles and labels
                    handles = [
                        plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1.5),  # 'Same value as BBS'
                        plt.Line2D([0], [0], color='magenta', linestyle=':', linewidth=1),  # 'Only fixed'
                        plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=1),  # 'Only mutable'
                    ]
                    labels = [
                        'MVP = BBS',
                        '$w(\mathbf{g}_{\\mathrm{new}})=0$',
                        '$w(\mathbf{g}_{\\mathrm{new}})=1$',
                    ]

                    ax_bbs.set_xlabel("Genotype dim 1")

                    # Add overlays without legend
                    add_overlays(ax_bbs, show_legend=False)

                    # Add combined legend
                    ax_bbs.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)

                    cb_bbs = plt.colorbar(cf_bbs, ax=ax_bbs)
                    cb_bbs.set_label('$V_{fn}(\\mathbf{g}_{\\mathrm{new}}) - V_{fn}(\\mathbf{g}^{BBS})$')
                    ax_bbs.set_title('MVP vs BBS')

                    return overlay_data

                # ==========================================
                # ============ PLOTTING ROUTINE ============
                # ==========================================

                # --- Plot 0: Delta V Heatmap (Top Left) ---
                ax = ax_list[0]
                mv_grid_values = grid_dict['stats']['v_port']
                levels = np.linspace(np.min(mv_grid_values), np.max(mv_grid_values), 50)
                cf = ax.contourf(self.X, self.Y, mv_grid_values, levels=levels, cmap='RdBu', alpha=0.8)
                cb = fig.colorbar(cf, ax=ax)
                cb.set_label('ΔV (Gain over Baseline)')
                ax.contour(self.X, self.Y, mv_grid_values, levels=[0], colors='k', linewidths=1, linestyles='--')
                
                cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # 2 decimals for Delta V
                
                add_overlays(ax, show_legend=False)
                ax.set_title("Portfolio Value Added")
                ax.set_ylabel("Genotype dim 2")
                ax.grid(True, alpha=0.3)

                # --- Plot 1: Dual Axis Gain & Adaptation (Top Right in minimal, or 0,1 in extensive) ---
                ax1 = ax_list[1]
                bar_colors = [plot_styles[n][1] for n in strat_names]

                # Primary Axis: Economic Gain
                ax1.bar(strat_names, gains, edgecolor='k', alpha=0.6, color=bar_colors, label='Economic Gain')
                ax1.set_ylabel("Gain % (Normalized)")
                ax1.tick_params(axis='y')
                ax1.axhline(0, color='k', linewidth=0.8)
                ax1.set_ylim(0, 1.05 * 100)

                # Secondary Axis: Adaptation Rate
                ax2 = ax1.twinx()
                ax2.plot(strat_names, adaptions2, color='darkblue', marker='D', linestyle='None', 
                        markersize=8, label='Adaptation Rate')
                ax2.plot(strat_names, adaptions1, color='darkred', marker='D', linestyle='None', 
                        markersize=8, label='Adaptation Rate')
                ax2.set_ylabel("Adaptation Rate (Share)", color='darkred')
                ax2.tick_params(axis='y')
                ax2.set_ylim(0, 1.05)

                ax1.set_title("Strategy Performance: Gain vs. Adoption")
                ax1.set_xticks(range(len(strat_names)), labels=strat_names, rotation=45, ha='right')

                # --- EXTENSIVE PLOTS (Indices 2 through 11) ---
                if plot_extensive:
                    
                    # --- Plot 2: Portfolio Mean ---
                    ax = ax_list[2]
                    cf = ax.contourf(self.X, self.Y, grid_dict['stats']['mean_port'], 50, cmap='viridis', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Mean Yield')
                    add_overlays(ax)
                    ax.set_title("Portfolio Mean")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 3: Portfolio Variance ---
                    ax = ax_list[3]
                    # Reversed magma: Dark=Low Var
                    cf = ax.contourf(self.X, self.Y, grid_dict['stats']['var_port'], 50, cmap='magma_r', alpha=0.8) 
                    fig.colorbar(cf, ax=ax, label='Variance')
                    add_overlays(ax)
                    ax.set_title("Portfolio Variance")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 4: Single Variety Mean ---
                    ax = ax_list[4]
                    cf = ax.contourf(self.X, self.Y, U_grid, 50, cmap='viridis', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Mean Yield')
                    add_overlays(ax)
                    ax.set_title("Single Variety Mean")
                    ax.set_ylabel("Genotype dim 2")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 5: Single Variety Variance ---
                    ax = ax_list[5]
                    cf = ax.contourf(self.X, self.Y, Var_grid, 50, cmap='magma_r', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Variance')
                    add_overlays(ax)
                    ax.set_title("Single Variety Variance")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 6: Yield EV1 ---
                    ax = ax_list[6]
                    z_ev1 = self.p * (self.beta1[0] * self.X + self.beta1[1] * self.Y)
                    cf = ax.contourf(self.X, self.Y, z_ev1, 50, cmap='viridis', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Yield EV1')
                    add_overlays(ax)
                    ax.set_title("Single Variety Yield EV1")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 7: Yield EV2 ---
                    ax = ax_list[7]
                    z_ev2 = (1 - self.p) * (self.beta2[0] * self.X + self.beta2[1] * self.Y)
                    cf = ax.contourf(self.X, self.Y, z_ev2, 50, cmap='viridis', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Yield EV2')
                    add_overlays(ax)
                    ax.set_title("Single Variety Yield EV2")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 8: Adoption Share ---
                    ax = ax_list[8]
                    cf = ax.contourf(self.X, self.Y, grid_dict['stats']['w_c'], 50, cmap='RdYlGn', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Adoption Rate')
                    add_overlays(ax)
                    ax.set_title("Possible adoption shares")
                    ax.set_ylabel("Genotype dim 2")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 9: Weight Mutable ---
                    ax = ax_list[9]
                    cf = ax.contourf(self.X, self.Y, w_new_grid, 50, cmap='RdYlGn', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Weight mutable')
                    add_overlays(ax)
                    ax.set_title("Possible adoption shares g_mut")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 10: Covariance ---
                    ax = ax_list[10]
                    cov_min, cov_max = np.min(Cov_grid), np.max(Cov_grid)
                    
                    # Handle normalization safely
                    if cov_min < 0 < cov_max:
                        divnorm = TwoSlopeNorm(vmin=cov_min, vcenter=0., vmax=cov_max)
                    else:
                        divnorm = Normalize(vmin=cov_min, vmax=cov_max)

                    cf = ax.contourf(self.X, self.Y, Cov_grid, 50, cmap='RdBu_r', norm=divnorm, alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Covariance')
                    ax.contour(self.X, self.Y, Cov_grid, levels=[0], colors='k', linewidths=1, linestyles='--')
                    
                    add_overlays(ax)
                    ax.set_title("Covariance (Cand vs Fixed)")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 11: Correlation ---
                    ax = ax_list[11]
                    ax.axis('off')  # Hide axis, ticks, border

                    overlay_data = compute_analytical_overlay(grid_dict)
                    # As standalone figure:
                    plot_mvp_conditions(ax_list[12], ax_list[13], grid_dict, overlay_data)

                    for idx in [14, 15]:
                        ax_list[idx].axis('off')  # Hide axes, ticks, and rectangle
                    
                    # Build Custom Legend Handles
                    legend_elements = [
                        Line2D([0], [0], color='k', linestyle='--', label='Breeding Ellipse'),
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=18, 
                               markeredgecolor='k', label='Fixed Anchor'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
                               label='Original Mutable'),
                    ]
                    
                    # Add Strategy Markers
                    for s_name, (p_mk, p_col) in plot_styles.items():
                        legend_elements.append(
                            Line2D([0], [0], marker=p_mk, color='w', markerfacecolor=p_col, 
                                   markersize=12, label=s_name)
                        )

                    # Draw the legend in the center of the subplot
                    ax.legend(handles=legend_elements, loc='center', fontsize=9, frameon=True, borderpad=0.7)
                    ax.set_title("Legend", fontweight='bold')

                    

                    # Use subplots_adjust instead of tight_layout for manual control over spacing
                    # hspace controls vertical space (height), wspace controls horizontal (width)
                    fig.subplots_adjust(
                        top=0.92,
                        bottom=0.05,
                        left=0.05,
                        right=0.95,
                        hspace=0.4,
                        wspace=0.3
                    )

                    # Indices of plots that are spatial heatmaps (skipping the bar chart at index 1 
                    # and the legend at index 11)
                    spatial_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]

                    for idx in spatial_indices:
                        if idx < len(ax_list):
                            ax_list[idx].set_xlim([0, 1])
                            ax_list[idx].set_ylim([0, 1])

                    result_text = output_buffer.getvalue()
            
            # If multiple scenarios run, 'fig' will be the last one generated, 
            # which is correct for the GUI (it runs one at a time).
            return result_text, fig