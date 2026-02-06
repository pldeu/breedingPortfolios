import gurobipy as gp
import io
from contextlib import redirect_stdout
from matplotlib.figure import Figure 
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
from gurobipy import GRB
from matplotlib.colors import TwoSlopeNorm, Normalize
from GurobiPortfolioOptimizer import GurobiPortfolioOptimizer

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

    def get_metrics_for_genotype(self, grid, g_cand):
        """
        Helper to get (Variance, Mean) of the OPTIMAL portfolio 
        constructed using the given candidate genotype g_cand.
        """
        # We need to run the portfolio weight optimization (inner loop)
        # for this specific candidate to get the Portfolio Variance/Mean.
        # We can reuse _get_standard_output or similar.

        if self.replace:
            # 2-Asset Mode
            res = self._get_standard_output([grid['g_fixed'], g_cand])
        else:
            # 3-Asset Mode
            res = self._get_standard_output([grid['g_fixed'], grid['g_mutable'], g_cand])

        # Extract Portfolio Mean and Variance
        # Your dict keys are: 'mean_yield', 'var_portfolio'
        # Note: 'stats_portfolio' might be nested depending on which helper you used.
        # Based on your file, _get_standard_output returns nested stats, 
        # but _get_standard_output_3_assets might not?
        # Let's handle the specific dict structure from your file:

        if 'stats_portfolio' in res:
            return res['stats_portfolio']['var_portfolio'], res['stats_portfolio']['mean_yield']
        else:
            # Fallback for 3-asset result structure if it differs
            # You might need to adjust this key access based on your exact return dict
            # Assuming you might have added keys to _get_standard_output_3_assets
            # If not, you might have to calculate them manually here.

            # Quick calculation if keys missing:
            w = res['weights']['env_1']  # Assuming constant weights
            # ... (Manual calculation if needed, but standard dicts usually have it)
            # For now, let's assume you ensure the dict has 'mean_yield' and 'var_portfolio'
            return 0.0, 0.0  # Replace with actual key access
        

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
                            'Ginv': self.Ginv, 'R': self.R, 'replace': self.replace}
            grid_dict['stats'] = self.get_exact_grid_stats(grid_dict)

            return grid_dict

    def solve_nise_frontier(self, grid=None, tol=1e-4):
        """
        NISE Algorithm to find vertices of the efficient frontier.
        """
        frontier = []
        if grid == None:
            grid = self.grid_dict
            if grid == None:
                grid = self._compute_grid()

        # 1. Helper to solve for a specific Gamma and return (Var, Mean, Gamma, x_cand)
        def solve_point(gamma_v):
            x_cand = self.find_optimal_action_constrained(grid, gamma_v)

            # Now calculate the specific Portfolio Variance/Mean for this candidate
            # We MUST use the same gamma to find the optimal weights
            orig_g = self.gamma
            self.gamma = gamma_v

            # Use the single-point calculator
            if self.replace:
                details = self._get_standard_output([grid['g_fixed'], x_cand])
                res = self.calculate_stats_from_dict(details)
                var_p, mean_p = res['variance'], res['mean']
            else:
                details = self._get_standard_output([grid['g_fixed'], grid['g_mutable'], x_cand])
                # Note: You need to ensure _get_standard_output_3_assets returns these stats!
                # If it currently only returns weights, you must update it to return mean/var.
                # Assuming it returns a 'hindsight_score' or similar, we might need to recalc:
                # (You can just call calculate_stats_from_dict(details) if compatible)
                res = self.calculate_stats_from_dict(details)
                var_p, mean_p = res['variance'], res['mean']

            self.gamma = orig_g
            return (var_p, mean_p, gamma_v, x_cand)

        # 2. Anchors
        p_min_var = solve_point(1e6)  # Large Gamma
        p_max_ret = solve_point(0.0)  # Zero Gamma

        frontier.append(p_min_var)
        frontier.append(p_max_ret)

        # 3. Recursive NISE
        def refine(p1, p2):
            v1, m1, _, _ = p1
            v2, m2, _, _ = p2

            # Slope
            if abs(v2 - v1) < 1e-9: return
            slope = (m2 - m1) / (v2 - v1)
            gamma_new = 2 * slope

            if gamma_new < 0: return  # Concavity error or numerical noise

            p_new = solve_point(gamma_new)
            v_new, m_new = p_new[0], p_new[1]

            # Check improvement (Vertical distance to segment)
            expected_mean = m1 + slope * (v_new - v1)
            if (m_new - expected_mean) > tol:
                frontier.append(p_new)
                # Sort by variance to ensure correct left/right recursion
                # (Optional, but helps logical flow if implementing iteratively)
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
        weights = [w[i].X for i in range(n)] if m.Status == GRB.OPTIMAL else [1.0] + [0.0] * (n - 1)


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

    def run(self, strategies=None, plot_extensive=False, easy_base=False, no_plot=False, replace=False, pareto=False, plot_right=False):
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

            def strat_max_investability(grid):
                """Maximize Candidate Share (Adaptation)."""
                return None, grid['optimizer'].strat_max_investability()

            def strat_BB(grid):
                """BeatBest: Standalone Max."""
                return None, grid['optimizer'].strat_BB()

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
                    f"{'Single Component':<25} | {'MV':<8} | {'Gain':<8} |{'Mean':<8}  | {'Var':<8} | {'g1':<8} | {'g2':<8}  ")

                def print_pure_row(row_name, g_vec, ref_mv):
                    g = np.array(g_vec)  # Ensure numpy array
                    mu, var, mv, _, _ = self._get_yields_and_stats(g[0], g[1])
                    gain = mv - ref_mv
                    # Print row matching the table format
                    print(
                        f"{row_name:<25} | {mv:<8.4f} | {gain:<8.4f} | {mu:<8.4f} | {var:<8.4f} | {g[0]:<8.4f} | {g[1]:<8.4f} ")

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

                fig = Figure(figsize=(15.55, 9.6), dpi=100) 
                axs = fig.subplots(3, 4)
                                
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
                        loc = 'lower left' if (g_fixed[0] > 0 or g_fixed[1] > 0) else 'lower right'
                        ax_target.legend(loc=loc, fontsize='small', framealpha=0.8)

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
                ax.set_title(f"{label}\nPortfolio Value Added")
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
                    cf = ax.contourf(self.X, self.Y, grid_dict['stats']['mean_port'], 50, cmap='magma_r', alpha=0.8) 
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
                    ax.set_title("Possible adoption shares g_new")
                    ax.set_xlabel("Genotype dim 1")
                    ax.set_ylabel("Genotype dim 2")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 9: Weight Mutable ---
                    ax = ax_list[9]
                    cf = ax.contourf(self.X, self.Y, w_new_grid, 50, cmap='RdYlGn', alpha=0.8)
                    fig.colorbar(cf, ax=ax, label='Weight mutable')
                    add_overlays(ax)
                    ax.set_title("Possible adoption shares g_mut")
                    ax.set_xlabel("Genotype dim 1")
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
                    ax.set_title("Covariance (Cand vs Fixed)\nBlue = Good Hedge")
                    ax.set_xlabel("Genotype dim 1")
                    ax.grid(True, alpha=0.3)

                    # --- Plot 11: Correlation ---
                    ax = ax_list[11]
                    ax.axis('off')  # Hide axis, ticks, border
                    
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
                    ax.legend(handles=legend_elements, loc='center', fontsize=12, frameon=True, borderpad=0.7)
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
                spatial_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                for idx in spatial_indices:
                    if idx < len(ax_list):
                        ax_list[idx].set_xlim([-1, 1])
                        ax_list[idx].set_ylim([-1, 1])

                # ------------------------------------------


                result_text = output_buffer.getvalue()
        
        # If multiple scenarios run, 'fig' will be the last one generated, 
        # which is correct for the GUI (it runs one at a time).
        return result_text, fig
