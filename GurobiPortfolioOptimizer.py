import gurobipy as gp
from gurobipy import GRB
import numpy as np


class GurobiPortfolioOptimizer:
    def __init__(self, grid_params):
        """
        grid_params: dict containing 'p', 'c', 'gamma', 'g_fixed', 'g_mutable',
                     'R', 'Ginv', 'replace'
        """
        self.p = grid_params['p']
        self.c = grid_params['c']
        self.gamma = grid_params['gamma']
        self.g_fixed = np.array(grid_params['g_fixed'])
        self.g_mutable = np.array(grid_params['g_mutable'])
        self.R = grid_params['R']
        self.Ginv = grid_params['Ginv']
        self.replace = grid_params['replace']
        self.sigma2 = float(grid_params.get('sigma', 0.0)) ** 2
        self.candidate_breeding_var = float(grid_params.get('candidate_breeding_var', 0.0))
        self.known_eta = np.asarray(grid_params.get('known_eta', np.zeros(2)), dtype=float)
        self.breeding_timing = grid_params.get('breeding_timing', 'unknown')
        self.stochastic_active = bool(
            self.sigma2 > 0 or self.candidate_breeding_var > 0
            or grid_params.get('breeding_sd', 0.0) > 0
        )

        # Helper: Fixed/Mutable Yield Vectors (Env 1, Env 2)
        # Yield formula: y1 = x + c*y; y2 = c*x + y
        self.yf = self._get_yield_vec(self.g_fixed[0], self.g_fixed[1])
        self.ym = self._get_yield_vec(self.g_mutable[0], self.g_mutable[1])

        self.POB_model = None

    def _get_yield_vec(self, x, y):
        """Returns array [y_env1, y_env2]"""
        return np.array([x + self.c * y, self.c * x + y])

    def _add_mahalanobis_constraint(self, model, x, y):
        """Adds (g - g_mut)^T Ginv (g - g_mut) <= R^2"""
        # Center points (u = g - g_mutable)
        # Note: Mahalanobis is usually around the original population mean or mutable point
        # Assuming g_mutable is the center for the ellipse as per your snippet
        cx = self.g_mutable[0] + self.known_eta[0]
        cy = self.g_mutable[1] + self.known_eta[1]

        u1 = x - cx
        u2 = y - cy

        # d^2 = G00*u1^2 + 2*G01*u1*u2 + G11*u2^2
        model.addQConstr(
            self.Ginv[0, 0] * u1 * u1 +
            2 * self.Ginv[0, 1] * u1 * u2 +
            self.Ginv[1, 1] * u2 * u2 <= self.R ** 2,
            name="Mahalanobis"
        )

    def strat_BB(self):
        """
        BeatBest: Maximize Standalone Utility of Candidate.
        """
        result = self.strat_B4P(w_f_given=0.0, w_m_given=0.0)
        return result

    def strat_B4P(self, w_f_given=None, w_m_given=None):
        """
        Maximize Portfolio Utility (Joint Optimization).
        """
        m = gp.Model("B4P")
        m.setParam("OutputFlag", 0)
        m.setParam("NonConvex", 2)
        m.setParam("MIPGap", 1e-4)
        m.setParam("TimeLimit", 60)

        # --- 1. Explicit Bounds Calculation (CRITICAL FIX) ---
        (xb_min, xb_max), (yb_min, yb_max) = self._get_ellipse_bounds()

        # --- 2. Add Variables with Bounds ---
        # Genotype variables (Bounded by Ellipse)
        gx = m.addVar(lb=xb_min, ub=xb_max, name="gx")
        gy = m.addVar(lb=yb_min, ub=yb_max, name="gy")

        # Weight variables
        wf = m.addVar(lb=0, ub=1, name="wf")
        wc = m.addVar(lb=0, ub=1, name="wc")
        wm = m.addVar(lb=0, ub=1, name="wm") if not self.replace else m.addVar(lb=0, ub=0, name="wm")

        if w_f_given != None:
            m.addConstr(wf == w_f_given)

        if w_m_given != None:
            m.addConstr(wm == w_m_given)

        # Auxiliary variables for Bilinear terms: z = w * g
        # Since w is in [0, 1], z must be between [min(0, g_min), max(0, g_max)]
        zx_min, zx_max = min(0, xb_min), max(0, xb_max)
        zy_min, zy_max = min(0, yb_min), max(0, yb_max)

        zx = m.addVar(lb=zx_min, ub=zx_max, name="zx")
        zy = m.addVar(lb=zy_min, ub=zy_max, name="zy")

        # --- 3. Constraints ---
        m.addConstr(wf + wc + wm == 1)

        self._add_mahalanobis_constraint(m, gx, gy)

        # Bilinear definitions
        m.addQConstr(zx == wc * gx)
        m.addQConstr(zy == wc * gy)

        # --- 4. Objective ---
        term_wc_yc1 = zx + self.c * zy
        term_wc_yc2 = self.c * zx + zy
        term_wf_yf1 = wf * self.yf[0]
        term_wf_yf2 = wf * self.yf[1]

        term_wm_ym1 = wm * self.ym[0]
        term_wm_ym2 = wm * self.ym[1]

        Yp1 = term_wf_yf1 + term_wm_ym1 + term_wc_yc1
        Yp2 = term_wf_yf2 + term_wm_ym2 + term_wc_yc2

        mu_p = self.p * Yp1 + (1 - self.p) * Yp2
        E_Yp2 = self.p * (Yp1 * Yp1) + (1 - self.p) * (Yp2 * Yp2)

        noise_var = self.sigma2 * (wf * wf + wc * wc + wm * wm) + self.candidate_breeding_var * wc * wc
        obj = mu_p - 0.5 * self.gamma * (E_Yp2 - mu_p * mu_p + noise_var)
        m.setObjective(obj, GRB.MAXIMIZE)

        # --- 5. Optimize ---
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            g_opt = np.array([gx.X, gy.X])
            w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
            self.POB_model = m
            return self._build_result(g_opt, w_opt)
        elif m.Status == GRB.TIME_LIMIT:
            print('Struggling with start_B4P or B4M')
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)
            m.setParam("TimeLimit", 180)
            m.optimize()
        return None
    
    def change_gamma_B4P(self, gamma):
        """
        Updates the gamma value of the existing Gurobi model and re-optimizes.
        This leverages Gurobi's native warm-starting to speed up NISE calculations.
        """

        # 1. Fallback: If strat_B4P hasn't been run yet, run it from scratch
        if getattr(self, 'POB_model', None) is None:
            self.gamma = gamma
            return self.strat_B4P()

        m = self.POB_model

        # 2. Retrieve existing variables by name
        gx = m.getVarByName("gx")
        gy = m.getVarByName("gy")
        wf = m.getVarByName("wf")
        wc = m.getVarByName("wc")
        wm = m.getVarByName("wm")
        zx = m.getVarByName("zx")
        zy = m.getVarByName("zy")

        # 3. Rebuild the Objective with the NEW gamma
        term_wc_yc1 = zx + self.c * zy
        term_wc_yc2 = self.c * zx + zy
        term_wf_yf1 = wf * self.yf[0]
        term_wf_yf2 = wf * self.yf[1]

        term_wm_ym1 = wm * self.ym[0]
        term_wm_ym2 = wm * self.ym[1]

        Yp1 = term_wf_yf1 + term_wm_ym1 + term_wc_yc1
        Yp2 = term_wf_yf2 + term_wm_ym2 + term_wc_yc2

        mu_p = self.p * Yp1 + (1 - self.p) * Yp2
        E_Yp2 = self.p * (Yp1 * Yp1) + (1 - self.p) * (Yp2 * Yp2)

        # Apply the new objective
        noise_var = self.sigma2 * (wf * wf + wc * wc + wm * wm) + self.candidate_breeding_var * wc * wc
        obj = mu_p - 0.5 * gamma * (E_Yp2 - mu_p * mu_p + noise_var)
        m.setObjective(obj, GRB.MAXIMIZE)

        # 4. Re-optimize
        m.optimize()

        # 5. Extract and return results
        if m.Status == GRB.OPTIMAL:
            g_opt = np.array([gx.X, gy.X])
            w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
            return self._build_result(g_opt, w_opt)
            
        elif m.Status == GRB.TIME_LIMIT:
            print(f'Struggling with change_gamma_B4P for gamma={gamma}. Increasing time limit.')
            m.setParam("TimeLimit", 180)
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                g_opt = np.array([gx.X, gy.X])
                w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
                return self._build_result(g_opt, w_opt)
                
        # Return None if it fails to find an optimal solution even after time extension
        return None

    def strat_B4M(self):
        """
        Maximize Portfolio Mean Yield (Risk Neutral).
        """
        original_gamma = self.gamma
        self.gamma = 0# 1e-8  # Effectively maximize Mean

        # result is already in the correct dict format
        result = self.strat_B4P(w_f_given=0.0, w_m_given=0.0)

        self.gamma = original_gamma  # Restore
        return result
    
    def strat_max_investability_old(self, warm_start_data=None):
        """
        Optimized MPEC solver with Variable Rescaling and Tight Dual Bounds.
        """
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np

        m = gp.Model("Max_Investability")
        
        # --- 1. Solver Parameters for Hard MPEC Instances ---
        m.setParam("OutputFlag", 0)
        m.setParam("NonConvex", 2)
        m.setParam("MIPFocus", 3)        # Focus on BestBound (Crucial for proving optimality)
        m.setParam("PreMIQCPForm", 2)    # Stronger LP relaxation for Quadratic constraints
        m.setParam("NumericFocus", 1)    # Slight numerical safety without sacrificing too much speed
        m.setParam("TimeLimit", 60)

        if warm_start_data != None:
            m.setParam("OutputFlag", 0)
            print(' strat_max_investability_old with warm start')

        # --- 2. Constants & Scaling ---
        # Bounds for Yields/Returns (Keep these tight!)
        Y_BOUND = 3.0 
        
        # Calculate scaling factor.
        # If gamma is 1,000,000, we want to solve for "lambda_scaled" which is approx 1.0
        # Equation: lambda_scaled = (1/gamma) * real_lambda
        scale_factor = 1.0 / self.gamma if self.gamma > 1e-6 else 1.0
        
        # --- 3. Variables ---

        # A. Genotype (Upper Level)
        (xb_min, xb_max), (yb_min, yb_max) = self._get_ellipse_bounds()
        gx = m.addVar(lb=xb_min, ub=xb_max, name="gx")
        gy = m.addVar(lb=yb_min, ub=yb_max, name="gy")

        # B. Weights (Lower Level)
        wf = m.addVar(lb=0.0, ub=1.0, name="wf")
        wc = m.addVar(lb=0.0, ub=1.0, name="wc")
        wm_ub = 0.0 if self.replace else 1.0
        wm = m.addVar(lb=0.0, ub=wm_ub, name="wm")

        # C. SCALED Dual Variables (The Key Fix)
        # Instead of 'lambda' being free and huge, 'lam_s' is small and bounded.
        # Theoretical max for lam_s is approx Max(Covariance) + Max(Mean/gamma) ~ 4 + 0.
        # We use [-50, 50] to be strictly safe but mathematically much tighter than infinity.
        lam_s = m.addVar(lb=-50.0, ub=50.0, name="lam_scaled") 
        eta_f_s = m.addVar(lb=0.0, ub=50.0, name="eta_f_scaled")
        eta_c_s = m.addVar(lb=0.0, ub=50.0, name="eta_c_scaled")
        eta_m_s = m.addVar(lb=0.0, ub=50.0, name="eta_m_scaled")

        # D. Auxiliary Variables
        yc1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="yc1")
        yc2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="yc2")
        
        # zc variables (z = w * y). Bounds = 1 * Y_BOUND
        zc1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="zc1")
        zc2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="zc2")

        R1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="R1")
        R2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="R2")

        mu_p = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="mu_p")
        mu_c = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="mu_c")
        
        # Cross term E[R * yc]. Max magnitude approx Y_BOUND^2 = 9
        E_R_yc = m.addVar(lb=-10.0, ub=10.0, name="E_R_yc")

        # --- 4. Constraints ---

        # A. Ellipse & Definitions
        self._add_mahalanobis_constraint(m, gx, gy)
        m.addConstr(yc1 == gx + self.c * gy)
        m.addConstr(yc2 == self.c * gx + gy)

        # B. Quadratic Linking (z = w * y)
        m.addQConstr(zc1 == wc * yc1)
        m.addQConstr(zc2 == wc * yc2)

        # C. Returns & Budget
        m.addConstr(R1 == wf * self.yf[0] + wm * self.ym[0] + zc1)
        m.addConstr(R2 == wf * self.yf[1] + wm * self.ym[1] + zc2)
        m.addConstr(wf + wc + wm == 1, name="Budget")

        # D. SCALED Stationarity (KKT)
        # Original:  mu - gamma * Cov = lambda - eta
        # Scaled:    (1/gamma)*mu - Cov = (1/gamma)*lambda - (1/gamma)*eta
        # Using Vars: scale_factor*mu - Cov = lam_s - eta_s
        
        m.addConstr(mu_p == self.p * R1 + (1 - self.p) * R2)
        m.addConstr(mu_c == self.p * yc1 + (1 - self.p) * yc2)
        m.addQConstr(E_R_yc == self.p * (R1 * yc1) + (1 - self.p) * (R2 * yc2))

        # -- Fixed Variety F --
        mu_f = self.p * self.yf[0] + (1 - self.p) * self.yf[1]
        E_R_yf = self.p * R1 * self.yf[0] + (1 - self.p) * R2 * self.yf[1]
        # Linear constraint (since yf is const)
        m.addConstr(
            scale_factor * mu_f - (E_R_yf - mu_p * mu_f) == lam_s - eta_f_s,
            name="KKT_Stat_F"
        )

        # -- Fixed Variety M --
        if not self.replace:
            mu_m = self.p * self.ym[0] + (1 - self.p) * self.ym[1]
            E_R_ym = self.p * R1 * self.ym[0] + (1 - self.p) * R2 * self.ym[1]
            m.addConstr(
                scale_factor * mu_m - (E_R_ym - mu_p * mu_m) == lam_s - eta_m_s,
                name="KKT_Stat_M"
            )

        # -- Candidate Variety C --
        # Quadratic constraint (mu_p * mu_c term)
        m.addQConstr(
            scale_factor * mu_c - (E_R_yc - mu_p * mu_c) == lam_s - eta_c_s,
            name="KKT_Stat_C"
        )

        # E. Complementary Slackness (SOS1)
        # We must use the SCALED eta variables here
        m.addSOS(GRB.SOS_TYPE1, [wf, eta_f_s])
        m.addSOS(GRB.SOS_TYPE1, [wc, eta_c_s])
        if not self.replace:
            m.addSOS(GRB.SOS_TYPE1, [wm, eta_m_s])

        # --- 5. Warm Start & Optimization ---
        if warm_start_data:
            try:
                # Genotypes
                g_start = warm_start_data['genotypes'][-1]
                gx.Start = g_start[0]
                gy.Start = g_start[1]
                # Weights
                w_list = warm_start_data['weights']['env_1']
                if self.replace:
                    wf.Start = w_list[0]
                    wc.Start = w_list[1] if len(w_list) < 3 else w_list[2]
                else:
                    wf.Start = w_list[0]
                    wm.Start = w_list[1]
                    wc.Start = w_list[2]
            except:
                pass

        m.setObjective(wc, GRB.MAXIMIZE)
        m.optimize()

        # --- HANDLING RESULTS ---
        if m.Status == GRB.OPTIMAL:
            g_opt = np.array([gx.X, gy.X])
            w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
            return self._build_result(g_opt, w_opt)
        elif m.Status == GRB.TIME_LIMIT and not self.stochastic_active:
            m.write('hard_invest.lp')
            print("couldn't solve strat_max_investability_old, try strat_max_investability_old with warmstart")
            return self.strat_max_investability_old(self.strat_max_share(steps=100))

        if self.stochastic_active and m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                "MaxMarket could not certify an optimal stochastic solution; "
                "the legacy fallback is disabled for stochastic settings "
                f"(Gurobi status {m.Status}, solution_count {m.SolCount}, "
                f"gap {m.MIPGap if m.SolCount else 'unavailable'})."
            )
        if m.Status == GRB.INFEASIBLE:
            print("\n!!! Model is INFEASIBLE. Calculating IIS... !!!")
            m.computeIIS()
            m.write("model_iis.ilp")
            print("Conflict written to 'model_iis.ilp'. Open this file to see which constraints conflict.")
            print("Likely causes: 1) 'gx' bounds conflict with Ellipse. 2) Stationarity impossible (dominance).")
            return self.strat_max_share(steps=100)
            
        print("couldn't solve strat_max_investability_old, try strat_max_share")

        return self.strat_max_share(steps=100)
    
    def strat_max_investability(self, warm_start_data=None, min_improvement=1e-4):
        """
        Maximizes investability (wc). 
        If no candidate can beat the parents (Infeasible), returns the optimal 
        parent-only portfolio with wc=0 (Status Quo).
        """

        # --- STEP 0: Solve Benchmark (Parents Only) & Prepare Fallback ---
        # We need the full details of the benchmark to use as a fallback.
        bench_util_low, bench_weights, _ = self._solve_parent_benchmark()

        # If we can't improve by at least epsilon, the solver will be infeasible.
        if bench_util_low > 0:
            target_utility = bench_util_low * (1 + min_improvement)
        else:
            target_utility = bench_util_low * (1 - min_improvement)

        best_portfolio = self.strat_B4P()
        if best_portfolio is None:
            raise RuntimeError("Could not certify the best-utility portfolio for MaxMarket.")
        best_utility = float(best_portfolio["mean_variance"])
        # If even the globally best-utility portfolio cannot reach the
        # improvement threshold, the only meaningful MaxMarket result is the
        # parent benchmark with zero candidate share. Returning the best-utility
        # portfolio here can incorrectly assign a positive candidate share.
        if best_utility < target_utility - 1e-8:
            candidate = self.g_mutable + self.known_eta
            status_quo = self._build_result(
                candidate,
                {
                    "wf": bench_weights["wf"],
                    "wm": bench_weights["wm"],
                    "wc": 0.0,
                },
            )
            status_quo["solver_certificate"] = {
                "status": "OPTIMAL_STATUS_QUO",
                "objective_value": 0.0,
                "objective_bound": 0.0,
                "relative_gap": 0.0,
                "requested_relative_gap": 1e-4,
                "baseline_utility": float(bench_util_low),
                "best_achievable_utility": best_utility,
                "required_utility": float(target_utility),
                "scope": "no candidate portfolio reaches the required utility threshold",
            }
            return status_quo

        m = gp.Model("Max_Investability_Robust")
        
        # ... (Solver Parameters: OutputFlag, NonConvex, etc.) ...
        m.setParam("OutputFlag", 0)
        m.setParam("NonConvex", 2)
        m.setParam("MIPFocus", 3)
        m.setParam("NumericFocus", 1)
        m.setParam("MIPGap", 1e-4)
        #m.setParam("TimeLimit", 120)

        # ... (Define Variables & Constraints exactly as before) ...
       # --- Constants & Scaling ---
        Y_BOUND = 3.0
        # Scaling factor for KKT (same as before)
        scale_factor = 1.0 / self.gamma if self.gamma > 1e-6 else 1.0

        # --- 1. Variables ---

        # A. Genotype
        (xb_min, xb_max), (yb_min, yb_max) = self._get_ellipse_bounds()
        gx = m.addVar(lb=xb_min, ub=xb_max, name="gx")
        gy = m.addVar(lb=yb_min, ub=yb_max, name="gy")

        # B. Weights
        wf = m.addVar(lb=0.0, ub=1.0, name="wf")
        wc = m.addVar(lb=0.0, ub=1.0, name="wc")
        wm_ub = 0.0 if self.replace else 1.0
        wm = m.addVar(lb=0.0, ub=wm_ub, name="wm")

        # C. Scaled Duals
        lam_s = m.addVar(lb=-50.0, ub=50.0, name="lam_scaled") 
        eta_f_s = m.addVar(lb=0.0, ub=50.0, name="eta_f_scaled")
        eta_c_s = m.addVar(lb=0.0, ub=50.0, name="eta_c_scaled")
        eta_m_s = m.addVar(lb=0.0, ub=50.0, name="eta_m_scaled")

        # D. Aux & Stats
        yc1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="yc1")
        yc2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="yc2")
        zc1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="zc1")
        zc2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="zc2")

        R1 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="R1")
        R2 = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="R2")
        
        # Statistics for Utility Calculation
        mu_p = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="mu_p")
        mu_c = m.addVar(lb=-Y_BOUND, ub=Y_BOUND, name="mu_c")
        E_R_yc = m.addVar(lb=-10.0, ub=10.0, name="E_R_yc")
        
        # NEW: Explicit Variance Variable for the Utility Constraint
        # Var = E[R^2] - (E[R])^2
        E_R2 = m.addVar(lb=0, ub=100.0, name="E_R2") 

        # --- 2. Constraints ---

        # ... (Standard Ellipse, Definitions, KKT - SAME AS BEFORE) ...
        self._add_mahalanobis_constraint(m, gx, gy)
        m.addConstr(yc1 == gx + self.c * gy)
        m.addConstr(yc2 == self.c * gx + gy)
        m.addQConstr(zc1 == wc * yc1)
        m.addQConstr(zc2 == wc * yc2)
        m.addConstr(R1 == wf * self.yf[0] + wm * self.ym[0] + zc1)
        m.addConstr(R2 == wf * self.yf[1] + wm * self.ym[1] + zc2)
        m.addConstr(wf + wc + wm == 1, name="Budget")

        # Stats defs
        m.addConstr(mu_p == self.p * R1 + (1 - self.p) * R2)
        m.addConstr(mu_c == self.p * yc1 + (1 - self.p) * yc2)
        m.addQConstr(E_R_yc == self.p * (R1 * yc1) + (1 - self.p) * (R2 * yc2))
        
        # NEW: Second Moment of Returns definition
        m.addQConstr(E_R2 == self.p * (R1 * R1) + (1 - self.p) * (R2 * R2)
                     + self.sigma2 * (wf * wf + wm * wm + wc * wc)
                     + self.candidate_breeding_var * wc * wc)

        # KKT Stationarity (Scaled)
        mu_f = self.p * self.yf[0] + (1 - self.p) * self.yf[1]
        E_R_yf = self.p * R1 * self.yf[0] + (1 - self.p) * R2 * self.yf[1]
        m.addConstr(scale_factor * mu_f - (E_R_yf - mu_p * mu_f) - self.sigma2 * wf
                    == lam_s - eta_f_s)

        if not self.replace:
            mu_m = self.p * self.ym[0] + (1 - self.p) * self.ym[1]
            E_R_ym = self.p * R1 * self.ym[0] + (1 - self.p) * R2 * self.ym[1]
            m.addConstr(scale_factor * mu_m - (E_R_ym - mu_p * mu_m) - self.sigma2 * wm
                        == lam_s - eta_m_s)

        m.addQConstr(scale_factor * mu_c - (E_R_yc - mu_p * mu_c)
                     - (self.sigma2 + self.candidate_breeding_var) * wc
                     == lam_s - eta_c_s)

        m.addSOS(GRB.SOS_TYPE1, [wf, eta_f_s])
        m.addSOS(GRB.SOS_TYPE1, [wc, eta_c_s])
        if not self.replace:
            m.addSOS(GRB.SOS_TYPE1, [wm, eta_m_s])
            
        # --- 3. THE FIX: Strict Utility Improvement ---
        # Utility = Mean - (Gamma/2) * Variance
        # Variance = E[R^2] - Mean^2
        # So: Mean - 0.5*Gamma*(E_R2 - Mean^2) >= Target
        
        # We rearrange to avoid non-convex -Mean^2 if possible, but Gurobi handles it.
        # Expression: mu_p - 0.5 * self.gamma * (E_R2 - mu_p * mu_p) >= target_utility
        
        # Note: Gurobi's Python interface allows direct quadratic expressions in addQConstr
        utility_constraint = m.addQConstr(
            mu_p - 0.5 * self.gamma * (E_R2 - mu_p * mu_p) >= target_utility,
            name="Strict_Improvement"
        )

        # --- Optimize ---
        if warm_start_data:
            try:
                # Genotypes
                g_start = warm_start_data['genotypes'][-1]
                gx.Start = g_start[0]
                gy.Start = g_start[1]
                # Weights
                w_list = warm_start_data['weights']['env_1']
                if self.replace:
                    wf.Start = w_list[0]
                    wc.Start = w_list[1] if len(w_list) < 3 else w_list[2]
                else:
                    wf.Start = w_list[0]
                    wm.Start = w_list[1]
                    wc.Start = w_list[2]
            except:
                pass


        # --- Optimize ---
        m.setObjective(wc, GRB.MAXIMIZE)
        m.optimize()

        # --- HANDLING THE RESULT ---
        
        # CASE A: Success! We found a candidate that beats the market.
        if m.Status == GRB.OPTIMAL:
            g_opt = np.array([gx.X, gy.X])
            w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
            result = self._build_result(g_opt, w_opt)
            result["solver_certificate"] = {
                "status": "OPTIMAL",
                "objective_value": float(wc.X),
                "objective_bound": float(m.ObjBound),
                "relative_gap": float(m.MIPGap),
                "requested_relative_gap": float(m.Params.MIPGap),
                "scope": "global optimum of the bounded KKT reformulation",
            }
            return result
        if self.stochastic_active:
            raise RuntimeError(
                "MaxMarket could not certify an optimal stochastic solution; "
                "the legacy fallback is disabled for stochastic settings "
                f"(Gurobi status {m.Status}, solution_count {m.SolCount}, "
                f"gap {m.MIPGap if m.SolCount else 'unavailable'})."
            )
        if m.Status == GRB.INFEASIBLE:
            m.remove(utility_constraint)
            print('removed utility constraint')
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                #print('Removed utility constraint')
                g_opt = np.array([gx.X, gy.X])
                w_opt = {'wf': wf.X, 'wc': wc.X, 'wm': wm.X if not self.replace else 0.0}
                return self._build_result(g_opt, w_opt)
        
        # CASE B: Infeasible (Cannot beat parents). Return Status Quo.
            # We return the benchmark solution where wc=0
            #print("No candidate can beat the parents. Returning Status Quo (wc=0).")
            
            # Construct the result dictionary manually using benchmark data
        print('help', m.Status, self.g_fixed, self.g_mutable)
        return self.strat_max_investability_old()

    def _solve_parent_benchmark(self):
        """
        Solves the standard QP for parents (F and M) only.
        Returns: (Utility_Value, Weights_Dict, Stats_Dict)
        """
        import gurobipy as gp
        from gurobipy import GRB
        
        m = gp.Model("Benchmark")
        m.setParam("OutputFlag", 0)
        
        # Weights for parents
        wf = m.addVar(lb=0, ub=1, name="wf")
        wm = m.addVar(lb=0, ub=1, name="wm")
        
        m.addConstr(wf + wm == 1)
        if self.replace:
            m.addConstr(wm == 0)
            
        # Calculate Portfolio Statistics directly
        # Means
        mu_f = self.p * self.yf[0] + (1 - self.p) * self.yf[1]
        mu_m = self.p * self.ym[0] + (1 - self.p) * self.ym[1]
        
        # Second Moments E[Y^2] for Variance calculation
        # E[Y^2] = p*y1^2 + (1-p)*y2^2
        E_f2 = self.p * self.yf[0]**2 + (1 - self.p) * self.yf[1]**2
        E_m2 = self.p * self.ym[0]**2 + (1 - self.p) * self.ym[1]**2
        E_fm = self.p * (self.yf[0]*self.ym[0]) + (1 - self.p) * (self.yf[1]*self.ym[1])
        
        # Portfolio Mean & Second Moment
        # mu_p = wf*mu_f + wm*mu_m
        # E[R^2] = wf^2*E_f2 + wm^2*E_m2 + 2*wf*wm*E_fm
        
        port_mean = wf * mu_f + wm * mu_m
        port_E_R2 = ((wf*wf * E_f2) + (wm*wm * E_m2) + (2 * wf * wm * E_fm)
                     + self.sigma2 * (wf * wf + wm * wm))
        
        # Var = E[R^2] - Mean^2
        # Obj = Mean - 0.5 * gamma * Var
        # Obj = Mean - 0.5 * gamma * (E[R^2] - Mean^2)
        # Obj = Mean - 0.5 * gamma * E[R^2] + 0.5 * gamma * Mean^2
        
        # Gurobi allows quadratic objectives. 
        # Note: port_mean is linear, port_E_R2 is quadratic. port_mean^2 is quadratic.
        
        obj = port_mean - 0.5 * self.gamma * (port_E_R2 - port_mean * port_mean)
        
        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            # Calculate final stats for return
            w_f_val = wf.X
            w_m_val = wm.X
            final_mean = w_f_val * mu_f + w_m_val * mu_m
            final_E2 = (w_f_val**2 * E_f2) + (w_m_val**2 * E_m2) + (2 * w_f_val * w_m_val * E_fm)
            final_var = final_E2 - final_mean**2 + self.sigma2 * (w_f_val**2 + w_m_val**2)
            
            return (
                m.ObjVal, 
                {'wf': w_f_val, 'wm': w_m_val}, 
                {'mean': final_mean, 'var': final_var}
            )
        else:
            # Should not happen for a simple QP, but safe fallback:
            # Return all-in on F
            mu_f_val = mu_f
            var_f_val = E_f2 - mu_f**2 + self.sigma2
            util = mu_f_val - 0.5 * self.gamma * var_f_val
            raise RuntimeError(
                "Parent benchmark failed to certify an optimum "
                f"(Gurobi status {m.Status}, solution_count {m.SolCount})."
            )

    def _get_benchmark_utility(self):
        """
        Solves the standard QP for the parents (F and M) only.
        Returns the optimal Mean-Variance Utility.
        """
        import gurobipy as gp
        from gurobipy import GRB
        
        m = gp.Model("Benchmark")
        m.setParam("OutputFlag", 0)
        
        wf = m.addVar(lb=0, ub=1)
        wm = m.addVar(lb=0, ub=1)
        
        m.addConstr(wf + wm == 1)
        if self.replace:
            m.addConstr(wm == 0) # Just F if replacing
            
        # Portfolio returns
        # R = wf*Yf + wm*Ym
        # Exp = p * (wf*yf0 + wm*ym0) + (1-p) * ...
        
        # Easier way: Mean and Var directly
        mu_f = self.p * self.yf[0] + (1 - self.p) * self.yf[1]
        mu_m = self.p * self.ym[0] + (1 - self.p) * self.ym[1]
        
        # Covariances
        # We can construct the 2x2 covariance matrix Q
        # var(F) = p*yf0^2 + (1-p)*yf1^2 - mu_f^2
        # cov(F,M) = p*yf0*ym0 + (1-p)*yf1*ym1 - mu_f*mu_m
        
        def get_cov(y1, m1, y2, m2):
            E_xy = self.p * y1[0]*y2[0] + (1-self.p)*y1[1]*y2[1]
            return E_xy - m1*m2
            
        var_f = get_cov(self.yf, mu_f, self.yf, mu_f) + self.sigma2
        var_m = get_cov(self.ym, mu_m, self.ym, mu_m) + self.sigma2
        cov_fm = get_cov(self.yf, mu_f, self.ym, mu_m)
        
        port_mean = wf * mu_f + wm * mu_m
        port_var = wf*wf*var_f + wm*wm*var_m + 2*wf*wm*cov_fm
        
        obj = port_mean - 0.5 * self.gamma * port_var
        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()
        
        return m.ObjVal

    def strat_max_share(self, tolerance=1e-5, steps=15):
        """
        Binary-search Max Share approximation; not a global optimum certificate.
        """
        if self.stochastic_active:
            raise RuntimeError(
                "strat_max_share is an approximate binary-search route and has not been "
                "updated for stochastic variance terms. Use the MaxMarket MPEC route instead."
            )
        # --- 1. Constants & Bounds ---
        (xb_min, xb_max), (yb_min, yb_max) = self._get_ellipse_bounds()
        yf, ym = self.yf, self.ym

        # Pre-calc Fixed/Mutable constants
        mu_f_const = np.dot([self.p, 1 - self.p], yf)
        var_f_const = np.dot([self.p, 1 - self.p], yf ** 2) - mu_f_const ** 2

        mu_m_const, var_m_const, cov_fm_const = 0.0, 0.0, 0.0
        if not self.replace:
            mu_m_const = np.dot([self.p, 1 - self.p], ym)
            var_m_const = np.dot([self.p, 1 - self.p], ym ** 2) - mu_m_const ** 2
            cov_fm_const = np.dot([self.p, 1 - self.p], yf * ym) - mu_f_const * mu_m_const

        # --- 2. Model Initialization ---
        models = {}
        regimes = ['fixed_only'] if self.replace else ['fixed_only', 'mutable_only', 'mixed']
        
        # SCALING FACTOR: Compresses huge coeffs (1e5) down to ~1.0 for stability
        # We divide the objective by this, then multiply the result back.
        OBJ_SCALE = 1e5 

        for regime in regimes:
            m = gp.Model(f"Persistent_{regime}")
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)
            m.setParam("TimeLimit", 60)

            # Variables
            gx = m.addVar(lb=xb_min, ub=xb_max, name="gx")
            gy = m.addVar(lb=yb_min, ub=yb_max, name="gy")
            self._add_mahalanobis_constraint(m, gx, gy)

            # Regime-specific setup
            wf_var, wm_var = None, None
            if regime == 'mixed':
                # FIX 1: Explicit Upper Bounds [0, 1] are MANDATORY for quadratic stability
                wf_var = m.addVar(lb=0.0, ub=1.0, name="wf")
                wm_var = m.addVar(lb=0.0, ub=1.0, name="wm")
                budget_constr = m.addConstr(wf_var + wm_var == 0, name="Budget")

            models[regime] = {
                'm': m, 'gx': gx, 'gy': gy,
                'wf': wf_var, 'wm': wm_var,
                'budget_constr': budget_constr if regime == 'mixed' else None
            }

        # --- 3. Binary Search Loop ---
        low = 0.0
        high = 1.0 - 1e-6

        best_g = np.array([self.g_fixed[0], self.g_fixed[1]])
        best_w_opt = {'wf': 1.0, 'wc': 0.0} if self.replace else {'wf': 1.0, 'wm': 0.0, 'wc': 0.0}
        
        last_good_gx = self.g_fixed[0]
        last_good_gy = self.g_fixed[1]

        for _ in range(steps):
            if low > 0.99999:
                break

            alpha = (low + high) / 2
            remaining = 1.0 - alpha

            max_gap = -float('inf')
            current_iter_best_g = None
            current_iter_best_w = None

            # --- Solve Regimes ---
            for regime in regimes:
                setup = models[regime]
                m, gx, gy = setup['m'], setup['gx'], setup['gy']
                wf_var, wm_var = setup['wf'], setup['wm']

                # A. Apply Warm Start
                gx.Start = last_good_gx
                gy.Start = last_good_gy

                # B. Update Weights & Constraints
                if regime == 'fixed_only':
                    wf_curr, wm_curr = remaining, 0.0
                elif regime == 'mutable_only':
                    wf_curr, wm_curr = 0.0, remaining
                else:  # mixed
                    setup['budget_constr'].RHS = remaining
                    # Important: Update bounds for tighter presolve if remaining < 1
                    wf_var.UB = remaining
                    wm_var.UB = remaining
                    wf_curr, wm_curr = wf_var, wm_var

                # C. Construct Objective (Gap)
                yc1 = gx + self.c * gy
                yc2 = self.c * gx + gy
                mu_c = self.p * yc1 + (1 - self.p) * yc2
                E_yc2 = self.p * (yc1 * yc1) + (1 - self.p) * (yc2 * yc2)
                var_c = E_yc2 - mu_c * mu_c

                E_yc_yf = self.p * (yc1 * yf[0]) + (1 - self.p) * (yc2 * yf[1])
                cov_cf = E_yc_yf - mu_c * mu_f_const

                # Marginal Utilities
                term_cov_fp = alpha * cov_cf + wf_curr * var_f_const
                if not self.replace: term_cov_fp += wm_curr * cov_fm_const
                mu_marginal_f = mu_f_const - self.gamma * term_cov_fp

                term_cov_cp = alpha * var_c + wf_curr * cov_cf
                if not self.replace:
                    E_yc_ym = self.p * (yc1 * ym[0]) + (1 - self.p) * (yc2 * ym[1])
                    cov_cm = E_yc_ym - mu_c * mu_m_const
                    term_cov_cp += wm_curr * cov_cm
                mu_marginal_c = mu_c - self.gamma * term_cov_cp

                # D. Regime Constraints
                if 'regime_constr' in setup:
                    m.remove(setup['regime_constr'])

                if not self.replace:
                    mu_marginal_m = 0.0
                    term_cov_mp = alpha * cov_cm + wf_curr * cov_fm_const + wm_curr * var_m_const
                    mu_marginal_m = mu_m_const - self.gamma * term_cov_mp

                    if regime == 'mixed':
                        setup['regime_constr'] = m.addConstr(mu_marginal_f == mu_marginal_m, name="RegimeBalance")
                        mu_active_rest = mu_marginal_f
                    elif regime == 'fixed_only':
                        setup['regime_constr'] = m.addConstr(mu_marginal_f >= mu_marginal_m, name="RegimeBalance")
                        mu_active_rest = mu_marginal_f
                    elif regime == 'mutable_only':
                        setup['regime_constr'] = m.addConstr(mu_marginal_m >= mu_marginal_f, name="RegimeBalance")
                        mu_active_rest = mu_marginal_m
                else:
                    mu_active_rest = mu_marginal_f

                # E. Set Scaled Objective
                # FIX 2: We divide the expression by OBJ_SCALE to keep gradients well-behaved
                raw_obj = mu_marginal_c - mu_active_rest
                m.setObjective(raw_obj / OBJ_SCALE, GRB.MAXIMIZE)
                
                m.optimize()

                # --- RESULT HANDLING ---
                # Note: We must un-scale the gap when reading it back
                if m.Status == GRB.OPTIMAL:
                    gap = m.ObjVal * OBJ_SCALE 
                    if gap > max_gap:
                        max_gap = gap
                        current_iter_best_g = np.array([gx.X, gy.X])
                        w_res = {'wc': alpha}
                        if regime == 'fixed_only': w_res['wf'] = remaining
                        elif regime == 'mutable_only': 
                            if not self.replace: w_res['wm'] = remaining
                        else:
                            w_res['wf'] = wf_var.X
                            if not self.replace: w_res['wm'] = wm_var.X
                        current_iter_best_w = w_res

                # --- RESCUE BLOCK ---
                elif m.Status == GRB.TIME_LIMIT or m.Status == GRB.NUMERIC:
                    print(f'Struggling with {regime} at alpha={alpha:.4f}... Defense Mode.')
                    m.resetParams()
                    m.setParam("OutputFlag", 0)
                    m.setParam("NonConvex", 2)
                    m.setParam("NumericFocus", 3)
                    m.setParam("Presolve", 2)
                    m.setParam("TimeLimit", 180)
                    m.optimize()
                    
                    # Fallback: Epsilon Relaxation
                    if (m.Status != GRB.OPTIMAL) and (regime == 'mixed'):
                        print("  -> Attempting Relaxation...")
                        if 'regime_constr' in setup: m.remove(setup['regime_constr'])
                        
                        # Use scaled range to match model scale
                        epsilon = 1e-6
                        expr = mu_marginal_f - mu_marginal_m
                        # Range constraint: -eps <= expr <= eps
                        setup['regime_constr'] = m.addRange(expr, -epsilon, epsilon, name="RegimeRelaxed")
                        m.optimize()

                    if m.Status == GRB.OPTIMAL:
                        gap = m.ObjVal * OBJ_SCALE
                        if gap > max_gap:
                            max_gap = gap
                            current_iter_best_g = np.array([gx.X, gy.X])
                            w_res = {'wc': alpha}
                            if regime == 'mixed':
                                w_res['wf'] = wf_var.X
                                if not self.replace: w_res['wm'] = wm_var.X
                            current_iter_best_w = w_res
                            
            # --- Decision ---
            if max_gap >= -1e-7:
                best_g = current_iter_best_g
                best_w_opt = current_iter_best_w
                low = alpha
                last_good_gx = best_g[0]
                last_good_gy = best_g[1]
            else:
                high = alpha

        if self.replace and 'wm' in best_w_opt: del best_w_opt['wm']
        return self._build_result(best_g, best_w_opt)

    def _build_result(self, g_cand, w_dict):
        """
        Formats the output to match the required dictionary structure.
        Calculates Mean and Variance explicitly using the solver's result.
        """
        if g_cand is None:
            return None

        # 1. Organize Genotypes and Weights based on mode
        if self.replace:
            # Mode: Fixed + Candidate
            g_list = [self.g_fixed, g_cand]
            weights = [w_dict.get('wf', 0.0), w_dict.get('wc', 0.0)]
        else:
            # Mode: Fixed + Mutable + Candidate
            g_list = [self.g_fixed, self.g_mutable, g_cand]
            weights = [w_dict.get('wf', 0.0), w_dict.get('wm', 0.0), w_dict.get('wc', 0.0)]

        # 2. Compute Yield Matrix Y (Rows=Env, Cols=Genotypes)
        # Yield formula: y1 = x + c*y; y2 = c*x + y
        Y = []
        for g in g_list:
            Y.append([g[0] + self.c * g[1], self.c * g[0] + g[1]])
        Y = np.array(Y).T  # Shape (2, N_assets)

        # 3. Compute Statistics
        probs = np.array([self.p, 1.0 - self.p])
        w_vec = np.array(weights)

        # Individual Expected Returns (y_bar)
        # (2, N) -> (N,)
        y_bar = Y.T @ probs

        # Covariance Matrix (Sigma)
        # E[Y * Y.T] - E[Y] * E[Y].T
        # We need element-wise weighting for covariance
        # Sigma_ij = E[Y_i * Y_j] - mu_i * mu_j

        # Vectorized Covariance Calculation:
        # Weighted inner product of centered yields is safer
        Y_centered = Y - y_bar  # Broadcasting subtract
        # Sigma = sum(p_k * (Y_ki - mu_i)(Y_kj - mu_j))
        Sigma = (Y_centered * probs[:, None]).T @ Y_centered

        # Portfolio Stats
        port_mean = np.dot(w_vec, y_bar)
        Sigma += np.eye(len(w_vec)) * self.sigma2
        Sigma[-1, -1] += self.candidate_breeding_var
        port_var = w_vec.T @ Sigma @ w_vec
        obj_val = port_mean - 0.5 * self.gamma * port_var

        # 4. Return Dictionary
        return {
            "genotypes": [g.tolist() for g in g_list],
            "weights": {
                "env_1": weights,
                "env_2": weights,  # Assuming symmetric weights for environments as per standard model
            },
            "mean_variance": float(obj_val),
            "mean": float(port_mean),
            "variance": float(port_var)
        }

    def strat_hindsight(self):
        """
        Hindsight / State-Contingent Strategy:
        Optimizes the Genotype AND specific weights for each environment simultaneously.

        Allows the portfolio to 'react' to the environment (perfect foresight),
        but penalizes the resulting variance in total portfolio returns.
        """
        m = gp.Model("Hindsight_StateContingent")
        m.setParam("OutputFlag", 0)
        m.setParam("NonConvex", 2)
        m.setParam("MIPGap", 1e-4)
        m.setParam("TimeLimit", 60)

        # --- 1. Bounds & Genotype Variables ---
        (xb_min, xb_max), (yb_min, yb_max) = self._get_ellipse_bounds()

        gx = m.addVar(lb=xb_min, ub=xb_max, name="gx")
        gy = m.addVar(lb=yb_min, ub=yb_max, name="gy")

        self._add_mahalanobis_constraint(m, gx, gy)

        # --- 2. State-Contingent Weights ---
        # Environment 1 Weights
        wc1 = m.addVar(lb=0, ub=1, name="wc1")
        wf1 = m.addVar(lb=0, ub=1, name="wf1")
        wm1 = m.addVar(lb=0, ub=1, name="wm1") if not self.replace else m.addVar(lb=0, ub=0, name="wm1")

        # Environment 2 Weights
        wc2 = m.addVar(lb=0, ub=1, name="wc2")
        wf2 = m.addVar(lb=0, ub=1, name="wf2")
        wm2 = m.addVar(lb=0, ub=1, name="wm2") if not self.replace else m.addVar(lb=0, ub=0, name="wm1")

        # Budget Constraints (Separate for each Env)
        if self.replace:
            m.addConstr(wc1 + wf1 == 1, "Budget1")
            m.addConstr(wc2 + wf2 == 1, "Budget2")
        else:
            m.addConstr(wc1 + wf1 + wm1 == 1, "Budget1")
            m.addConstr(wc2 + wf2 + wm2 == 1, "Budget2")

        # --- 3. Bilinear Terms (Interaction between Weight and Genotype) ---
        # We need auxiliary variables z = w * g for each environment
        # Because w is different, z will be different for Env1 and Env2

        # Bounds for Z
        z_xmin, z_xmax = min(0, xb_min), max(0, xb_max)
        z_ymin, z_ymax = min(0, yb_min), max(0, yb_max)

        # Env 1 Interaction
        zx1 = m.addVar(lb=z_xmin, ub=z_xmax, name="zx1")
        zy1 = m.addVar(lb=z_ymin, ub=z_ymax, name="zy1")
        m.addQConstr(zx1 == wc1 * gx)
        m.addQConstr(zy1 == wc1 * gy)

        # Env 2 Interaction
        zx2 = m.addVar(lb=z_xmin, ub=z_xmax, name="zx2")
        zy2 = m.addVar(lb=z_ymin, ub=z_ymax, name="zy2")
        m.addQConstr(zx2 == wc2 * gx)
        m.addQConstr(zy2 == wc2 * gy)

        # --- 4. Construct Portfolio Yields ---
        # Candidate Yield contributions (wc * yc)
        # yc1 = gx + c*gy  ->  wc1*yc1 = zx1 + c*zy1
        # yc2 = c*gx + gy  ->  wc2*yc2 = c*zx2 + zy2
        term_c_env1 = zx1 + self.c * zy1
        term_c_env2 = self.c * zx2 + zy2

        # Fixed Asset Contributions
        term_f_env1 = wf1 * self.yf[0]
        term_f_env2 = wf2 * self.yf[1]

        # Mutable Asset Contributions
        term_m_env1 = wm1 * self.ym[0]
        term_m_env2 = wm2 * self.ym[1]

        # Total Portfolio Yields (Expressions)
        Yp1 = term_f_env1 + term_m_env1 + term_c_env1
        Yp2 = term_f_env2 + term_m_env2 + term_c_env2

        # --- 5. Objective: Utility of the Portfolio ---
        # Yield noise is a variance penalty, not a shock the farmer observes.
        # Environment-contingent weights may respond to the environment, while
        # independent yield noise remains unobserved and contributes sigma^2
        # times the squared portfolio weights to conditional variance.
        mu_p = self.p * Yp1 + (1 - self.p) * Yp2

        # Second moment across environments plus conditional yield-risk variance.
        E_Yp2 = self.p * (Yp1 * Yp1) + (1 - self.p) * (Yp2 * Yp2)
        yield_risk = self.sigma2 * (
            self.p * (wf1 * wf1 + wm1 * wm1 + wc1 * wc1)
            + (1 - self.p) * (wf2 * wf2 + wm2 * wm2 + wc2 * wc2)
        )
        breeding_risk = self.candidate_breeding_var * (
            self.p * wc1 * wc1 + (1 - self.p) * wc2 * wc2
        )
        obj = mu_p - 0.5 * self.gamma * (
            E_Yp2 - mu_p * mu_p + yield_risk + breeding_risk
        )

        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()

        # Retry timed-out solves and evaluate the retry status below. The old
        # flow discarded a successful 180-second retry and raised regardless.
        if m.Status == GRB.TIME_LIMIT:
            print('Struggling with start_hindsight; retrying with 180 seconds')
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)
            m.setParam("MIPGap", 1e-4)
            m.setParam("TimeLimit", 180)
            m.optimize()

        # --- 6. Format Output ---
        if m.Status == GRB.OPTIMAL:
            # Extract Results
            g_opt = np.array([gx.X, gy.X])

            # Extract Weights per environment
            w1_list = [wf1.X, wm1.X, wc1.X] if not self.replace else [wf1.X, wc1.X]
            w2_list = [wf2.X, wm2.X, wc2.X] if not self.replace else [wf2.X, wc2.X]

            # Calculate final stats numerically for verification/return
            # (Re-calculating utilizing the solved values)
            val_Yp1 = Yp1.getValue()
            val_Yp2 = Yp2.getValue()

            final_mean = self.p * val_Yp1 + (1 - self.p) * val_Yp2
            final_var = (
                self.p * (val_Yp1 ** 2) + (1 - self.p) * (val_Yp2 ** 2)
                - final_mean ** 2
                + self.sigma2 * (
                    self.p * (wf1.X ** 2 + wm1.X ** 2 + wc1.X ** 2)
                    + (1 - self.p) * (wf2.X ** 2 + wm2.X ** 2 + wc2.X ** 2)
                )
                + self.candidate_breeding_var * (
                    self.p * wc1.X ** 2 + (1 - self.p) * wc2.X ** 2
                )
            )
            final_obj = final_mean - 0.5 * self.gamma * final_var

            # Genotype list for output
            if self.replace:
                g_list = [self.g_fixed, g_opt]
            else:
                g_list = [self.g_fixed, self.g_mutable, g_opt]

            return {
                "genotypes": [g.tolist() for g in g_list],
                "weights": {
                    "env_1": w1_list,
                    "env_2": w2_list
                },
                "mean_variance": float(final_obj),
                "mean": float(final_mean),
                "variance": float(final_var),
                "solver_certificate": {
                    "status": "OPTIMAL",
                    "objective_value": float(m.ObjVal),
                    "objective_bound": float(m.ObjBound),
                    "relative_gap": float(m.MIPGap),
                    "requested_relative_gap": float(m.Params.MIPGap),
                    "scope": "global optimum for the known breeding outcome and analytic yield-risk variance",
                },
            }
        raise RuntimeError(
            "Clairvoyance did not certify a global optimum for the known-outcome "
            f"model (Gurobi status {m.Status}, solution_count {m.SolCount}, "
            f"gap {m.MIPGap if m.SolCount else 'unavailable'})."
        )

    def strat_hindsight_full_information(self, states):
        """Globally optimize a full-information policy over a finite state sample.

        The target genotype is common across states and must lie in the breeding
        ellipse. Each state has its own adoption weights after the environment,
        variety yield shocks, and candidate breeding shock are observed. The
        objective is the ex-ante mean-variance utility across all states.

        Gurobi's spatial branch-and-bound globally solves this nonconvex QCQP.
        A result is returned only at OPTIMAL status; callers must still interpret
        it as optimal for the supplied finite Monte Carlo approximation.
        """
        if not states:
            raise ValueError("Full-information Clairvoyance needs at least one state.")
        expected_assets = 2 if self.replace else 3
        probabilities = []
        checked_states = []
        for k, state in enumerate(states):
            try:
                env = int(state['env'])
                prob = float(state['prob'])
                eta = np.asarray(state['eta'], dtype=float)
                eps = np.asarray(state['eps'], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed full-information state at index {k}.") from exc
            if env not in (0, 1):
                raise ValueError(f"State {k} has environment {env}; expected 0 or 1.")
            if not np.isfinite(prob) or prob < 0:
                raise ValueError(f"State {k} probability must be finite and nonnegative.")
            if eta.shape != (2,) or not np.all(np.isfinite(eta)):
                raise ValueError(f"State {k} eta must contain two finite values.")
            if eps.shape != (expected_assets,) or not np.all(np.isfinite(eps)):
                raise ValueError(
                    f"State {k} eps must contain {expected_assets} finite variety shocks."
                )
            probabilities.append(prob)
            checked_states.append({'env': env, 'prob': prob, 'eta': eta, 'eps': eps})
        if not np.isclose(sum(probabilities), 1.0, atol=1e-9, rtol=0.0):
            raise ValueError("Full-information state probabilities must sum to one.")
        states = checked_states

        m = gp.Model("Clairvoyance_FullInformation_SAA")
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        m.Params.MIPGap = 1e-4
        m.Params.TimeLimit = 600

        G = np.linalg.inv(self.Ginv)
        dx = self.R * np.sqrt(max(float(G[0, 0]), 0.0))
        dy = self.R * np.sqrt(max(float(G[1, 1]), 0.0))
        # The breeding target is selected before eta is realized in either
        # timing mode. Constrain the target itself to the admissible genotype
        # box and breeding ellipse; realized genotypes may lie outside the box
        # under the additive shock specified by the stochastic model.
        tx_lb, tx_ub = max(-1.0, self.g_mutable[0] - dx), min(
            1.0, self.g_mutable[0] + dx
        )
        ty_lb, ty_ub = max(-1.0, self.g_mutable[1] - dy), min(
            1.0, self.g_mutable[1] + dy
        )
        tx = m.addVar(lb=tx_lb, ub=tx_ub, name="target_x")
        ty = m.addVar(lb=ty_lb, ub=ty_ub, name="target_y")
        ux, uy = tx - self.g_mutable[0], ty - self.g_mutable[1]
        m.addQConstr(
            self.Ginv[0, 0] * ux * ux + 2 * self.Ginv[0, 1] * ux * uy
            + self.Ginv[1, 1] * uy * uy <= self.R ** 2,
            name="target_breeding_ellipse",
        )

        scenario_returns = []
        scenario_probs = []
        weight_vars = []
        for k, state in enumerate(states):
            env = int(state["env"])
            beta = (1.0, self.c) if env == 0 else (self.c, 1.0)
            eta = np.asarray(state["eta"], dtype=float)
            eps = np.asarray(state["eps"], dtype=float)
            wc = m.addVar(lb=0.0, ub=1.0, name=f"wc_{k}")
            wf = m.addVar(lb=0.0, ub=1.0, name=f"wf_{k}")
            wm = (m.addVar(lb=0.0, ub=1.0, name=f"wm_{k}")
                  if not self.replace else None)
            if self.replace:
                m.addConstr(wf + wc == 1.0, name=f"budget_{k}")
                eps_f, eps_c = eps[0], eps[1]
                parent_mutable = 0.0
            else:
                m.addConstr(wf + wm + wc == 1.0, name=f"budget_{k}")
                eps_f, eps_m, eps_c = eps[0], eps[1], eps[2]
                parent_mutable = wm * (self.ym[env] + eps_m)

            # z = wc * target. Since wc is in [0, 1], these bounds must
            # include every feasible target value (known shocks can shift
            # target coordinates outside the realized-genotype box).
            ztx = m.addVar(lb=min(0.0, tx_lb), ub=max(0.0, tx_ub),
                           name=f"wc_target_x_{k}")
            zty = m.addVar(lb=min(0.0, ty_lb), ub=max(0.0, ty_ub),
                           name=f"wc_target_y_{k}")
            m.addQConstr(ztx == wc * tx, name=f"candidate_product_x_{k}")
            m.addQConstr(zty == wc * ty, name=f"candidate_product_y_{k}")

            candidate_const = beta[0] * eta[0] + beta[1] * eta[1] + eps_c
            ret_expr = (
                wf * (self.yf[env] + eps_f)
                + parent_mutable
                + wc * candidate_const
                + beta[0] * ztx + beta[1] * zty
            )
            ret = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"return_{k}")
            m.addConstr(ret == ret_expr, name=f"return_definition_{k}")
            scenario_returns.append(ret)
            scenario_probs.append(float(state["prob"]))
            weight_vars.append((env, wf, wm, wc))

        mu = gp.quicksum(prob * ret for prob, ret in zip(scenario_probs, scenario_returns))
        second_moment = gp.quicksum(
            prob * ret * ret for prob, ret in zip(scenario_probs, scenario_returns)
        )
        variance = second_moment - mu * mu
        m.setObjective(mu - 0.5 * self.gamma * variance, GRB.MAXIMIZE)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                "Full-information Clairvoyance did not certify a global optimum "
                f"within the requested relative gap (Gurobi status {m.Status}, "
                f"solution_count {m.SolCount}, "
                f"gap {m.MIPGap if m.SolCount else 'unavailable'})."
            )

        probs = np.asarray(scenario_probs, dtype=float)
        returns = np.asarray([ret.X for ret in scenario_returns], dtype=float)
        mean = float(np.dot(probs, returns))
        var = float(np.dot(probs, (returns - mean) ** 2))
        env_weights = {}
        for env, key in ((0, "env_1"), (1, "env_2")):
            selected = [item for item in weight_vars if item[0] == env]
            env_mass = sum(s['prob'] for s in states if s['env'] == env)
            if env_mass > 0:
                local_probabilities = [s['prob'] / env_mass for s in states if s['env'] == env]
            else:
                local_probabilities = [1.0 / len(selected)] * len(selected) if selected else []
            if self.replace:
                avg = [sum(prob * item[1].X for prob, item in zip(local_probabilities, selected)),
                       sum(prob * item[3].X for prob, item in zip(local_probabilities, selected))]
            else:
                avg = [sum(prob * item[1].X for prob, item in zip(local_probabilities, selected)),
                       sum(prob * item[2].X for prob, item in zip(local_probabilities, selected)),
                       sum(prob * item[3].X for prob, item in zip(local_probabilities, selected))]
            env_weights[key] = avg

        reported_candidate = np.array([tx.X, ty.X], dtype=float)
        genotypes = ([self.g_fixed.tolist(), reported_candidate.tolist()] if self.replace else
                     [self.g_fixed.tolist(), self.g_mutable.tolist(), reported_candidate.tolist()])
        try:
            solver_bound = float(m.ObjBound)
        except (AttributeError, gp.GurobiError):
            solver_bound = float(m.ObjVal)
        try:
            solver_gap = float(m.MIPGap)
        except (AttributeError, gp.GurobiError):
            solver_gap = abs(solver_bound - float(m.ObjVal)) / max(1.0, abs(float(m.ObjVal)))
        full_information_policy = []
        for state, ret, vars_for_state in zip(states, scenario_returns, weight_vars):
            env, wf, wm, wc = vars_for_state
            weights = ([float(wf.X), float(wc.X)] if self.replace else
                       [float(wf.X), float(wm.X), float(wc.X)])
            full_information_policy.append({
                'env': int(state['env']),
                'prob': float(state['prob']),
                'eta': np.asarray(state['eta']).astype(float).tolist(),
                'eps': np.asarray(state['eps']).astype(float).tolist(),
                'weights': weights,
                'return': float(ret.X),
            })
        return {
            "genotypes": genotypes,
            "weights": env_weights,
            "mean_variance": float(mean - 0.5 * self.gamma * var),
            "mean": mean,
            "variance": var,
            "full_information": True,
            "state_count": len(states),
            "target": [float(tx.X), float(ty.X)],
            "full_information_policy": full_information_policy,
            "solver_certificate": {
                'status': 'OPTIMAL',
                'objective_value': float(m.ObjVal),
                'objective_bound': solver_bound,
                'relative_gap': solver_gap,
                'absolute_bound_difference': abs(solver_bound - float(m.ObjVal)),
                'requested_relative_gap': float(m.Params.MIPGap),
                'scope': 'within requested relative gap for the supplied finite outcome sample',
            },
        }

    def _get_ellipse_bounds(self):
        """
        Calculates explicit bounds [min, max] for x and y based on the
        Mahalanobis constraint.
        Formula: Extent_i = R * sqrt((Ginv^-1)_ii)
        """
        # 1. Invert Ginv to get Covariance Matrix G
        try:
            G = np.linalg.inv(self.Ginv)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix: use very large bounds
            return (-1e4, 1e4), (-1e4, 1e4)

        # 2. Calculate Extents
        # The bounding box half-width is R * sqrt(Diagonal of Inverse Ginv)
        dx = self.R * np.sqrt(G[0, 0])
        dy = self.R * np.sqrt(G[1, 1])

        # 3. Center point (g_mutable or g_fixed)
        cx = self.g_mutable[0] + self.known_eta[0]
        cy = self.g_mutable[1] + self.known_eta[1]

        # 4. Return explicit bounds
        x_bounds = (cx - dx, cx + dx)
        y_bounds = (cy - dy, cy + dy)

        if x_bounds[0] < -1:
            x_bounds = (-1, x_bounds[1])
        if x_bounds[1] > 1:
            x_bounds = (x_bounds[0], 1)
        if y_bounds[0] < -1:
            y_bounds = (-1, y_bounds[1])
        if y_bounds[1] > 1:
            y_bounds = (y_bounds[0], 1)
        return x_bounds, y_bounds
