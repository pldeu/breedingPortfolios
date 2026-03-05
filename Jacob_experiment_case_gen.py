"""
experiment_sweep_runner.py

A single-file class that encapsulates the functionality from your notebook snippets:
- configuration / grids
- discrete input generation
- running sweeps (calling a provided RunnerClass)
- a few selection procedures (greedy max-min, OR-Tools CBC linearization, Gurobi quadratic, Pyomo)
- running detailed plots for selected scenarios
- exporting a LaTeX snippet per scenario

Optional external solvers (ortools, gurobipy, pyomo) are imported only if available; the class will automatically fall back to the greedy method when a requested solver is missing.

Usage (short):
from experiment_sweep_runner import ExperimentSweepRunner
s = ExperimentSweepRunner()
inputs = s.generate_discrete_inputs(n=1000)
results = s.run_sweep(inputs, RunnerClass=ExperimentRunner)
selected = s.select_diverse_scenarios(k=10, method='greedy')
s.run_detailed_experiments(selected, RunnerClass=ExperimentRunner)
s.export_latex_report(selected, out_dir='scenarios')

"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from typing import Optional, Callable, Dict, Any
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =============================
# Default configuration 
# =============================
PARAM_GRID_DEFAULT = {
    'p': np.arange(0.5, 0.6, 0.4),
    'c': np.arange(0.1, 0.2, 0.4),
    #'gamma': np.array([21.0]),
    'r_g': np.arange(-0.4, 0.8, 1.45),
    'R': np.arange(0.1, 0.2, 0.2),
}
GENOTYPE_GRID_VALS_DEFAULT = np.arange(0.0, 1.01, 0.05)

def _process_single_row(args):
            """Worker function to process a single row. Must be picklable (top-level function)."""
            idx, row, RunnerClass, strategies, keep_errors = args
            
            try:
                g_fixed_vec = np.array([row['g_fix_1'], row['g_fix_2']])
                g_mutable_vec = np.array([row['g_mut_1'], row['g_mut_2']])

                current_pairs = [{
                    "label": f"Sim_{idx}",
                    "g_fixed": g_fixed_vec,
                    "g_mutable": g_mutable_vec,
                }]

                runner = RunnerClass(
                    p=row['p'],
                    c=row['c'],
                    gamma=row['gamma'],
                    r_g=row['r_g'],
                    R=row['R'],
                    scenario_pairs=current_pairs,
                )

                run_output = runner.run(strategies=strategies, plot_extensive=False, easy_base=True, no_plot=True)

                # Extract results robustly
                res_list = run_output[0].get('Results', run_output[0]) if isinstance(run_output[0], dict) else run_output[0]

                def get_stat(name, stat_key):
                    try:
                        for item in res_list:
                            if item.get('name') == name:
                                if stat_key in item['stats'].keys():
                                    return item['stats'][stat_key]
                                else:
                                    return item['out_dict'][stat_key]
                    except Exception:
                        pass
                    return 0.0

                
                util_port = get_stat('MVP', 'mean_variance')
                
                weights = get_stat('MVP', 'weights')

                

                row_result = row.to_dict()
                
                row_result['Weights'] = weights

                for strat_name in strategies.keys():
                    if strat_name !='MVP':
                        util_other_strat = get_stat(strat_name, 'mean_variance')
                        advantage = util_port - util_other_strat
                        row_result['MVP-'+strat_name] = advantage #/ (util_port + 2)
                        row_result["MV_"+strat_name] = util_other_strat 
                        row_result["V_"+strat_name] = get_stat(strat_name, 'variance') 
                        row_result["M_"+strat_name] = get_stat(strat_name, 'mean')  
                    else:
                        util_other_strat = get_stat(strat_name, 'mean_variance')
                        advantage = util_port - util_other_strat
                        row_result["MV_"+strat_name] = util_other_strat 
                        row_result["V_"+strat_name] = get_stat(strat_name, 'variance') 
                        row_result["M_"+strat_name] = get_stat(strat_name, 'mean')  


                return row_result

            except Exception as e:
                if keep_errors:
                    return {**row.to_dict(), 'Out_Advantage': np.nan, 'Out_Risk': np.nan, 'Out_Yield': np.nan, '_error': str(e)}
                return None

class ExperimentSweepRunner:
    """Container class for configuration, sweep, selection and reporting.

    Parameters
    ----------
    param_grid : dict, optional
        Grid of scalar parameters to sample from. Defaults to PARAM_GRID_DEFAULT.
    genotype_vals : array-like, optional
        Values to sample genotype components from.
    sweep_size : int, optional
        Default number of configurations to generate if generate_discrete_inputs called without n.
    random_state : Optional[int]
        Seed for reproducibility.
    """

    def __init__(
        self,
        param_grid: Optional[dict] = None,
        genotype_vals: Optional[np.ndarray] = None,
        sweep_size: int = 10000,
        target_scenarios: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.param_grid = param_grid or PARAM_GRID_DEFAULT
        self.genotype_vals = genotype_vals if genotype_vals is not None else GENOTYPE_GRID_VALS_DEFAULT
        self.sweep_size = sweep_size
        self.target_scenarios = target_scenarios
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Internal state
        self.df_inputs: Optional[pd.DataFrame] = None
        self.df_results: Optional[pd.DataFrame] = None
        self.df_selected: Optional[pd.DataFrame] = None

    # -------------------------
    # 1. Discrete input generation
    # -------------------------
    def generate_discrete_inputs(self, n: Optional[int] = None) -> pd.DataFrame:
        """Generate discrete input rows.
        
        If n (or self.sweep_size) is greater than or equal to the total number of
        possible unique combinations in the grid, this method will Enumerate 
        the full Cartesian product instead of random sampling.
        """
        n = n or self.sweep_size

        # 1. Prepare lists of values for every dimension
        scalar_keys = list(self.param_grid.keys())
        scalar_values = [self.param_grid[k] for k in scalar_keys]
        
        # Genotype dimensions (4 separate variables sampled from genotype_vals)
        geno_keys = ['g_fix_1', 'g_fix_2', 'g_mut_1', 'g_mut_2']
        geno_values = [self.genotype_vals] * 4

        # 2. Calculate total cardinality of the search space
        total_combinations = 1
        for v in scalar_values:
            total_combinations *= len(v)
        for v in geno_values:
            total_combinations *= len(v)

        # 3. Decide strategy: Enumeration vs Sampling
        if n >= total_combinations:
            print(f"Requested sweep_size ({n}) >= Total Space ({total_combinations}).")
            print("Enumerating all possible combinations (Full Grid)...")
            
            # Create full Cartesian product
            all_keys = scalar_keys + geno_keys
            all_lists = scalar_values + geno_values
            
            # itertools.product is memory efficient generator, 
            # pd.DataFrame creation will consume memory proportional to total_combinations
            grid_product = itertools.product(*all_lists)
            
            df = pd.DataFrame(grid_product, columns=all_keys)
            
            # Apply rounding to match the original random sampling behavior
            df = df.round(3)
            
        else:
            # Fallback to original random sampling if n < total space
            samples = []
            print(f"Requested sweep_size ({n}) < Total Space ({total_combinations}).")
            for _ in range(n):
                row = {k: float(np.random.choice(v)) for k, v in self.param_grid.items()}
                row['g_fix_1'] = float(np.random.choice(self.genotype_vals))
                row['g_fix_2'] = float(np.random.choice(self.genotype_vals))
                row['g_mut_1'] = float(np.random.choice(self.genotype_vals))
                row['g_mut_2'] = float(np.random.choice(self.genotype_vals))

                # Round to 1 decimal place to reproduce original behaviour
                for k in list(row.keys()):
                    row[k] = round(row[k], 3)

                samples.append(row)
            
            df = pd.DataFrame(samples).drop_duplicates().reset_index(drop=True)

        self.df_inputs = df
        return df

    # -------------------------
    # 2. Run sweep
    # -------------------------
    from functools import partial
    from typing import Optional, Dict, Any, Callable
    from tqdm import tqdm

    
    def run_sweep(
            self,
            input_df: Optional[pd.DataFrame],
            RunnerClass: Callable,
            strategies: Optional[Dict[str, Any]] = None,
            progress: bool = True,
            keep_errors: bool = False,
            n_jobs: Optional[int] = None,
        ) -> pd.DataFrame:
        """Run RunnerClass for every row in input_df and collect results.
        
        Args:
            input_df: Input dataframe with parameters
            RunnerClass: Class to instantiate for each row
            strategies: Strategy configuration dict
            progress: Show progress bar
            keep_errors: Keep rows that error out (with NaN values)
            n_jobs: Number of parallel processes. None = all CPUs, 1 = sequential (original behavior)
        
        Returns:
            DataFrame with appended Out_Advantage, Out_Risk, Out_Yield columns.
        """

                
        if input_df is None:
            if self.df_inputs is None:
                raise ValueError("No input dataframe provided and none generated previously.")
            input_df = self.df_inputs

        strategies = strategies or {'MVP': None, 'BeatBest': None}
        
        # Prepare arguments for workers
        args_list = [
            (idx, row, RunnerClass, strategies, keep_errors)
            for idx, row in input_df.iterrows()
        ]
        
        # Sequential execution (original behavior)
        if n_jobs == 1:
            results = []
            iterator = args_list
            if progress:
                iterator = tqdm(args_list, desc="Running sweep")
            
            for args in iterator:
                result = _process_single_row(args)
                if result is not None:
                    results.append(result)
        
        # Parallel execution
        else:
            n_processes = n_jobs or os.cpu_count()
            
            with multiprocessing.Pool(processes=n_processes) as pool:
                if progress:
                    results = list(tqdm(
                        pool.imap(_process_single_row, args_list),
                        total=len(args_list),
                        desc=f"Running sweep ({n_processes} cores)"
                    ))
                else:
                    results = pool.map(_process_single_row, args_list)
            
            # Filter out None results (errors when keep_errors=False)
            results = [r for r in results if r is not None]

        df_results = pd.DataFrame(results).reset_index(drop=True)
        self.df_results = df_results
        return df_results

    # -------------------------
    # 3. Selection methods
    # -------------------------
    def select_diverse_scenarios(
        self,
        df: Optional[pd.DataFrame] = None,
        k: Optional[int] = None,
        method: str = 'greedy',
        **kwargs,
    ) -> pd.DataFrame:
        """Select k diverse scenarios from df using method.

        method: one of {'greedy', 'gurobi'}; falls back to 'greedy' if solver missing.
        """
        df = df if df is not None else self.df_results
        if df is None or df.empty:
            warnings.warn("No results available to select from. Returning empty DataFrame.")
            return pd.DataFrame()

        k = k or self.target_scenarios
        if k <= 0:
            raise ValueError("k must be positive")

        required_cols = ['Criterion1', 'Criterion2', 'Criterion3']
        if not all(c in df.columns for c in required_cols):
            raise ValueError(f"Input dataframe must contain columns {required_cols}")

        if method == 'greedy':
            selected = self._select_greedy(df, k)
        elif method == 'gurobi':
            if not _HAS_GUROBI:
                warnings.warn("gurobi not available; falling back to greedy")
                selected = self._select_greedy(df, k)
            else:
                selected = self._select_gurobi(df, k)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        self.df_selected = selected.reset_index(drop=True)
        return self.df_selected

    def _select_greedy(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        X = df[['Criterion1', 'Criterion2', 'Criterion3']].values
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(X)

        selected_indices = []
        remaining = list(range(len(df)))

        # Extremes: max advantage
        idx_best = df['Criterion1'].idxmax()
        selected_indices.append(remaining.index(idx_best))

        if k > 1:
            idx_worst = df['Criterion1'].idxmin()
            if idx_worst != idx_best:
                selected_indices.append(remaining.index(idx_worst))

        while len(selected_indices) < k and len(selected_indices) < len(df):
            cur = normalized[selected_indices]
            min_dists = []
            for i in range(len(normalized)):
                if i in selected_indices:
                    min_dists.append(-1)
                else:
                    dists = np.linalg.norm(cur - normalized[i], axis=1)
                    min_dists.append(np.min(dists))

            next_idx = int(np.argmax(min_dists))
            selected_indices.append(next_idx)

        return df.iloc[selected_indices].copy()
    
    def _select_gurobi(self, df: pd.DataFrame, k: int, distance_percentile: float = 99.0) -> pd.DataFrame:

        # ===== HELPER FUNCTIONS =====

        def _greedy_max_min_dispersion(D: np.ndarray, k: int) -> list:
            """
            Greedy heuristic for max-min dispersion.
            Start with two most distant points, iteratively add point that maximizes min distance.
            """
            N = D.shape[0]
            
            # Start with two most distant points
            i_max, j_max = np.unravel_index(np.argmax(D), D.shape)
            selected = [i_max, j_max]
            
            # Iteratively add points
            for _ in range(k - 2):
                best_point = -1
                best_min_dist = -1
                
                # For each unselected point, compute min distance to selected set
                for i in range(N):
                    if i in selected:
                        continue
                    
                    # Min distance from i to all selected points
                    min_dist_i = min(D[i, s] for s in selected)
                    
                    if min_dist_i > best_min_dist:
                        best_min_dist = min_dist_i
                        best_point = i
                
                selected.append(best_point)
            
            return selected


        def _compute_min_distance(D: np.ndarray, selected: list) -> float:
            """Compute minimum pairwise distance in selected set."""
            if len(selected) < 2:
                return 0.0
            
            min_dist = float('inf')
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    min_dist = min(min_dist, D[selected[i], selected[j]])
            
            return min_dist
        
        """
        Optimized max-min dispersion for large N (>10,000) and small k (<20).
        
        Parameters:
        -----------
        distance_percentile : float
            Only consider distances above this percentile (default 99.0 = top 1%)
        """
        if not _HAS_GUROBI:
            raise RuntimeError("gurobi not available")
        
        X = df[['Criterion1', 'Criterion2', 'Criterion3']].values
        N = len(df)
        
        # Compute distance matrix
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        
        # ===== STEP 1: Pre-filter distances (HUGE speedup) =====
        # Only consider pairs with distances in top percentile
        threshold = np.percentile(D, distance_percentile)
        
        # Find pairs exceeding threshold
        high_dist_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if D[i, j] >= threshold:
                    high_dist_pairs.append((i, j, D[i, j]))
        
        print(f"N = {N}, k = {k}")
        print(f"Total possible pairs: {N*(N-1)//2:,}")
        print(f"Pairs considered (top {100-distance_percentile:.1f}%): {len(high_dist_pairs):,}")
        print(f"Reduction: {100*(1-len(high_dist_pairs)/(N*(N-1)//2)):.1f}%")
        
        # ===== STEP 2: Greedy warm start =====
        # Start with two most distant points, then add points maximizing min distance
        warm_start = _greedy_max_min_dispersion(D, k)
        print(f"Warm start min distance: {_compute_min_distance(D, warm_start):.4f}")
        
        # ===== STEP 3: Build model =====
        model = gp.Model('max_min_dispersion_fast')
        model.Params.OutputFlag = 1
        model.Params.MIPFocus = 1
        
        # Use MVar for faster variable creation
        x = model.addMVar(N, vtype=GRB.BINARY, name='x')
        z = model.addVar(lb=0, name='min_distance')
        
        # Selection constraint
        model.addConstr(x.sum() == k, name='selection')
        
        # ===== STEP 4: Add constraints only for high-distance pairs =====
        # Big-M formulation: z <= D[i,j] when both x[i] and x[j] are selected
        M = np.max(D) + 1
        
        # Group constraints by chunks for better performance
        for i, j, d_ij in high_dist_pairs:
            model.addConstr(z <= d_ij + M * (2 - x[i] - x[j]))
        
        # ===== STEP 5: Set warm start =====
        for i in range(N):
            x[i].Start = 1 if i in warm_start else 0
        z.Start = _compute_min_distance(D, warm_start)
        
        # ===== STEP 6: Optional - Add valid inequalities for small k =====
        # For small k, we can add symmetry breaking or clique constraints
        if k <= 20:
            # Symmetry breaking: order selected items by index (optional)
            # This helps when many items are similar
            pass  # Can be added if needed
        
        model.setObjective(z, GRB.MAXIMIZE)
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            selected = np.where(x.X > 0.5)[0].tolist()
            min_dist = _compute_min_distance(D, selected)
            print(f"Optimal min distance: {min_dist:.4f}")
            print(f"Improvement over warm start: {min_dist - _compute_min_distance(D, warm_start):.4f}")
            return df.iloc[selected].copy()
        else:
            raise RuntimeError(f"Optimization failed with status {model.Status}")

    def _select_gurobi_lazy_constraints(self, df: pd.DataFrame, k: int) -> pd.DataFrame:

        # ===== HELPER FUNCTIONS =====

        def _greedy_max_min_dispersion(D: np.ndarray, k: int) -> list:
            """
            Greedy heuristic for max-min dispersion.
            Start with two most distant points, iteratively add point that maximizes min distance.
            """
            N = D.shape[0]
            
            # Start with two most distant points
            i_max, j_max = np.unravel_index(np.argmax(D), D.shape)
            selected = [i_max, j_max]
            
            # Iteratively add points
            for _ in range(k - 2):
                best_point = -1
                best_min_dist = -1
                
                # For each unselected point, compute min distance to selected set
                for i in range(N):
                    if i in selected:
                        continue
                    
                    # Min distance from i to all selected points
                    min_dist_i = min(D[i, s] for s in selected)
                    
                    if min_dist_i > best_min_dist:
                        best_min_dist = min_dist_i
                        best_point = i
                
                selected.append(best_point)
            
            return selected


        def _compute_min_distance(D: np.ndarray, selected: list) -> float:
            """Compute minimum pairwise distance in selected set."""
            if len(selected) < 2:
                return 0.0
            
            min_dist = float('inf')
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    min_dist = min(min_dist, D[selected[i], selected[j]])
            
            return min_dist
        
        """
        Max-min dispersion using lazy constraints - fastest for very large N.
        Only adds minimum distance constraints when they're violated.
        """
        if not _HAS_GUROBI:
            raise RuntimeError("gurobi not available")
        
        X = df[['Out_Advantage', 'Out_Risk', 'Out_Yield']].values
        N = len(df)
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        
        model = gp.Model('max_min_lazy')
        model.Params.OutputFlag = 0
        model.Params.LazyConstraints = 1
        
        x = model.addMVar(N, vtype=GRB.BINARY, name='x')
        z = model.addVar(lb=0, ub=np.max(D), name='min_distance')
        
        model.addConstr(x.sum() == k)
        model.setObjective(z, GRB.MAXIMIZE)
        
        # Callback to add violated constraints lazily
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                x_val = model.cbGetSolution(x)
                z_val = model.cbGetSolution(z)
                
                # Find selected items
                selected = np.where(x_val > 0.5)[0]
                
                # Check if z violates any pairwise distance
                M = np.max(D) + 1
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        idx_i, idx_j = selected[i], selected[j]
                        if z_val > D[idx_i, idx_j] + 1e-6:  # Violated
                            # Add lazy constraint
                            model.cbLazy(z <= D[idx_i, idx_j] + M * (2 - x[idx_i] - x[idx_j]))
        
        # Warm start
        warm_start = _greedy_max_min_dispersion(D, k)
        for i in range(N):
            x[i].Start = 1 if i in warm_start else 0
        z.Start = _compute_min_distance(D, warm_start)
        
        model.optimize(callback)
        
        if model.Status == GRB.OPTIMAL:
            selected = np.where(x.X > 0.5)[0].tolist()
            return df.iloc[selected].copy()
        else:
            raise RuntimeError(f"Optimization failed with status {model.Status}")

    
    # -------------------------
    # 4. Run detailed experiments (with plotting)
    # -------------------------

    def extract_results_to_row_wide(self, output, scenario_params):
        """
        One row per scenario, one column per strategy
        """
        scenario_out = output[0]
        row = dict(scenario_params)

        # ---- Baseline ----
        baseline = scenario_out['Baseline']
        row['mv_Baseline'] = float(baseline['mean_variance'])
        row['w_Baseline'] = baseline['weights']
        row['mv_Optimum'] = float(max(scenario_out['Results'][i]['stats']['mean_variance'] for i in range(len(scenario_out['Results']))))

        # ---- Other strategies ----
        for res in scenario_out['Results']:
            name = res['name']
            if row['mv_Optimum'] == row['mv_Baseline']:
                row[f'w_{name}'] = 0
            else:
                row[f'mv_{name}'] = float((res['stats']['mean_variance'] - row['mv_Baseline']) / (row['mv_Optimum'] - row['mv_Baseline']))
            if type(res['out_dict']['weights']) == list:
                row[f'w_{name}'] = res['out_dict']['weights'][-1]
            else:
                row[f'w_{name}'] =  row['p'] * res['out_dict']['weights']['env_1'][-1] + (1 - row['p'] ) * res['out_dict']['weights']['env_2'][-1]

        return row

    
    def run_detailed_experiments(self, scenarios_df: pd.DataFrame, RunnerClass: Callable, strategies: Optional[Dict[str, Any]] = None, easy_base=False, replace=None, pareto=False, plot_right=False, num_runs=None, no_plot=False, dpi=100) -> None:
        if scenarios_df is None or scenarios_df.empty:
            warnings.warn("No scenarios to run for detailed experiments.")
            return
        
        if num_runs == None:
            num_runs = len(scenarios_df)

        strategies = strategies or {
            'Base': None,
            'Mean': None,
            'MVP': None,
            'BeatBest': None,
            'Clairvoyance': None,
            'Adopt': None
        }
        all_rows = []
        for it, (idx, row) in enumerate(scenarios_df.iterrows()):
            if it == num_runs:
                break
            if not no_plot:
                print(f"\n---> Running Selected Scenario #{idx}")
                print(f"     Params: p={row['p']}, c={row['c']}, gamma={row['gamma']}, r_g={row['r_g']}, R={row['R']}")
            g_fixed_vec = np.array([row['g_fix_1'], row['g_fix_2']])
            g_mutable_vec = np.array([row['g_mut_1'], row['g_mut_2']])

            current_pairs = [{
                "label": f"Selected_Scenario_{idx}",
                "g_fixed": g_fixed_vec,
                "g_mutable": g_mutable_vec,
            }]

            runner = RunnerClass(
                p=row['p'], c=row['c'], gamma=row['gamma'], r_g=row['r_g'], R=row['R'], scenario_pairs=current_pairs
            )

            output = runner.run(strategies=strategies, plot_extensive=True, easy_base=easy_base, replace=replace, pareto=pareto, plot_right=plot_right, no_plot=no_plot, dpi=dpi)
            if not no_plot:
                print("-" * 20)

            scenario_params = {
                'p': row['p'],
                'c': row['c'],
                'gamma': row['gamma'],
                'r_g': row['r_g'],
                'R': row['R'],
                'g_fix_1': current_pairs[0]["g_fixed"][0],
                'g_fix_2':current_pairs[0]["g_fixed"][1],
                'g_mut_1': current_pairs[0]["g_mutable"][0],
                'g_mut_2':current_pairs[0]["g_mutable"][1],
            }

            all_rows.append(
                self.extract_results_to_row_wide(output, scenario_params)
            )

        results_df = pd.DataFrame(all_rows)
        results_df.loc[:, results_df.columns.str.startswith(('mv_', 'w_'))] = \
        results_df.loc[:, results_df.columns.str.startswith(('mv_', 'w_'))].round(3)


        cols = list(scenario_params.keys())
        for s in strategies.keys():
            cols += [f'mv_{s}', f'w_{s}']
        results_df = results_df[cols]
        results_df = results_df.sort_values(
        by=['r_g', 'p', 'c', 'gamma'])
        return results_df

    # -------------------------
    # 5. Plotting helpers
    # -------------------------
    def plot_selection_context(self, df_results: Optional[pd.DataFrame] = None, final_scenarios: Optional[pd.DataFrame] = None, normalize=False) -> None:

        if normalize:

            X = df_results[['Criterion1', 'Criterion2', 'Criterion3']].values
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(X)
            df_results = pd.DataFrame(normalized, columns=['Criterion1', 'Criterion2', 'Criterion3'])

        df_results = df_results if df_results is not None else self.df_results
        final_scenarios = final_scenarios if final_scenarios is not None else self.df_selected

        if df_results is None or df_results.empty:
            warnings.warn("No results available for plotting")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_results, x='Criterion1', y='Criterion2', alpha=0.5, label='Sweep')
        if final_scenarios is not None and not final_scenarios.empty:
            sns.scatterplot(data=final_scenarios, x='Criterion1', y='Criterion2', color='red', s=100, label='Selected')
        plt.xlabel("Criterion1")
        plt.ylabel("Criterion2")
        plt.title("Scenario Selection from Discrete Grid")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(
            df_results['Criterion1'],
            df_results['Criterion2'],
            df_results['Criterion3'],
            label='All Scenarios',
            alpha=0.4,
            s=40
        )

        ax.scatter(
            final_scenarios['Criterion1'],
            final_scenarios['Criterion2'],
            final_scenarios['Criterion3'],
            label='Selected',
            marker='x',
            color='red',
            s=100,      # Increased size
            zorder=10,   # Force it to the front
            depthshade=False # Prevents the red from turning dark/grey based on distance
        )

        ax.set_xlabel("Criterion1")
        ax.set_ylabel("Criterion2")
        ax.set_zlabel("Criterion3")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.view_init(elev=25, azim=45)

        fig.tight_layout(rect=[0,0,1,0.95])
        ax.set_title("Scenario Selection from Discrete Grid", pad=20)
        ax.legend()
        plt.show()

    # -------------------------
    # 6. Export LaTeX report
    # -------------------------
    def export_latex_report(self, df_selected: Optional[pd.DataFrame] = None, out_dir: str = 'scenarios', tex_filename: str = 'scenarios.tex') -> str:
        df_selected = df_selected if df_selected is not None else self.df_selected
        if df_selected is None or df_selected.empty:
            warnings.warn("No selected scenarios to export")
            return ''

        os.makedirs(out_dir, exist_ok=True)
        tex_path = os.path.join(out_dir, tex_filename)

        with open(tex_path, 'w', encoding='utf-8') as f:
            j = 1
            for i, row in df_selected.iterrows():
                p = row['p']
                c = row['c']
                gamma = row['gamma']
                r_g = row['r_g']
                R = row['R']
                g_fix_1 = row['g_fix_1']
                g_fix_2 = row['g_fix_2']
                g_mut_1 = row['g_mut_1']
                g_mut_2 = row['g_mut_2']

                snippet = fr"""
\subsubsection*{{Scenario {j}:}}
\noindent\textit{{Parameters: $p={p}, c={c}, r_g={r_g}, \gamma={gamma}, R={R}, g_fixed = {{{{{g_fix_1}, {g_fix_2}}}}}, g_mutable=[{g_mut_1}, {g_mut_2}]$}}

\begin{{lstlisting}}
\end{{lstlisting}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{scenarios/sc{j}.png}}
    \caption{{Scenario {j}.}}
\end{{figure}}

"""
                f.write(snippet)
                j += 1

        return tex_path

    # -------------------------
    # Utilities
    # -------------------------
    def save_results_csv(self, df: Optional[pd.DataFrame] = None, path: str = 'sweep_results.csv') -> str:
        df = df if df is not None else self.df_results
        if df is None:
            raise ValueError('No dataframe to save')
        df.to_csv(path, index=False)
        return path


# If the file is run as a script, provide a quick example (won't run as-is because RunnerClass is not defined here)
if __name__ == '__main__':
    print('This module provides ExperimentSweepRunner. Import and use in your project.')
