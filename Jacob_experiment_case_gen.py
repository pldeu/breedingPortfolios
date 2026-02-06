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

import warnings
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =============================
# Default configuration 
# =============================
PARAM_GRID_DEFAULT = {
    'p': np.arange(0.1, 1.0, 0.4),
    'c': np.arange(0.1, 0.6, 0.4),
    'gamma': np.array([21.0, 500.0]),
    'r_g': np.arange(-0.9, 0.8, 0.45),
    'R': np.arange(0.1, 0.4, 0.2),
}
GENOTYPE_GRID_VALS_DEFAULT = np.arange(-0.7, 0.8, 0.35)

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
   