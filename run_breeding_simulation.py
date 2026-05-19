# %%
from Jacob_experiment_case_gen import ExperimentSweepRunner
from Jacob_experiment_base4 import ExperimentRunner
import pandas as pd
import numpy as np

sweep = ExperimentSweepRunner(sweep_size=10000000, random_state=42)

# 1) generate inputs
df_inputs = sweep.generate_discrete_inputs()
df_inputs['gamma'] =  pd.read_csv('results_jacob_long.csv').gamma.mean()

#df_inputs = df_inputs[:1000]

my_strategies = {
    #'Base': None,
    'Mean': None,   
    'PoB': None,
    'BeatBest': None,
    'Adopt': None,
    #'Clairvoyance': None
}


# 2) run sweep (pass your Runner class)
#df_results = sweep.run_sweep(df_inputs, RunnerClass=ExperimentRunner, strategies=my_strategies, n_jobs=5, replace=False)
df_results = pd.read_csv('results_jacob_circle.csv')

df_results['w'] = df_results.Weights.astype(str)

df_results['Criterion1'] = df_results['PoB-BeatBest']
df_results['Criterion2'] = df_results['PoB-Mean']
df_results['Criterion3'] = df_results['PoB-Adopt']

# 3) select diverse scenarios
selected = sweep.select_diverse_scenarios(df_results, k=5, method='greedy')

# 4) plot / inspect / run detailed experiments

sweep.plot_selection_context(df_results, selected, normalize=True)

#sweep.run_detailed_experiments(selected, RunnerClass=ExperimentRunner, strategies=my_strategies, replace=False, easy_base=False)

# 5) export LaTeX snippets
#sweep.export_latex_report(selected, out_dir='scenarios')
df_results.to_csv('results_jacob_circle.csv')
