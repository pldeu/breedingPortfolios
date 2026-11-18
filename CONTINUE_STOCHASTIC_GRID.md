# Run the stochastic paper grid on the faster computer

## Scope and grid size

Run the four paper conditions only. Each condition has `21^4 = 194,481` genotype cells: 21 values (0 to 1 by 0.05) for each of the fixed and mutable genotypes' two coordinates. Four conditions therefore total `4 x 21^4 = 777,924` jobs. There is no need to run the nine-condition factorial for this paper-grid rerun.

| Condition | Yield noise (`sigma`) | Breeding outcome SD |
| --- | ---: | ---: |
| Deterministic | 0 | 0 |
| Yield noise only | 0.10 | 0 |
| Known breeding outcome only | 0 | 0.05 |
| Both sources | 0.10 | 0.05 |

Other settings: `p=0.5`, `c=0.1`, `gamma=19.25`, `r_g=0`, `R=0.1`, internal `n=401`, and seed 42. The script uses all logical CPU cores and one Gurobi thread per worker.

## Existing completed run (older Clairvoyance model)

The completed CSV reported from the Desktop clone is at `C:\Users\pdeuker\Desktop\coding\breedingPortfolios\outputs\stochastic_paper_grid_full_rerun_20261117_v2\results.csv`, with `progress.json` alongside it. Its 777,924 unique job IDs are the expected four-condition grid. It has three duplicate job IDs; keep the last OK row per job ID, since the earlier OK rows were not MaxMarket-certified.

Treat that CSV as the older Clairvoyance definition. It does not contain the later statewise breeding-target choice, so do not use it as the result of the current god-mode model. Nine rows also report a Clairvoyance solver gap slightly above `1e-4`; flag those as above the requested tolerance in any analysis.

## Prepare the faster computer

Install Git, Git LFS, Python, the project packages, and Gurobi with a working license. In PowerShell:

```powershell
git lfs install
git clone https://github.com/pldeu/breedingPortfolios.git
cd .\breedingPortfolios
git lfs pull
git pull
```

The latest code uses a fresh output folder, `outputs/stochastic_paper_grid_god_mode_statewise_20261118/`. Do not copy the older CSV into that folder; the model definitions differ. The old run remains available for side-by-side comparison.

## Start or resume the current model

From the `breedingPortfolios` folder:

```powershell
python .\run_stochastic_paper_grid.py
```

The run resumes from that output folder if interrupted. Run the same command again after an interruption; it skips rows certified for the current model version and retries other rows. Keep the computer plugged in and prevent sleep while the run is active.

## What current Clairvoyance assumes

In each modeled future state, Clairvoyance sees the environment, breeding outcome, and every variety's yield shock. It then chooses that state's breeding target and portfolio, subject to the same genotype bounds and breeding ellipse. This is an oracle benchmark for a lucky guesser, not a physically implementable breeding plan.

With yield noise, the ex-ante score uses 128 antithetic outcome states. Without yield noise, the two environment states are enumerated directly. The result is an optimum for those states; the 128-state approximation is not a mathematical guarantee for the full continuous outcome distribution.

## Monitor progress

In another PowerShell window, from the project folder:

```powershell
Get-Content .\outputs\stochastic_paper_grid_god_mode_statewise_20261118\progress.json -Raw | ConvertFrom-Json | Format-List completed,total,succeeded,errors,workers,scenarios_per_second,estimated_remaining_seconds
```

The target is `total=777924`, `succeeded=777924`, and `errors=0`; the log should end with `FINISHED`. After completion, inspect the maximum `clairvoyance_solver_gap` in `results.csv` and flag any value above `1e-4`. The script records the requested target in `clairvoyance_solver_gap_target`; a completed job count alone does not establish every numerical gap is below target.

Keep `results.csv`, `progress.json`, the run log, and the exact code version together when transferring or archiving the results.
