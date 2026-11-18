# Continue the stochastic grid on another computer

## Which checkpoint to use

The old partial run is preserved in `outputs/stochastic_paper_grid_full_rerun_20261117_v2/`. It has 329,662 certified cells, but its Clairvoyance results only accounted for yield noise as a variance penalty. Do not use it as the checkpoint for the corrected god-mode run.

The corrected script writes a fresh grid to `outputs/stochastic_paper_grid_god_mode_20261118/`. This keeps results from the two Clairvoyance definitions separate. The old checkpoint is not deleted.

## Get the files

Install Git, Git LFS, Python, and the packages used by the project, including Gurobi's Python package. A working Gurobi license is required.

In PowerShell, run:

```powershell
git lfs install
git clone https://github.com/pldeu/breedingPortfolios.git
cd .\breedingPortfolios
git lfs pull
```

## Start or resume the corrected run

From the `breedingPortfolios` folder, run:

```powershell
python .\run_stochastic_paper_grid.py
```

If it stops, run the same command again. It resumes from the new output folder's `results.csv`, skipping cells certified under the corrected Clairvoyance model and retrying failures. The old v2 results remain separate.

The grid has 777,924 cells: 21 values (0 to 1 by 0.05) for each of four fixed/mutable genotype coordinates, under four conditions: deterministic; yield noise only (sigma 0.10); known breeding outcome only (breeding SD 0.05); and both (sigma 0.10, breeding SD 0.05). It uses all logical CPU cores available to Python and one Gurobi thread per worker.

## What Clairvoyance knows

For each possible future state, Clairvoyance observes the environment, every variety's realized yield shock, and the known breeding outcome before choosing adoption and portfolio weights. The breeding target remains common across states because it is chosen before the breeding outcome; the portfolio decision can vary by state.

When yield noise is active, the ex-ante value is estimated over 128 antithetic outcome states. Those are possible futures used to estimate expected performance; Clairvoyance sees the full realized state before acting in each one. Ordinary strategies are scored on the same states, but choose without observing yield shocks. These larger state-contingent optimizations may take longer than the old run.

## Check progress

In another PowerShell window, from the same folder, run:

```powershell
Get-Content .\outputs\stochastic_paper_grid_god_mode_20261118\progress.json -Raw | ConvertFrom-Json | Format-List completed,total,succeeded,errors,workers,scenarios_per_second,estimated_remaining_seconds
```

The run is complete when `completed` and `succeeded` both equal `777924`, and `errors` is `0`. The final console line should start with `FINISHED`.
