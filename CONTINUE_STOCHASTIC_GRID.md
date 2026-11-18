# Continue the stochastic grid on another computer

## Current checkpoint

The run is stopped. Its saved results are here:

`app/breedingPortfolios/outputs/stochastic_paper_grid_full_rerun_20261117_v2/`

The CSV currently has **329,662 certified cells out of 777,924**. It also has three unsuccessful rows; the resume script will retry those cells. The CSV parsed without malformed rows. Keep `results.csv` and `progress.json` together.

## Move the run

Copy the **whole** `app/breedingPortfolios` folder to the faster computer, including the output folder above. Do not open or edit `results.csv` in Excel. It is large (about 558 MB).

The computer needs Python, the packages used by this project (including Gurobi's Python package), and a working Gurobi license. Use the same environment that can run `experiment_core.py`.

## Resume it

Open PowerShell in the copied `breedingPortfolios` folder and run:

```powershell
python .\run_stochastic_paper_grid.py
```

That is the resume command too: if the run stops, run the same command again. The script reads the existing CSV, skips certified cells, and retries missing or unsuccessful cells. It uses all logical CPU cores available to Python, with one Gurobi thread per worker.

## Check progress

In another PowerShell window, from the same folder, run:

```powershell
Get-Content .\outputs\stochastic_paper_grid_full_rerun_20261117_v2\progress.json -Raw | ConvertFrom-Json | Format-List completed,total,succeeded,errors,workers,scenarios_per_second,estimated_remaining_seconds
```

The run is complete when `completed` and `succeeded` both equal `777924`, and `errors` is `0`. The final console line should also start with `FINISHED`.

## What this run includes

It evaluates all 21 values (0 to 1 by 0.05) for each of the four fixed/mutable genotype coordinates under four conditions: deterministic; yield noise only (sigma 0.10); known breeding outcome only (breeding SD 0.05); and both (sigma 0.10, breeding SD 0.05). It saves the strategy results and solver certificates in `results.csv`.
