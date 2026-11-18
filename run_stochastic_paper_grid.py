"""Run the full 21^4 stochastic-paper grid at medium noise and breeding levels.

The four known-timing conditions are deterministic, yield-noise only,
breeding-outcome only, and both sources. Unknown breeding timing is omitted.
Clairvoyance sees the environment, realized yield shocks, and known breeding
outcome. Its ex-ante value is estimated on 128 common outcome states when
yield noise is active. Results use a new directory so the prior analytic-risk
Clairvoyance checkpoint remains available for comparison.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import gurobipy as gp
import numpy as np

from experiment_core import ExperimentRunner


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs" / "stochastic_paper_grid_god_mode_20261118"
RESULTS = OUTDIR / "results.csv"
PROGRESS = OUTDIR / "progress.json"
README = OUTDIR / "README.md"
CLAIRVOYANCE_MODEL = "god_mode_known_eta_and_observed_yield_shocks_saa128_v1"
# The user authorized this follow-on grid to use all available logical cores.
WORKERS = os.cpu_count() or 1
N_INTERNAL = 401
RANDOM_STATE = 42
PARAMETERS = {"p": 0.5, "c": 0.1, "gamma": 19.25, "r_g": 0.0, "R": 0.1}
COORDINATES = tuple(round(i / 20, 2) for i in range(21))
GENOTYPES = list(itertools.product(COORDINATES, repeat=2))
# Keep every fixed/mutable combination, including coincident genotypes: the
# requested full grid is 21 choices for each of four genotype coordinates.
PAIRS = [(fixed, mutable) for fixed in GENOTYPES for mutable in GENOTYPES]
CONDITIONS = (
    ("deterministic", 0.0, 0.0),
    ("yield_noise_only", 0.10, 0.0),
    ("breeding_only_known", 0.0, 0.05),
    ("both_known", 0.10, 0.05),
)
CORE_STRATEGIES = ("Base", "BeatBest", "PoB", "BeatBestS", "Clairvoyance")
TOTAL = len(PAIRS) * len(CONDITIONS)


def _worker_init() -> None:
    gp.setParam("Threads", 1)
    gp.setParam("OutputFlag", 0)


def _weights_for(strategy_data: dict) -> tuple[list[float], list[float]]:
    weights = strategy_data["out_dict"]["weights"]
    if isinstance(weights, dict):
        return list(map(float, weights["env_1"])), list(map(float, weights["env_2"]))
    values = list(map(float, weights))
    return values, values


def _json_default(value):
    """Convert NumPy values in retained optimizer output to JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _run_one(job: tuple[int, int, str, float, float, tuple, tuple]) -> dict:
    job_id, pair_id, condition, sigma, breeding_sd, g_fixed, g_mutable = job
    row = {
        "job_id": job_id,
        "pair_id": pair_id,
        "condition": condition,
        "sigma": sigma,
        "breeding_sd": breeding_sd,
        "g_fixed_1": g_fixed[0],
        "g_fixed_2": g_fixed[1],
        "g_mutable_1": g_mutable[0],
        "g_mutable_2": g_mutable[1],
        "clairvoyance_model": CLAIRVOYANCE_MODEL,
        "status": "ERROR",
        "error": "",
        "maxmarket_error": "",
    }
    try:
        scenario = {
            "label": f"{condition}_{pair_id}",
            "g_fixed": np.asarray(g_fixed, dtype=float),
            "g_mutable": np.asarray(g_mutable, dtype=float),
        }
        runner = ExperimentRunner(
            **PARAMETERS,
            scenario_pairs=[scenario],
            replace=False,
            n=N_INTERNAL,
            sigma=sigma,
            breeding_sd=breeding_sd,
            breeding_timing="known",
            random_state=RANDOM_STATE,
        )
        _, scenario_data = runner.compute(
            strategy_keys=CORE_STRATEGIES,
            replace=False,
            easy_base=False,
        )
        if not scenario_data:
            raise RuntimeError("No scenario result was produced.")
        data = scenario_data[0]
        details = data["details_per_strategy"]
        base = data["base_line"]
        row["eta_1"] = float(runner.known_eta[0])
        row["eta_2"] = float(runner.known_eta[1])
        row["base_mean_variance"] = float(base["mean_variance"])

        for strategy in CORE_STRATEGIES:
            result = details[strategy]
            stats = result["stats"]
            prefix = strategy.lower()
            row[f"{prefix}_mean_variance"] = float(stats["mean_variance"])
            row[f"{prefix}_mean"] = float(stats["mean"])
            row[f"{prefix}_variance"] = float(stats["variance"])
            row[f"{prefix}_raw_gain_vs_base"] = float(stats["gain"])
            w1, w2 = _weights_for(result)
            row[f"{prefix}_weights_env1"] = json.dumps(w1, default=_json_default)
            row[f"{prefix}_weights_env2"] = json.dumps(w2, default=_json_default)
            row[f"{prefix}_genotypes"] = json.dumps(
                result["out_dict"]["genotypes"], default=_json_default
            )
            if strategy == "PoB":
                wc1, wc2 = w1[-1], w2[-1]
                row["pob_candidate_share_env1"] = wc1
                row["pob_candidate_share_env2"] = wc2
                row["pob_candidate_mixed"] = int(
                    (1e-6 < wc1 < 1.0 - 1e-6)
                    or (1e-6 < wc2 < 1.0 - 1e-6)
                )
            if strategy == "Clairvoyance":
                certificate = result["out_dict"].get("solver_certificate", {})
                row["clairvoyance_solver_gap"] = certificate.get("relative_gap", "")
                row["clairvoyance_solver_gap_target"] = certificate.get(
                    "requested_relative_gap", ""
                )
                row["clairvoyance_state_count"] = result["out_dict"].get(
                    "state_count", 2
                )

        # Run MaxMarket separately so an infeasible MaxMarket optimization
        # does not erase the other strategy results for this scenario.
        try:
            _, policy = runner._strat_max_investability(data["grid_dict"])
            if policy is None:
                raise RuntimeError("MaxMarket returned no policy.")
            certificate = policy.get("solver_certificate", {})
            row["maxmarket_solver_status"] = certificate.get("status", "")
            row["maxmarket_solver_gap"] = certificate.get("relative_gap", "")
            row["maxmarket_solver_bound"] = certificate.get("objective_bound", "")
            # Score the actual MaxMarket policy weights. Re-optimizing weights
            # for its chosen genotype would no longer measure max investability.
            maxmarket = runner.calculate_stats_from_dict(policy)
            states = data["grid_dict"].get("clairvoyance_states")
            if states is not None:
                maxmarket = runner._evaluate_policy_on_states(
                    maxmarket, states, candidate_is_new=True
                )
            row["maxmarket_mean_variance"] = float(maxmarket["mean_variance"])
            row["maxmarket_mean"] = float(maxmarket["mean"])
            row["maxmarket_variance"] = float(maxmarket["variance"])
            row["maxmarket_raw_gain_vs_base"] = float(
                maxmarket["mean_variance"] - base["mean_variance"]
            )
            w1, w2 = _weights_for({"out_dict": maxmarket})
            row["maxmarket_weights_env1"] = json.dumps(w1, default=_json_default)
            row["maxmarket_weights_env2"] = json.dumps(w2, default=_json_default)
            row["maxmarket_genotypes"] = json.dumps(
                maxmarket["genotypes"], default=_json_default
            )
        except Exception as exc:
            row["maxmarket_error"] = f"{type(exc).__name__}: {exc}"[:1000]

        # This ratio reproduces the paper's normalized-gain summary definition.
        # For known breeding outcomes it is diagnostic: ordinary strategies are
        # conditional on one eta draw while Clairvoyance averages its state sample.
        pob_gain = float(details["PoB"]["stats"]["gain"])
        clairvoyance_gain = float(details["Clairvoyance"]["stats"]["gain"])
        if abs(clairvoyance_gain) > 1e-4:
            row["pob_normalized_gain_diagnostic"] = pob_gain / clairvoyance_gain
        row["status"] = "OK"
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        return row


def _jobs(done: set[int]):
    job_id = 0
    for condition, sigma, breeding_sd in CONDITIONS:
        for pair_id, (g_fixed, g_mutable) in enumerate(PAIRS):
            if job_id not in done:
                yield (job_id, pair_id, condition, sigma, breeding_sd, g_fixed, g_mutable)
            job_id += 1


def _is_certified_row(row: dict[str, str]) -> bool:
    return (
        row.get("status") == "OK"
        and row.get("clairvoyance_model") == CLAIRVOYANCE_MODEL
        and not row.get("maxmarket_error")
        and row.get("maxmarket_solver_status") in {"OPTIMAL", "OPTIMAL_STATUS_QUO"}
        and row.get("clairvoyance_solver_gap") not in (None, "")
    )


def _write_progress(completed: int, ok: int, errors: int, started: float,
                    prior_elapsed: float, prior_rate: float,
                    run_start_completed: int) -> None:
    segment_elapsed = time.time() - started
    elapsed = prior_elapsed + segment_elapsed
    segment_completed = completed - run_start_completed
    rate = (
        segment_completed / segment_elapsed
        if segment_elapsed >= 60 and segment_completed > 0
        else prior_rate
    )
    payload = {
        "completed": completed,
        "total": TOTAL,
        "clairvoyance_model": CLAIRVOYANCE_MODEL,
        "succeeded": ok,
        "errors": errors,
        "fraction": completed / TOTAL,
        "elapsed_seconds": elapsed,
        "scenarios_per_second": rate,
        "estimated_remaining_seconds": (TOTAL - completed) / rate if rate > 0 else None,
        "workers": WORKERS,
        "conditions": [
            {"condition": name, "sigma": sigma, "breeding_sd": breeding_sd}
            for name, sigma, breeding_sd in CONDITIONS
        ],
        "updated_unix": time.time(),
    }
    temp = PROGRESS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(PROGRESS)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    previous = {}
    if PROGRESS.exists():
        try:
            previous = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    prior_elapsed = float(previous.get("elapsed_seconds", 0.0))
    prior_rate = float(previous.get("scenarios_per_second", 1.0 / 2.81))
    README.write_text(
        "# Stochastic paper-grid god-mode rerun (2026-11-18)\n\n"
        "The full grid has 21 values (0, 0.05, ..., 1) for each of four "
        "genotype coordinates, for 21^4 = 194,481 cells per condition. "
        "All fixed/mutable combinations are included. Four conditions use known "
        "breeding-outcome timing: deterministic; yield noise only (sigma=0.10); "
        "breeding outcome only (sd=0.05); and both sources (sigma=0.10, sd=0.05). "
        "Unknown timing is omitted. Parameters are p=0.5, c=0.1, gamma=19.25, "
        "r_g=0, R=0.1, internal n=401, seed=42. MaxMarket and Clairvoyance "
        "target relative MIP gap 1e-4. Strategy weights, genotype solutions, "
        "and per-scenario gains are retained. For Clairvoyance, each outcome\n"
        "state reveals the environment, yield shock for every variety, and the\n"
        "known breeding outcome. With sigma>0, ex-ante value is approximated\n"
        "using 128 antithetic states. Ordinary strategies are scored on the same\n"
        "state sample but do not observe yield shocks when choosing weights.\n"
        "The previous v2 checkpoint used analytic yield-risk Clairvoyance and\n"
        "is preserved separately; this run starts fresh to avoid mixing models.\n",
        encoding="utf-8",
    )
    persisted = set()
    if RESULTS.exists():
        with RESULTS.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _is_certified_row(row):
                    persisted.add(int(row["job_id"]))
    fields = [
        "job_id", "pair_id", "condition", "sigma", "breeding_sd",
        "g_fixed_1", "g_fixed_2", "g_mutable_1", "g_mutable_2",
        "clairvoyance_model", "clairvoyance_state_count",
        "eta_1", "eta_2", "status", "error",
        "maxmarket_error", "pob_candidate_share_env1", "pob_candidate_share_env2",
        "maxmarket_solver_status", "maxmarket_solver_gap", "maxmarket_solver_bound",
        "pob_candidate_mixed", "pob_normalized_gain_diagnostic",
        "clairvoyance_solver_gap", "clairvoyance_solver_gap_target",
    ]
    for strategy in (*CORE_STRATEGIES, "MaxMarket"):
        prefix = strategy.lower()
        fields.extend([
            f"{prefix}_mean_variance", f"{prefix}_mean", f"{prefix}_variance",
            f"{prefix}_raw_gain_vs_base", f"{prefix}_weights_env1",
            f"{prefix}_weights_env2", f"{prefix}_genotypes",
        ])
    needs_header = not RESULTS.exists() or RESULTS.stat().st_size == 0
    out = RESULTS.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    if needs_header:
        writer.writeheader()
        out.flush()
    completed = len(persisted)
    ok = completed
    errors = 0
    started = time.time()
    run_start_completed = completed
    _write_progress(completed, ok, errors, started, prior_elapsed, prior_rate,
                    run_start_completed)
    jobs = iter(_jobs(persisted))
    pending = set()
    window = WORKERS * 2
    print(f"START total={TOTAL} workers={WORKERS} completed={completed}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init) as pool:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < window:
                    try:
                        pending.add(pool.submit(_run_one, next(jobs)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    writer.writerow(row)
                    completed += 1
                    if _is_certified_row(row):
                        ok += 1
                    else:
                        errors += 1
                    if completed % 10 == 0 or completed == TOTAL:
                        out.flush()
                        _write_progress(
                            completed, ok, errors, started, prior_elapsed,
                            prior_rate, run_start_completed,
                        )
                        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
                        prior_rate = progress["scenarios_per_second"] or prior_rate
                        print(
                            f"PROGRESS completed={completed}/{TOTAL} ok={ok} "
                            f"errors={errors} rate={prior_rate:.4f}/s "
                            f"eta_hours={progress['estimated_remaining_seconds'] / 3600:.2f}",
                            flush=True,
                        )
    finally:
        out.flush()
        out.close()
        _write_progress(
            completed, ok, errors, started, prior_elapsed, prior_rate,
            run_start_completed,
        )
    print(f"FINISHED completed={completed}/{TOTAL} ok={ok} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
