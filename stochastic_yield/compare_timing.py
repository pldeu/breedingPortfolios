"""
Compare the deterministic model with farm-level stochastic variants, and the two timings
of the breeding outcome ('known' before adoption vs 'unknown' at adoption).

    python stochastic_yield/compare_timing.py [--sweep N]

Uses the full ExperimentRunner pipeline (Gurobi + Monte Carlo).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from experiment_core import ExperimentRunner  # noqa: E402

KEYS = ['Base', 'BeatBest', 'PoB', 'MaxMarket', 'Clairvoyance']
DEFAULT = dict(p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1, g_fixed=(0.6, 0.2), g_mutable=(-0.1, 0.2))


def run(params, spec, n=201):
    sc = [{"label": "s", "g_fixed": np.array(params['g_fixed']), "g_mutable": np.array(params['g_mutable'])}]
    r = ExperimentRunner(p=params['p'], c=params['c'], gamma=params['gamma'], r_g=params['r_g'],
                         R=params['R'], scenario_pairs=sc, n=n, stochastic=spec)
    _, data = r.compute(strategy_keys=KEYS)
    if not data:
        return None
    d = data[0]['details_per_strategy']
    out = {k: dict(gain=v['stats']['gain'], e_wc=v['out_dict'].get('e_wc', v['out_dict']['weights']['env_1'][-1]),
                   p_adopt=v['out_dict'].get('p_adopt', np.nan), g=np.array(v['out_dict']['genotypes'][-1]))
           for k, v in d.items()}
    clair = out['Clairvoyance']['gain']
    for k in out:
        out[k]['norm'] = out[k]['gain'] / clair if abs(clair) > 1e-4 else np.nan
    return out


def default_table():
    R = DEFAULT['R']
    configs = [
        ("deterministic", None),
        ("sigma=0.10", dict(sigma=0.10)),
        ("s=R/2, known", dict(breeding_sd=R / 2, timing='known')),
        ("s=R/2, unknown", dict(breeding_sd=R / 2, timing='unknown')),
        ("sigma=0.10, s=R/2, known", dict(sigma=0.10, breeding_sd=R / 2, timing='known')),
        ("sigma=0.10, s=R/2, unknown", dict(sigma=0.10, breeding_sd=R / 2, timing='unknown')),
    ]
    print("Default scenario of app.py. Gain = E[MV] - base; % = share of the Clairvoyance gain.")
    print(f"{'configuration':28s} {'BeatBest':>16s} {'PoB':>16s} {'MaxMarket':>16s} {'Clairv':>8s}"
          f" {'PoB E[w_c]':>10s} {'PoB P(adopt)':>12s}")
    for label, spec in configs:
        t = time.time()
        o = run(DEFAULT, spec, n=401)
        cells = " ".join(f"{o[k]['gain']:8.4f} ({100 * o[k]['norm']:4.0f}%)" for k in ('BeatBest', 'PoB', 'MaxMarket'))
        print(f"{label:28s} {cells} {o['Clairvoyance']['gain']:8.4f} {o['PoB']['e_wc']:10.3f}"
              f" {o['PoB']['p_adopt']:12.2f}   [{time.time() - t:.1f}s]")


def random_params(rng):
    return dict(p=rng.uniform(0.2, 0.8), c=rng.uniform(0.0, 0.6), gamma=float(rng.choice([5.0, 21.0, 50.0])),
                r_g=rng.uniform(-0.7, 0.7), R=rng.uniform(0.05, 0.3),
                g_fixed=tuple(rng.uniform(-0.2, 1.0, 2)), g_mutable=tuple(rng.uniform(-0.2, 1.0, 2)))


def sweep(n, sigma=0.10, rel_s=0.5, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    while len(rows) < n:
        pr = random_params(rng)
        res = {tm: run(pr, dict(sigma=sigma, breeding_sd=rel_s * pr['R'], timing=tm, n_draws=150,
                                n_clair_draws=30, n_states=60))
               for tm in ('known', 'unknown')}
        det = run(pr, None)
        if None in res.values() or det is None:
            continue
        rows.append((det, res['known'], res['unknown']))
    print(f"\nSweep: {n} random scenarios, sigma={sigma}, s={rel_s}*R  [{time.time() - t0:.0f}s]")
    print(f"{'':34s} {'deterministic':>14s} {'known':>10s} {'unknown':>10s}")

    def line(label, f, fmt="{:10.4f}"):
        vals = [np.nanmedian([f(r[i]) for r in rows]) for i in range(3)]
        print(f"{label:34s} {fmt.format(vals[0]):>14s} {fmt.format(vals[1]):>10s} {fmt.format(vals[2]):>10s}")

    line("median PoB gain", lambda o: o['PoB']['gain'])
    line("median BeatBest gain", lambda o: o['BeatBest']['gain'])
    line("median MaxMarket gain", lambda o: o['MaxMarket']['gain'])
    line("median Clairvoyance gain", lambda o: o['Clairvoyance']['gain'])
    line("median PoB share of Clairvoyance", lambda o: o['PoB']['norm'], "{:10.2f}")
    line("median PoB E[w_c]", lambda o: o['PoB']['e_wc'], "{:10.3f}")
    line("median BeatBest E[w_c]", lambda o: o['BeatBest']['e_wc'], "{:10.3f}")
    for i, lab in ((0, 'deterministic'), (1, 'known'), (2, 'unknown')):
        adv = np.array([r[i]['PoB']['gain'] - r[i]['BeatBest']['gain'] for r in rows])
        print(f"PoB - BeatBest gain, {lab:14s} median {np.median(adv):8.4f}, PoB better in"
              f" {100 * np.mean(adv > 1e-5):3.0f}%")
    voi = np.array([r[1]['PoB']['gain'] - r[2]['PoB']['gain'] for r in rows])
    print(f"value of knowing the outcome before adoption (PoB): median {np.median(voi):.4f},"
          f" min {voi.min():.4f}, max {voi.max():.4f}")
    dist = np.array([np.linalg.norm(r[1]['PoB']['g'] - r[2]['PoB']['g']) for r in rows])
    print(f"PoB target shift known vs unknown: median {np.median(dist):.4f} (genotype units)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', type=int, default=0)
    args = ap.parse_args()
    default_table()
    if args.sweep:
        sweep(args.sweep)
