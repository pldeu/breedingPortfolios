"""
Grid experiment: vary only g_fixed and g_mutable (other parameters at app.py defaults)
and compare deterministic, farm-noise and breeding-noise (known / unknown) variants.

    python stochastic_yield/grid_genotypes.py [--vals -0.2 0.1 0.4 0.7 1.0] [--jobs 4]

Writes stochastic_yield/grid_genotypes.csv and prints a summary.
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_timing import run  # noqa: E402

BASE = dict(p=0.5, c=0.1, gamma=21.0, r_g=-0.4, R=0.1)
CONFIGS = {
    'det': None,
    'sigma': dict(sigma=0.10),
    'known': dict(breeding_sd=0.05, timing='known', n_draws=150, n_clair_draws=30),
    'unknown': dict(breeding_sd=0.05, timing='unknown', n_clair_draws=30),
}
STRATS = ('BeatBest', 'BeatBestS', 'PoB', 'MaxMarket', 'Clairvoyance')


def one(args):
    gf, gm = args
    pr = dict(BASE, g_fixed=gf, g_mutable=gm)
    row = dict(gf1=gf[0], gf2=gf[1], gm1=gm[0], gm2=gm[1])
    for cname, spec in CONFIGS.items():
        try:
            o = run(pr, spec)
        except Exception as e:  # keep the sweep going, record the failure
            row[f'{cname}_error'] = str(e)[:80]
            continue
        if o is None:
            continue
        for s in STRATS:
            row[f'{cname}_{s}_gain'] = o[s]['gain']
            row[f'{cname}_{s}_wc'] = o[s]['e_wc']
    return row


def summarize(df):
    print(f"\n{len(df)} genotype pairs (p=0.5, c=0.1, gamma=21, r_g=-0.4, R=0.1)\n")
    print(f"{'':30s}" + "".join(f"{c:>10s}" for c in CONFIGS))
    for s in ('PoB', 'BeatBest', 'BeatBestS', 'MaxMarket', 'Clairvoyance'):
        print(f"{'median ' + s + ' gain':30s}" + "".join(f"{df[f'{c}_{s}_gain'].median():10.4f}" for c in CONFIGS))
    print(f"{'median PoB share w_c':30s}" + "".join(f"{df[f'{c}_PoB_wc'].median():10.3f}" for c in CONFIGS))
    for other in ('BeatBest', 'BeatBestS'):
        print(f"{'PoB better than ' + other:30s}" + "".join(
            f"{100 * np.mean(df[f'{c}_PoB_gain'] - df[f'{c}_{other}_gain'] > 1e-5):9.0f}%" for c in CONFIGS))
        print(f"{'  mean PoB - ' + other:30s}" + "".join(
            f"{(df[f'{c}_PoB_gain'] - df[f'{c}_{other}_gain']).mean():10.4f}" for c in CONFIGS))
    voi = df['known_PoB_gain'] - df['unknown_PoB_gain']
    print(f"\nvalue of knowing outcome (PoB): median {voi.median():.4f}, max {voi.max():.4f}, "
          f"negative (< -1e-3) in {np.mean(voi < -1e-3) * 100:.0f}%")
    # where does PoB beat BBS? by environmental tilt of the fixed variety
    tilt = (df.gf1 - df.gf2).abs()
    for lab, m in (("fixed balanced |g1-g2|<=0.3", tilt <= 0.3), ("fixed tilted  |g1-g2|>0.3", tilt > 0.3)):
        gap = (df.loc[m, 'det_PoB_gain'] - df.loc[m, 'det_BeatBestS_gain'])
        print(f"{lab}: n={m.sum():3d}, det mean PoB-BBS {gap.mean():.4f}, PoB better {100 * np.mean(gap > 1e-5):.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--vals', type=float, nargs='+', default=[-0.2, 0.1, 0.4, 0.7, 1.0])
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    pts = list(itertools.product(a.vals, repeat=2))
    pairs = [(gf, gm) for gf in pts for gm in pts if gf != gm]
    with Pool(a.jobs) as pool:
        rows = pool.map(one, pairs, chunksize=4)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_genotypes.csv')
    df.to_csv(out, index=False)
    summarize(df)
    print(f"\nsaved {out}")
