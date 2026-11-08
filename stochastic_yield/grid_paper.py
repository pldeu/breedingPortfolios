"""Genotype grid at the manuscript's settings (p=0.5, c=0.1, rho_g=0, R=0.1, lambda=19.25,
g in [0,1]^2), reporting the paper's metrics under deterministic and stochastic variants.
    python stochastic_yield/grid_paper.py [--step 0.25] [--jobs 4]"""
import argparse, itertools, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_genotypes as gg
gg.BASE = dict(p=0.5, c=0.1, gamma=19.25, r_g=0.0, R=0.1)

def norm_stats(df, c):
    out = {}
    clair = df[f'{c}_Clairvoyance_gain']
    for s in ('PoB', 'BeatBestS', 'BeatBest', 'MaxMarket'):
        g = df[f'{c}_{s}_gain'] / clair.where(clair.abs() > 1e-4)
        out[s] = (g.median(), 100 * (g > 0.25).mean())
    dom = {}
    for s in ('BeatBest', 'BeatBestS', 'MaxMarket'):
        a, b = df[f'{c}_PoB_gain'], df[f'{c}_{s}_gain']
        dom[s] = 100 * np.mean((a >= 1.2 * b) & (a > 1e-6) | ((b <= 1e-9) & (a > 1e-6)))
    return out, dom

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--step', type=float, default=0.25); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    vals = np.round(np.arange(0, 1 + 1e-9, a.step), 4)
    pts = list(itertools.product(vals, repeat=2))
    pairs = [(gf, gm) for gf in pts for gm in pts if gf != gm]
    with Pool(a.jobs) as pool:
        df = pd.DataFrame(pool.map(gg.one, pairs, chunksize=4))
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_paper.csv'), index=False)
    print(f"{len(df)} pairs, paper settings, step {a.step}")
    for c in gg.CONFIGS:
        out, dom = norm_stats(df, c)
        print(f"\n[{c}] median normalized gain / % >0.25:  " + "  ".join(f"{s} {m:.2f}/{p:.0f}%" for s, (m, p) in out.items()))
        print(f"[{c}] PoB >=20% better than: " + "  ".join(f"{s} {v:.1f}%" for s, v in dom.items()))
        print(f"[{c}] median raw gain PoB {df[f'{c}_PoB_gain'].median():.4f}, median PoB share {df[f'{c}_PoB_wc'].median():.2f}, "
              f"3-variety-type mixes: PoB w_c in (0,1): {100*np.mean((df[f'{c}_PoB_wc']>1e-3)&(df[f'{c}_PoB_wc']<0.999)):.0f}%")
