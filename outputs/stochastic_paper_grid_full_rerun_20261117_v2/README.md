# Stochastic paper-grid full rerun (2026-11-17)

The full grid has 21 values (0, 0.05, ..., 1) for each of four genotype coordinates, for 21^4 = 194,481 cells per condition. All fixed/mutable combinations are included. Four conditions use known breeding-outcome timing: deterministic; yield noise only (sigma=0.10); breeding outcome only (sd=0.05); and both sources (sigma=0.10, sd=0.05). Unknown timing is omitted. Parameters are p=0.5, c=0.1, gamma=19.25, r_g=0, R=0.1, internal n=401, seed=42. MaxMarket and Clairvoyance target relative MIP gap 1e-4. Strategy weights, genotype solutions, and per-scenario gains are retained. For known breeding outcomes, the
realized genotype is used directly; yield noise is integrated as a
variance penalty and is not revealed to the policy.
