# From a deterministic to a stochastic yield model

This note explores what it would mean to replace the yield model used in
`experiment_core.py` and `GurobiPortfolioOptimizer.py` by a stochastic one: which
sources of randomness are worth adding, what they change in each strategy, what they
leave unchanged, and how to implement them. Sections 1 to 5 are the exploration;
Sections 6 and 7 describe the stochastic mode now implemented (farm-level risk,
full-information Clairvoyance, both timings of the breeding outcome).
Claims marked [num] are checked with `prototype.py` (numpy only, no Gurobi needed):

```
python stochastic_yield/prototype.py            # default scenario of app.py
python stochastic_yield/prototype.py --sweep 150  # robustness over random scenarios (~25 s)
```

## 1. The current model and what is already random

For a genotype `g = (g0, g1)` the yield is deterministic given the environment:

```
y_1(g) = g0 + c g1        (environment 1, probability p)
y_2(g) = c g0 + g1        (environment 2, probability 1 - p)
```

A portfolio with area shares `w_i` has yield `Y_e = sum_i w_i y_e(g_i)` and is scored by
`U = E[Y] - (gamma/2) Var[Y]`. The candidate is chosen inside the ellipse
`(g - g_m)' G^-1 (g - g_m) <= R^2`. The only random element is the environment draw.

Four structural facts follow. They are not stated in the code, but they drive most of
what a stochastic extension changes.

**F1. Risk is rank one.** `Var[Y] = p(1-p)(Y_1 - Y_2)^2`, and `Y_1 - Y_2 = (1-c)(gbar0 - gbar1)`
with `gbar = sum_i w_i g_i`. The yield covariance matrix across varieties is
`p(1-p) d d'` with `d_i = y_1(g_i) - y_2(g_i)`, so every pair of varieties is perfectly
correlated (+1 or -1). Diversification in this model means *hedging*, never
*averaging out*. [num: identity holds to 3e-16]

**F2. A portfolio is equivalent to one "average" variety.** Because yield is linear in `g`,

```
U(w) = beta_bar' gbar - (gamma/2) p(1-p)(1-c)^2 (gbar0 - gbar1)^2,   beta_bar = p beta_1 + (1-p) beta_2
```

depends on the shares only through `gbar`. The market picks the best point of the
convex hull of available genotypes, and a candidate matters only insofar as it extends
that hull. Since `U` is linear in the mean, the optimum sits on an edge or vertex of the
hull's image, so **an optimal portfolio never uses more than two varieties**. [num: 0 of
200,000 random three-variety menus give a three-variety optimum.] This is also why the
three-asset interior branch in `get_exact_grid_stats` (`experiment_core.py:151`) is never
active: the 3x3 covariance determinant is always zero.

**F3. Under mean-variance, only two moments of the environment matter.** Replacing the
two-point environment by any continuous distribution of loadings `beta_t` with the same
mean and the same covariance `Omega = p(1-p)(1-c)^2 [[1,-1],[-1,1]]` gives identical
decisions. "Making the environment continuous" changes nothing by itself; what matters is
whether the covariance structure changes (Section 2, S1) or the objective stops being
mean-variance.

**F4. The ellipse is a selection-response set.** Under index selection on true breeding
values with intensity `i`, the response is `i G b / sqrt(b'Gb)`, whose locus over all
index weights `b` is exactly the ellipse boundary with `R = i`. With phenotypic
selection (phenotypic covariance `P`) the shape matrix becomes `G P^-1 G`. So the
breeding constraint already has a quantitative-genetics reading, and its stochastic
counterpart (realized vs expected response) is natural.

## 2. Sources of randomness worth adding

| | Source | Model | Key effect | Leaves unchanged |
|---|---|---|---|---|
| S1 | Richer environmental risk | `K` environments with probabilities `p_k`, or a common productivity shock, so `rank(Omega) = 2` | No perfect hedge; zero-variance portfolios disappear | Structure of the code (scenario-weighted sums) |
| S2 | Idiosyncratic yield noise (variety x year interaction, local disease, lodging) | `y_ie = beta_e' g_i + eps_ie`, `eps ~ N(0, sigma^2)` iid over varieties | Concentration penalty `(gamma/2) sigma^2 sum_i w_i^2`; any new variety has diversification value | BeatBest target |
| S3 | Breeding outcome uncertainty (Mendelian sampling, finite population, realized response) | `g_c = g_m + Delta + eta`, `eta ~ N(0, Sigma_eta)` | Certainty equivalence fails for PoB and MaxMarket; free disposal creates option value | BeatBest target (linear) |
| S4 | Information uncertainty at adoption (few trial years, prediction error) | farmers see `g_c_hat`; candidate gets extra variance `tau^2 w_c^2` | Novelty penalty, slower adoption, winner's curse in released varieties | |
| S5 | Parameter uncertainty (p, c shift with climate; 8 to 12 year release lag) | `p ~ Beta`, or a set `[p_lo, p_hi]` | Only matters across years or with robust / ambiguity-averse objectives | One-shot MV decisions if `E[p]` is known, since the environment is then marginally Bernoulli(`E[p]`) |
| S6 | Endogenous risk (pathogens adapt to widely grown varieties) | `Var(eps_i)` increasing in `w_i` | Monoculture penalty, stronger than S2 | |

S2 and S3 are the two with the largest conceptual payoff and the lowest implementation
cost, so the rest of this note focuses on them.

## 3. A unified linear-Gaussian formulation

```
y_it = beta_t' g_i + eps_it,     E[beta_t] = beta_bar,  Cov(beta_t) = Omega
Cov(eps_t) = sigma^2 [(1 - rho) I + rho 1 1']
g_c = g_m + Delta + eta,         Delta in the ellipse,  eta ~ N(0, Sigma_eta)
```

Given the genotypes,

```
E[Y]   = beta_bar' gbar
Var[Y] = gbar' Omega gbar  +  sigma_id^2 H(w)  +  rho sigma^2,     sigma_id^2 = (1 - rho) sigma^2,  H(w) = sum_i w_i^2
U      = beta_bar' gbar - (gamma/2) gbar' Omega gbar - (gamma/2) sigma_id^2 H(w)     (+ constant)
```

The current model is the special case `Omega` rank one, `sigma = 0`, `eta = 0`. Two
consequences:

* Noise that is common to all varieties (`rho = 1`) only adds a constant: it never
  changes a decision. What matters is the idiosyncratic part `sigma_id`.
* The Herfindahl term `H(w)` breaks F2: a mix of varieties is now strictly less risky
  than a single variety at `gbar`, and portfolios with three or more varieties appear.
  [num: 3% of optima use all three varieties at `sigma = 0.05`, 14% at `sigma = 0.10`.]

If exact expected utility is preferred over the mean-variance approximation, CARA utility
with normal `eps` gives a closed form that keeps the same Herfindahl term:
`CE = -(1/gamma) log( p exp(-gamma Y_1) + (1-p) exp(-gamma Y_2) ) - (gamma/2) sigma_id^2 H(w)`.

## 4. What changes per strategy

| Strategy | Under S2 (yield noise) | Under S3 (breeding noise) |
|---|---|---|
| BeatBest (max mean) | Target unchanged; evaluation changes | Target unchanged (objective linear in `g`) |
| BeatBestS (standalone MV) | Target unchanged (adds a constant) | Target can change (standalone variance is quadratic in `g`) |
| PoB | Shares more balanced; candidate adopted more often | Becomes a two-stage stochastic program: choose target, then the market re-optimizes per outcome |
| MaxMarket | KKT stationarity gains a `-sigma^2 w_i` term per asset | Natural objective becomes `E[w_c]` or `P(adopt)` |
| Clairvoyance | Must define what is foreseen (below) | Same |
| Baseline | Lower utility (two varieties, high `H`) | Unchanged |

**Clairvoyance needs a decision.** If the clairvoyant foresees only the environment type,
the benchmark variance gains `sigma^2 (p H_1 + (1-p) H_2)` and the clairvoyant also
diversifies within an environment. If it also foresees `eps`, the benchmark grows with
`sigma` (expected maximum of noisy yields), and the normalized gain used in
`compute()` (`best_possible_value` as denominator) becomes noise-driven. Foreseeing
the environment type only keeps the benchmark comparable.

## 5. Numerical findings

Default scenario of `app.py` (`p=0.5, c=0.1, gamma=21, r_g=-0.4, R=0.1, g_fix=(0.6,0.2), g_mut=(-0.1,0.2)`).
Gains are relative to the base portfolio. "Clone gain" is the gain from releasing an exact
copy of the mutable parent.

**Idiosyncratic noise (S2)**

| sigma | U base | PoB gain | BeatBest gain | Adoption region | Clone gain | PoB gain net of clone |
|---|---|---|---|---|---|---|
| 0.00 | 0.256 | 0.0377 | 0.0243 | 50% | 0.0000 | 0.0377 |
| 0.05 | 0.242 | 0.0375 | 0.0246 | 61% | 0.0020 | 0.0354 |
| 0.10 | 0.201 | 0.0367 | 0.0264 | 94% | 0.0090 | 0.0277 |
| 0.20 | 0.040 | 0.0635 | 0.0613 | 100% | 0.0452 | 0.0183 |

* The share of the breeding ellipse whose candidate the market adopts grows with
  `sigma`. This is robust: median over 150 random scenarios 50%, 74%, 100% for
  `sigma = 0, 0.1, 0.2`.
* Part of the gain is pure "one more variety" value with no genetic content (clone gain).
  Strategy comparisons under S2 should report gains net of the clone gain, otherwise
  every strategy looks better as `sigma` grows.
* In the default scenario the PoB advantage over BeatBest shrinks with `sigma`
  (0.0134 to 0.0022), but over random scenarios it shrinks in only 53% of cases. There is
  no general direction; it depends on how much of the PoB advantage comes from hedging
  environmental risk.

**Breeding outcome noise (S3)**, `eta ~ N(0, s^2 G)`, market re-optimizes after `g_c` is known:

| s | Strategy | V(target) | E[V] | P(adopt) | E[w_c] |
|---|---|---|---|---|---|
| 0.05 | PoB (deterministic target) | 0.0377 | 0.0374 | 0.98 | 0.36 |
| 0.05 | BeatBest | 0.0243 | 0.0251 | 0.87 | 0.36 |
| 0.10 | PoB (deterministic target) | 0.0377 | 0.0386 | 0.85 | 0.30 |
| 0.10 | BeatBest | 0.0243 | 0.0294 | 0.72 | 0.29 |

In this scenario the stochastic optimum coincides with the deterministic PoB target.

Over 150 random scenarios with `s = R`:

* **Noise hurts good targets and helps poor ones.** For targets with a large
  deterministic gain (> 0.02) noise lowers expected value in 81% of cases (concavity near
  the optimum). For targets with zero or small gain it raises expected value in 67 to 70%
  of cases: the
  market discards bad outcomes and adopts good ones, so breeding has option value. The
  deterministic model can never show this.
* **The deterministic PoB target is usually still near-optimal** (median regret 0), but in
  17% of scenarios its relative regret exceeds 5%. The five worst cases are those where the
  deterministic model finds no improving candidate and therefore aims at `g_m`, while the
  stochastic optimum aims at a boundary point with a 27 to 51% adoption probability.
* The median PoB advantage over BeatBest shrinks (0.0030 to 0.0024) because BeatBest
  targets gain more option value.

Caveats: one market model (three varieties), mean-variance utility, `s = R` is a heavy
noise case, and noisy outcomes can land outside the ellipse (which is realistic for
realized response but is part of the measured option value).

## 6. Implemented stochastic mode

### Decisions

| Question | Decision | Consequence in the code |
|---|---|---|
| Whose risk? | **The farm.** `w` are the area shares of a representative farm; `gamma` is the farmer's risk aversion. | `sigma` is the farm-level variety-specific shock and enters every variance as `sigma^2 sum_i w_i^2`. |
| What does Clairvoyance see? | **Everything**: environment, every farm-level shock `eps_i`, and the breeding outcome `eta`. | New full-information benchmark (`stochastic_core.clairvoyance_full_info`). |
| When is the breeding outcome known? | **Before adoption** (`timing='known'`, default). The reverse is available as `timing='unknown'`. | `known`: Monte Carlo over `eta` with re-optimized shares. `unknown`: closed form with `tau^2`. |

### Usage

```python
from experiment_core import ExperimentRunner
runner = ExperimentRunner(p, c, gamma, r_g, R, scenario_pairs,
                          stochastic=dict(sigma=0.10, breeding_sd=0.05, timing='known'))
text, data = runner.compute(strategy_keys=['Base', 'BeatBest', 'PoB', 'MaxMarket', 'Clairvoyance'])
```

`stochastic=None` (or `sigma = breeding_sd = 0`) reproduces the deterministic results exactly
(checked byte for byte on the text report). Optional keys: `n_draws` (200), `n_states` (100),
`n_clair_draws` (50), `max_targets` (1500), `seed`. The Streamlit app (`stream.py`) has
sliders for `sigma`, `breeding_sd` and the timing. The text report gains a table with
`E[MV]`, `sd(MV)`, `P(adopt)` and `E[w_c]` per strategy.

### What each strategy does

| Strategy | `timing='known'` (outcome seen before adoption) | `timing='unknown'` (adopt on the target) |
|---|---|---|
| Base | Gurobi QP with `sigma` | same |
| BeatBest | Gurobi, target unchanged (objective linear in `g`) | same |
| BeatBestS | Gurobi, target unchanged (noise adds a constant) | same |
| PoB | target maximizing `E_eta[V(t + eta)]`, Monte Carlo search | Gurobi with `sigma^2 H(w) + tau^2 w_c^2` (exact) |
| MaxMarket | target maximizing `E_eta[w_c(t + eta)]`, Monte Carlo search | Gurobi MPEC with the extra terms in the KKT conditions |
| Clairvoyance | full information, same in both timings | same |

Evaluation: under `known`, each strategy's target is evaluated with fresh breeding draws
(separate from those used to choose targets, which avoids optimizer's-curse bias), with
farm shares re-optimized per draw. Under `unknown`, the ex-ante value includes
`tau^2 = p b1' S b1 + (1-p) b2' S b2`, `S = s^2 G`.

Monte Carlo searches use the feasible grid points (coarse subsample, then local grid
refinement), followed by a continuous pattern search inside the ellipse. They use the
same box `[-1, 1]^2` as the Gurobi models (`_get_ellipse_bounds`), which the plotting grid
(up to 1.2) does not.

Clairvoyance with full information has a closed form per state: with state-contingent
shares the portfolio yield `Y_k` can be anything between the worst and best variety in
state `k`, and maximizing `mean - gamma/2 var` gives `Y_k = clip(mu + 1/gamma, lo_k, hi_k)`.
This reproduces the Gurobi hindsight model exactly in the deterministic case.

### Files

* `stochastic_core.py`: spec, exact batched market solver, evaluation, Monte Carlo
  target search, full-information Clairvoyance.
* `experiment_core.py`, `GurobiPortfolioOptimizer.py`: `sigma^2` and `tau^2` terms in all
  default strategies (B4P, B4M, BeatBestS, MaxMarket MPEC, benchmarks, grid statistics).
  Not extended: `strat_max_share` and the SplitEnv / SplitEffort strategies (their targets
  stay deterministic; they are still evaluated under the stochastic model).
* `stochastic_yield/test_stochastic.py`: 8 consistency tests, each comparing two
  independent computations (Gurobi vs enumeration, closed form vs simulation, box
  Clairvoyance vs Gurobi hindsight, MPEC shares vs true market response, value of
  information >= 0, inactive spec = deterministic).
* `stochastic_yield/compare_timing.py`: the tables below.

### Results

Default scenario (`p=0.5, c=0.1, gamma=21, r_g=-0.4, R=0.1`), `s = R/2`. Gain is
`E[MV] - base`; the percentage is the share of the Clairvoyance gain.

| Configuration | BeatBest | PoB | MaxMarket | Clairvoyance | PoB `E[w_c]` |
|---|---|---|---|---|---|
| deterministic | 0.0243 (44%) | 0.0377 (69%) | 0.0000 (0%) | 0.0547 | 0.37 |
| sigma = 0.10 | 0.0263 (19%) | 0.0367 (26%) | 0.0357 (26%) | 0.1392 | 0.39 |
| s = R/2, known | 0.0246 (39%) | 0.0368 (59%) | 0.0353 (56%) | 0.0628 | 0.35 |
| s = R/2, unknown | 0.0201 (32%) | 0.0346 (55%) | 0.0044 (7%) | 0.0628 | 0.36 |
| sigma = 0.10, s = R/2, known | 0.0277 (20%) | 0.0366 (26%) | 0.0351 (25%) | 0.1400 | 0.35 |
| sigma = 0.10, s = R/2, unknown | 0.0235 (17%) | 0.0333 (24%) | 0.0328 (23%) | 0.1400 | 0.36 |

30 random scenarios (`python stochastic_yield/compare_timing.py --sweep 30`, about 2
minutes), `sigma = 0.10`, `s = R/2`, medians:

| | deterministic | known | unknown |
|---|---|---|---|
| PoB gain | 0.071 | 0.079 | 0.061 |
| BeatBest gain | 0.048 | 0.059 | 0.042 |
| MaxMarket gain | 0.028 | 0.078 | 0.057 |
| Clairvoyance gain | 0.124 | 0.227 | 0.227 |
| PoB gain as share of Clairvoyance | 0.65 | 0.39 | 0.26 |
| PoB candidate share `E[w_c]` | 0.63 | 0.51 | 0.41 |
| PoB better than BeatBest | 70% | 77% | 77% |
| PoB minus BeatBest gain | 0.0009 | 0.0020 | 0.0020 |

* **Knowing the outcome before adoption is worth a lot**: for PoB the gain under
  `known` is never lower beyond Monte Carlo noise (median difference 0.014, up to 0.11),
  roughly a fifth of the median PoB gain. Adopting before the outcome is known lowers the candidate's share
  (0.51 to 0.41) because farms treat the uncertain variety as riskier (novelty penalty).
* **The PoB target hardly depends on the timing** (median shift 0.009 genotype units);
  the timing mainly changes how much of it farms adopt.
* **PoB keeps its edge over BeatBest** and slightly widens it under farm-level risk
  (better in 77% of scenarios vs 70% deterministic).
* **MaxMarket is defined differently per timing**: `known` maximizes the expected share
  over outcomes (and lands close to PoB); `unknown` keeps the MPEC that maximizes the share
  subject to a minimal improvement, which tends to give small utility gains. Compare
  their shares, not their gains.

## 7. Farm-level risk: implications and open points

1. **Size of `sigma`.** At farm level the variety-specific shock includes the variety x
   location x year interaction and, when each variety sits on its own field, field-level
   error. Neither averages out within a farm, so `sigma` is larger than it would be for
   national supply. Shocks shared by all varieties on the farm (weather beyond the
   environment type, management) do not change decisions and need not be modeled.
   Calibration: variance components of variety x location x year interaction and plot
   residual in multi-environment trial data of the target crop.
2. **Normalized gains shrink.** Because Clairvoyance sees every farm-level shock, its gain
   grows quickly with `sigma` (0.055 to 0.139 in the default scenario), while breeding
   strategies cannot exploit those shocks. Gains expressed as a share of Clairvoyance
   (`compute()` with `easy_base=False`, used in the plots) fall from 69% to 26% for PoB.
   Absolute gains are the more stable comparison.
3. **Clairvoyance does not always pick the best variety.** Under mean-variance, yields
   above `mu + 1/gamma` lower utility, so the clairvoyant deliberately caps them. This
   already holds in the existing Gurobi hindsight model (interior shares in environment 1)
   and binds more often with farm-level noise. An expected-utility or "always pick the
   best" benchmark would avoid it.
4. **Number of varieties.** With `sigma > 0` every additional variety lowers farm risk,
   so the model favors spreading area. Real farms grow few varieties (seed logistics,
   management), so a fixed cost per variety or a cap on the number of varieties may be
   needed before drawing conclusions about adoption.
5. **Market share = representative farm.** With identical farms, a variety's market
   share equals its share on the representative farm. Farms that differ in `p`,
   `gamma` or `sigma` would make market share an average of farm-level shares, which
   smooths adoption. Not implemented.
6. **`unknown` timing treats `eta` as one-year risk.** The breeding outcome is permanent
   for a variety, so over several seasons it does not average out the way environment
   and farm shocks do. A multi-year objective would penalize it more than `tau^2` does.
