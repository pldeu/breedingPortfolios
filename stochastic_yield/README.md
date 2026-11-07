# From a deterministic to a stochastic yield model

This note explores what it would mean to replace the yield model used in
`experiment_core.py` and `GurobiPortfolioOptimizer.py` by a stochastic one: which
sources of randomness are worth adding, what they change in each strategy, what they
leave unchanged, and how to implement them. Claims marked [num] are checked with
`prototype.py` (numpy only, no Gurobi needed):

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

## 6. Implementation path in this codebase

**Step 0, centralize the yield model.** The formula `y1 = g0 + c g1, y2 = c g0 + g1` is
re-implemented in roughly 80 lines across `experiment_core.py` and
`GurobiPortfolioOptimizer.py`. A single `YieldModel` object exposing `mean(g)`,
`cov(g_i, g_j)`, `env_yields(g)` and `sample(g, n)` is a prerequisite for any of the
steps below and reduces the risk of inconsistent variants.

**Step 1, S2 in closed form (low effort).** Add a parameter `sigma_id` and add
`sigma_id^2` on the diagonal only:

* `_get_yields_and_stats` (`experiment_core.py:348`): `var_y += sigma_id**2`.
* `get_exact_grid_stats` (`experiment_core.py:66`): `v_ff, v_mm, v_cc += sigma_id**2`;
  covariances unchanged. The interior three-asset branch then becomes active.
* `_get_standard_output` (`experiment_core.py:436`): `Sigma += sigma_id**2 * I`.
* `w_fix_grid` formulas (`experiment_core.py:247`, `:970`): add `sigma_id**2` to `Var_fix` and `Var_grid`.
* `calculate_stats_from_dict` (`experiment_core.py:410`) and `strat_hindsight`
  (`GurobiPortfolioOptimizer.py:944`): add `sigma_id**2 * (p H_1 + (1-p) H_2)`.
* `strat_B4P` / `change_gamma_B4P` (`GurobiPortfolioOptimizer.py:58`, `:137`): add
  `sigma_id**2 * (wf*wf + wm*wm + wc*wc)` to the variance. This is convex, so solve
  times should not suffer.
* `strat_max_investability` (`GurobiPortfolioOptimizer.py:383`) and
  `_solve_parent_benchmark`: subtract `sigma_id**2 * w_i` in each stationarity
  constraint and add the `H(w)` term to the utility constraint.

**Step 2, S1 via more environments.** Generalize the two-term `p, 1-p` sums to
`sum_k p_k`. Gurobi handles this as a scenario model with no new nonconvexity, and it
lets you add, for example, a severe year that scales all yields down.

**Step 3, S3 (breeding noise).** Two options:

* *Grid / Monte Carlo (cheap):* `E[V(t + eta)]` is the `v_port` surface from
  `get_exact_grid_stats` averaged over `eta`, which for Gaussian `eta` is a blur of that
  surface. The grid must extend about `3 s` beyond the ellipse.
* *Sample average approximation in Gurobi:* first-stage target `(gx, gy)` in the
  ellipse; `K` outcome scenarios with their own weights `w^k` and bilinear terms
  `wc^k (gx + eta_k)`. Model size grows linearly in `K` but the problem stays nonconvex,
  so expect `K` in the tens to be practical.

Outputs then become distributions (expected gain, sd, `P(adopt)`, `E[w_c]`, quantiles)
rather than points, which affects `subplot_registry.py` and the text report.

**Step 4, S4.** Add `tau^2 w_c^2` to the variance seen by the market at adoption.

**Step 5, simulation-based evaluation.** Simulate years and farms with the chosen
portfolios to validate the analytics and to report downside measures (CVaR, probability
of yield below a threshold), which mean-variance hides.

## 7. Modeling decisions to make first

1. **Whose risk?** A farm faces temporal, spatial and idiosyncratic risk; national supply
   faces mostly temporal risk because spatial and plot-level noise average out over
   many farms. If `w_i` are aggregate area shares, `sigma_id` should be the variety x
   year interaction (which does not average out), not plot error. `gamma` should match
   the same level.
2. **Timing of information.** Is the breeding outcome known before the market adopts
   (recourse, S3) or only afterwards (extra risk, S4)? The two pull adoption in
   opposite directions.
3. **Definition of Clairvoyance** (Section 4).
4. **Calibration.** `sigma_id^2` relative to the structured GxE implied by `c` can be
   estimated from variety x year and variety x location x year variance components in
   multi-environment trial data of the target crop. `Sigma_eta` can be set from the
   segregation variance of the crosses (roughly proportional to `G`) and the number of
   lines tested per cycle.
