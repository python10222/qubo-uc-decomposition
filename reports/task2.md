# Task 2: QUBO builder

Implemented `src/qubo.py` with `build_qubo(instance, T, B, lambda1, lambda2,
encoding="linear", penalty="slack")`. The returned object exposes `Q`, `offset`,
`variables`, `num_variables`, `encode(u, grid_indices)`, `decode(bits)`,
`true_cost(u, p)`, `is_feasible(u, p)`, and `energy(bits)`.

The upper-triangular convention is E = offset + Σ_(i≤j) Q[i,j] x_i x_j;
there is no implicit factor of two. `energy` evaluates this expanded polynomial,
including its constant, using compensated summation. The physical objective is
reused from Task 1. Start-up is S_it u_it(1-u_i,t-1), with the original initial
status at hour 1. Off-unit penalties apply once per set output bit. Slack uses
q/(2^B−1), with M_t computed by the loader using integer bit lengths. Decoding
returns rational powers and surplus; grid membership and slack checks are exact.
Float inputs to `is_feasible` are interpreted as decimal values, so callers should
use decoded Fractions for noninteger grid levels.

Must-run commitments are substituted as constants and logged. Fixed-output
unit-hours get zero output bits. In general the free variable count is
Σ_it (1−must_run_it+B_it) + Σ_t M_t. All ten committed benchmarks have no such
reductions, and the requested N T (1+B) + Σ_t M_t identity passes for all 50 cases.
Only the requested linear/slack model is implemented; other options raise errors.
T selects the first T hours and preserves the original initial status.

## Validation

- All 50 Task 1 discretized optima (ten instances, B=2–6) encode with zero
  exact slack residual and feasible dispatch. Expanded energy agrees with
  `results/baselines.csv` z_disc within relative 1e-6; maximum observed relative
  error: **3.79621509839e-12**. Saved certificates are audited by the
  existing Task 1 tests; no full benchmark optimization was rerun.
- Exhaustive enumeration includes every commitment, output and slack bit on
  2-unit and 3-unit, 2-hour reductions at B=2 (16 and 22 free bits). Large
  penalties λ1=λ2=10^6 give a feasible ground state matching the proven-optimal
  Task 1 SCIP discretized model. Tiny penalties λ1=λ2=10^−9 give an infeasible
  ground state below z_disc. This does not claim exactness of the table's
  estimated weights on full benchmarks.
- Reduction decision: take the first 2 or 3 actual units and first 2 hours of
  `10_0_1_w`, keeping their limits, fitted costs, start-up costs and initial
  status. The full-system demand exceeds reduced capacity, so set reduced
  demand to reduced capacity minus 1 MW per hour. This explicitly derived
  stress case keeps surplus bits small enough for exhaustive enumeration;
  it is not a new OR-LIB benchmark or a replacement for its raw data.
- Additional tests compare the expanded polynomial against the direct physical
  cost and penalties on 200 seeded states, with hourly costs, initial status,
  must-run transitions and fixed-output units. They also cover horizon slicing,
  zero slack bits, invalid inputs and exact off-grid rejection.
- Run `python -m pytest -q` before `python -m experiments.qubo_report`.
- Final full suite: **84 passed** (including 54 new QUBO tests). The initial
  full run had 83 passes and one pre-existing audit failure from last-digit
  least-squares differences against saved JSON. The audit test now compares
  only fitted coefficients and residuals with rtol=1e-12, atol=1e-10;
  raw data and every other preprocessing field remain exact. No production
  loader, model, raw data, baseline or dependency changes were needed.
- Commands used the retained project Python environment at
  `/workspace/.cloud-setup/qubo-uc-decomposition/venv/bin/python`, with
  `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. No unresolved test failures.

## Coefficients and counts (T=24)

λ1 = 10 max_it(a_it+S_it); λ2 = 10 b_max/Δ_min, with b_max the fitted linear
fuel coefficient and Δ_min the smallest positive unit-hour output step.
Min/max below are absolute **nonzero** upper-triangular coefficients, including
linear diagonal terms and excluding the offset. Absent matrix entries are zero;
the minimum over the entire matrix would therefore be zero. Values have nine
significant digits. Reproduce with `python -m experiments.qubo_report`.

| Instance | B | Variables | λ1 | λ2 | Min nonzero abs Q | Max abs Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10_0_1_w | 2 | 990 | 35240.1975 | 35.4124865 | 15.7388829 | 66545507.9 |
| 10_0_1_w | 3 | 1258 | 35240.1975 | 82.6291353 | 6.74523553 | 141526913 |
| 10_0_1_w | 4 | 1522 | 35240.1975 | 177.062433 | 3.14777658 | 279281617 |
| 10_0_1_w | 5 | 1786 | 35240.1975 | 365.929028 | 1.5231177 | 555156077 |
| 10_0_1_w | 6 | 2050 | 35240.1975 | 743.662217 | 0.749470615 | 1.10706724e+09 |
| 10_0_2_w | 2 | 1008 | 42765.8333 | 31.3208045 | 13.9203576 | 59925691.6 |
| 10_0_2_w | 3 | 1276 | 42765.8333 | 73.0818772 | 5.96586752 | 199985807 |
| 10_0_2_w | 4 | 1543 | 42765.8333 | 156.604022 | 2.78407151 | 425968465 |
| 10_0_2_w | 5 | 1807 | 42765.8333 | 323.648313 | 1.34713138 | 839883002 |
| 10_0_2_w | 6 | 2072 | 42765.8333 | 657.736894 | 0.662874169 | 1.74354955e+09 |
| 10_0_3_w | 2 | 993 | 41031.037 | 29.6414635 | 13.1739838 | 60759677.9 |
| 10_0_3_w | 3 | 1258 | 41031.037 | 69.1634148 | 5.64599305 | 119919808 |
| 10_0_3_w | 4 | 1526 | 41031.037 | 148.207317 | 2.63479676 | 418993558 |
| 10_0_3_w | 5 | 1792 | 41031.037 | 306.295123 | 1.27490166 | 831759711 |
| 10_0_3_w | 6 | 2056 | 41031.037 | 622.470733 | 0.627332561 | 1.65265502e+09 |
| 10_0_4_w | 2 | 992 | 44051.7527 | 31.8247205 | 14.1443202 | 59412481.6 |
| 10_0_4_w | 3 | 1258 | 44051.7527 | 74.2576812 | 6.06185153 | 126145481 |
| 10_0_4_w | 4 | 1522 | 44051.7527 | 159.123603 | 2.82886405 | 248900930 |
| 10_0_4_w | 5 | 1788 | 44051.7527 | 328.855445 | 1.36880518 | 870776302 |
| 10_0_4_w | 6 | 2055 | 44051.7527 | 668.319131 | 0.673539059 | 1.74796374e+09 |
| 10_0_5_w | 2 | 988 | 42430.7305 | 35.165824 | 15.6292551 | 66658147.8 |
| 10_0_5_w | 3 | 1253 | 42430.7305 | 82.0535892 | 6.69825218 | 128633878 |
| 10_0_5_w | 4 | 1521 | 42430.7305 | 175.82912 | 3.12585102 | 279832985 |
| 10_0_5_w | 5 | 1786 | 42430.7305 | 363.380181 | 1.51250856 | 556282546 |
| 10_0_5_w | 6 | 2050 | 42430.7305 | 738.482303 | 0.744250243 | 1.10934278e+09 |
| 20_0_1_w | 2 | 1738 | 42954.9344 | 37.7083079 | 16.7592479 | 268302420 |
| 20_0_1_w | 3 | 2250 | 42954.9344 | 87.9860517 | 7.18253484 | 580979499 |
| 20_0_1_w | 4 | 2759 | 42954.9344 | 188.541539 | 3.35184959 | 1.9072022e+09 |
| 20_0_1_w | 5 | 3265 | 42954.9344 | 389.652515 | 1.6218627 | 3.75635568e+09 |
| 20_0_1_w | 6 | 3771 | 42954.9344 | 791.874466 | 0.798059426 | 7.77045206e+09 |
| 20_0_2_w | 2 | 1727 | 45303.4438 | 34.2569242 | 15.2252996 | 243464478 |
| 20_0_2_w | 3 | 2236 | 45303.4438 | 79.9328231 | 6.52512842 | 497307912 |
| 20_0_2_w | 4 | 2740 | 45303.4438 | 171.284621 | 3.04505993 | 980019359 |
| 20_0_2_w | 5 | 3244 | 45303.4438 | 353.988217 | 1.47341609 | 1.94685481e+09 |
| 20_0_2_w | 6 | 3749 | 45303.4438 | 719.395408 | 0.725014268 | 4.20144901e+09 |
| 20_0_3_w | 2 | 1741 | 39917.8925 | 30.6820974 | 13.6364877 | 231379560 |
| 20_0_3_w | 3 | 2254 | 39917.8925 | 71.5915606 | 5.84420903 | 484454819 |
| 20_0_3_w | 4 | 2763 | 39917.8925 | 153.410487 | 2.72729755 | 1.5370864e+09 |
| 20_0_3_w | 5 | 3268 | 39917.8925 | 317.04834 | 1.3196601 | 3.1516088e+09 |
| 20_0_3_w | 6 | 3772 | 39917.8925 | 644.324046 | 0.649356559 | 6.25823429e+09 |
| 20_0_4_w | 2 | 1752 | 41659.5397 | 32.426805 | 14.4119133 | 238338458 |
| 20_0_4_w | 3 | 2259 | 41659.5397 | 75.662545 | 6.17653429 | 778604623 |
| 20_0_4_w | 4 | 2764 | 41659.5397 | 162.134025 | 2.88238267 | 1.5586125e+09 |
| 20_0_4_w | 5 | 3269 | 41659.5397 | 335.076985 | 1.39470129 | 3.26423477e+09 |
| 20_0_4_w | 6 | 3774 | 41659.5397 | 680.962905 | 0.686281587 | 6.67927312e+09 |
| 20_0_5_w | 2 | 1727 | 41402.795 | 34.3168455 | 15.2519313 | 242672137 |
| 20_0_5_w | 3 | 2236 | 41402.795 | 80.0726394 | 6.53654199 | 495553969 |
| 20_0_5_w | 4 | 2740 | 41402.795 | 171.584227 | 3.05038626 | 976485942 |
| 20_0_5_w | 5 | 3245 | 41402.795 | 354.607403 | 1.47599335 | 2.09869346e+09 |
| 20_0_5_w | 6 | 3749 | 41402.795 | 720.653755 | 0.726282444 | 4.18480886e+09 |
