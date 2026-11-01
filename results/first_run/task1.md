# Task 1: continuous and discretized UC baselines

Generated 2026-11-01T22:04:29.678794+03:00. Base commit: `39b64a31950d826514795f4f9ef115ddfa85964c`.

## Data and preprocessing

All ten committed OR-LIB files were loaded using gzip magic-byte detection. Their UnitCommitment.jl input format version is **0.3**. None has a generator `Type` field; every generator is treated as thermal, with an explicit per-generator audit entry. Each instance has one bus, 24 hourly loads, and 10 or 20 units.

The raw cost form is a convex **piecewise-linear total-cost curve with five breakpoints** per unit. The breakpoints lie on convex quadratics to floating-point precision. We fit `a + b*p + c*p²` by least squares at the original breakpoints, enforcing `c >= 0`, then use those same coefficients in both reference models. The intercept `a` is charged only when on; the first breakpoint cost is not used as a second no-load charge. Linear curves use `c = 0`.

Pmin is rounded up, Pmax down, and aggregate bus demand up to q = 1 MW. All benchmark capacity endpoints were already integral; all 240 hourly demands rounded upward. Original and rounded arrays, per-unit/per-hour coefficients, RMSE, maximum absolute fit errors, raw file SHA-256 hashes, assumptions, and every dropped field (including its value) are in `results/preprocessing.json`. All raw generator parameters are static in these ten files; demand varies hourly. No benchmark unit is marked must-run; the loader/model support hourly must-run flags.

| Instance | Units | Maximum breakpoint fit error ($) | Maximum unit fit RMSE ($) | Dropped-field entries |
|---|---:|---:|---:|---:|
| 10_0_1_w | 10 | 1.455e-11 | 1.177e-11 | 81 |
| 10_0_2_w | 10 | 1.637e-11 | 1.177e-11 | 81 |
| 10_0_3_w | 10 | 1.819e-11 | 1.621e-11 | 81 |
| 10_0_4_w | 10 | 1.819e-11 | 1.271e-11 | 81 |
| 10_0_5_w | 10 | 1.364e-11 | 1.168e-11 | 81 |
| 20_0_1_w | 20 | 1.637e-11 | 1.403e-11 | 161 |
| 20_0_2_w | 20 | 1.637e-11 | 1.327e-11 | 161 |
| 20_0_3_w | 20 | 1.455e-11 | 1.190e-11 | 161 |
| 20_0_4_w | 20 | 1.819e-11 | 1.509e-11 | 161 |
| 20_0_5_w | 20 | 1.455e-11 | 1.162e-11 | 161 |

Dropped fields in every instance: generator `Bus` (network allocation), `Initial power (MW)` (only initial on/off status is retained), `Minimum uptime (h)`, `Minimum downtime (h)`, `Reserve eligibility`, `Startup delays (h)`, `Ramp up limit (MW)`, `Ramp down limit (MW)`, and the complete `Reserves` object. Each raw generator has one startup category; its cost is retained. For inputs with multiple categories, the loader takes the cheapest hourly cost and logs the discarded category structure. Initial-status duration is reduced to its sign; minimum up/down carryover is outside scope. Source and license text are retained as metadata.

## Models and optimality

Both models retain binary commitment, exact startup transitions (including initial status), demand ≥ load, output bounds, and must-run constraints. The continuous model uses SCIP with a convex perspective epigraph `c*p² <= v*u`, exact for binary commitment. The discrete model uses an integer index `k = 0,…,2^B_i−1`, equivalent to all binary-weighted bit patterns, with `k <= (2^B_i−1)*u` and `p = Pmin*u + Δ_i*k`. Convex adjacent-level chord inequalities evaluate the quadratic exactly at every grid point, so SCIP solves an exact MILP for this restricted quadratic model.

For the full discrete runs, SCIP certifies hourly dispatch patterns in increasing production cost. At fixed hourly commitments, dispatch is independent of every other hour. A separate SCIP master selects generated patterns with the original startup costs and initial status. Each hour also has an optimistic unseen-pattern option, priced at a proven lower bound on all patterns not yet generated and allowing arbitrary commitments. This makes the master a relaxation of the entire grid model. Whenever its optimum uses an unseen option, another pattern is generated for that hour. Termination requires a proven-optimal master using only known, feasible patterns: its lower bound then equals a feasible full-model cost and proves global optimality. No bit pattern is removed from the full feasible set. Every pricing solve and master solve uses SCIP and retains its certificate.

Discrete solve time is the sum of all hourly pricing and master SCIP solve times. Build time includes all hourly models and master rebuilds. Component timings, pattern counts, master bounds, and the final empty unseen-option set are in each run JSON. The direct 24-hour grid MILP remains available through `solve(build_model(instance, B))` and is cross-checked against pattern generation and enumeration on small instances.

`B_i = min(B, floor(log2((Pmax−Pmin)/q + 1)))`; `Δ_i = q*floor((Pmax−Pmin)/(q*(2^B_i−1)))`. No bit pattern is excluded. For zero-width ranges, B_i = 0 and p = Pmin*u (step stored as 0). Bit depths and steps are recorded by generator in every CSV row; hourly vectors are used if they vary. The prescribed integer steps need not reach Pmax, and grids for successive B are not necessarily nested. Consequently G_disc need not decrease monotonically with B.

Only SCIP `optimal` status with reported gap exactly zero is accepted as a baseline. This is a numerical optimality certificate under SCIP tolerances, not an exact-arithmetic proof. Relative and absolute stopping gaps are zero; feasibility tolerance is 1e-8. Costs are recomputed from the returned schedules, and demand, bounds, must-run, startup, and grid membership are checked independently. Full schedules, solver bounds, versions, CPU model, settings, and timings are retained in `results/runs/`. Continuous results are solved once per instance and reused across its five rows. Build time and SCIP solve time are separate.

The runner used 3 worker processes with one SCIP thread per solve. Timing reflects concurrent worker contention and is not a solver speed comparison.

## Results

**The required B=6 grid for `20_0_2_w` is infeasible.** Its total encodable capacity is 2,752 MW, while rounded demands in hours 10–13 are 2,860, 2,874, 2,912, and 2,862 MW (shortfalls 108, 122, 160, and 110 MW). SCIP proves the direct 24-hour model infeasible. This is also certified independently by the capacity bound; there is no finite z_disc or G_disc to report. The grid rule is preserved. The corresponding run JSON contains the infeasibility certificate and SCIP timing.

| Instance | B | z_cont | z_disc | G_disc ($) | G_disc (%) | cont seconds | disc seconds | cont gap | disc gap | B_i (g0, g1, …) | disc status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 10_0_1_w | 2 | 1893764.135291 | 1908771.638632 | 15007.503341 | 0.792470 | 2.436 | 14.254 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2 | optimal |
| 10_0_1_w | 3 | 1893764.135291 | 1908296.162652 | 14532.027361 | 0.767362 | 2.436 | 12.602 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3 | optimal |
| 10_0_1_w | 4 | 1893764.135291 | 1927351.844178 | 33587.708887 | 1.773595 | 2.436 | 9.490 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4 | optimal |
| 10_0_1_w | 5 | 1893764.135291 | 1963818.256388 | 70054.121097 | 3.699200 | 2.436 | 7.351 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5 | optimal |
| 10_0_1_w | 6 | 1893764.135291 | 1974568.394544 | 80804.259253 | 4.266860 | 2.436 | 5.380 | 0 | 0 | 6,6,5,6,5,6,6,6,6,6 | optimal |
| 10_0_2_w | 2 | 1278409.952444 | 1291746.457394 | 13336.504950 | 1.043210 | 1.670 | 18.763 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2 | optimal |
| 10_0_2_w | 3 | 1278409.952444 | 1287932.831683 | 9522.879239 | 0.744900 | 1.670 | 12.496 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3 | optimal |
| 10_0_2_w | 4 | 1278409.952444 | 1288690.445486 | 10280.493042 | 0.804162 | 1.670 | 8.986 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4 | optimal |
| 10_0_2_w | 5 | 1278409.952444 | 1311299.154376 | 32889.201932 | 2.572665 | 1.670 | 7.846 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5 | optimal |
| 10_0_2_w | 6 | 1278409.952444 | 1324360.811906 | 45950.859462 | 3.594376 | 1.670 | 7.927 | 0 | 0 | 5,6,6,6,6,6,6,6,6,6 | optimal |
| 10_0_3_w | 2 | 1811059.541008 | 1825020.359964 | 13960.818956 | 0.770865 | 2.536 | 12.815 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2 | optimal |
| 10_0_3_w | 3 | 1811059.541008 | 1821161.530364 | 10101.989356 | 0.557794 | 2.536 | 13.263 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3 | optimal |
| 10_0_3_w | 4 | 1811059.541008 | 1827711.542949 | 16652.001941 | 0.919462 | 2.536 | 10.747 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4 | optimal |
| 10_0_3_w | 5 | 1811059.541008 | 1857806.396489 | 46746.855481 | 2.581188 | 2.536 | 5.435 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5 | optimal |
| 10_0_3_w | 6 | 1811059.541008 | 1902654.948005 | 91595.406997 | 5.057559 | 2.536 | 3.965 | 0 | 0 | 6,6,5,6,6,6,6,6,6,6 | optimal |
| 10_0_4_w | 2 | 1846103.661369 | 1858109.535719 | 12005.874350 | 0.650336 | 5.716 | 16.473 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2 | optimal |
| 10_0_4_w | 3 | 1846103.661369 | 1860027.276396 | 13923.615027 | 0.754216 | 5.716 | 21.514 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3 | optimal |
| 10_0_4_w | 4 | 1846103.661369 | 1861712.442379 | 15608.781010 | 0.845499 | 5.716 | 9.320 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4 | optimal |
| 10_0_4_w | 5 | 1846103.661369 | 1897345.760245 | 51242.098876 | 2.775689 | 5.716 | 8.916 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5 | optimal |
| 10_0_4_w | 6 | 1846103.661369 | 1912872.565907 | 66768.904538 | 3.616747 | 5.716 | 7.639 | 0 | 0 | 6,5,6,6,5,6,6,6,6,6 | optimal |
| 10_0_5_w | 2 | 1902335.717430 | 1912574.796645 | 10239.079215 | 0.538237 | 3.324 | 15.859 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2 | optimal |
| 10_0_5_w | 3 | 1902335.717430 | 1914691.125692 | 12355.408262 | 0.649486 | 3.324 | 16.360 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3 | optimal |
| 10_0_5_w | 4 | 1902335.717430 | 1923427.313890 | 21091.596460 | 1.108721 | 3.324 | 16.163 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4 | optimal |
| 10_0_5_w | 5 | 1902335.717430 | 1944278.081027 | 41942.363597 | 2.204782 | 3.324 | 9.311 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5 | optimal |
| 10_0_5_w | 6 | 1902335.717430 | 1950847.221774 | 48511.504344 | 2.550102 | 3.324 | 10.209 | 0 | 0 | 6,5,5,6,6,6,6,6,6,6 | optimal |
| 20_0_1_w | 2 | 2906313.315851 | 2917799.247501 | 11485.931650 | 0.395206 | 6.626 | 56.965 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 | optimal |
| 20_0_1_w | 3 | 2906313.315851 | 2921429.341002 | 15116.025151 | 0.520110 | 6.626 | 57.685 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | optimal |
| 20_0_1_w | 4 | 2906313.315851 | 2947011.975869 | 40698.660018 | 1.400353 | 6.626 | 93.249 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 | optimal |
| 20_0_1_w | 5 | 2906313.315851 | 3000298.108086 | 93984.792235 | 3.233815 | 6.626 | 72.710 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | optimal |
| 20_0_1_w | 6 | 2906313.315851 | 3004350.225250 | 98036.909399 | 3.373240 | 6.626 | 14.836 | 0 | 0 | 6,6,6,6,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 | optimal |
| 20_0_2_w | 2 | 3843567.627477 | 3850514.963487 | 6947.336010 | 0.180752 | 7.999 | 36.389 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 | optimal |
| 20_0_2_w | 3 | 3843567.627477 | 3862680.649018 | 19113.021541 | 0.497273 | 7.999 | 40.962 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | optimal |
| 20_0_2_w | 4 | 3843567.627477 | 3893973.044906 | 50405.417429 | 1.311423 | 7.999 | 36.719 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 | optimal |
| 20_0_2_w | 5 | 3843567.627477 | 3996678.503678 | 153110.876201 | 3.983561 | 7.999 | 15.165 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | optimal |
| 20_0_2_w | 6 | 3843567.627477 |  |  |  | 7.999 | 0.053 | 0 |  | 6,6,5,6,5,6,6,6,5,6,6,6,6,6,6,6,6,6,6,6 | infeasible |
| 20_0_3_w | 2 | 2922331.474494 | 2934502.394353 | 12170.919859 | 0.416480 | 9.511 | 121.721 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 | optimal |
| 20_0_3_w | 3 | 2922331.474494 | 2940277.550154 | 17946.075660 | 0.614101 | 9.511 | 48.566 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | optimal |
| 20_0_3_w | 4 | 2922331.474494 | 2955285.768397 | 32954.293903 | 1.127671 | 9.511 | 86.648 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 | optimal |
| 20_0_3_w | 5 | 2922331.474494 | 2969310.186772 | 46978.712278 | 1.607576 | 9.511 | 86.260 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | optimal |
| 20_0_3_w | 6 | 2922331.474494 | 2996219.725241 | 73888.250747 | 2.528401 | 9.511 | 79.873 | 0 | 0 | 6,6,6,6,6,6,5,6,6,6,6,6,6,6,6,6,6,6,6,6 | optimal |
| 20_0_4_w | 2 | 2445908.491601 | 2459430.308372 | 13521.816771 | 0.552834 | 5.999 | 36.375 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 | optimal |
| 20_0_4_w | 3 | 2445908.491601 | 2471148.559290 | 25240.067689 | 1.031930 | 5.999 | 30.268 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | optimal |
| 20_0_4_w | 4 | 2445908.491601 | 2481668.840620 | 35760.349019 | 1.462048 | 5.999 | 28.128 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 | optimal |
| 20_0_4_w | 5 | 2445908.491601 | 2536273.417228 | 90364.925627 | 3.694534 | 5.999 | 45.087 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | optimal |
| 20_0_4_w | 6 | 2445908.491601 | 2543494.363527 | 97585.871926 | 3.989760 | 5.999 | 39.121 | 0 | 0 | 6,5,6,6,5,6,5,6,6,6,6,6,6,6,6,6,6,6,6,6 | optimal |
| 20_0_5_w | 2 | 3652075.657185 | 3665547.752124 | 13472.094939 | 0.368889 | 7.995 | 97.041 | 0 | 0 | 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 | optimal |
| 20_0_5_w | 3 | 3652075.657185 | 3671993.982988 | 19918.325803 | 0.545397 | 7.995 | 132.062 | 0 | 0 | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | optimal |
| 20_0_5_w | 4 | 3652075.657185 | 3703308.257942 | 51232.600757 | 1.402835 | 7.995 | 104.856 | 0 | 0 | 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 | optimal |
| 20_0_5_w | 5 | 3652075.657185 | 3746312.490374 | 94236.833189 | 2.580364 | 7.995 | 59.270 | 0 | 0 | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | optimal |
| 20_0_5_w | 6 | 3652075.657185 | 3784662.854467 | 132587.197282 | 3.630461 | 7.995 | 18.923 | 0 | 0 | 5,6,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 | optimal |

## Validation, execution, and scope

59 of 60 unique solves are proven optimal; 1 certified infeasible; 0 execution failures. Infeasible and unproven objective/gap cells are empty, never presented as costs. See `results/failures.json`.

Task 1 final validation: **28 tests passed**, including the complete artifact audit. Before the full experiment, all 26 then-existing loader/model tests passed. A subsequent explicit startup-coupling regression confirms that the pattern master can prefer a costlier hourly dispatch to avoid a startup. An additional artifact test checks every saved schedule, certificate, CSV cost/gap, effective bit depth, and preprocessing record. The tests cover all ten real files, misleading gzip extensions, rounding (including fractional q), convex/linear/constrained fits, hourly parameters, dropped-field audit, effective bits, zero-width ranges, startup/initial status, must-run and surplus generation, infeasibility, and refusal to accept interrupted solves. Tiny discrete optima, including eight randomized cases, are checked against exhaustive physical-level enumeration and the direct SCIP model; continuous optima are checked against commitment enumeration plus independent scalar dispatch minimization.

Initial monolithic coarse-grid smoke solves hit 90-second limits; a strengthened variant also hit a 120-second limit. Those unproven values were discarded. The certified pattern approach above was used for the final sweep.

QUBO construction and QUBO-dependent tests from AGENTS.md are deferred by the explicit Task 1 scope. No z_Q, z_S, penalty gaps, heuristic results, or QUBO certificates are claimed.

Reproduce with the pinned environment in `requirements.lock`:

```sh
python -m pytest -q tests/test_loader.py tests/test_models.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.baselines --workers 3
python -m pytest -q
```

The runner validates source/data fingerprints before resuming cached solves. Use `--time-limit SECONDS` for bounded runs; any unproven solve leaves empty cells and produces a nonzero exit.
