# AGENTS.md — QUBO unit commitment error decomposition

Instructions for any coding agent working in this repository. Read `docs/stage3_research_question.md` and `docs/stage4_uc_qubo_derivation.md` before writing code; they define the problem and every formula.

## Research question

When solving QUBO-formulated unit commitment problems on 24-hour, 10- and 20-unit OR-LIB benchmarks using quantum and quantum-inspired methods, how much of the optimality gap stems from output discretization, penalty-based constraint enforcement, and solver suboptimality, and how does that attribution shift across varying bit depths, penalty weights, and instance sizes?

## The four costs and the decomposition (do not change these definitions)

- `z_cont`: proven-optimal cost of the preprocessed instance with continuous output.
- `z_disc`: proven-optimal cost of the same instance with each unit's output restricted to its bit grid.
- `z_Q`: true cost of the QUBO ground state (exact on reduced instances only).
- `z_S`: true cost of the best feasible state returned by a heuristic solver.
- Gap terms: `G_disc = z_disc - z_cont`, `G_pen = z_Q - z_disc`, `G_solv = z_S - z_Q`. They must sum to `z_S - z_cont` exactly; assert this in code.
- If a QUBO ground state is infeasible, record `G_pen = NaN` and set an `infeasible_ground_state` flag. Never fold infeasibility into a cost.

## Fixed modelling decisions

1. **Instances.** OR-LIB sets from UnitCommitment.jl: `or-lib/10_0_{1..5}_w` and `or-lib/20_0_{1..5}_w`, 24 hours, one bus. The JSON files are committed in `data/raw/` as either `<name>.json` or `<name>.json.gz` (e.g. `data/raw/10_0_1_w.json`); the loader must accept both, detecting gzip by its first two bytes (0x1f 0x8b) rather than by the file name. Originally from `https://axavier.org/UnitCommitment.jl/0.4/instances/or-lib/<name>.json.gz`. Read them from `data/raw/`; never generate or invent instance data. If a file is missing, stop and report.
   Field mapping (UnitCommitment.jl input format): `Generators` → units with `"Type": "Thermal"`; `Production cost curve (MW)` and `Production cost curve ($)` give a convex piecewise-linear total-cost curve: first MW point = Pmin, last MW point = Pmax, cost at the first point = no-load cost a_i (+ b_i·Pmin from the fit). `Startup costs ($)`/`Startup delays (h)` → use the cheapest category as S_i. `Initial status (h)` > 0 means on at t = 0. `Must run?` = true → fix u_it = 1 in every model and in the QUBO (log it). Demand D_t = `Load (MW)` of the bus (sum over buses if more than one). If any of these fields is time-varying, log it and use it per hour.
2. **Model scope.** Keep: demand balance (≥ demand), output limits, on/off, start-up cost, initial on/off status (needed for the hour-1 start-up term). Drop and list in the run log: minimum up/down times, ramp limits, reserves, network, multiple start-up categories (use the cheapest category's cost). Log every dropped field per instance.
3. **Common power quantum q (default 1 MW).** Round Pmin and Pmax to multiples of q (Pmin up, Pmax down) and demand up to a multiple of q. Set each unit's step Δ_i = q · floor((Pmax_i − Pmin_i) / (q (2^B − 1))), with Δ_i ≥ q. Then every output level, the demand and the slack step (Δ_s = q) are integer multiples of q, so slack can represent any surplus exactly. `z_cont` is computed on the rounded instance, so all four costs refer to the same data.
4. **Cost function.** Report the cost form in the raw files. If it is a piecewise-linear curve, fit a convex quadratic a + b·p + c·p² by least squares over its breakpoints (c ≥ 0) and use that same function in every model. Log the fit error per unit. If the curve is linear, set c = 0.
5. **Slack bits per hour.** M_t = ceil(log2((Σ_i Pmax_i − D_t)/q + 1)).
6. **Encodings.** `linear` = binary weights 2^k (main). `log` = logarithmic levels after Christeson et al. (secondary; only after the main grid is done).
7. **Penalty types.** `slack` = squared demand penalty with slack bits (main). `unbalanced` = −α·h + β·h² with h = Σp − D, no slack (secondary).

## Solvers

- MILP/MIQP for `z_cont` and `z_disc`: SCIP via `pyscipopt` (handles the quadratic cost). HiGHS (`highspy`) is allowed when c = 0. Require proven optimality; record gap and time.
- Exact QUBO ground state on reduced instances: brute force when n ≤ 22, otherwise SCIP on the QUBO.
- Heuristics: simulated annealing (`dwave-samplers` `SimulatedAnnealingSampler`), simulated bifurcation (`simulated-bifurcation`). QAOA (one library, Qiskit or PennyLane) on reduced instances only, added last.
- Every heuristic run: fixed seed, ≥ 10 seeds per setting, log all solver parameters.
-    - Simulated bifurcation must run in float64. simulated-bifurcation 2.0.0 fails in float64 unless torch's default dtype is set to float64 for the duration of the call and restored afterwards; wrap every SB call this way. Never fall back to float32 silently. Record the dtype in every result row.
   - Record CPU model, thread count and library versions with every timing result.

## Engineering rules

- Python 3.11+, packages from `requirements.txt` only; pin versions in `requirements.lock` after the first successful install.
- Layout: `src/` (loader, models, qubo, solvers, decompose), `tests/`, `experiments/` (configs), `results/` (CSV + JSON, one row per run), `reports/`.
- Tests that must pass before any experiment: (a) energy of the discretized optimum equals `z_disc` to 1e-6 with slack penalties; (b) on a 2-unit, 2-hour instance, brute-force ground state equals `z_disc` when penalties exceed the threshold; (c) the three gap terms sum to the total gap.
- Never report a number you did not compute. If something cannot run (missing package, no internet, timeout), say so in the report and leave the cell empty.
- Each task ends with a short markdown report in `reports/` listing what ran, what failed, and any decision you had to make that this file does not cover.
