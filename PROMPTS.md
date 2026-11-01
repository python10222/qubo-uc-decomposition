# Prompts for Stages 5 and 6

Run them in order. Each Codex task is small enough to review in one sitting; check its report before starting the next.

## Stage 5a — Math review (Astra, chat; attach both files in `docs/`)

```
Attached: a research question (stage3) and a derivation of a unit-commitment QUBO (stage4). Act as a referee. Do not write experiment code.

1. Check every formula in stage4 Sections 2–5 and 7 for errors (signs, limits, the claim that the encoding enforces output bounds, the start-up term, both demand penalties).
2. Re-derive the exactness thresholds in Section 5: the smallest λ1 and λ2 that make the penalties exact. State the assumptions; say whether "λ2 grows roughly like 2^B" holds, fails, or needs qualifying.
3. Check that the decomposition in stage3 Section 2 is exact (the three terms sum to the total gap) and identify any case where a term could be negative or undefined.
4. Check the certificate argument in stage3 Section 2 (penalty error certified as zero when no solver finds energy below z_disc). When does it hold, and when does it fail?
5. List anything that would make the results misleading.

Answer item by item. Mark each finding OK / ERROR (with correction) / CAVEAT.
```

## Codex Task 1 — Data and classical baselines

```
Read AGENTS.md and docs/. Implement:
1. src/loader.py: read the 10 OR-LIB instances from data/raw/ (see AGENTS.md), parse them, apply the preprocessing in AGENTS.md (rounding to q, cost fit, dropped fields logged).
2. src/models.py: the continuous model (z_cont) and the discretized model for a given B (z_disc), both solved with SCIP to proven optimality.
3. results/baselines.csv: one row per instance × B ∈ {2,3,4,5,6} with z_cont, z_disc, G_disc (absolute and % of z_cont), solve times, optimality gaps.
4. reports/task1.md: cost form found in the raw data, fit errors, dropped fields, and the table.
Add tests. Do not build the QUBO yet.
```

## Codex Task 2 — QUBO builder and correctness tests

```
Read AGENTS.md and docs/stage4. Implement src/qubo.py: build the QUBO for a given instance, horizon, B, λ1, λ2, encoding=linear, penalty=slack, exactly as derived in stage4 Sections 2–4 (start-up as the quadratic term u_t(1−u_{t−1}), off-unit penalty b(1−u), squared demand penalty with slack bits, steps from AGENTS.md).
Also implement: decode(bits) → (u, p, s), true_cost(u, p), is_feasible(u, p), energy(bits).
Tests: (a) energy of the decoded discretized optimum equals z_disc to 1e-6; (b) on reduced instances of 2–3 units and 2 hours, brute-force ground state equals the discretized optimum when λ is large, and becomes infeasible when λ is tiny; (c) variable count equals N·T·(1+B) + Σ_t M_t.
Report in reports/task2.md.
```

## Codex Task 3 — Feasibility numbers (this is Stage 5b)

```
Using src/qubo.py:
1. Table of QUBO variable counts for N ∈ {10, 20}, T = 24, B ∈ {2..6}.
2. On reduced instances (first 3–5 units of 10_0_1_w, T ∈ {2, 4}, B ∈ {2..5}), find the exact minimum λ2 (with λ1 fixed at 10 × max(a_i + S_i)) and then the exact minimum λ1 (with λ2 at 10 × its minimum), by bisection to 1% relative, using exact ground states (brute force if n ≤ 22, else SCIP). "Exact" means the ground state is feasible and its true cost equals z_disc. Compare with λ2 ≈ b_max/Δ and λ1 ≈ a_i + S_i from stage4 Section 5, and report how the minimum λ2 scales with B.
3. Time simulated annealing (dwave-samplers, num_reads=100, default schedule) and simulated bifurcation (agents=128, default settings) on the full 10- and 20-unit QUBO at B = 4, λ at 2 × the estimated minimum; report time, best energy, true cost of best feasible sample, feasibility rate.
4. Largest reduced instance SCIP solves to proven optimality as a QUBO within 10 minutes.
Report all numbers and the code in reports/task3.md. Do not tune anything to make results look better.
```

Bring the Task 3 report back to Claude before Task 4: its numbers fix the experiment grid.

## Codex Task 4 — Heuristic solvers and the decomposition

```
Implement src/solvers.py (SA, SB, with seeds and logged parameters) and src/decompose.py (G_disc, G_pen, G_solv per AGENTS.md, with the sum assertion, the infeasible-ground-state flag, and the certificate check: flag any sample with energy below z_disc). Implement experiments/run.py taking a config file (instances, B list, λ multipliers, solvers, seeds) and writing one row per run to results/. Run a small smoke test only (one instance, B ∈ {3,4}, λ ∈ {1,2}×min, 3 seeds). Report in reports/task4.md.
```

## Codex Task 5 — QAOA arm (reduced instances only)

```
Add QAOA (choose Qiskit or PennyLane, add it to requirements.txt) on reduced instances of ≤ 25 qubits, depth p ∈ {1, 2, 3}, statevector simulation, ≥ 10 seeds of the classical optimiser. Return the best feasible sample's true cost as z_S and compute the decomposition against exact z_Q. Report in reports/task5.md.
```

The full experiment grid (all instances, all factors) is configured after Task 3, with Claude.
