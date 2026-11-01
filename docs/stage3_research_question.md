# Stage 3 — Research question (1 Nov 2026)

**Final question (Yousif's wording).** When solving QUBO-formulated unit commitment problems on 24-hour, 10- and 20-unit OR-LIB benchmarks using quantum and quantum-inspired methods, how much of the optimality gap stems from output discretization, penalty-based constraint enforcement, and solver suboptimality, and how does that attribution shift across varying bit depths, penalty weights, and instance sizes?

Optimality gap here means the gap to the continuous MILP optimum, z_S − z_cont (Section 2). The encoding and penalty-type factors below are secondary: run them on the 10-unit set once the core grid is done.

**Original draft (superseded).** When unit commitment is rewritten as a QUBO and solved with quantum and quantum-inspired heuristics, how much of the cost gap to the true optimum comes from discretizing generator output, from enforcing constraints with penalties, and from the solver's search, and how does that split change with bit depth, penalty weight and instance size on standard multi-period benchmarks?

**Defined result.** For OR-LIB/UC instances with 10 and 20 units over 24 hours, an exact three-way decomposition of each solver's optimality gap, reported across bit depth, penalty weight, encoding, penalty type and solver, plus a tested rule for choosing bits and penalty weight that minimises the total gap.



## 1. Where the two closest papers stand

| Paper | What could be read | What it shows | Effect on the question |
| --- | --- | --- | --- |
| Christeson, Khodaei, Eskandarpour 2024 (ISGT Europe) | Not the paper (paywalled) or the thesis (embargoed until 9 Dec 2026). Read: Hasanzadeh survey summary; thesis abstract | Monolithic QUBO with generation levels on logarithmic steps and quadratic penalties; fewer binaries and less time, "accuracy and optimality" still need work. No error split mentioned anywhere | Logarithmic encoding becomes one level of the encoding factor, so the study answers "does it lose more through discretization?" directly |
| Müller, Dukalski, Phillipson 2026 (IEEE TPWRS) | Abstract; Hasanzadeh survey summary; Carrillo-Muñoz et al. cite it without detail | Slack variables vs unbalanced penalization on a quantum annealer. Unbalanced penalization uses fewer variables but penalizes feasible solutions unevenly; quality falls with size. Instance sizes and any formulation-vs-solver split are not visible | Unbalanced penalization becomes one level of the penalty factor. Uneven penalties shift the QUBO ground state, which is exactly what the penalty term measures |
| Carrillo-Muñoz et al. 2026 (arXiv 2609.36999), read in full | Full text | Commitment only, single period, 4–20 generators, fixed penalty weight 0.5; reports feasibility and optimal probability, no error split | Not a threat |

**Residual risk.** Müller et al. may separate the QUBO ground state from annealer output on its own instances. Even then it has no discretization term, no bit-depth axis and no standard multi-period thermal benchmark, so the three-way split would still be new. To close the risk, get the full text: ask the authors for a copy (ResearchGate "request full text" or email), or a university contact with IEEE Xplore access. Recheck Christeson's thesis after 9 December 2026.

## 2. The decomposition (what makes the result exact)

For one instance and one setting, define four costs:

| Symbol | Meaning | How it is computed |
| --- | --- | --- |
| z_cont | Optimal cost of the original MILP (continuous output) | HiGHS or SCIP, to proven optimality |
| z_disc | Optimal cost of the same MILP with each unit's output restricted to the B-bit grid | Same solver, proven optimality |
| z_Q | True cost of the QUBO's ground state (the best state under cost + penalties) | Exact on reduced instances; certified bound on full ones (below) |
| z_S | True cost of the best feasible state a heuristic solver returns | SA, SB, QAOA, repeated over seeds |

The total gap telescopes into three parts that add up exactly:

```latex
z_S - z_{cont} = \underbrace{(z_{disc} - z_{cont})}_{\text{discretization}} + \underbrace{(z_Q - z_{disc})}_{\text{penalty}} + \underbrace{(z_S - z_Q)}_{\text{solver}}
```

- **Discretization error** ≥ 0 always; it shrinks as bits increase.
- **Penalty error** is 0 when penalties are exact (the ground state is the discretized optimum). It is positive when penalties distort the landscape (for example unbalanced penalization shifting the ground state to a feasible but costlier plan). If the ground state is infeasible, the penalty is too weak; that case is reported as a separate infeasibility flag, not folded into a cost.
- **Solver error** ≥ 0; it is what the hardware or heuristic loses.

**Certifying z_Q without solving the QUBO exactly.** With slack-variable penalties, the discretized optimum (with its slack bits set correctly) has energy exactly z_disc, provided the slack grid can represent the demand surplus exactly (same step size as the output grid, demand scaled onto it; Stage 5 checks this). So any state with energy below z_disc must be infeasible, which proves the penalty weight is too small. On full-size instances, penalty error is therefore 0 whenever no solver finds such a state, and flagged otherwise. Exact ground states (needed for unbalanced penalization) are computed on reduced instances: the same units with short horizons, solved by SCIP or brute force.

## 3. Scope

| Decision | Choice | Why |
| --- | --- | --- |
| Instances | OR-LIB/UC 10-unit and 20-unit sets, 24 h (UnitCommitment.jl); reduced-horizon copies (T = 2–6) for exact ground states; 50-unit as a stretch | Standard, public, multi-period, ≥ 10 units: exactly what the gap requires |
| UC model | Demand balance, output limits, start-up costs. Minimum up/down times as an extension only | Keeps the QUBO small enough to sweep; start-up cost fits a QUBO with no slack (cost × u_t(1 − u_{t−1})) |
| Classical reference | Tight MILP after Morales-España et al. 2013 | Gives z_cont and z_disc fast and certified |
| Factor: bit depth B | 2, 3, 4, 5, 6 bits per unit-hour | Spans coarse to fine |
| Factor: penalty weight | 0.5, 1, 2, 5, 10 × the smallest weight that makes penalties exact | Brackets the threshold; tests whether bigger is worse for the solver |
| Factor: encoding | Linear (uniform steps) vs logarithmic (after Christeson) | Answers the obvious reviewer question |
| Factor: penalty type | Slack variables vs unbalanced penalization (after Müller, and arXiv 2211.13914) | Same |
| Solvers | Simulated annealing (dwave-neal); simulated bifurcation (`simulated-bifurcation`, stand-in for Toshiba SBM); QAOA in simulation on reduced instances (≤ ~25 qubits) | All run on a laptop; covers annealing-style, quantum-inspired and gate-based arms |
| Repetitions | ≥ 10 seeds per setting; report means with 95% intervals | Solver error is random, so it needs spread, not one run |
| Out of scope | Quantum hardware, speed or "advantage" claims, network constraints, stochastic UC | Simulator-only constraint; keeps the question single |

**Rough size, to be checked in Stage 5:** per unit-hour, 1 on/off bit plus B output bits; per hour, about B + log₂N slack bits for the demand constraint. For 10 units × 24 h × 4 bits this is about 1,200 + 190 ≈ 1,400 binaries; for 20 units about 2,800. Both are routine for SA and SB. A 3-unit × 2-hour × 2-bit reduced instance is about 24 qubits, within statevector simulation.

## 4. Hypotheses (each can come out false)

1. **H1 — discretization dominates at low bit depth.** At B ≤ 3, discretization error is the largest of the three terms on both instance sizes.
2. **H2 — solver error dominates at high bit depth and larger size.** At B ≥ 5, and more so at 20 units than 10, solver error is the largest term.
3. **H3 — the penalty–solver trade-off.** Penalty error reaches zero at a threshold weight, but solver error rises as the weight grows past it, so the weight that minimises the total gap sits just above the threshold, not far above it.
4. **H4 — encoding.** Logarithmic encoding reaches a given discretization error with fewer bits than linear encoding, but loses more to solver error per bit (wider coefficient range).
5. **H5 — penalty type.** Unbalanced penalization trades lower variable count for a non-zero penalty error that grows with instance size.

## 5. What counts as success

- The decomposition is computed for every OR-LIB 10-unit instance across the full B × weight grid, with all three terms summing to the measured total gap.
- The same is done for the 20-unit set on at least the slack/linear configuration.
- Each hypothesis is confirmed or refuted with confidence intervals.
- A one-line rule for choosing B and penalty weight is stated and tested on held-out instances.

A null result (for example, solver error dominates everywhere) still answers the question.

## 6. Main risks

| Risk | Mitigation |
| --- | --- |
| Müller et al. or Christeson et al. already report part of the split | Get both full texts first (Section 1). The three-way split on standard multi-period instances stays new either way |
| Exact ground states only on reduced instances | Certificate argument covers full instances for slack penalties; state clearly which results are exact and which are certified |
| Thin quantum arm | Frame the contribution around the formulation every QUBO solver shares; QAOA on reduced instances shows the decomposition applies to gate-based solvers too |
| Scope creep from five factors | Full grid on 10 units only; 20 units on the main configuration; extensions only once the core is done |

## 7. Next stages

- **Stage 4 (deep reading).** Core set to load into Gemini Notebook: Ajagekar & You 2019 (UC QUBO, Tables 3–4); Morales-España et al. 2013 (tight MILP); Braun et al. 2023 (binary expansion, bit choice); Halffmann et al. 2022 (compact UC QUBO); Ribes & González Grandón 2025 (penalty tuning); Hellstern et al. 2026 (finding vs certifying); arXiv 2211.13914 (unbalanced penalization); Müller et al. 2026 and Christeson et al. 2024 once obtained. Then work through the UC-to-QUBO derivation step by step with Claude.
- **Stage 5 (feasibility, GPT-6 Astra).** Confirm variable counts per B, N and T; minimum exact penalty weight; runtime per SA/SB run; largest reduced instance SCIP can solve exactly.
- **Stage 6 (simulation, Codex).** Pipeline: instance loader → MILP (z_cont, z_disc) → QUBO builder (linear/log, slack/unbalanced) → solvers → decomposition table.

## Sources

- [Müller, Dukalski & Phillipson 2026, Maastricht record (abstract)](https://cris.maastrichtuniversity.nl/en/publications/quantum-annealing-for-optimizing-unit-scheduling-in-renewable-ene/)
- [Christeson 2025 PhD thesis, University of Denver (abstract; full text embargoed to 9 Dec 2026)](https://digitalcommons.du.edu/etd/2663)
- [Carrillo-Muñoz et al. 2026, arXiv 2609.36999](https://arxiv.org/pdf/2609.36999)
- [Phillipson & Müller 2026, "Quantum Approaches for the Unit Commitment Problem – a Literature Survey" (Springer CCIS 2744)](https://link.springer.com/chapter/10.1007/978-3-032-13855-2_1)
- [Unbalanced penalization, arXiv 2211.13914](https://arxiv.org/pdf/2211.13914)
- [UnitCommitment.jl benchmark instances (OR-LIB/UC)](https://anl-ceeesa.github.io/UnitCommitment.jl/0.2/instances)
- Hasanzadeh, Rajabi & Kargarian, arXiv 2601.01777, refs. [12] and [20] (read from the PDF)
