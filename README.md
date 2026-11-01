# QUBO unit commitment error decomposition

Research code for: *When solving QUBO-formulated unit commitment problems on 24-hour, 10- and 20-unit OR-LIB benchmarks using quantum and quantum-inspired methods, how much of the optimality gap stems from output discretization, penalty-based constraint enforcement, and solver suboptimality, and how does that attribution shift across varying bit depths, penalty weights, and instance sizes?*

- `AGENTS.md`: rules and fixed decisions for coding agents (Codex reads this automatically).
- `docs/`: the research question (Stage 3) and the derivation of the QUBO (Stage 4).
- `PROMPTS.md`: the Stage 5a review prompt for Astra and the Codex tasks, in order.
- `requirements.txt`: Python packages.

Everything runs on simulators. No quantum hardware accounts are needed.

## Task 1 baselines

Use the versions pinned in `requirements.lock` (the lock selects CPU PyTorch for
later tasks). With `uv` installed:

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend cpu --require-hashes requirements.lock
.venv/bin/python -m pytest -q tests/test_loader.py tests/test_models.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.baselines --workers 3
.venv/bin/python -m pytest -q
```

The runner reads the ten committed files, solves each continuous reference once,
and solves the five grids B=2,…,6 with SCIP. It accepts only proven-optimal
solutions with zero reported gap. Infeasible grids receive an explicit status,
a SCIP certificate, and empty discrete objective/gap fields. The grid solver uses certified hourly dispatch
patterns and a SCIP master retaining all startup coupling; the report explains
the global optimality certificate. The direct grid model is also available as
`solve(build_model(instance, B))` in `src.models`.

Outputs are `results/baselines.csv`, `results/preprocessing.json`, individual
schedule/certificate records in `results/runs/`, and `reports/task1.md`. Completed
runs can be resumed when model-source and raw-data fingerprints match. Optional
`--time-limit SECONDS` bounds each reference solve; an unproven solve produces an
empty result and a failure record, never an optimality claim. Full-suite tests
also validate the committed result artifacts. No QUBO is built in Task 1.

The current grid rule rounds input powers to integer MW, then uses exactly B
bits for every nonzero-width unit with step `(Pmax-Pmin)/(2**B-1)`. Its highest
level equals Pmax exactly. Saved integer indices and scaled outputs permit exact
grid and surplus checks; the slack step is `1/(2**B-1)` MW. CSV steps are exact
rational strings. The superseded whole-MW-grid results and source snapshots are
archived under `results/first_run/`; see "Change from the first run" in the report.
