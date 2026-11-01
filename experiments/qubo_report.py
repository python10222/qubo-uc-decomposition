"""Reproduce Task 2's coefficient/count table from committed benchmark data.

Run after tests: python -m experiments.qubo_report
"""
import csv
import json
from pathlib import Path

from src.loader import load_all
from src.qubo import build_qubo

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'results/baselines.csv').open() as stream:
        baselines = {(r['instance'], int(r['B'])): float(r['z_disc']) for r in csv.DictReader(stream)}
    rows = []
    max_relative_error = 0.
    for instance in load_all():
        for B in range(2, 7):
            delta_min = min((hi-lo)/(2**B-1) for u in instance.units
                            for lo, hi in zip(u.pmin, u.pmax) if hi > lo)
            lambda1 = 10*max(a+s for u in instance.units for a, s in zip(u.cost[:, 0], u.startup))
            lambda2 = 10*max(u.cost[:, 1].max() for u in instance.units)/delta_min
            q = build_qubo(instance, instance.hours, B, lambda1, lambda2)
            expected = len(instance.units)*instance.hours*(1+B)+sum(q.slack_bits)
            assert q.num_variables == expected
            data = json.loads((ROOT/f'results/runs/{instance.name}_B{B}.json').read_text())
            bits = q.encode(data['on'], data['grid_indices'])
            reference = baselines[instance.name, B]
            error = abs(q.energy(bits)-reference)/abs(reference)
            assert error <= 1e-6
            max_relative_error = max(max_relative_error, error)
            coefficients = [abs(value) for value in q.Q.values()]
            rows.append(f'| {instance.name} | {B} | {q.num_variables} | {lambda1:.9g} | '
                        f'{lambda2:.9g} | {min(coefficients):.9g} | {max(coefficients):.9g} |')
    report = f'''# Task 2: QUBO builder

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
  error: **{max_relative_error:.12g}**. Saved certificates are audited by the
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
''' + '\n'.join(rows) + '\n'
    (ROOT/'reports/task2.md').write_text(report)
    print(f'Wrote 50 rows; maximum relative energy error {max_relative_error:.12g}')


if __name__ == '__main__':
    main()
