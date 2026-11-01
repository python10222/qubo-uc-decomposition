"""Independent checks of the expanded polynomial and its exact ground states."""
import csv
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from src.loader import load_all, slack_grid
from src.models import build_model, solve
from src.qubo import build_qubo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def instances():
    return load_all()


def weights(instance, B):
    delta = min((hi-lo)/(2**B-1) for u in instance.units
                for lo, hi in zip(u.pmin, u.pmax) if hi > lo)
    return (10*max(float(a+s) for u in instance.units for a, s in zip(u.cost[:, 0], u.startup)),
            10*max(float(u.cost[:, 1].max()) for u in instance.units)/delta)


@pytest.mark.parametrize('B', range(2, 7))
@pytest.mark.parametrize('index', range(10))
def test_all_saved_optima_and_variable_counts(instances, index, B):
    instance = instances[index]
    qubo = build_qubo(instance, instance.hours, B, *weights(instance, B))
    data = json.loads((ROOT/f'results/runs/{instance.name}_B{B}.json').read_text())
    with (ROOT/'results/baselines.csv').open() as stream:
        row = next(r for r in csv.DictReader(stream) if r['instance'] == instance.name and int(r['B']) == B)
    bits = qubo.encode(data['on'], data['grid_indices'])
    u, p, s = qubo.decode(bits)
    assert qubo.is_feasible(u, p)
    assert all(sum(p[:, t])-int(instance.demand[t]) == s[t] for t in range(instance.hours))
    assert qubo.energy(bits) == pytest.approx(float(row['z_disc']), rel=1e-6)
    assert qubo.true_cost(u, p) == pytest.approx(float(row['z_disc']), rel=1e-12)
    _, M = slack_grid(instance, B)
    assert qubo.num_variables == len(instance.units)*instance.hours*(1+B)+sum(M)


def brute_force(qubo):
    """Enumerate every full bit string, including slack, in bounded batches."""
    assert qubo.num_variables <= 22
    best, state = float('inf'), None
    shifts = np.arange(qubo.num_variables, dtype=np.uint64)
    for start in range(0, 2**qubo.num_variables, 65536):
        ids = np.arange(start, min(start+65536, 2**qubo.num_variables), dtype=np.uint64)
        bits = ((ids[:, None] >> shifts) & 1).astype(float)
        energies = np.full(len(ids), qubo.offset)
        for (i, j), value in qubo.Q.items():
            energies += value*bits[:, i]*bits[:, j]
        at = int(energies.argmin())
        if energies[at] < best:
            best, state = energies[at], bits[at].astype(int)
    assert best == pytest.approx(qubo.energy(state), abs=1e-4)
    return best, state


@pytest.mark.parametrize('N', [2, 3])
def test_reduced_ground_states(instances, N):
    # Preserve actual benchmark units, costs and initial status. With fewer
    # plants the original demand is impossible; use capacity minus 1 MW as an
    # explicit stress reduction. Small surplus keeps exhaustive slack affordable.
    source = instances[0]
    units = tuple(replace(u, pmin=u.pmin[:2], pmax=u.pmax[:2], cost=u.cost[:2],
                          startup=u.startup[:2], must_run=u.must_run[:2]) for u in source.units[:N])
    instance = replace(source, name=f'reduced_{N}', units=units,
                       demand=sum(u.pmax for u in units)-1)
    reference = solve(build_model(instance, 2))
    large = build_qubo(instance, 2, 2, 1e6, 1e6)
    energy, bits = brute_force(large)
    u, p, _ = large.decode(bits)
    assert large.is_feasible(u, p)
    assert energy == pytest.approx(reference.objective, abs=1e-4)
    assert large.true_cost(u, p) == pytest.approx(reference.objective, abs=1e-6)
    tiny = build_qubo(instance, 2, 2, 1e-9, 1e-9)
    energy, bits = brute_force(tiny)
    u, p, _ = tiny.decode(bits)
    assert not tiny.is_feasible(u, p)
    assert energy < reference.objective


def test_polynomial_matches_direct_formula_with_hourly_fixed_units(instances):
    source = instances[0]
    first = replace(source.units[0], pmin=np.array([3., 4.]), pmax=np.array([3., 8.]),
                    cost=np.array([[2., 3., .25], [5., 1., .5]]), startup=np.array([7., 11.]),
                    must_run=np.array([True, False]), initial_on=False)
    second = replace(first, name='second', pmin=np.array([1., 2.]), pmax=np.array([5., 2.]),
                     must_run=np.array([False, True]), initial_on=True)
    instance = replace(source, units=(first, second), demand=np.array([5., 6.]))
    q = build_qubo(instance, 2, 2, 17., 23.)
    assert len(q.fixed_on) == 2
    assert len(q.b) == 4  # Exactly two nonzero-width unit-hours.
    rng = np.random.default_rng(19)
    for _ in range(200):
        bits = rng.integers(0, 2, q.num_variables)
        u, p, s = q.decode(bits)
        off = sum(bits[j]*(1-u[i, t]) for (i, t, _), j in q.b.items())
        residual = sum((sum(p[:, t])-int(instance.demand[t])-s[t])**2 for t in range(2))
        assert q.energy(bits) == pytest.approx(q.true_cost(u, p)+17*off+23*float(residual), abs=1e-9)
    result = solve(build_model(instance, 2))
    bits = q.encode(result.on, result.grid_indices)
    u, p, s = q.decode(bits)
    assert q.is_feasible(u, p)
    assert q.energy(bits) == pytest.approx(result.objective, abs=1e-9)
    u[0, 0] = 0
    assert not q.is_feasible(u, p)
    u[0, 0] = 1
    p[0, 1] += Fraction(1, 100)
    assert not q.is_feasible(u, p)


def test_horizon_zero_slack_and_input_validation(instances):
    x = instances[0]
    x = replace(x, demand=sum(u.pmax for u in x.units))
    q = build_qubo(x, 1, 2, 0, 0)
    assert q.T == 1 and len(q.e) == 0
    u = np.ones((len(x.units), 1), dtype=int)
    indices = np.full(u.shape, 3)
    bits = q.encode(u, indices)
    on, p, s = q.decode(bits)
    assert q.is_feasible(on, p) and s[0] == 0
    for kwargs in ({'T': 0}, {'B': 0}, {'lambda1': -1}, {'lambda2': float('nan')},
                   {'encoding': 'log'}, {'penalty': 'unbalanced'}):
        args = dict(instance=x, T=1, B=2, lambda1=1., lambda2=1.)
        args.update(kwargs)
        with pytest.raises(ValueError):
            build_qubo(**args)
    with pytest.raises(ValueError):
        q.decode(np.zeros(q.num_variables+1))
    with pytest.raises(ValueError):
        q.energy(np.full(q.num_variables, .5))
    with pytest.raises(ValueError):
        q.encode(u, indices-.5)
    with pytest.raises(ValueError):
        q.encode(u, indices-1)  # Demand shortfall.
