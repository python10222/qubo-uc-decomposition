from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from src.loader import Instance, Unit, full_range_grid
from src.models import (build_model, solve, solve_continuous, solve_discretized,
                        true_cost, validate_solution)


def tiny():
    return Instance('tiny',1.,np.array([4.,6.]),(
        Unit('g0',np.array([1.,1.]),np.array([5.,5.]),np.array([[2.,1.,.4]]*2),
             np.array([3.,3.]),False,np.array([False,False])),
        Unit('g1',np.array([2.,2.]),np.array([6.,6.]),np.array([[1.,2.,.1]]*2),
             np.array([4.,4.]),True,np.array([False,False]))),{})


def brute_grid(instance,B):
    """Enumerate physical levels, independently of the SCIP formulation."""
    choices = []
    for unit in instance.units:
        grid = full_range_grid(unit,B,instance.q)
        for t in range(instance.hours):
            levels = [(1,Fraction(int(unit.pmin[t]))+grid.step[t]*k) for k in range(grid.levels[t]+1)]
            choices.append(levels if unit.must_run[t] else [(0,Fraction(0))]+levels)
    best = float('inf')
    for state in product(*choices):
        u = np.array([v[0] for v in state]).reshape(len(instance.units),instance.hours)
        p = np.array([v[1] for v in state]).reshape(u.shape)
        if np.all(p.sum(axis=0) >= instance.demand):
            best = min(best,true_cost(instance,u,p.astype(float)))
    return best


def brute_continuous(instance):
    """Enumerate commitments and minimize each 2-unit hourly dispatch in 1D."""
    best = float('inf')
    for state in product([0,1],repeat=4):
        u = np.array(state).reshape(2,2)
        p = np.zeros((2,2))
        for t in range(2):
            low = np.array([v.pmin[t]*u[i,t] for i,v in enumerate(instance.units)])
            high = np.array([v.pmax[t]*u[i,t] for i,v in enumerate(instance.units)])
            demand = max(instance.demand[t],low.sum())
            if high.sum() < demand or any(v.must_run[t] and not u[i,t] for i,v in enumerate(instance.units)):
                break
            left,right = max(low[0],demand-high[1]),min(high[0],demand-low[1])
            def cost(x):
                return sum(v.cost[t,1]*z+v.cost[t,2]*z*z
                           for v,z in zip(instance.units,[x,demand-x]))
            x = left if left == right else minimize_scalar(cost,bounds=(left,right),method='bounded').x
            p[:,t] = [x,demand-x]
        else:
            best = min(best,true_cost(instance,u,p))
    return best


@pytest.mark.parametrize('B',[1,2,3,6])
def test_discrete_matches_enumeration(B):
    instance = tiny()
    if B == 6:
        instance = replace(instance,demand=instance.demand[:1],units=tuple(
            replace(u,pmin=u.pmin[:1],pmax=u.pmax[:1],cost=u.cost[:1],startup=u.startup[:1],must_run=u.must_run[:1]) for u in instance.units))
    solution = solve_discretized(instance,B)
    assert solution.status == 'optimal' and solution.gap == 0
    assert solution.objective == pytest.approx(brute_grid(instance,B),abs=1e-6)
    assert solution.metadata['scip_threads'] == 1


def test_continuous_matches_independent_enumeration():
    instance = tiny()
    continuous = solve_continuous(instance)
    assert continuous.objective == pytest.approx(brute_continuous(instance),abs=1e-6)
    assert solve_discretized(instance,2).objective >= continuous.objective-1e-6


@pytest.mark.parametrize('B',[None,2])
def test_initial_status_must_run_surplus_and_fixed_output(B):
    instance = tiny()
    unit = replace(instance.units[0],pmin=np.array([5.,5.]),pmax=np.array([5.,5.]),
                   must_run=np.array([True,True]))
    instance = replace(instance,units=(unit,),demand=np.array([0.,2.]))
    solution = solve(build_model(instance,B))
    np.testing.assert_array_equal(solution.on,[[1,1]])
    np.testing.assert_array_equal(solution.power,[[5,5]])
    np.testing.assert_array_equal(solution.startup,[[1,0]])
    assert solution.objective == pytest.approx(2*(2+5+.4*25)+3)
    already_on = replace(instance,units=(replace(unit,initial_on=True),))
    assert solve(build_model(already_on,B)).objective == pytest.approx(solution.objective-3)


def test_hourly_parameters_and_linear_cost():
    instance = tiny()
    first = replace(instance.units[0],pmin=np.array([2.,3.]),pmax=np.array([6.,7.]),
                    cost=np.array([[2.,1.,0.],[3.,2.,0.]]),startup=np.array([3.,8.]),
                    must_run=np.array([True,False]))
    instance = replace(instance,units=(first,instance.units[1]))
    assert solve_discretized(instance,2).objective == pytest.approx(brute_grid(instance,2),abs=1e-6)
    assert solve_continuous(instance).objective == pytest.approx(brute_continuous(instance),abs=1e-6)


def test_infeasible_and_unproven_are_not_baselines():
    instance = replace(tiny(),demand=np.array([100.,100.]))
    with pytest.raises(RuntimeError,match='infeasible'): solve_continuous(instance)
    with pytest.raises(RuntimeError,match='did not prove optimality'):
        solve_continuous(tiny(),time_limit=0)


@pytest.mark.parametrize('seed',range(8))
def test_pattern_certificate_against_direct_model_and_enumeration(seed):
    rng = np.random.default_rng(seed)
    instance = tiny()
    units = tuple(replace(unit,
                  cost=np.array([[rng.uniform(0,4),rng.uniform(.5,3),rng.uniform(0,.8)] for _ in range(2)]),
                  startup=rng.uniform(0,15,size=2),initial_on=bool(rng.integers(2)))
                  for unit in instance.units)
    instance = replace(instance,units=units,demand=rng.integers(0,10,size=2).astype(float))
    pattern = solve_discretized(instance,2)
    direct = solve(build_model(instance,2))
    assert pattern.objective == pytest.approx(brute_grid(instance,2),abs=1e-6)
    assert pattern.objective == pytest.approx(direct.objective,abs=1e-6)
    certificate = pattern.metadata['pattern_generation']
    assert certificate['final_unseen_hours'] == []
    assert certificate['master_certificates'][-1]['unseen_hours'] == []
    assert all(c['gap'] == 0 for c in certificate['master_certificates'])


def test_pattern_infeasibility_time_limit_and_zero_cost():
    with pytest.raises(RuntimeError,match='infeasible'):
        solve_discretized(replace(tiny(),demand=np.array([100.,100.])),2)
    with pytest.raises(RuntimeError,match='did not prove optimality'):
        solve_discretized(tiny(),2,time_limit=0)
    instance = tiny()
    instance = replace(instance,units=tuple(replace(u,cost=np.zeros((2,3)),startup=np.zeros(2)) for u in instance.units))
    assert solve_discretized(instance,2).objective == 0


def test_pattern_master_preserves_startup_coupling():
    # Cheapest hourly dispatch switches plants, but startup makes staying on best.
    first = Unit('a',np.full(2,5.),np.full(2,5.),np.array([[1.,0.,0.],[10.,0.,0.]]),
                 np.full(2,20.),True,np.zeros(2,dtype=bool))
    second = replace(first,name='b',cost=np.array([[10.,0.,0.],[1.,0.,0.]]),initial_on=False)
    instance = Instance('startup_coupling',1.,np.full(2,5.),(first,second),{})
    result = solve_discretized(instance,2)
    assert result.objective == pytest.approx(11.,abs=1e-6)
    np.testing.assert_array_equal(result.on,[[1,1],[0,0]])
    assert result.objective == pytest.approx(solve(build_model(instance,2)).objective,abs=1e-6)


def test_exact_certificate_rejects_off_grid_ticks_and_false_demand():
    instance = tiny()
    result = solve_discretized(instance,3)
    assert result.power_denominator == 7
    assert np.issubdtype(result.grid_indices.dtype,np.integer)
    bad = replace(result,power_scaled=result.power_scaled.copy())
    bad.power_scaled[0,0] += 1
    with pytest.raises(AssertionError,match='exact grid'):
        validate_solution(instance,bad,3)
    bad = replace(result,grid_indices=result.grid_indices.copy())
    bad.grid_indices[0,0] = 8
    with pytest.raises(AssertionError,match='index outside'):
        validate_solution(instance,bad,3)
    bad = replace(result,power_scaled=result.power_scaled.astype(float))
    with pytest.raises(AssertionError,match='integers'):
        validate_solution(instance,bad,3)
