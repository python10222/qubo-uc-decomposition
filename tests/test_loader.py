from fractions import Fraction
import gzip
import json

import numpy as np
import pytest

from src.loader import (INSTANCE_NAMES, full_range_grid, fit_cost, load_all,
                        load_instance, round_power, slack_grid)


def raw_fixture():
    return {"Parameters": {"Version": "0.3", "Time (h)": 2},
            "Buses": {"b1": {"Load (MW)": [2.1, 3.0]}},
            "Generators": {"g0": {"Production cost curve (MW)": [1.1, 4.9],
                "Production cost curve ($)": [5.2, 12.8], "Initial status (h)": -3,
                "Startup costs ($)": [5, 2], "Startup delays (h)": [1, 3],
                "Must run?": [True, False], "Ramp up limit (MW)": 2}},
            "Reserves": {"r": {"Amount (MW)": 1}}}


def write_raw(tmp_path, raw, compressed=False, name="instance.json"):
    path = tmp_path/name
    payload = json.dumps(raw).encode()
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    return path


@pytest.mark.parametrize("compressed,name", [(True,"misnamed.json"), (False,"misnamed.json.gz")])
def test_magic_rounding_and_audit(tmp_path, compressed, name):
    instance = load_instance(write_raw(tmp_path, raw_fixture(), compressed, name))
    unit, = instance.units
    np.testing.assert_array_equal(instance.demand, [3, 3])
    np.testing.assert_array_equal(unit.pmin, [2, 2])
    np.testing.assert_array_equal(unit.pmax, [4, 4])
    np.testing.assert_allclose(unit.cost, [[3, 2, 0]]*2, atol=1e-12)
    np.testing.assert_array_equal(unit.startup, [2, 2])
    assert not unit.initial_on
    assert instance.audit['data_format_version'] == '0.3'
    assert 'treated as thermal' in instance.audit['assumptions'][0]
    dropped = {item['field'] for item in instance.audit['dropped_fields']}
    assert dropped == {'Reserves','Generators.g0.Startup costs ($)',
                       'Generators.g0.Startup delays (h)','Generators.g0.Ramp up limit (MW)'}
    assert 'Generators.g0.Must run?' in instance.audit['time_varying_fields']


def test_hourly_curves_and_cheapest_startup(tmp_path):
    raw = raw_fixture()
    raw['Generators']['g0']['Production cost curve (MW)'] = [[1.1, 2.1],[4.9, 6.9]]
    raw['Generators']['g0']['Production cost curve ($)'] = [[5.2, 7.2],[12.8, 16.8]]
    raw['Generators']['g0']['Startup costs ($)'] = [[8, 1], [3, 4]]
    unit, = load_instance(write_raw(tmp_path, raw)).units
    np.testing.assert_array_equal(unit.pmin, [2,3])
    np.testing.assert_array_equal(unit.pmax, [4,6])
    np.testing.assert_array_equal(unit.startup, [3,1])
    np.testing.assert_allclose(unit.cost, [[3,2,0]]*2,atol=1e-12)


def test_fit_convex_quadratic_and_constraint_boundary():
    x = np.array([10.,20.,30.,40.,50.])
    coeff, audit = fit_cost(x, 7+2*x+.3*x*x)
    np.testing.assert_allclose(coeff,[7,2,.3],atol=1e-11)
    assert audit['max_abs_error'] < 1e-10
    assert audit['raw_cost_form'] == 'piecewise_linear'
    coeff, audit = fit_cost(x, 7+2*x-.01*x*x)
    assert coeff[2] == 0
    expected = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),7+2*x-.01*x*x,rcond=None)[0]
    np.testing.assert_allclose(coeff[:2],expected,atol=1e-11)
    assert audit['max_abs_error'] > 0


def test_all_committed_data_and_grids():
    instances = load_all()
    assert tuple(i.name for i in instances) == INSTANCE_NAMES
    for instance in instances:
        assert instance.hours == 24
        assert len(instance.units) == int(instance.name.split('_')[0])
        assert len(instance.audit['assumptions']) == len(instance.units)
        for unit in instance.units:
            for B in range(2,7):
                grid = full_range_grid(unit,B,instance.q)
                slack_step,slack_bits = slack_grid(instance,B)
                for t in range(instance.hours):
                    assert grid.bits[t] == (B if unit.pmax[t] > unit.pmin[t] else 0)
                    levels = [Fraction(int(unit.pmin[t]))+grid.step[t]*k for k in range(int(grid.levels[t])+1)]
                    assert levels[-1] == int(unit.pmax[t])
                    assert all((level/slack_step).denominator == 1 for level in levels)
                    assert (Fraction(int(instance.demand[t]))/slack_step).denominator == 1
                    max_ticks = sum(int(u.pmax[t]) for u in instance.units)-int(instance.demand[t])
                    max_ticks *= 2**B-1
                    assert 2**int(slack_bits[t])-1 >= max_ticks
                    assert slack_bits[t] == 0 or 2**(int(slack_bits[t])-1)-1 < max_ticks


def test_small_range_and_zero_bits(tmp_path):
    raw = raw_fixture()
    raw['Generators']['g0']['Production cost curve (MW)'] = [2.,3.]
    unit, = load_instance(write_raw(tmp_path,raw)).units
    grid = full_range_grid(unit,6,1.)
    np.testing.assert_array_equal(grid.bits,[6,6])
    np.testing.assert_array_equal(grid.step,[Fraction(1,63),Fraction(1,63)])
    raw['Generators']['g0']['Production cost curve (MW)'] = [2.]
    raw['Generators']['g0']['Production cost curve ($)'] = [7.]
    unit, = load_instance(write_raw(tmp_path,raw)).units
    grid = full_range_grid(unit,6,1.)
    np.testing.assert_array_equal(grid.bits,[0,0])
    np.testing.assert_array_equal(grid.levels,[0,0])
    np.testing.assert_array_equal(grid.step,[0,0])


def test_fractional_quantum():
    np.testing.assert_allclose(round_power([.3,.31],.1,up=True),[.3,.4])
    np.testing.assert_allclose(round_power([.3,.31],.1,up=False),[.3,.3])
    for q in [0,-1,float('nan')]:
        with pytest.raises(ValueError): round_power([1],q,up=True)


def test_missing_and_invalid_data(tmp_path):
    with pytest.raises(FileNotFoundError,match='10_0_1_w'): load_all(tmp_path)
    raw = raw_fixture()
    raw['Generators']['g0']['Production cost curve (MW)'] = [1.1,1.9]
    with pytest.raises(ValueError,match='empty output interval'):
        load_instance(write_raw(tmp_path,raw))
    raw = raw_fixture()
    raw['Buses']['b1']['Load (MW)'] = [1]
    with pytest.raises(ValueError,match='hourly values'): load_instance(write_raw(tmp_path,raw))
    raw = raw_fixture()
    raw['Generators']['g0']['Type'] = 'Wind'
    with pytest.raises(ValueError,match='non-thermal'): load_instance(write_raw(tmp_path,raw))


def test_integer_assertion_and_capacity_regression():
    from dataclasses import replace
    instance = load_all()[6]
    for B in range(2,7):
        capacity = sum(Fraction(int(u.pmin[0]))+full_range_grid(u,B).step[0]*int(full_range_grid(u,B).levels[0]) for u in instance.units)
        assert capacity == 3343
        assert capacity >= max(instance.demand)
    unit = replace(instance.units[0],pmin=np.array([1.5]),pmax=np.array([5.]))
    with pytest.raises(AssertionError,match='integers'): full_range_grid(unit,2)
    with pytest.raises(ValueError,match='q = 1'): load_all(q=.5)
