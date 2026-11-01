"""Audit the committed experiment artifacts independently of CSV generation."""
import csv
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

from src.loader import load_all, full_range_grid, slack_grid
from experiments.baselines import fingerprint
from src.models import Solution, true_cost, validate_solution

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_artifacts_are_complete_and_certified():
    instances = {instance.name:instance for instance in load_all()}
    with (ROOT/'results/baselines.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert {(r['instance'],int(r['B'])) for r in rows} == {
        (name,B) for name in instances for B in range(2,7)}
    assert json.loads((ROOT/'results/failures.json').read_text()) == []
    results = {}
    for instance in instances.values():
        for B in (None,2,3,4,5,6):
            tag = 'cont' if B is None else f'B{B}'
            data = json.loads((ROOT/f'results/runs/{instance.name}_{tag}.json').read_text())
            assert data['fingerprint'] == fingerprint(instance,B)
            assert data['status'] == 'optimal' and data['gap'] == 0
            values = {field.name:data[field.name] for field in fields(Solution)}
            for key in ('on','power','startup','grid_indices','power_scaled'):
                if values[key] is None: continue
                values[key] = np.array(values[key])
                assert np.isfinite(values[key]).all()
            solution = Solution(**values)
            assert solution.on.shape == (len(instance.units),instance.hours)
            assert np.isin(solution.on,[0,1]).all()
            validate_solution(instance,solution,B)
            cost = true_cost(instance,solution.on,solution.power)
            assert cost == pytest.approx(data['objective'],abs=1e-6,rel=0)
            assert cost == pytest.approx(data['dual_bound'],abs=1e-4,rel=0)
            assert solution.solve_seconds > 0
            assert solution.metadata['cpu_model']
            assert solution.metadata['scip_threads'] == 1
            if B is not None:
                _,slack_bits = slack_grid(instance,B)
                for t in range(instance.hours):
                    ticks = sum(int(v) for v in solution.power_scaled[:,t])-int(instance.demand[t])*(2**B-1)
                    assert 0 <= ticks <= 2**int(slack_bits[t])-1
                certificate = solution.metadata['pattern_generation']
                assert certificate['final_unseen_hours'] == []
                assert certificate['master_certificates'][-1]['unseen_hours'] == []
                for hour in certificate['pricing_certificates']:
                    assert all(c['status'] in ('optimal','infeasible') for c in hour)
                    assert all(c['gap'] == 0 for c in hour if c['status']=='optimal')
            results[instance.name,B] = solution
    for row in rows:
        instance,B = instances[row['instance']],int(row['B'])
        continuous,discrete = results[instance.name,None],results[instance.name,B]
        assert float(row['z_cont']) == continuous.objective
        bits = json.loads(row['bit_depths'])
        for unit in instance.units:
            np.testing.assert_array_equal(np.broadcast_to(bits[unit.name],(instance.hours,)),
                                          full_range_grid(unit,B,instance.q).bits)
        slack_step,slack_bits = slack_grid(instance,B)
        assert row['slack_step_mw'] == str(slack_step)
        assert json.loads(row['slack_bits_by_hour']) == slack_bits.tolist()
        assert row['top_levels_equal_pmax'] == 'True'
        assert float(row['z_disc']) == discrete.objective
        gap = discrete.objective-continuous.objective
        assert gap >= -1e-4
        assert float(row['G_disc_abs']) == pytest.approx(gap,abs=1e-8,rel=0)
        assert float(row['G_disc_pct']) == pytest.approx(100*gap/continuous.objective,abs=1e-10,rel=0)
        for prefix,result in [('cont',continuous),('disc',discrete)]:
            assert row[f'{prefix}_status'] == 'optimal'
            assert float(row[f'{prefix}_optimality_gap']) == 0
            assert float(row[f'{prefix}_solve_seconds']) == result.solve_seconds
    audit = json.loads((ROOT/'results/preprocessing.json').read_text())
    assert set(audit) == set(instances)
    for name,instance in instances.items():
        assert audit[name] == instance.audit
