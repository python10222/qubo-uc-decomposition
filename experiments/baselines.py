"""Run with python -m experiments.baselines; only certified solves enter CSV."""
import argparse
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from src.loader import load_all, full_range_grid, slack_grid
from src.models import build_model, solve, Solution, true_cost, validate_solution, solve_discretized, runtime_metadata

ROOT = Path(__file__).resolve().parents[1]
BITS = (2,3,4,5,6)


def json_value(value):
    if isinstance(value,Fraction):
        return str(value)
    if isinstance(value,np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def fingerprint(instance,B):
    digest = hashlib.sha256()
    for source in ('src/loader.py','src/models.py'):
        digest.update((ROOT/source).read_bytes())
    digest.update(json.dumps([instance.audit['sha256'],instance.q,B]).encode())
    return digest.hexdigest()


def grid_shortfalls(instance, B):
    for unit in instance.units:
        full_range_grid(unit,B,instance.q)  # asserts exact top-level equality
    maximum = [sum(int(u.pmax[t]) for u in instance.units) for t in range(instance.hours)]
    return [{'hour':t+1,'capacity_mw':maximum[t],'demand_mw':int(instance.demand[t]),
             'shortfall_mw':int(instance.demand[t])-maximum[t]}
            for t in range(instance.hours) if maximum[t] < int(instance.demand[t])]


def certify_grid_infeasibility(instance, B, shortfalls, time_limit):
    built = build_model(instance,B)
    model = built.model
    if time_limit is not None:
        model.setParam('limits/time',time_limit)
    model.optimize()
    if str(model.getStatus()) != 'infeasible':
        raise RuntimeError(f'{instance.name} B={B}: infeasibility not certified ({model.getStatus()})')
    metadata = runtime_metadata(model)
    metadata['method'] = 'direct_scip_infeasibility'
    return {'status':'infeasible','objective':None,'dual_bound':None,'gap':None,
            'solve_seconds':model.getSolvingTime(),'build_seconds':built.build_seconds,
            'on':None,'power':None,'startup':None,
            'bit_depths':{u.name:full_range_grid(u,B,instance.q).bits.tolist() for u in instance.units},
            'metadata':metadata,'infeasibility_certificate':shortfalls}


def run_one(instance,B,directory,time_limit):
    tag = 'cont' if B is None else f'B{B}'
    path = directory/f'{instance.name}_{tag}.json'
    key = fingerprint(instance,B)
    if path.exists():
        cached = json.loads(path.read_text())
        if (cached.get('fingerprint') == key and cached.get('status') == 'infeasible'
                and B is not None and cached.get('infeasibility_certificate') == grid_shortfalls(instance,B)
                and cached['infeasibility_certificate']):
            return instance.name,B,cached
        if cached.get('fingerprint') == key and cached.get('status') == 'optimal' and cached.get('gap') == 0:
            solution = Solution(**{k:np.array(v) if k in ('on','power','startup','grid_indices','power_scaled') and v is not None else v
                                  for k,v in cached.items() if k in Solution.__dataclass_fields__})
            validate_solution(instance,solution,B)
            if abs(true_cost(instance,solution.on,solution.power)-solution.objective)>1e-6:
                raise AssertionError('cached cost mismatch')
            return instance.name,B,cached
    shortfalls = grid_shortfalls(instance,B) if B is not None else []
    result = (certify_grid_infeasibility(instance,B,shortfalls,time_limit) if shortfalls
              else asdict(solve(build_model(instance),time_limit=time_limit) if B is None
                          else solve_discretized(instance,B,time_limit=time_limit)))
    result.update({'instance':instance.name,'B':B,'fingerprint':key,
                   'completed_utc':datetime.now(timezone.utc).isoformat(),
                   'time_limit_seconds':time_limit})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,default=json_value,indent=2)+'\n')
    temporary.replace(path)
    return instance.name,B,result


def compact_hourly(values):
    return values[0] if len(set(values))==1 else values


def write_outputs(instances,solved,failures,output,workers):
    rows = []
    for instance in instances:
        cont = solved.get((instance.name,None),{})
        for B in BITS:
            disc = solved.get((instance.name,B),{})
            zc,zd = cont.get('objective'),disc.get('objective')
            gap = zd-zc if zc is not None and zd is not None else None
            if gap is not None and gap < -1e-4:
                raise AssertionError('discrete feasible set cannot beat continuous optimum')
            bits = {unit.name:compact_hourly(full_range_grid(unit,B,instance.q).bits.tolist()) for unit in instance.units}
            steps = {unit.name:compact_hourly(full_range_grid(unit,B,instance.q).step.tolist()) for unit in instance.units}
            slack_step,slack_bits = slack_grid(instance,B)
            metadata = cont.get('metadata',disc.get('metadata',{}))
            rows.append({'instance':instance.name,'B':B,'q_mw':instance.q,
                'z_cont':zc,'z_disc':zd,'G_disc_abs':gap,
                'G_disc_pct':100*gap/zc if gap is not None and zc else None,
                'cont_solve_seconds':cont.get('solve_seconds'), 'disc_solve_seconds':disc.get('solve_seconds'),
                'cont_build_seconds':cont.get('build_seconds'),'disc_build_seconds':disc.get('build_seconds'),
                'cont_status':cont.get('status','failed_or_pending'),'disc_status':disc.get('status','failed_or_pending'),
                'cont_optimality_gap':cont.get('gap'),'disc_optimality_gap':disc.get('gap'),
                'cont_dual_bound':cont.get('dual_bound'),'disc_dual_bound':disc.get('dual_bound'),
                'bit_depths':json.dumps(bits,separators=(',',':')),
                'steps_mw':json.dumps(steps,separators=(',',':'),default=json_value),
                'slack_step_mw':str(slack_step),'slack_bits_by_hour':json.dumps(slack_bits.tolist()),
                'grid_rule':'full_range_rational_v2','top_levels_equal_pmax':True,
                'data_format_version':instance.audit['data_format_version'],
                'cpu_model':metadata.get('cpu_model'), 'scip_threads':metadata.get('scip_threads'),
                'runner_workers':workers, 'library_versions':json.dumps(metadata.get('libraries',{}),sort_keys=True),
                'scip_version':metadata.get('scip_version'),'python_version':metadata.get('python'),
                'solver_parameters':json.dumps(metadata.get('parameters',{}),sort_keys=True)})
    with (output/'baselines.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    report = ROOT/'reports/task1.md'
    lines = ['# Task 1: continuous and discretized UC baselines', '',
        f'Generated {datetime.now(ZoneInfo("Asia/Riyadh")).isoformat()}. Base commit: `{subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}`.', '',
        '## Data and preprocessing', '',
        'All ten committed OR-LIB files were loaded using gzip magic-byte detection. '
        'Their UnitCommitment.jl input format version is **0.3**. None has a generator `Type` field; '
        'every generator is treated as thermal, with an explicit per-generator audit entry. '
        'Each instance has one bus, 24 hourly loads, and 10 or 20 units.', '',
        'The raw cost form is a convex **piecewise-linear total-cost curve with five breakpoints** per unit. '
        'The breakpoints lie on convex quadratics to floating-point precision. We fit `a + b*p + c*p²` '
        'by least squares at the original breakpoints, enforcing `c >= 0`, then use those same coefficients '
        'in both reference models. The intercept `a` is charged only when on; the first breakpoint cost '
        'is not used as a second no-load charge. Linear curves use `c = 0`.', '',
        'Pmin is rounded up, Pmax down, and aggregate bus demand up to q = 1 MW. '
        'All benchmark capacity endpoints were already integral; all 240 hourly demands rounded upward. '
        'Original and rounded arrays, per-unit/per-hour coefficients, RMSE, maximum absolute fit errors, '
        'raw file SHA-256 hashes, assumptions, and every dropped field (including its value) are in '
        '`results/preprocessing.json`. All raw generator parameters are static in these ten files; demand varies hourly. '
        'No benchmark unit is marked must-run; the loader/model support hourly must-run flags.', '',
        '| Instance | Units | Maximum breakpoint fit error ($) | Maximum unit fit RMSE ($) | Dropped-field entries |',
        '|---|---:|---:|---:|---:|']
    for instance in instances:
        fits = [f for unit in instance.audit['units'].values() for f in unit['fits_by_hour']]
        lines.append(f"| {instance.name} | {len(instance.units)} | {max(f['max_abs_error'] for f in fits):.3e} | {max(f['rmse'] for f in fits):.3e} | {len(instance.audit['dropped_fields'])} |")
    lines += ['', 'Dropped fields in every instance: generator `Bus` (network allocation), `Initial power (MW)` '
        '(only initial on/off status is retained), `Minimum uptime (h)`, `Minimum downtime (h)`, '
        '`Reserve eligibility`, `Startup delays (h)`, `Ramp up limit (MW)`, `Ramp down limit (MW)`, '
        'and the complete `Reserves` object. Each raw generator has one startup category; its cost is retained. '
        'For inputs with multiple categories, the loader takes the cheapest hourly cost and logs the discarded category structure. '
        'Initial-status duration is reduced to its sign; minimum up/down carryover is outside scope. '
        'Source and license text are retained as metadata.', '',
        '## Models and optimality', '',
        'Both models retain binary commitment, exact startup transitions (including initial status), '
        'demand ≥ load, output bounds, and must-run constraints. The continuous model uses SCIP with '
        'a convex perspective epigraph `c*p² <= v*u`, exact for binary commitment. '
        'The discrete model uses an integer index `k = 0,…,2^B_i−1`, equivalent to all binary-weighted '
        'bit patterns, with `k <= (2^B_i−1)*u` and `p = Pmin*u + Δ_i*k`. '
        'Convex adjacent-level chord inequalities evaluate the quadratic exactly at every grid point, '
        'so SCIP solves an exact MILP for this restricted quadratic model.', '',
        'For the full discrete runs, SCIP certifies hourly dispatch patterns in increasing production cost. '
        'At fixed hourly commitments, dispatch is independent of every other hour. A separate SCIP master '
        'selects generated patterns with the original startup costs and initial status. Each hour also has '
        'an optimistic unseen-pattern option, priced at a proven lower bound on all patterns not yet generated '
        'and allowing arbitrary commitments. This makes the master a relaxation of the entire grid model. '
        'Whenever its optimum uses an unseen option, another pattern is generated for that hour. '
        'Termination requires a proven-optimal master using only known, feasible patterns: its lower bound '
        'then equals a feasible full-model cost and proves global optimality. No bit pattern is removed from '
        'the full feasible set. Every pricing solve and master solve uses SCIP and retains its certificate.', '',
        'Discrete solve time is the sum of all hourly pricing and master SCIP solve times. '
        'Build time includes all hourly models and master rebuilds. Component timings, pattern counts, '
        'master bounds, and the final empty unseen-option set are in each run JSON. '
        'The direct 24-hour grid MILP remains available through `solve(build_model(instance, B))` and is '
        'cross-checked against pattern generation and enumeration on small instances.', '',
        'Every nonzero-width unit uses B bits and `Δ_i = (Pmax_i−Pmin_i)/(2^B−1)`; '
        'fixed-output units use zero bits. All rounded bounds and demands are asserted integer. '
        'The loader stores steps as exact Fractions. The SCIP grid equation is written with integer '
        'coefficients: `L*p = L*Pmin*u + (Pmax−Pmin)*k`, where `L=2^B−1`. '
        'Demand balance is imposed on the same scaled integer expression.', '',
        'Discrete artifacts retain integer `grid_indices`, `power_scaled`, and `power_denominator=L`. '
        'SCIP integer values are decoded within tolerance, and raw floating-point output must agree with '
        'the decoded schedule within 1e-5 MW. Grid membership, bounds and demand surplus are then checked '
        'with exact integer arithmetic. Floating-point equality is never used to decide grid membership.', '',
        'Slack step is `Δ_s=1/L` MW. Every output, demand and surplus has integer scaled ticks. '
        'Slack bits are `M_t=ceil(log2((ΣPmax−D_t)/Δ_s+1))`, evaluated exactly as '
        'the bit length of the nonnegative integer maximum surplus in ticks. Exact steps and hourly slack '
        'bit counts appear in the CSV. No QUBO is constructed.', '',
        'Only SCIP `optimal` status with reported gap exactly zero is accepted as a baseline. '
        'This is a numerical optimality certificate under SCIP tolerances, not an exact-arithmetic proof. '
        'Relative and absolute stopping gaps are zero; feasibility tolerance is 1e-8. '
        'Costs are recomputed from the returned schedules, and demand, bounds, must-run, startup, and grid '
        'membership are checked independently. Full schedules, solver bounds, versions, CPU model, settings, '
        'and timings are retained in `results/runs/`. Continuous results are solved once per instance '
        'and reused across its five rows. Build time and SCIP solve time are separate.', '',
        f'The runner used {workers} worker processes with one SCIP thread per solve. '
        'Timing reflects concurrent worker contention and is not a solver speed comparison.', '',
        '## Results', '',
        '![Discretization gap (%) versus bit depth, with separate panels for 10-unit and 20-unit instances.](figures/gdisc_vs_B.png)', '',
        '| Instance | B | z_cont | z_disc | G_disc ($) | G_disc (%) | cont seconds | disc seconds | cont gap | disc gap | B_i (g0, g1, …) | disc status |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        def fmt(key,digits=6):
            return '' if row[key] is None else f'{row[key]:.{digits}f}'
        bit_text = ','.join(str(v) for v in json.loads(row['bit_depths']).values())
        lines.append(f"| {row['instance']} | {row['B']} | {fmt('z_cont')} | {fmt('z_disc')} | {fmt('G_disc_abs')} | {fmt('G_disc_pct')} | {fmt('cont_solve_seconds',3)} | {fmt('disc_solve_seconds',3)} | {fmt('cont_optimality_gap',0)} | {fmt('disc_optimality_gap',0)} | {bit_text} | {row['disc_status']} |")
    endpoint_checks = sum(i.hours*len(i.units)*len(BITS) for i in instances)
    lines += ['', '## Change from the first run', '',
        'The superseded rule floored each step to whole MW and reduced bit depths on narrow ranges. '
        'That could lower the top encodable output as B increased, removing physical capacity. '
        'For example, `10_0_1_w/g0` has Pmin=32 and Pmax=118 MW. At B=6 the old rule gave '
        'Δ=floor(86/63)=1 MW and a top level of 95 MW, losing 23 MW. The new step is '
        '86/63 MW and the exact top level is 118 MW.', '',
        'For `20_0_2_w` at B=6, the old aggregate grid capacity was 2,752 MW, versus '
        '3,343 MW of rounded physical capacity. Hour 12 demand was 2,912 MW, causing a 160 MW '
        'shortfall and the old infeasibility certificate. The new grid reaches all 3,343 MW. '
        'The old CSV, preprocessing audit, schedules, report, and implementation snapshots are preserved '
        'under `results/first_run/`. The new source fingerprint invalidates the old solve cache; '
        'all continuous and discrete references are rerun.', '',
        f'**Exact endpoint check: {endpoint_checks:,} unit-hour-B cases passed.** '
        'For every unit and B=2,…,6, `Pmin+Δ*(2^B−1)=Pmax` exactly (fixed-output units have '
        'zero bits). This assertion is enforced in the loader using Fractions and covered by tests. '
        'All output and demand values are on the exact slack lattice.', '',
        'Different B grids are not generally nested, so monotonicity is an observation, not an enforced '
        'constraint or a test requirement. Adjacent changes below use the unrounded computed costs; '
        'a 1e-5 dollar tolerance is used only to distinguish numerical ties.', '',
        '| Instance | Decreases | Ties | Increases | G_disc at B=2 (%) | G_disc at B=6 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    increase_notes = []
    adjacent_changes = []
    for instance in instances:
        group = [r for r in rows if r['instance']==instance.name]
        if all(r['G_disc_abs'] is not None for r in group):
            changes = [right['G_disc_abs']-left['G_disc_abs'] for left,right in zip(group,group[1:])]
            adjacent_changes.extend(changes)
            falls = sum(v < -1e-5 for v in changes)
            ties = sum(abs(v) <= 1e-5 for v in changes)
            rises = sum(v > 1e-5 for v in changes)
            lines.append(f"| {instance.name} | {falls} | {ties} | {rises} | {group[0]['G_disc_pct']:.6f} | {group[-1]['G_disc_pct']:.6f} |")
            for left,right in zip(group,group[1:]):
                if right['G_disc_abs']-left['G_disc_abs'] > 1e-5:
                    increase_notes.append(f"Increase observed for {instance.name}, B={left['B']}→{right['B']}: "
                                 f"{left['G_disc_abs']:.9f}→{right['G_disc_abs']:.9f} dollars.")
        else:
            lines.append(f"| {instance.name} | pending | | | | |")
    lines.extend(['',*increase_notes])
    if len(adjacent_changes) == 40:
        lines += [f"Observed over all 40 adjacent B comparisons: "
                  f"{sum(v < -1e-5 for v in adjacent_changes)} decreases, "
                  f"{sum(abs(v) <= 1e-5 for v in adjacent_changes)} ties, and "
                  f"{sum(v > 1e-5 for v in adjacent_changes)} increases. No result was adjusted.", '']
    lines += ['', '## Validation, execution, and scope', '',
        f'{sum(r["status"] == "optimal" for r in solved.values())} of 60 unique solves are proven optimal; '
        f'{sum(r["status"] == "infeasible" for r in solved.values())} certified infeasible; '
        f'{len(failures)} execution failures. '
        'Infeasible and unproven objective/gap cells are empty, never presented as costs. See `results/failures.json`.', '',
        'Validation covers all ten input files, integer rounding, cost fitting, dropped fields, '
        'hourly parameters, exact endpoint equality for every unit/hour/B, narrow ranges retaining all B bits, '
        'fixed output, exact slack representability and minimal slack bits, startup coupling, must-run, '
        'independent tiny-instance enumeration, direct SCIP cross-checks, and rejection of invalid integer '
        'schedule certificates. Artifact tests independently verify all saved schedules, objectives, bounds, '
        'CSV values, source fingerprints, and preprocessing records.', '',
        'QUBO construction and QUBO-dependent tests from AGENTS.md are deferred by the explicit Task 1 scope. '
        'No z_Q, z_S, penalty gaps, heuristic results, or QUBO certificates are claimed.', '',
        'Reproduce with the pinned environment in `requirements.lock`:', '',
        '```sh', 'python -m pytest -q tests/test_loader.py tests/test_models.py',
        f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.baselines --workers {workers}',
        'python -m pytest -q', '```', '', 'The runner validates source/data fingerprints before resuming cached solves. '
        'Use `--time-limit SECONDS` for bounded runs; any unproven solve leaves empty cells and produces a nonzero exit.']
    validation_path = output/'validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if validation.get('run_fingerprints') == sorted(r['fingerprint'] for r in solved.values()):
            lines += ['', f"Final artifact validation: **{validation['tests']} tests passed**, "
                      f"{validation['failures']} failures, {validation['errors']} errors, "
                      f"{validation['skipped']} skipped ({validation['seconds']:.2f} seconds). "
                      'The validation record is in `results/validation.json`.']
    report.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--time-limit',type=float,default=None)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    instances = load_all()
    output = ROOT/'results'
    directory = output/'runs'
    directory.mkdir(parents=True,exist_ok=True)
    (output/'preprocessing.json').write_text(json.dumps({i.name:i.audit for i in instances},indent=2)+'\n')
    solved, failures = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one,i,B,directory,args.time_limit):(i.name,B)
                   for i in instances for B in (None,*BITS)}
        for future in as_completed(futures):
            name,B = futures[future]
            try:
                name,B,result = future.result()
                solved[name,B] = result
                value = '' if result['objective'] is None else f" {result['objective']:.9f}"
                print(f"{name} B={B}: {result['status']}{value}; {result['solve_seconds']:.3f}s",flush=True)
            except Exception as error:
                failures.append({'instance':name,'B':B,'error':str(error),'traceback':traceback.format_exc()})
                print(f'{name} B={B}: FAILED {error}',flush=True)
            write_outputs(instances,solved,failures,output,args.workers)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
