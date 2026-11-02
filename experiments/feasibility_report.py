"""Render Task 3 solely from saved measurements."""
import csv
import json
import numpy as np
from experiments.feasibility import OUT, ROOT, reduced
from src.loader import load_all
from src.qubo import build_qubo


def read(pattern):
    return [json.loads(p.read_text()) for p in sorted(OUT.glob(pattern))]


def fmt(v):
    if v is None: return ''
    if isinstance(v,float): return f'{v:.9g}'
    return str(v)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in rows])+'\n'


def main():
    thresholds=read('*_threshold.json'); heuristic=read('*_SA_*.json')+read('*_SB_*.json')
    probes=[r for p in sorted(OUT.glob('*_l[12]_*.json')) if (r:=json.loads(p.read_text()))]
    references={(r['instance'],r['B']):r['z_disc'] for r in thresholds}
    for r in probes:
        r['G_pen']=r['true_cost']-references[r['instance'],r['B']] if r['status']=='optimal' and r.get('feasible') else None
    lines=['# Task 3: feasibility','',
        'All measurements use the committed OR-LIB data and `src/qubo.py` with linear encoding and slack penalties. The starting commit after the explicitly requested sync was `d12e80c`. Raw measurements, schedules and runtime metadata are in [`results/task3`](../results/task3/); the experiment code is [`experiments/feasibility.py`](../experiments/feasibility.py).',
        '', '## Method and timing conditions','',
        'No solver hyperparameters were tuned. SCIP uses the repository’s fixed single-thread, zero-gap settings, feasibility tolerance 1e-8 and random seed shift 0. Each QUBO solve has a 600-second SCIP limit. Its binary variables are exactly the QUBO bits; continuous auxiliary variables lift affine output, squared residual/cost terms and binary products without imposing physical UC constraints. Thus infeasible schedules remain in the search space. This is an equivalent representation of the QUBO, not a constrained UC substitute. The returned bit string is independently evaluated with `QUBO.energy` and rational feasibility checks. Proven optimal means SCIP status `optimal`, zero gap, subject to floating-point solver tolerances, not a formal exact-arithmetic certificate.',
        '', 'Brute force is used when n ≤ 22; the requested proportional reductions all exceed that cutoff. A test compares the lifted SCIP model against exhaustive enumeration. The existing 2-unit/2-hour brute-force test, discretized-optimum energy checks and a computed four-cost telescoping check also pass.',
        '', 'Timing is wall time with `time.perf_counter`. Exact-QUBO timing includes model construction and optimization; the raw records also separate build and SCIP time. Heuristic timing includes the sampler call and its internal conversion, but excludes QUBO construction and post-run feasibility evaluation. The threshold, heuristic and size sweeps run sequentially; lightweight artifact inspection and validation can occur during a sweep. Wall times describe this shared cloud runtime. All coefficients and solver arithmetic use float64; decoded grid checks use exact rational arithmetic. Every timing record includes CPU, thread configuration, library versions and dtype. The five requested seeds override AGENTS.md’s general ten-seed rule.', '']
    if probes:
        meta=probes[0]['metadata']
        lines += ['Runtime metadata shared by the measurements (also embedded in every record):','', '```json',json.dumps(meta,indent=2),'```','',
                  'The container CPU quota is 2 CPU equivalents (`cpu.max = 200000 100000`); logical CPU count is host visibility. OMP/OpenBLAS and PyTorch use 2 threads; SCIP uses 1. No GPU is used.','']
    lines+=['## Reduced data','',
        'For each hour, demand is computed with integer arithmetic as `ceil(D_full × kept_capacity / full_capacity)`, and asserted no larger than kept capacity. Unit order is the loader/raw-file order. Limits, fitted quadratic costs `(a,b,c)`, cheapest start-up costs, must-run flags and initial on/off states are unchanged. The audit field in each saved data JSON retains the source preprocessing log, including dropped minimum up/down times, ramp limits, reserves, network and extra start-up categories.','']
    for data in read('*_data.json'):
        lines += [f"### {data['name']}",'',f"Demand (MW): `{data['demand']}`. Full per-hour data: [`JSON`](../results/task3/{data['name']}_data.json).",'']
        rows=[]
        for u in data['units']:
            rows.append([u['name'],u['initial_on'],u['pmin'],u['pmax'],u['startup'],u['must_run'],u['cost']])
        lines += [table(['Unit','Initially on','Pmin by hour','Pmax by hour','Start-up by hour','Must run','(a,b,c) by hour'],rows)]
    lines += ['## Exact penalty thresholds','',
        'For each setting, solve the hard discrete model to proven optimality for z_disc. For λ₂, hold λ₁ = 10 max(a+S); start at zero and the Section 5 estimate, double the upper endpoint until the returned exact ground state is feasible and has true cost equal to z_disc (absolute tolerance 1e-5), then bisect until `(upper−lower)/upper ≤ 1%`. For λ₁, hold λ₂ at 10 times the certified feasible upper endpoint of its bracket and repeat. The lower endpoint fails this criterion and the upper endpoint passes. At an exact tie, feasible and infeasible minimizers may coexist; brackets approximate the transition/infimum and do not claim uniqueness or that every tied ground state is feasible. Zero is reported explicitly if it passes.',
        '', 'Δ_min is the smallest positive **unit-output** grid step, as in Task 2; it is not the much smaller common slack/aggregate lattice step. The comparison estimates are λ₂ = b_max/Δ_min and λ₁ = max(a+S). Brackets and every individual solve, including returned bits, are saved. An infeasible exact ground state has `infeasible_ground_state=true` and G_pen=NaN (blank in the CSV); infeasibility is never treated as an objective gap. All derived ground-state gaps and per-probe timings are in [`ground_states.csv`](../results/task3/ground_states.csv).','']
    rows=[]
    for r in thresholds:
        x=r.get('lambda2_bracket');y=r.get('lambda1_bracket')
        rows.append([r['k'],r['T'],r['B'],r['n'],r['z_disc'],'' if x is None else f'({x[0]:.9g}, {x[1]:.9g}]',r['estimate_lambda2'],None if not x else x[1]/r['estimate_lambda2'],'' if y is None else f'({y[0]:.9g}, {y[1]:.9g}]',r['estimate_lambda1'],None if not y else y[1]/r['estimate_lambda1'],r.get('failure','')])
    lines += [table(['k','T','B','n','z_disc','λ₂ bracket','λ₂ estimate','upper/estimate','λ₁ bracket','λ₁ estimate','upper/estimate','Failure'],rows)]
    witnesses=[]
    source=load_all()[0]
    for r in thresholds:
        bracket=r.get('lambda2_bracket')
        if not bracket or bracket[0]==0: continue
        candidates=[p for p in probes if p['instance']==r['instance'] and p['B']==r['B'] and p['lambda2']==bracket[0] and p['lambda1']==10*r['estimate_lambda1']]
        if not candidates: continue
        p=candidates[-1]
        inst=reduced(source,r['k'],r['T'])
        q=build_qubo(inst,inst.hours,r['B'],p['lambda1'],p['lambda2'])
        u,power,slack=q.decode(p['bits'])
        residual=[sum(power[:,t])-int(inst.demand[t])-slack[t] for t in range(inst.hours)]
        p1=sum(p['bits'][j]*(1-int(u[i,t])) for (i,t,_),j in q.b.items())
        witnesses.append([r['k'],r['T'],r['B'],[str(v) for v in residual],p1,r['z_disc']-q.true_cost(u,power)])
    lines += ['### Lower-endpoint witnesses','',
        'Signed supply − demand − slack residuals at the failing λ₂ endpoint explain why the unit-step estimate can miss the transition. Combinations of different unit grids can change aggregate supply by much less than Δ_min; a small residual incurs a squared penalty while changing which dispatch combination is cheapest. Adjacent full-range grids have denominators 3, 7, 15 and 31 and are not nested, so changing B also changes the attainable near-demand combinations. Start-up/no-load choices and quadratic fuel costs also lie outside the linear one-unit estimate. The rows below report computed witnesses rather than assuming a doubling law.','',
        table(['k','T','B','Exact residuals (MW)','Off-output bit violations','z_disc − witness true cost'],witnesses)]
    fits=[]
    for k in [3,4,5]:
      for T in [2,4]:
        rs=sorted([r for r in thresholds if r['k']==k and r['T']==T and 'lambda2_bracket' in r],key=lambda r:r['B'])
        if len(rs)<2: continue
        Bs=np.array([r['B'] for r in rs]); vs=np.array([r['lambda2_bracket'][1] for r in rs])
        slope,intercept=np.polyfit(Bs,np.log(vs),1)
        pred=intercept+slope*Bs
        denom=np.sum((np.log(vs)-np.log(vs).mean())**2)
        fits.append([k,T,len(rs),vs[1]/vs[0] if len(rs)>1 else None,vs[2]/vs[1] if len(rs)>2 else None,vs[3]/vs[2] if len(rs)>3 else None,float(np.exp(slope)),float(1-np.sum((np.log(vs)-pred)**2)/denom) if denom else None])
    lines += ['### Scaling with bit depth','', 'Fit `log(λ₂ upper) = intercept + slope × B` separately for each reduction; `exp(slope)` is the fitted multiplicative change per added bit. Adjacent ratios are also shown because a four-point exponential fit can hide lattice effects. The Section 5 asymptotic prediction is about 2× per added bit; its finite-B estimate ratios are 7/3, 15/7 and 31/15.','',table(['k','T','Points','B3/B2','B4/B3','B5/B4','Fitted ratio/bit','R² (log)'],fits)]
    if fits:
        centered=[]
        for k in [3,4,5]:
            for T in [2,4]:
                rs=[r for r in thresholds if r['k']==k and r['T']==T and 'lambda2_bracket' in r]
                if len(rs)!=4: continue
                x=np.array([r['B'] for r in rs],float);y=np.log([r['lambda2_bracket'][1] for r in rs])
                centered.extend(zip(x-x.mean(),y-y.mean()))
        if centered:
            x,y=np.array(centered).T;slope=float(x@y/(x@x))
            within_r2=float(1-np.sum((y-slope*x)**2)/np.sum(y*y))
            lines += [f'A pooled log-linear fit with a separate intercept for each reduction gives **{np.exp(slope):.6g}× per added bit** (within-reduction R²={within_r2:.6g}). This averages strongly different and sometimes non-monotone responses.','']
    if fits:
        lines += ['The data do **not** support a consistent doubling rule over B=2–5: several adjacent thresholds decrease, and the fitted ratios range from '+fmt(min(r[6] for r in fits))+' to '+fmt(max(r[6] for r in fits))+'. Low R² in several reductions makes a single exponential law a poor description.','']
    if thresholds:
        ratios2=[r['lambda2_bracket'][1]/r['estimate_lambda2'] for r in thresholds if 'lambda2_bracket' in r]
        ratios1=[r['lambda1_bracket'][1]/r['estimate_lambda1'] for r in thresholds if 'lambda1_bracket' in r]
        lines += [f'Across these settings, λ₂ upper/estimate ranges from {min(ratios2):.5g} to {max(ratios2):.5g}; λ₁ upper/estimate ranges from {min(ratios1):.5g} to {max(ratios1):.5g}. The λ₁ estimate is much closer to the measured thresholds.','']
    if probes:
        lines += [f"Exact bisection probes completed: {len(probes)}; proven optimal: {sum(r['status']=='optimal' and r['gap']==0 for r in probes)}. Total probe wall time: {sum(r['wall_seconds'] for r in probes):.3f} s. Per-probe timing and metadata are in the linked raw records.",'']
    lines += ['## Full-instance heuristic trials','',
        'B=4, λ₁=2 max(a+S), λ₂=2 b_max/Δ_min. SA uses dwave-samplers, 100 reads, default schedule, seeds 0–4. SB uses 128 agents, default ballistic mode, 10,000 maximum steps, unheated, default early stopping (sampling period 50, convergence threshold 50), seeds 0–4, CPU float64. Every SB call temporarily sets PyTorch’s default dtype to float64 and restores it in `finally`. `best_only=False` only retains all final agents for feasibility measurement; `verbose=False` only suppresses progress bars. No feasibility repair, resampling or parameter changes are applied.',
        '', 'Feasibility means the decoded **physical schedule** satisfies on/off, limits, exact grid membership and demand. Slack bits need not be optimal for a physically feasible sample. The additional zero-penalty count requires physical feasibility and exact supply−demand−slack=0 in every hour. Best energy includes the QUBO offset; best feasible cost is minimized over feasible samples, independently of energy rank. Rates count all returned reads/agents, including duplicates. A blank cost/gap means **no sample was feasible**. Full z_disc values are the committed, proven-optimal Task 1 certificates, re-encoded and independently costed. These trials alone cannot prove a full QUBO ground state feasible, even if no energy below z_disc is found (the certification wording in the stage documents is insufficient for that conclusion).','']
    lines += [table(['Instance','Solver','Seed','n','λ₁','λ₂','Wall s','Best energy','Feasible / total','Rate','Zero-penalty samples','Best feasible true cost','z_disc','Absolute gap','Gap %'],[[r['instance'],r['solver'],r['seed'],r['n'],r['lambda1'],r['lambda2'],r['wall_seconds'],r['best_energy'],f"{r['feasible_count']}/{r['samples']}",r['feasibility_rate'],r.get('zero_penalty_count'),r['best_feasible_cost'],r['z_disc'],r['gap_absolute'],r['gap_percent']] for r in heuristic])]
    aggregates=[]
    for name in ['10_0_1_w','20_0_1_w']:
        for method in ['SA','SB']:
            rs=[r for r in heuristic if r['instance']==name and r['solver']==method]
            if not rs: continue
            costs=[r['best_feasible_cost'] for r in rs if r['best_feasible_cost'] is not None]
            aggregates.append([name,method,len(rs),sum(r['wall_seconds'] for r in rs)/len(rs),min(r['best_energy'] for r in rs),sum(r['feasible_count'] for r in rs),sum(r['samples'] for r in rs),min(costs) if costs else None])
    lines += [table(['Instance','Solver','Seeds','Mean wall s','Best energy across seeds','Feasible samples','Total samples','Best feasible cost'],aggregates)]
    for r in aggregates:
        if r[5]==0:
            lines += [f'{r[0]}, {r[1]}: **no feasible sample** among {r[6]} returned samples.','']
    if heuristic and all(r.get('zero_penalty_count')==0 for r in heuristic):
        lines += ['No returned sample had zero total QUBO penalty. Physical feasibility and exact satisfaction of the encoded slack equation are therefore distinct outcomes in these trials.','']
    if heuristic and not any(r['feasible_count'] for r in heuristic):
        lines += ['**No sample was feasible in any recorded trial.** Consequently there is no best feasible true cost or gap to z_disc to report; an infeasible sample’s cost is not substituted.','']
    lines += ['## Largest reduced QUBO within ten minutes','',
        'A fixed size ladder uses B=5 and Task 2 weights (10× both estimates): k=5 with T=4,6,8,12,16,20,24, then k=6,7,8,9 with T=24. Each keeps the same proportional-demand rule. Stop at the first solve not proven optimal within 600 wall seconds. This measures the largest success on this stated ladder, not a global size limit over all possible reductions or weights. If a run times out, its feasibility column describes only the incumbent, not a proven ground state. No solver settings are changed between attempts. The complete data for these additional reductions are saved under [`capacity_data`](../results/task3/capacity_data/).','']
    capacity=read('capacity.json')
    if capacity:
        cs=capacity[0]
        lines += [table(['Instance','B','n','Wall s','SCIP s','Status','Gap (fraction)','Feasible'],[[r['instance'],r['B'],r['n'],r['wall_seconds'],r.get('solve_seconds'),r['status'],r['gap'],r.get('feasible')] for r in cs])]
        ok=[r for r in cs if r['status']=='optimal' and r['gap']==0 and r['wall_seconds']<=600]
        if ok:
            largest=max(ok,key=lambda r:r['n']);lines += [f"Largest on the ladder: **{largest['instance']}, B={largest['B']}, n={largest['n']}, {largest['wall_seconds']:.3f} s**.",'']
        else: lines += ['No ladder member was proven optimal within ten minutes.','']
    log_path=OUT/'capacity.log'
    if log_path.exists() and 'without GMP' in log_path.read_text():
        lines += ['SCIP emitted `Cannot set feasibility tolerance to small value 1e-11 without GMP - using 1e-10.` during the size sweep. No settings were changed in response. The configured model feasibility tolerance is 1e-8; this warning concerns SCIP’s internal request. The complete output is retained in [`capacity.log`](../results/task3/capacity.log).','']
    lines += ['## Coefficient range','',
        'Recomputed for all ten instances, B=2–6, with the Task 2 weights λ₁=10 max(a+S), λ₂=10 b_max/Δ_min. The plotted ratio is max/min absolute nonzero upper-triangular Q coefficient, including diagonal linear terms and excluding the constant offset.','',
        '![Coefficient range versus bit depth](figures/coef_range_vs_B.png)','',
        'Numeric minima, maxima and ratios: [`coefficients.json`](../results/task3/coefficients.json).','',
        '## Reproduction and validation','',
        'Use the environment pinned by `requirements.lock`. No dependencies were added.','',
        '```bash','export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/task3-mpl',
        'python -m pytest -q','python -m experiments.feasibility thresholds','python -m experiments.feasibility heuristic','python -m experiments.feasibility capacity','python -m experiments.feasibility coefficients','python -m experiments.feasibility diagnostics','python -m experiments.feasibility_report','```','',
        'The runner checkpoints every solve and resumes completed measurements. Use a fresh `results/task3` directory to rerun, rather than mixing measurements from different environments. Raw JSON contains hardware/library/dtype metadata for every timing. Heuristic bit strings are retained in compressed NumPy archives.','']
    failures=[r for r in thresholds if 'failure' in r]
    lines += [f"Threshold settings completed: {len(thresholds)-len(failures)}/{len(thresholds)}. Unresolved thresholds: {len(failures)}. Full heuristic trials recorded: {len(heuristic)}/20. Missing or timed-out results are left blank, not estimated.",'']
    validation=read('validation.json')
    if validation:
        v=validation[0]
        lines += [f"Final validation: **{v['tests_passed']} tests passed**. The artifact audits independently re-evaluate {v['threshold_endpoints']} threshold endpoints and all {v['heuristic_samples']} saved heuristic samples, including exact rational feasibility, energies and feasible costs.",'']
    (ROOT/'reports/task3.md').write_text('\n'.join(lines))
    for name,rows in [('thresholds',thresholds),('heuristics',heuristic),('ground_states',probes)]:
        if not rows: continue
        keys=sorted(set().union(*(r.keys() for r in rows))-{'bits'})
        with (OUT/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

if __name__=='__main__': main()
