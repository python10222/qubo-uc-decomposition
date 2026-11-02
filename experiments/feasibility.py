"""Reproducible Task 3 measurements; run with OMP/OPENBLAS_NUM_THREADS=2.

SCIP solves an exact lifted representation of the unconstrained binary
polynomial: affine powers/residuals, convex-square epigraphs and binary ANDs.
No physical UC constraints are imposed on its QUBO search space.
"""
from dataclasses import asdict, replace
import importlib.metadata as md
import json
import os
from pathlib import Path
import time
import argparse
import numpy as np
from pyscipopt import Model, quicksum
from src.loader import load_all
from src.models import build_model, solve, runtime_metadata
from src.qubo import build_qubo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/task3'
OUT.mkdir(exist_ok=True)


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, default=serial, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metadata():
    m = runtime_metadata(Model())
    m.update(dtype='float64', omp_threads=os.environ.get('OMP_NUM_THREADS'),
             openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    m['libraries'].update({n: md.version(n) for n in ['torch','dimod','dwave-samplers','simulated-bifurcation','matplotlib','pandas']})
    return m


def estimates(inst, B):
    d = min((hi-lo)/(2**B-1) for u in inst.units for lo,hi in zip(u.pmin,u.pmax) if hi>lo)
    return float(max(a+s for u in inst.units for a,s in zip(u.cost[:,0],u.startup))), float(max(u.cost[:,1].max() for u in inst.units)/d)


def reduced(source, k, T):
    units = tuple(replace(u, **{f:getattr(u,f)[:T] for f in ['pmin','pmax','cost','startup','must_run']}) for u in source.units[:k])
    cap = [sum(int(u.pmax[t]) for u in units) for t in range(T)]
    allcap = [sum(int(u.pmax[t]) for u in source.units) for t in range(T)]
    demand = np.array([(int(source.demand[t])*cap[t]+allcap[t]-1)//allcap[t] for t in range(T)])
    assert np.all(demand <= cap)
    return replace(source,name=f'10_0_1_w_k{k}_T{T}', units=units,demand=demand)


def exact(q, limit=600):
    start=time.perf_counter()
    if q.num_variables<=22:
        best=float('inf'); state=None
        for lo in range(0,2**q.num_variables,65536):
            x=((np.arange(lo,min(lo+65536,2**q.num_variables),dtype=np.uint64)[:,None]>>np.arange(q.num_variables,dtype=np.uint64))&1).astype(np.float64)
            e=np.full(len(x),q.offset)
            for (i,j),v in q.Q.items(): e+=v*x[:,i]*x[:,j]
            at=e.argmin()
            if e[at]<best: best,state=e[at],x[at].astype(int)
        return dict(status='optimal',gap=0,wall_seconds=time.perf_counter()-start,method='brute_force',bits=state.tolist(),energy=q.energy(state),metadata=metadata())
    m=Model('unconstrained_QUBO'); m.hideOutput()
    for key,val in {'limits/time':limit,'limits/gap':0.,'limits/absgap':0.,'parallel/maxnthreads':1,'numerics/feastol':1e-8,'randomization/randomseedshift':0}.items(): m.setParam(key,val)
    x=[m.addVar(vtype='B',name=f'x{j}') for j in range(q.num_variables)]
    def product(a,b):
        if isinstance(a,int): return a*b
        if isinstance(b,int): return a*b
        z=m.addVar(lb=0,ub=1)
        m.addCons(z<=a); m.addCons(z<=b); m.addCons(z>=a+b-1)
        return z
    def square(affine, weight, name):
        if not weight: return 0
        r=m.addVar(lb=None,name=name)
        m.addCons(r==affine)
        v=m.addVar(lb=0)
        m.addCons(v>=r*r)
        return float(weight)*v
    obj=[]; powers={}; on={}
    for i,u in enumerate(q.instance.units):
        for t in range(q.T):
            on[i,t]=x[q.u[i,t]] if (i,t) in q.u else 1
            bits=[(j,k) for (ii,tt,k),j in q.b.items() if (ii,tt)==(i,t)]
            p=int(u.pmin[t])*on[i,t]+quicksum(float(q.grids[i].step[t]*2**k)*x[j] for j,k in bits)
            powers[i,t]=p
            a,b,c=map(float,u.cost[t]); prev=on[i,t-1] if t else int(u.initial_on)
            obj.extend([a*on[i,t],b*p,square(p,c,f'p{i}_{t}'),float(u.startup[t])*(on[i,t]-product(on[i,t],prev))])
            obj.extend(q.lambda1*(x[j]-product(x[j],on[i,t])) for j,k in bits)
    for t in range(q.T):
        r=quicksum(powers[i,t] for i in range(len(q.instance.units)))-int(q.instance.demand[t])-quicksum(float(q.slack_step*2**k)*x[j] for (tt,k),j in q.e.items() if tt==t)
        obj.append(square(r,q.lambda2,f'residual{t}'))
    m.setObjective(quicksum(obj),'minimize')
    build=time.perf_counter()-start
    m.optimize()
    row=dict(status=str(m.getStatus()),gap=float(m.getGap()),wall_seconds=time.perf_counter()-start,build_seconds=build,solve_seconds=m.getSolvingTime(),method='SCIP_lifted_QUBO',dual_bound=m.getDualbound(),metadata=metadata())
    if m.getNSols():
        bits=np.array([round(m.getVal(v)) for v in x]); row.update(bits=bits.tolist(),energy=q.energy(bits),solver_objective=m.getObjVal())
        assert abs(row['energy']-m.getObjVal()) < max(1e-3,abs(row['energy'])*1e-7), row
    return row


def probe(inst,B,l1,l2,label):
    path=OUT/f'{inst.name}_B{B}_{label}.json'
    if path.exists(): return json.loads(path.read_text())
    q=build_qubo(inst,inst.hours,B,l1,l2)
    row=exact(q)
    row.update(instance=inst.name,B=B,n=q.num_variables,lambda1=l1,lambda2=l2)
    if 'bits' in row:
        u,p,s=q.decode(row['bits']); row.update(feasible=q.is_feasible(u,p),true_cost=q.true_cost(u,p),zero_slack=all(sum(p[:,t])-int(inst.demand[t])==s[t] for t in range(q.T)))
        row['infeasible_ground_state']=row['status']=='optimal' and not row['feasible']
    save(path,row)
    print(path.name,row['status'],round(row['wall_seconds'],3),row.get('feasible'),flush=True)
    return row


def thresholds():
    source=load_all()[0]
    for k in [3,4,5]:
      for T in [2,4]:
        inst=reduced(source,k,T)
        save(OUT/f'{inst.name}_data.json',asdict(inst))
        for B in [2,3,4,5]:
            resultpath=OUT/f'{inst.name}_B{B}_threshold.json'
            if resultpath.exists(): continue
            a,b=estimates(inst,B)
            started=time.perf_counter(); sol=solve(build_model(inst,B),time_limit=600)
            ref=dict(z_disc=sol.objective,wall_seconds=time.perf_counter()-started,metadata=metadata(),on=sol.on,grid_indices=sol.grid_indices)
            save(OUT/f'{inst.name}_B{B}_reference.json',ref)
            q=build_qubo(inst,T,B,10*a,10*b)
            assert abs(q.energy(q.encode(sol.on,sol.grid_indices))-sol.objective)<1e-6
            summary=dict(instance=inst.name,k=k,T=T,B=B,n=q.num_variables,z_disc=sol.objective,estimate_lambda1=a,estimate_lambda2=b)
            def bisect(axis,fixed,initial):
                count=0
                def check(v):
                    nonlocal count
                    count+=1
                    r=probe(inst,B,fixed if axis==2 else v,v if axis==2 else fixed,f'l{axis}_{count:02d}')
                    if r['status']!='optimal' or r['gap']!=0: raise RuntimeError(f"{axis}: {r['status']}; no exact threshold")
                    return r['feasible'] and abs(r['true_cost']-sol.objective)<1e-5
                lo=0.; hi=initial
                if check(0): return [0.,0.]
                while not check(hi):
                    lo=hi; hi*=2
                    if count>=30: raise RuntimeError('unable to bracket threshold')
                while (hi-lo)/hi > .01:
                    mid=(lo+hi)/2
                    if check(mid): hi=mid
                    else: lo=mid
                return [lo,hi]
            try:
                summary['lambda2_bracket']=bisect(2,10*a,b)
                summary['lambda1_bracket']=bisect(1,10*summary['lambda2_bracket'][1],a)
            except RuntimeError as e: summary['failure']=str(e)
            save(resultpath,summary)


def heuristic():
    import torch
    import simulated_bifurcation as sb
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod
    torch.set_num_threads(2)
    for inst in [load_all()[0],load_all()[5]]:
      a,b=estimates(inst,4); q=build_qubo(inst,24,4,2*a,2*b)
      reference=json.loads((ROOT/f'results/runs/{inst.name}_B4.json').read_text())
      z=q.true_cost(*q.decode(q.encode(reference['on'],reference['grid_indices']))[:2])
      bqm=dimod.BinaryQuadraticModel.from_qubo(q.Q,offset=q.offset)
      mat=np.zeros((q.num_variables,q.num_variables),dtype=np.float64)
      for (i,j),v in q.Q.items(): mat[i,j]=v
      for method in ['SA','SB']:
       for seed in range(5):
        path=OUT/f'{inst.name}_{method}_{seed}.json'
        if path.exists(): continue
        meta=metadata();meta['torch_threads']=torch.get_num_threads()
        start=time.perf_counter()
        if method=='SA':
            samples=SimulatedAnnealingSampler().sample(bqm,num_reads=100,seed=seed)
            elapsed=time.perf_counter()-start
            xs=np.array([[r.sample[i] for i in range(q.num_variables)] for r in samples.data(sorted_by=None) for _ in range(r.num_occurrences)])
            parameters=dict(num_reads=100,seed=seed,schedule='default',num_sweeps=1000,num_sweeps_per_beta=1,beta_schedule_type='geometric',beta_range='automatic',initial_states_generator='random',randomize_order=False,proposal_acceptance_criteria='Metropolis',info=samples.info)
        else:
            old=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64); torch.manual_seed(seed)
                xs,energies=sb.minimize(mat,q.offset,domain='binary',dtype=torch.float64,agents=128,best_only=False,verbose=False)
                elapsed=time.perf_counter()-start
                assert xs.dtype==torch.float64
                xs=xs.numpy()
                returned_energies=energies.numpy()
                assert np.allclose(returned_energies,[q.energy(x) for x in xs],rtol=1e-8,atol=1e-3)
            finally: torch.set_default_dtype(old)
            parameters=dict(agents=128,seed=seed,max_steps=10000,mode='ballistic',heated=False,early_stopping=True,sampling_period=50,convergence_threshold=50,timeout=None,best_only=False,verbose=False,environment=sb.get_env())
        feasible=[]; energies=[]; zero_penalty_count=0
        for x in xs:
            u,p,slack=q.decode(x); energies.append(q.energy(x))
            if q.is_feasible(u,p):
                feasible.append(q.true_cost(u,p))
                zero_penalty_count+=int(all(sum(p[:,t])-int(inst.demand[t])==slack[t] for t in range(q.T)))
        best=min(feasible) if feasible else None
        row=dict(instance=inst.name,B=4,n=q.num_variables,solver=method,seed=seed,lambda1=2*a,lambda2=2*b,wall_seconds=elapsed,best_energy=min(energies),samples=len(xs),feasible_count=len(feasible),feasibility_rate=len(feasible)/len(xs),zero_penalty_count=zero_penalty_count,best_feasible_cost=best,z_disc=z,gap_absolute=None if best is None else best-z,gap_percent=None if best is None else 100*(best-z)/z,metadata=meta,parameters=parameters)
        save(path,row); np.savez_compressed(OUT/f'{inst.name}_{method}_{seed}_samples.npz',samples=xs)
        print(path.name,elapsed,len(feasible),flush=True)


def coefficients():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for inst in load_all():
        ratios=[]
        for B in range(2,7):
            a,b=estimates(inst,B); q=build_qubo(inst,24,B,10*a,10*b)
            vals=np.abs(list(q.Q.values())); ratio=float(vals.max()/vals.min()); ratios.append(ratio)
            rows.append(dict(instance=inst.name,B=B,lambda1=10*a,lambda2=10*b,minimum=float(vals.min()),maximum=float(vals.max()),ratio=ratio))
        ax.plot(range(2,7),ratios,marker='o',label=inst.name)
    ax.set(yscale='log',xlabel='Output bits B',ylabel='max |Q| / min nonzero |Q|',xticks=range(2,7),title='Coefficient range at Task 2 penalty settings')
    ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
    (ROOT/'reports/figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'reports/figures/coef_range_vs_B.png',dpi=180)
    save(OUT/'coefficients.json',rows)

def diagnostics():
    instances={i.name:i for i in load_all()}
    for path in list(OUT.glob('*_SA_*.json'))+list(OUT.glob('*_SB_*.json')):
        row=json.loads(path.read_text())
        inst=instances[row['instance']]
        q=build_qubo(inst,24,4,row['lambda1'],row['lambda2'])
        with np.load(path.with_name(path.stem+'_samples.npz')) as archive:
            samples=archive['samples']
        count=0
        for bits in samples:
            u,p,slack=q.decode(bits)
            if q.is_feasible(u,p) and all(sum(p[:,t])-int(inst.demand[t])==slack[t] for t in range(q.T)):
                count+=1
        row['zero_penalty_count']=count
        save(path,row)


def capacity():
    """Fixed size ladder, not a solver-parameter search; stop at first timeout."""
    source=load_all()[0]
    ladder=[(5,t) for t in [4,6,8,12,16,20,24]]+[(k,24) for k in [6,7,8,9]]
    rows=[]
    (OUT/'capacity_data').mkdir(exist_ok=True)
    for k,T in ladder:
        inst=reduced(source,k,T)
        save(OUT/'capacity_data'/f'{inst.name}.json',asdict(inst))
        a,b=estimates(inst,5)
        row=probe(inst,5,10*a,10*b,'capacity')
        rows.append(row)
        save(OUT/'capacity.json',rows)
        if row['status']!='optimal' or row['gap']!=0 or row['wall_seconds']>600: break


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['thresholds','heuristic','coefficients','capacity','diagnostics']);args=parser.parse_args()
    globals()[args.action]()
