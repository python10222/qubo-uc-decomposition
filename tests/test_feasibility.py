import numpy as np
from src.loader import load_all
from src.qubo import build_qubo
from experiments.feasibility import reduced, exact, estimates


def test_proportional_reduction():
    src=load_all()[0]
    for k in [3,4,5]:
        for T in [2,4]:
            inst=reduced(src,k,T)
            kept=sum(u.pmax[:T] for u in src.units[:k])
            total=sum(u.pmax[:T] for u in src.units)
            assert np.array_equal(inst.demand,np.ceil(src.demand[:T]*kept/total))
            assert np.all(inst.demand<=kept)
            for u,v in zip(src.units,inst.units):
                assert u.initial_on==v.initial_on
                assert np.array_equal(u.cost[:T],v.cost)


def test_lifted_qubo_matches_direct_enumeration():
    # Force the SCIP path for a small polynomial; compare its independent
    # expanded energy against exhaustive enumeration of every bit string.
    from dataclasses import replace
    src=load_all()[0]
    inst=reduced(src,1,1)
    inst=replace(inst,demand=inst.units[0].pmax-1)
    q=build_qubo(inst,1,2,100,10)
    brute=exact(q)
    # Add isolated binary variables to exceed the brute force cutoff.
    q.variables.extend([('isolated',i) for i in range(23-len(q.variables))])
    scip=exact(q,30)
    assert scip['status']=='optimal'
    assert abs(scip['energy']-brute['energy'])<1e-6


def test_cost_decomposition_on_exact_reduction():
    from src.models import build_model, solve
    inst=reduced(load_all()[0],1,1)
    cont=solve(build_model(inst))
    disc=solve(build_model(inst,2))
    a,b=estimates(inst,2)
    q=build_qubo(inst,1,2,10*a,100*b)
    ground=exact(q,30)
    assert ground['status']=='optimal'
    u,p,_=q.decode(ground['bits'])
    assert q.is_feasible(u,p)
    zq=q.true_cost(u,p)
    assert abs(zq-disc.objective)<1e-6
    encoded=q.encode(disc.on,disc.grid_indices)
    assert abs(q.energy(encoded)-disc.objective)<1e-6
    # A distinct, feasible full-output sample supplies the fourth cost.
    sample=q.encode(np.ones((1,1)),np.full((1,1),3))
    zs=q.true_cost(*q.decode(sample)[:2])
    gdisc=disc.objective-cont.objective
    gpen=zq-disc.objective
    gsolv=zs-zq
    assert abs((gdisc+gpen+gsolv)-(zs-cont.objective))<1e-9


def test_saved_threshold_endpoints():
    """Audit saved certificates without rerunning the costly optimization."""
    import json
    from experiments.feasibility import OUT
    rows=list(OUT.glob('*_threshold.json'))
    if not rows:
        return  # Available after the threshold experiment.
    assert len(rows)==24
    source=load_all()[0]
    for path in rows:
        r=json.loads(path.read_text())
        inst=reduced(source,r['k'],r['T'])
        assert 'failure' not in r
        for axis in [1,2]:
            lo,hi=r[f'lambda{axis}_bracket']
            assert 0 <= lo <= hi and (hi-lo)/hi<=.01
            probes=[json.loads(p.read_text()) for p in OUT.glob(f"{r['instance']}_B{r['B']}_l{axis}_*.json")]
            for endpoint,expect in [(lo,False),(hi,True)]:
                row=next(p for p in probes if p[f'lambda{axis}']==endpoint)
                assert row['status']=='optimal' and row['gap']==0
                q=build_qubo(inst,inst.hours,r['B'],row['lambda1'],row['lambda2'])
                u,p,_=q.decode(row['bits'])
                passed=q.is_feasible(u,p) and abs(q.true_cost(u,p)-r['z_disc'])<1e-5
                assert passed==expect
                assert abs(q.energy(row['bits'])-row['energy'])<1e-6
                assert row['metadata']['dtype']=='float64'
                if axis==2:
                    assert row['lambda1']==10*r['estimate_lambda1']
                else:
                    assert row['lambda2']==10*r['lambda2_bracket'][1]


def test_saved_heuristic_samples():
    import json
    from experiments.feasibility import OUT
    instances={i.name:i for i in load_all()}
    for path in list(OUT.glob('*_SA_*.json'))+list(OUT.glob('*_SB_*.json')):
        row=json.loads(path.read_text())
        inst=instances[row['instance']]
        a,b=estimates(inst,4)
        assert row['lambda1']==2*a and row['lambda2']==2*b
        q=build_qubo(inst,24,4,2*a,2*b)
        with np.load(path.with_name(path.stem+'_samples.npz')) as archive:
            samples=archive['samples']
        assert len(samples)==row['samples']==(100 if row['solver']=='SA' else 128)
        feasible=[];energies=[]
        for bits in samples:
            u,p,_=q.decode(bits)
            energies.append(q.energy(bits))
            if q.is_feasible(u,p): feasible.append(q.true_cost(u,p))
        assert abs(min(energies)-row['best_energy'])<1e-6
        assert len(feasible)==row['feasible_count']
        assert len(feasible)/len(samples)==row['feasibility_rate']
        assert row['metadata']['dtype']=='float64'
        if feasible:
            best=min(feasible)
            assert abs(best-row['best_feasible_cost'])<1e-6
            assert abs(best-row['z_disc']-row['gap_absolute'])<1e-6
        else:
            assert row['best_feasible_cost'] is None
            assert row['gap_absolute'] is None and row['gap_percent'] is None
