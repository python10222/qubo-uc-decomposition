"""SCIP reference models, with hard constraints and no QUBO penalties."""
from dataclasses import dataclass, replace, field
import importlib.metadata
import platform
import time

import numpy as np
from pyscipopt import Model, quicksum

from .loader import Instance, full_range_grid, assert_integer_power


@dataclass
class UCModel:
    model: Model
    instance: Instance
    B: int | None
    on: dict
    power: dict
    startup: dict
    grids: dict
    build_seconds: float
    index: dict = field(default_factory=dict)


@dataclass
class Solution:
    status: str
    objective: float
    dual_bound: float
    gap: float
    solve_seconds: float
    build_seconds: float
    on: np.ndarray
    power: np.ndarray
    startup: np.ndarray
    bit_depths: dict
    metadata: dict
    grid_indices: np.ndarray | None = None
    power_scaled: np.ndarray | None = None
    power_denominator: int = 1


def runtime_metadata(model):
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as stream:
            cpu = next(line.split(":", 1)[1].strip() for line in stream
                       if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu_model": cpu, "logical_cpu_count": __import__('os').cpu_count(),
            "scip_threads": 1, "python": platform.python_version(),
            "libraries": {name: importlib.metadata.version(name)
                          for name in ("numpy", "scipy", "pyscipopt")},
            "scip_version": f"{model.getMajorVersion()}.{model.getMinorVersion()}.{model.getTechVersion()}",
            "parameters": {"limits/gap": 0.0, "limits/absgap": 0.0,
                           "numerics/feastol": 1e-8, "parallel/maxnthreads": 1,
                           "randomization/randomseedshift": 0}}


def build_model(instance: Instance, B: int | None = None, *, verbose=False):
    """Continuous MIQCP or exact grid MILP.

    The integer grid index is equivalent to all B_i binary weights; every index
    0..2**B_i-1 is allowed. Convex adjacent-level chords give the *exact* cost
    on this grid, not an approximation. For continuous output, use the convex
    perspective epigraph c*p**2 <= quadratic_cost*u (exact when u is binary).
    """
    started = time.perf_counter()
    model = Model(f"{instance.name}_{'cont' if B is None else B}")
    if not verbose:
        model.hideOutput()
    for key, value in {"limits/gap": 0.0, "limits/absgap": 0.0,
                       "numerics/feastol": 1e-8, "parallel/maxnthreads": 1,
                       "randomization/randomseedshift": 0}.items():
        model.setParam(key, value)
    assert_integer_power(instance.demand)
    on, power, startup, grids, indices, scaled_power = {}, {}, {}, {}, {}, {}
    objective = []
    for i, unit in enumerate(instance.units):
        grid = full_range_grid(unit, B, instance.q) if B is not None else None
        if grid is not None:
            grids[unit.name] = grid
        for t in range(instance.hours):
            label = f"{unit.name}_{t}"
            u = on[i, t] = model.addVar(f"u_{label}", vtype="B", lb=int(unit.must_run[t]))
            p = power[i, t] = model.addVar(f"p_{label}", ub=float(unit.pmax[t]))
            y = startup[i, t] = model.addVar(f"y_{label}", vtype="B")
            previous = on[i, t-1] if t else int(unit.initial_on)
            model.addCons(y >= u - previous)
            model.addCons(y <= u)
            model.addCons(y <= 1 - previous)
            model.addCons(p >= float(unit.pmin[t]) * u)
            model.addCons(p <= float(unit.pmax[t]) * u)
            a, b, c = map(float, unit.cost[t])
            objective.extend([a*u, b*p, float(unit.startup[t])*y])
            if grid is not None:
                kmax, delta = int(grid.levels[t]), float(grid.step[t])
                if kmax:
                    k = indices[i,t] = model.addVar(f"index_{label}", vtype="I", ub=kmax)
                    model.addCons(k <= kmax * u)
                    scaled_power[i,t] = (grid.denominator*int(unit.pmin[t])*u
                                         + (int(unit.pmax[t])-int(unit.pmin[t]))*k)
                    model.addCons(grid.denominator*p == scaled_power[i,t])
                else:
                    scaled_power[i,t] = grid.denominator*int(unit.pmin[t])*u
                    model.addCons(grid.denominator*p == scaled_power[i,t])
                if c:
                    v = model.addVar(f"quadratic_{label}", ub=c*float(unit.pmax[t])**2)
                    objective.append(v)
                    if not kmax:
                        model.addCons(v >= c*float(unit.pmin[t])**2*u)
                    for k in range(kmax):
                        lo = float(int(unit.pmin[t]) + grid.step[t]*k)
                        hi = float(int(unit.pmin[t]) + grid.step[t]*(k+1))
                        model.addCons(v >= c*(lo+hi)*p - c*lo*hi*u)
            elif c:
                v = model.addVar(f"quadratic_{label}", ub=c*float(unit.pmax[t])**2)
                objective.append(v)
                model.addCons(c*p*p <= v*u)
                # Valid perspective tangents strengthen the root relaxation.
                for point in np.linspace(unit.pmin[t], unit.pmax[t], 9):
                    model.addCons(v >= 2*c*float(point)*p - c*float(point)**2*u)
    for t, demand in enumerate(instance.demand):
        if B is None:
            model.addCons(quicksum(power[i,t] for i in range(len(instance.units))) >= float(demand))
        else:
            model.addCons(quicksum(scaled_power[i,t] for i in range(len(instance.units)))
                          >= (2**B-1)*int(demand))
    model.setObjective(quicksum(objective), "minimize")
    return UCModel(model, instance, B, on, power, startup, grids,
                   time.perf_counter()-started,indices)


def true_cost(instance, on, power):
    total = 0.0
    for i, unit in enumerate(instance.units):
        previous = np.r_[int(unit.initial_on), on[i, :-1]]
        a, b, c = unit.cost.T
        total += np.sum(a*on[i] + b*power[i] + c*power[i]**2
                        + unit.startup*on[i]*(1-previous))
    return float(total)


def extract_exact_grid(built, on, raw_power, tolerance=1e-5):
    """Decode SCIP integer indices, then reconstruct the exact rational output.

    SCIP returns floats: its binary/integer values are accepted only within the
    stated tolerance of an integer. Membership, capacity and surplus checks use
    the resulting integers, never equality between floating-point MW values.
    Raw SCIP output must also agree with the reconstructed dispatch in tolerance.
    """
    L = 2**built.B-1
    indices = np.zeros(on.shape,dtype=np.int64)
    scaled = np.zeros(on.shape,dtype=np.int64)
    for i,unit in enumerate(built.instance.units):
        for t in range(built.instance.hours):
            if (i,t) in built.index:
                raw = built.model.getVal(built.index[i,t])
                if abs(raw-round(raw)) > tolerance:
                    raise AssertionError("nonintegral SCIP grid index")
                indices[i,t] = round(raw)
            scaled[i,t] = L*int(unit.pmin[t])*int(on[i,t]) + (int(unit.pmax[t])-int(unit.pmin[t]))*int(indices[i,t])
    power = scaled.astype(float)/L
    if np.max(np.abs(raw_power-power)) > tolerance:
        raise AssertionError("SCIP output disagrees with exact grid index")
    return power,indices,scaled


def solve(built: UCModel, *, time_limit=None) -> Solution:
    model, instance = built.model, built.instance
    if time_limit is not None:
        model.setParam("limits/time", time_limit)
    model.optimize()
    status = str(model.getStatus())
    if status != "optimal" or model.getGap() != 0.0:
        raise RuntimeError(f"{instance.name} B={built.B}: SCIP did not prove optimality "
                           f"(status={status}, gap={model.getGap()}, time={model.getSolvingTime()})")
    shape = (len(instance.units), instance.hours)
    def values(variables):
        return np.array([[model.getVal(variables[i, t]) for t in range(shape[1])]
                         for i in range(shape[0])])
    on = np.rint(values(built.on))
    power = values(built.power)
    startup = np.rint(values(built.startup))
    indices = scaled = None
    if built.B is not None:
        power,indices,scaled = extract_exact_grid(built,on,power)
    objective = true_cost(instance, on, power)
    if not np.isclose(objective, model.getObjVal(), rtol=0, atol=1e-4):
        raise AssertionError(f"SCIP objective differs from independently evaluated cost: "
                             f"{model.getObjVal()} vs {objective}")
    metadata = runtime_metadata(model)
    metadata['final_scip_seconds'] = model.getSolvingTime()
    total_scip_time = model.getSolvingTime()
    result = Solution(status, objective, model.getDualbound(), model.getGap(),
                      total_scip_time, built.build_seconds, on, power, startup,
                      {name: grid.bits.tolist() for name, grid in built.grids.items()},
                      metadata,indices,scaled,1 if built.B is None else 2**built.B-1)
    validate_solution(instance, result, built.B)
    return result


def validate_solution(instance, result, B=None, tolerance=1e-5):
    """Check rational grid membership and surplus using integer arithmetic."""
    shape = (len(instance.units),instance.hours)
    for array in (result.on,result.power,result.startup):
        if np.shape(array) != shape or not np.all(np.isfinite(array)):
            raise AssertionError("invalid schedule shape or non-finite value")
    if not np.isin(result.on,[0,1]).all():
        raise AssertionError("nonbinary commitment")
    if B is None and np.any(result.power.sum(axis=0) < instance.demand-tolerance):
        raise AssertionError("demand shortfall")
    if B is not None:
        L = 2**B-1
        if result.power_denominator != L:
            raise AssertionError("wrong power denominator")
        for array in (result.grid_indices,result.power_scaled):
            if np.shape(array) != shape or not np.issubdtype(np.asarray(array).dtype,np.integer):
                raise AssertionError("exact grid certificate must contain integers")
        for t in range(instance.hours):
            surplus = sum(int(result.power_scaled[i,t]) for i in range(len(instance.units)))-L*int(instance.demand[t])
            if surplus < 0:
                raise AssertionError("exact demand shortfall")
    for i,unit in enumerate(instance.units):
        u,p = result.on[i],result.power[i]
        if np.any(p < unit.pmin*u-tolerance) or np.any(p > unit.pmax*u+tolerance):
            raise AssertionError("output bounds")
        if np.any(u[unit.must_run] != 1):
            raise AssertionError("must-run commitment")
        expected = u*(1-np.r_[int(unit.initial_on),u[:-1]])
        if not np.array_equal(result.startup[i],expected):
            raise AssertionError("startup indicators")
        if B is not None:
            grid = full_range_grid(unit,B,instance.q)
            for t in range(instance.hours):
                k = int(result.grid_indices[i,t])
                ticks = int(result.power_scaled[i,t])
                lo,hi,on = int(unit.pmin[t]),int(unit.pmax[t]),int(u[t])
                if not 0 <= k <= int(grid.levels[t])*on:
                    raise AssertionError("grid index outside allowed range")
                if ticks != L*lo*on+(hi-lo)*k:
                    raise AssertionError("output off exact grid")
                if not L*lo*on <= ticks <= L*hi*on:
                    raise AssertionError("exact output bounds")
                # This only checks the float display of the exact certificate.
                if abs(p[t]-ticks/L) > tolerance:
                    raise AssertionError("float output disagrees with exact certificate")


def solve_continuous(instance, **kwargs):
    return solve(build_model(instance), **kwargs)


def solve_discretized(instance, B, *, time_limit=None):
    """Solve the exact grid model by SCIP-certified hourly pattern generation.

    At fixed commitments, output dispatch is independent across hours. For each
    hour SCIP enumerates commitment patterns in nondecreasing optimal dispatch
    cost, excluding only previously represented patterns from that enumeration.
    A SCIP master selects known patterns or an optimistic 'unseen' option whose
    price is a certified lower bound on every not-yet-generated pattern. That
    option allows any commitments, so the master is a relaxation of the full UC
    model. Once its proven optimum uses only known patterns, the same solution
    is feasible for the full grid model, proving global optimality. No physical
    bit value is excluded: generated patterns remain available in the master.
    """
    started = time.perf_counter()
    build_seconds, scip_seconds = 0., 0.
    hourly, patterns, bounds, exhausted, certificates = [], [], [], [], []
    for t in range(instance.hours):
        units = tuple(replace(unit,pmin=unit.pmin[t:t+1],pmax=unit.pmax[t:t+1],
                              cost=unit.cost[t:t+1],startup=np.zeros(1),
                              must_run=unit.must_run[t:t+1]) for unit in instance.units)
        built = build_model(replace(instance,units=units,demand=instance.demand[t:t+1]),B)
        hourly.append(built)
        build_seconds += built.build_seconds
        patterns.append([])
        bounds.append(None)
        exhausted.append(False)
        certificates.append([])

    def set_remaining(model):
        if time_limit is not None:
            remaining = time_limit-(time.perf_counter()-started)
            if remaining <= 0:
                raise RuntimeError(f'{instance.name} B={B}: pattern generation did not prove optimality (timelimit)')
            model.setParam('limits/time',remaining)

    def generate(t):
        nonlocal scip_seconds
        built = hourly[t]
        model = built.model
        set_remaining(model)
        model.optimize()
        scip_seconds += model.getSolvingTime()
        status = str(model.getStatus())
        record = {'status':status,'solve_seconds':model.getSolvingTime()}
        certificates[t].append(record)
        if status == 'infeasible':
            exhausted[t] = True
            if not patterns[t]:
                raise RuntimeError(f'{instance.name} B={B}: infeasible at hour {t}')
            return
        if status != 'optimal' or model.getGap() != 0:
            raise RuntimeError(f'{instance.name} B={B}: hourly pricing did not prove optimality '
                               f'(hour={t}, status={status}, gap={model.getGap()})')
        u = np.array([round(model.getVal(built.on[i,0])) for i in range(len(instance.units))])
        p = np.array([model.getVal(built.power[i,0]) for i in range(len(instance.units))])
        exact_p,k,ticks = extract_exact_grid(built,u[:,None],p[:,None])
        p = exact_p[:,0]
        if sum(int(v) for v in ticks[:,0]) < (2**B-1)*int(built.instance.demand[0]):
            raise AssertionError("exact hourly demand shortfall")
        cost = true_cost(built.instance,u[:,None],p[:,None])
        if abs(cost-model.getObjVal()) > 1e-4:
            raise AssertionError('hourly pricing objective mismatch')
        patterns[t].append((u,p,cost,k[:,0],ticks[:,0]))
        bounds[t] = min(cost,model.getDualbound())-1e-7
        record.update({'gap':0.,'dual_bound':model.getDualbound(),'cost':cost,'on':u.tolist(),
                       'grid_indices':k[:,0].tolist(),'power_scaled':ticks[:,0].tolist()})
        model.freeTransform()
        model.addCons(quicksum(1-built.on[i,0] if on else built.on[i,0]
                               for i,on in enumerate(u)) >= 1)

    for t in range(instance.hours):
        generate(t)
    iterations = 0
    master_certificates = []
    while True:
        iterations += 1
        build_started = time.perf_counter()
        master = Model(f'{instance.name}_B{B}_pattern_master')
        master.hideOutput()
        for key,value in {'limits/gap':0.,'limits/absgap':0.,'numerics/feastol':1e-8,
                          'parallel/maxnthreads':1,'randomization/randomseedshift':0}.items():
            master.setParam(key,value)
        on, y, choices, unseen = {}, {}, {}, {}
        terms = []
        for t in range(instance.hours):
            choices[t] = [master.addVar(vtype='B') for _ in patterns[t]]
            unseen[t] = master.addVar(vtype='B',ub=0 if exhausted[t] else 1)
            master.addCons(quicksum(choices[t])+unseen[t] == 1)
            terms.extend(float(pattern[2])*x for pattern,x in zip(patterns[t],choices[t]))
            terms.append(float(bounds[t])*unseen[t])
            for i,unit in enumerate(instance.units):
                on[i,t] = master.addVar(vtype='B',lb=int(unit.must_run[t]))
                known = quicksum(int(pattern[0][i])*x for pattern,x in zip(patterns[t],choices[t]))
                master.addCons(on[i,t] >= known)
                master.addCons(on[i,t] <= known+unseen[t])
                y[i,t] = master.addVar(vtype='B')
                previous = on[i,t-1] if t else int(unit.initial_on)
                master.addCons(y[i,t] >= on[i,t]-previous)
                master.addCons(y[i,t] <= on[i,t])
                master.addCons(y[i,t] <= 1-previous)
                terms.append(float(unit.startup[t])*y[i,t])
        master.setObjective(quicksum(terms),'minimize')
        build_seconds += time.perf_counter()-build_started
        set_remaining(master)
        master.optimize()
        scip_seconds += master.getSolvingTime()
        if str(master.getStatus()) != 'optimal' or master.getGap() != 0:
            raise RuntimeError(f'{instance.name} B={B}: pattern master did not prove optimality '
                               f'(status={master.getStatus()}, gap={master.getGap()})')
        missing = [t for t in range(instance.hours) if master.getVal(unseen[t]) > .5]
        master_certificates.append({'objective':master.getObjVal(),'dual_bound':master.getDualbound(),
                                    'gap':master.getGap(),'unseen_hours':missing,
                                    'solve_seconds':master.getSolvingTime()})
        if not missing:
            break
        for t in missing:
            generate(t)
    output_on = np.zeros((len(instance.units),instance.hours))
    power = np.zeros_like(output_on)
    startups = np.zeros_like(output_on)
    indices = np.zeros(output_on.shape,dtype=np.int64)
    scaled = np.zeros(output_on.shape,dtype=np.int64)
    for t in range(instance.hours):
        selected = next(k for k,x in enumerate(choices[t]) if master.getVal(x) > .5)
        output_on[:,t],power[:,t],_,indices[:,t],scaled[:,t] = patterns[t][selected]
        startups[:,t] = [round(master.getVal(y[i,t])) for i in range(len(instance.units))]
    objective = true_cost(instance,output_on,power)
    if abs(objective-master.getObjVal()) > 1e-4:
        raise AssertionError('pattern master objective mismatch')
    metadata = runtime_metadata(master)
    metadata['method'] = 'certified_hourly_pattern_generation'
    metadata['wall_seconds'] = time.perf_counter()-started
    metadata['pattern_generation'] = {'iterations':iterations,'patterns_per_hour':list(map(len,patterns)),
        'unseen_cost_bounds':bounds,'exhausted_hours':exhausted,
        'pricing_certificates':certificates,'master_certificates':master_certificates,
        'final_unseen_hours':[]}
    result = Solution('optimal',objective,master.getDualbound(),master.getGap(),scip_seconds,
                      build_seconds,output_on,power,startups,
                      {u.name:full_range_grid(u,B,instance.q).bits.tolist() for u in instance.units},metadata,
                      indices,scaled,2**B-1)
    validate_solution(instance,result,B)
    return result
