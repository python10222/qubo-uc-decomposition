"""Load the committed UnitCommitment.jl instances and audit preprocessing.

All numerical unit fields have shape (hours,); cost coefficients have shape
(hours, 3) in a, b, c order. Curves follow the v0.3 convention: each breakpoint
may itself be a vector over time. Compression is detected from magic bytes.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import gzip
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
INSTANCE_NAMES = tuple(f"{n}_0_{i}_w" for n in (10, 20) for i in range(1, 6))
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Unit:
    name: str
    pmin: np.ndarray
    pmax: np.ndarray
    cost: np.ndarray
    startup: np.ndarray
    initial_on: bool
    must_run: np.ndarray


@dataclass(frozen=True)
class Instance:
    name: str
    q: float
    demand: np.ndarray
    units: tuple[Unit, ...]
    audit: dict

    @property
    def hours(self):
        return len(self.demand)


@dataclass(frozen=True)
class Grid:
    bits: np.ndarray
    step: np.ndarray  # exact Fraction values
    denominator: int
    levels: np.ndarray  # maximum index, inclusive; zero for fixed-output units


def round_power(values, q, *, up):
    """Decimal rounding avoids float noise at exact quantum boundaries."""
    quantum = Decimal(str(q))
    if not quantum.is_finite() or quantum <= 0:
        raise ValueError("q must be finite and positive")
    mode = ROUND_CEILING if up else ROUND_FLOOR
    return np.array([float((Decimal(str(v)) / quantum).to_integral_value(
        rounding=mode) * quantum) for v in values])


def assert_integer_power(values):
    if not all(np.isfinite(v) and Fraction(str(v)).denominator == 1 for v in values):
        raise AssertionError("rounded powers and demands must be integers")


def full_range_grid(unit: Unit, B: int, q: float = 1.0) -> Grid:
    if isinstance(B, bool) or not isinstance(B, int) or B < 1:
        raise ValueError("B must be a positive integer")
    if q != 1:
        raise ValueError("the full-range rule requires q = 1 MW")
    assert_integer_power(unit.pmin)
    assert_integer_power(unit.pmax)
    widths = [int(hi)-int(lo) for lo,hi in zip(unit.pmin,unit.pmax)]
    if any(w < 0 for w in widths):
        raise ValueError("negative rounded output range")
    denominator = 2**B-1
    bits = np.array([B if w else 0 for w in widths])
    levels = np.array([denominator if w else 0 for w in widths])
    step = np.array([Fraction(w,denominator) for w in widths],dtype=object)
    for lo,hi,delta,kmax in zip(unit.pmin,unit.pmax,step,levels):
        assert Fraction(int(lo)) + delta*int(kmax) == int(hi)
    return Grid(bits,step,denominator,levels)


def slack_grid(instance: Instance, B: int):
    """Exact slack step and minimal bits; no QUBO is constructed."""
    grids = [full_range_grid(u,B,instance.q) for u in instance.units]
    assert_integer_power(instance.demand)
    denominator = grids[0].denominator
    capacities = [sum(int(u.pmax[t]) for u in instance.units) for t in range(instance.hours)]
    surplus_ticks = [(cap-int(d))*denominator for cap,d in zip(capacities,instance.demand)]
    if any(v < 0 for v in surplus_ticks):
        raise ValueError("demand exceeds total capacity")
    # For integer n >= 0, ceil(log2(n+1)) = n.bit_length().
    return Fraction(1,denominator), np.array([v.bit_length() for v in surplus_ticks])


def fit_cost(power, costs):
    """Least squares quadratic with c >= 0, using a scaled power coordinate.

    The unconstrained LS optimum is used if convex; otherwise the constrained
    optimum lies at c=0 and is the linear LS fit. Truly linear inputs use c=0.
    """
    x, y = np.asarray(power, float), np.asarray(costs, float)
    if x.ndim != 1 or x.shape != y.shape or not len(x):
        raise ValueError("cost curve must contain matching breakpoint vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite cost curve")
    if np.any(x < 0) or (len(x) > 1 and np.any(np.diff(x) <= 0)):
        raise ValueError("MW breakpoints must be nonnegative and strictly increasing")
    if len(x) == 1:
        coeff = np.array([y[0], 0., 0.])
        form = "fixed_output"
    else:
        center, scale = float(x.mean()), float(np.ptp(x))
        z = (x - center) / scale
        linear = np.linalg.lstsq(np.column_stack([np.ones(len(x)), z]), y, rcond=None)[0]
        linear_error = np.max(np.abs(linear[0] + linear[1] * z - y))
        form = "linear" if linear_error <= 1e-10 * max(1., np.max(np.abs(y))) else "piecewise_linear"
        if form == "linear":
            A, C, E = *linear, 0.
        else:
            A, C, E = np.linalg.lstsq(np.column_stack([np.ones(len(x)), z, z*z]), y, rcond=None)[0]
            if E < 0:
                A, C, E = *linear, 0.
        coeff = np.array([A - C*center/scale + E*center**2/scale**2,
                          C/scale - 2*E*center/scale**2, E/scale**2])
    residual = coeff[0] + coeff[1]*x + coeff[2]*x*x - y
    return coeff, {"raw_cost_form": form, "coefficients_abc": coeff.tolist(),
                   "rmse": float(np.sqrt(np.mean(residual**2))),
                   "max_abs_error": float(np.max(np.abs(residual)))}


def load_instance(path, q=1.0) -> Instance:
    if q != 1:
        raise ValueError("the full-range rule requires q = 1 MW")
    path = Path(path)
    payload = path.read_bytes()
    raw = json.loads(gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload)
    params = raw["Parameters"]
    version = str(params["Version"])
    if version != "0.3":
        raise ValueError(f"unsupported UnitCommitment.jl format {version}")
    hours = params["Time (h)"]
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("Time (h) must be a positive integer")
    audit = {"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
             "data_format_version": version, "q_mw": q, "dropped_fields": [],
             "time_varying_fields": [], "assumptions": [], "units": {},
             "source": raw.get("SOURCE"), "license": raw.get("LICENSE")}

    def series(value, field, boolean=False):
        if isinstance(value, list):
            audit["time_varying_fields"].append(field)
            if len(value) != hours:
                raise ValueError(f"{field}: expected {hours} hourly values")
            result = np.asarray(value)
        else:
            result = np.full(hours, value)
        if result.ndim != 1:
            raise ValueError(f"{field}: expected scalar or hourly vector")
        if boolean and not all(isinstance(v, (bool, np.bool_)) for v in result):
            raise ValueError(f"{field}: expected booleans")
        result = result.astype(bool if boolean else float)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{field}: non-finite values")
        return result

    def dropped(field, value, reason):
        audit["dropped_fields"].append({"field": field, "reason": reason, "value": value})
        LOG.info("%s: dropping %s (%s)", path.name, field, reason)

    for field, value in raw.items():
        if field not in {"Parameters", "Generators", "Buses", "SOURCE", "LICENSE"}:
            dropped(field, value, "reserves/network/other constraints outside model scope")
    for field, value in params.items():
        if field not in {"Version", "Time (h)"}:
            dropped(f"Parameters.{field}", value, "unused parameter")
    loads = []
    for name, bus in raw["Buses"].items():
        loads.append(series(bus["Load (MW)"], f"Buses.{name}.Load (MW)"))
        for field, value in bus.items():
            if field != "Load (MW)":
                dropped(f"Buses.{name}.{field}", value, "network outside model scope")
    if not loads:
        raise ValueError("no buses")
    original_demand = np.sum(loads, axis=0)
    if np.any(original_demand < 0):
        raise ValueError("negative demand")
    demand = round_power(original_demand, q, up=True)
    assert_integer_power(demand)
    audit["grid_rule"] = "full_range_rational_v2"
    audit["integer_power_assertion"] = True
    audit["demand_raw_mw"] = original_demand.tolist()
    audit["demand_rounded_mw"] = demand.tolist()
    units = []
    for name, gen in raw["Generators"].items():
        prefix = f"Generators.{name}"
        if "Type" not in gen:
            audit["assumptions"].append(f"{prefix}: no Type field in v0.3; treated as thermal")
        elif gen["Type"] != "Thermal":
            raise ValueError(f"{prefix}: unsupported non-thermal generator")
        powers = np.array([series(v, f"{prefix}.Production cost curve (MW)[{k}]")
                           for k, v in enumerate(gen["Production cost curve (MW)"])]).T
        costs = np.array([series(v, f"{prefix}.Production cost curve ($)[{k}]")
                          for k, v in enumerate(gen["Production cost curve ($)"])]).T
        if powers.shape != costs.shape or powers.ndim != 2 or not powers.shape[1]:
            raise ValueError(f"{prefix}: mismatched/empty cost curve")
        coefficients, fits = zip(*(fit_cost(p, c) for p, c in zip(powers, costs)))
        pmin = round_power(powers[:, 0], q, up=True)
        pmax = round_power(powers[:, -1], q, up=False)
        assert_integer_power(pmin)
        assert_integer_power(pmax)
        if np.any(pmin > pmax):
            raise ValueError(f"{prefix}: empty output interval after rounding")
        categories = np.array([series(v, f"{prefix}.Startup costs ($)[{k}]")
                               for k, v in enumerate(gen["Startup costs ($)"])])
        if not categories.size or np.any(categories < 0):
            raise ValueError(f"{prefix}: startup costs must be nonnegative and nonempty")
        startup = categories.min(axis=0)
        if len(categories) > 1:
            dropped(f"{prefix}.Startup costs ($)", gen["Startup costs ($)"],
                    "multiple categories replaced by cheapest cost per hour")
        initial = gen["Initial status (h)"]
        if not isinstance(initial, (int, float)) or not np.isfinite(initial):
            raise ValueError(f"{prefix}: initial status must be a finite scalar")
        must_run = series(gen.get("Must run?", False), f"{prefix}.Must run?", boolean=True)
        for field, value in gen.items():
            if field not in {"Type", "Production cost curve (MW)", "Production cost curve ($)",
                             "Startup costs ($)", "Initial status (h)", "Must run?"}:
                dropped(f"{prefix}.{field}", value, "outside simplified UC scope")
        audit["units"][name] = {"fits_by_hour": list(fits), "pmin_raw_mw": powers[:, 0].tolist(),
            "pmax_raw_mw": powers[:, -1].tolist(), "pmin_rounded_mw": pmin.tolist(),
            "pmax_rounded_mw": pmax.tolist(), "startup_cost_by_hour": startup.tolist(),
            "initial_status_h": initial, "initial_on": initial > 0,
            "must_run_by_hour": must_run.tolist()}
        units.append(Unit(name, pmin, pmax, np.array(coefficients), startup, initial > 0, must_run))
    if not units:
        raise ValueError("no thermal generators")
    name = path.name.removesuffix(".gz").removesuffix(".json")
    audit["time_varying_fields"] = sorted(set(audit["time_varying_fields"]))
    return Instance(name, float(q), demand, tuple(units), audit)


def load_all(raw_dir=RAW_DIR, q=1.0):
    """Resolve all ten files before parsing any; never fabricate missing data."""
    paths = []
    for name in INSTANCE_NAMES:
        candidates = [Path(raw_dir) / f"{name}{ext}" for ext in (".json", ".json.gz")]
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            raise FileNotFoundError(f"missing OR-LIB instance: {name} in {raw_dir}")
        if len(existing) > 1:
            raise ValueError(f"ambiguous instance: both .json and .json.gz exist for {name}")
        paths.append(existing[0])
    return tuple(load_instance(p, q) for p in paths)
