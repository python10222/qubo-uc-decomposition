"""Stage 4 linear/slack QUBO, with exact full-range and surplus lattices.

Q is upper triangular: E(x) = offset + sum(Q[i,j]*x[i]*x[j]).
Diagonal entries are linear biases (x*x = x). Must-run commitments are
substituted as constants, not enforced by a finite penalty. Arrays use (unit,
hour) order; decode returns Fraction-valued power and slack for exact checks.
"""
from dataclasses import replace
from fractions import Fraction
import logging
import math

import numpy as np

from .loader import Instance, full_range_grid, slack_grid
from .models import true_cost

LOG = logging.getLogger(__name__)


class QUBO:
    def __init__(self, instance: Instance, T: int, B: int, lambda1: float,
                 lambda2: float, encoding="linear", penalty="slack"):
        if encoding != "linear" or penalty != "slack":
            raise ValueError("only encoding='linear', penalty='slack' is implemented")
        if isinstance(T, bool) or not isinstance(T, int) or not 1 <= T <= instance.hours:
            raise ValueError("T must be an integer within the instance horizon")
        if any(not math.isfinite(v) or v < 0 for v in (lambda1, lambda2)):
            raise ValueError("penalty weights must be finite and nonnegative")
        self.instance = replace(instance, demand=instance.demand[:T], units=tuple(
            replace(u, pmin=u.pmin[:T], pmax=u.pmax[:T], cost=u.cost[:T],
                    startup=u.startup[:T], must_run=u.must_run[:T]) for u in instance.units))
        self.T, self.B = T, B
        self.lambda1, self.lambda2 = float(lambda1), float(lambda2)
        self.grids = [full_range_grid(u, B, instance.q) for u in self.instance.units]
        self.slack_step, self.slack_bits = slack_grid(self.instance, B)
        self.denominator = 2**B - 1
        self.variables, self.u, self.b, self.e = [], {}, {}, {}
        self.fixed_on = {}
        self.Q, self.offset = {}, 0.0

        def variable(label):
            index = len(self.variables)
            self.variables.append(label)
            return index

        # Affine expressions are dictionaries: None denotes the constant.
        commitments, powers = {}, {}
        for i, unit in enumerate(self.instance.units):
            for t in range(T):
                if unit.must_run[t]:
                    self.fixed_on[i, t] = 1
                    commitments[i, t] = {None: 1.0}
                    LOG.info("%s: fixing %s hour %d on", instance.name, unit.name, t + 1)
                else:
                    self.u[i, t] = variable(("u", i, t))
                    commitments[i, t] = {self.u[i, t]: 1.0}
                p = {j: float(unit.pmin[t])*v for j, v in commitments[i, t].items()}
                for k in range(int(self.grids[i].bits[t])):
                    index = self.b[i, t, k] = variable(("b", i, t, k))
                    p[index] = float(self.grids[i].step[t] * 2**k)
                powers[i, t] = p
        for t, count in enumerate(self.slack_bits):
            for m in range(int(count)):
                self.e[t, m] = variable(("e", t, m))

        for i, unit in enumerate(self.instance.units):
            for t in range(T):
                u, p = commitments[i, t], powers[i, t]
                a, b, c = unit.cost[t]
                self._product(u, {None: 1}, a)
                self._product(p, {None: 1}, b)
                self._product(p, p, c)
                previous = commitments[i, t-1] if t else {None: int(unit.initial_on)}
                self._product(u, {None: 1}, unit.startup[t])
                self._product(u, previous, -unit.startup[t])
                for k in range(int(self.grids[i].bits[t])):
                    bit = {self.b[i, t, k]: 1}
                    self._product(bit, {None: 1}, self.lambda1)
                    self._product(bit, u, -self.lambda1)
        for t in range(T):
            residual = {None: -float(self.instance.demand[t])}
            for i in range(len(self.instance.units)):
                for j, value in powers[i, t].items():
                    residual[j] = residual.get(j, 0) + value
            for m in range(int(self.slack_bits[t])):
                residual[self.e[t, m]] = -float(self.slack_step * 2**m)
            self._product(residual, residual, self.lambda2)
        self.Q = {pair: float(value) for pair, value in self.Q.items() if value != 0}

    @property
    def num_variables(self):
        return len(self.variables)

    def _product(self, left, right, scale):
        for i, a in left.items():
            for j, b in right.items():
                value = float(scale)*a*b
                if i is None and j is None:
                    self.offset += value
                else:
                    pair = (j, j) if i is None else (i, i) if j is None else tuple(sorted((i, j)))
                    self.Q[pair] = self.Q.get(pair, 0.0) + value

    def _bits(self, bits):
        bits = np.asarray(bits)
        if bits.shape != (self.num_variables,) or not np.isin(bits, [0, 1]).all():
            raise ValueError("bits must be a binary vector of length num_variables")
        return bits.astype(np.int8)

    def decode(self, bits):
        """Return commitment integers and exact Fraction power/slack arrays."""
        bits = self._bits(bits)
        u = np.ones((len(self.instance.units), self.T), dtype=np.int8)
        for (i, t), j in self.u.items():
            u[i, t] = bits[j]
        p = np.empty(u.shape, dtype=object)
        for i, unit in enumerate(self.instance.units):
            for t in range(self.T):
                k = sum(int(bits[self.b[i, t, k]]) * 2**k
                        for k in range(int(self.grids[i].bits[t])))
                p[i, t] = Fraction(int(unit.pmin[t])*int(u[i, t])) + self.grids[i].step[t]*k
        s = np.array([self.slack_step * sum(int(bits[self.e[t, m]]) * 2**m
                      for m in range(int(self.slack_bits[t]))) for t in range(self.T)], dtype=object)
        return u, p, s

    def encode(self, u, grid_indices):
        """Encode a feasible schedule using exact integer indices, including slack.

        Task 1 saves these indices in each discrete solution JSON. This avoids
        recovering bit decisions from rounded floating-point dispatch values.
        """
        u, indices = np.asarray(u), np.asarray(grid_indices)
        shape = (len(self.instance.units), self.T)
        if u.shape != shape or indices.shape != shape or not np.isin(u, [0, 1]).all():
            raise ValueError("invalid commitment or index shape")
        if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
            raise ValueError("grid indices must be integers")
        bits = np.zeros(self.num_variables, dtype=np.int8)
        for (i, t), j in self.u.items():
            bits[j] = u[i, t]
        for i, unit in enumerate(self.instance.units):
            for t in range(self.T):
                k = int(indices[i, t])
                if unit.must_run[t] and u[i, t] != 1:
                    raise ValueError("must-run unit is off")
                if not 0 <= k <= int(self.grids[i].levels[t])*int(u[i, t]):
                    raise ValueError("index outside committed output grid")
                for m in range(int(self.grids[i].bits[t])):
                    bits[self.b[i, t, m]] = (k >> m) & 1
        _, p, _ = self.decode(bits)
        for t in range(self.T):
            ticks = (sum(p[:, t]) - int(self.instance.demand[t])) / self.slack_step
            if ticks.denominator != 1 or not 0 <= ticks < 2**int(self.slack_bits[t]):
                raise ValueError("surplus is negative or outside the slack lattice")
            for m in range(int(self.slack_bits[t])):
                bits[self.e[t, m]] = (int(ticks) >> m) & 1
        return bits

    def true_cost(self, u, p):
        """Task 1's physical objective, without penalties (even if infeasible)."""
        return true_cost(self.instance, np.asarray(u, float), np.asarray(p, float))

    def is_feasible(self, u, p):
        """Check commitments, limits, exact grid membership and demand.

        Pass decode's Fraction outputs for exact checks. Float inputs are treated
        as their decimal values, without rounding onto the grid.
        """
        u, p = np.asarray(u), np.asarray(p, dtype=object)
        shape = (len(self.instance.units), self.T)
        if u.shape != shape or p.shape != shape or not np.isin(u, [0, 1]).all():
            return False
        try:
            p = np.array([[v if isinstance(v, Fraction) else Fraction(str(v))
                           for v in row] for row in p], dtype=object)
        except (ValueError, OverflowError, ZeroDivisionError):
            return False
        for i, unit in enumerate(self.instance.units):
            for t in range(self.T):
                if unit.must_run[t] and u[i, t] != 1:
                    return False
                lo, hi = int(unit.pmin[t])*int(u[i, t]), int(unit.pmax[t])*int(u[i, t])
                if not lo <= p[i, t] <= hi:
                    return False
                delta = self.grids[i].step[t]
                if delta and ((p[i, t]-lo)/delta).denominator != 1:
                    return False
        return all(sum(p[:, t]) >= int(self.instance.demand[t]) for t in range(self.T))

    def energy(self, bits):
        """Evaluate the expanded QUBO, including the constant offset."""
        bits = self._bits(bits)
        return math.fsum([self.offset] + [v for (i, j), v in self.Q.items() if bits[i] and bits[j]])


def build_qubo(instance, T, B, lambda1, lambda2, encoding="linear", penalty="slack"):
    return QUBO(instance, T, B, lambda1, lambda2, encoding, penalty)
