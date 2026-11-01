# Stage 4 — From unit commitment to QUBO, and where each error enters (1 Nov 2026)

**Result of this stage.** A complete derivation of the QUBO the experiment will build, showing the exact step at which each of the three error terms is created, plus one structural prediction: the penalty weight needed for exactness grows roughly like 2^B, which is the mechanism behind hypotheses H2 and H3.

Every formula here is derived, not quoted. The thresholds in Section 5 are first-order estimates for Stage 5 (Astra) to check numerically.

## 0. The physical picture first

Each hour, a grid operator must decide two things for every power plant: is it on, and if so, how hard does it run. The plan must meet demand every hour at the lowest total cost. Starting a cold plant costs extra.

A QUBO solver can only flip switches (bits) and minimise a score that depends on single bits and pairs of bits. So three translations are needed:
1. **Continuous output → switches.** Output becomes a sum of bits. *This is where discretization error enters.*
2. **Hard rules → fines.** Rules become penalty terms added to the score. *This is where penalty error enters.*
3. **Finding the lowest score.** A heuristic searches the switch landscape. *This is where solver error enters.*

## 1. The original problem (continuous MILP)

Units i = 1…N, hours t = 1…T. Decision variables: on/off u_{it} ∈ {0,1}, output p_{it} ≥ 0, start-up indicator y_{it} ∈ {0,1}.

```latex
\min \sum_{t=1}^{T}\sum_{i=1}^{N}\left[a_i u_{it} + b_i p_{it} + c_i p_{it}^2 + S_i y_{it}\right]
```

subject to

```latex
\sum_{i} p_{it} \ge D_t \quad \forall t \qquad\qquad P^{\min}_i u_{it} \le p_{it} \le P^{\max}_i u_{it} \quad \forall i,t \qquad\qquad y_{it} \ge u_{it}-u_{i,t-1}
```

a_i is the no-load cost, b_i and c_i the linear and quadratic fuel cost, S_i the start-up cost, D_t demand. (Whether OR-LIB data carry a quadratic term is checked in Stage 5; the derivation works either way.) Its optimal cost is **z_cont**.

## 2. Step 1 — Output becomes bits (discretization)

With B bits per unit-hour, write output as the minimum level plus a binary-weighted number of steps:

```latex
p_{it} = P^{\min}_i u_{it} + \Delta_i \sum_{k=0}^{B-1} 2^k\, b_{itk}, \qquad \Delta_i = \frac{P^{\max}_i - P^{\min}_i}{2^B - 1}
```

Structural points:
- **The bounds come for free.** The largest value is P^min + Δ(2^B − 1) = P^max, so the output limits can never be broken. No penalty is needed for them.
- **One rule remains:** a unit that is off must produce nothing, so every bit must be 0 when u = 0, i.e. b_{itk}(1 − u_{it}) = 0.
- **Output is now on a grid** with spacing Δ_i. The best plan restricted to that grid is a feasible plan of the original problem, so its cost z_disc can never beat z_cont.

```latex
G_{disc} = z_{disc} - z_{cont} \ge 0
```

Halving Δ costs one extra bit per unit-hour. Logarithmic encoding (Christeson et al.) spaces the levels unevenly instead; it is a different choice of the weights in the sum, so the same derivation applies.

## 3. Step 2 — The objective in bits

- No-load and linear fuel cost are linear in the bits.
- Quadratic fuel cost c_i p² expands into products of pairs of bits, which a QUBO allows.
- **Start-up cost needs no extra variable and no penalty.** A start happens exactly when the unit is on now and was off before, so y_{it} = u_{it}(1 − u_{i,t−1}), and the cost S_i(u_{it} − u_{it}u_{i,t−1}) is already quadratic. This term is exact.

## 4. Step 3 — Rules become fines (penalties)

**(a) Off means zero output.** A fine that is zero when the rule holds and positive otherwise, with no extra variables:

```latex
P_1 = \sum_{i,t,k} b_{itk}\,(1-u_{it})
```

**(b) Meet demand.** This is an inequality, and QUBOs only handle equalities through squares. Two ways:

*Slack variables (standard).* Introduce a surplus s_t ≥ 0 encoded in M slack bits, and fine any mismatch:

```latex
s_t = \Delta_s \sum_{m=0}^{M-1} 2^m e_{tm}, \qquad P_2 = \sum_t \Big(\sum_i p_{it} - D_t - s_t\Big)^2
```

The fine is zero at a feasible plan **only if the slack can represent the surplus exactly**: Δ_s must match the output grid and D_t must sit on it. If units have different Δ_i, the surplus values form an irregular lattice and some feasible plans carry a small leftover fine. This is a hidden source of penalty error and must be designed out (common step size, or demand rounded onto the grid) and checked in Stage 5.

*Unbalanced penalization (Müller et al.; arXiv 2211.13914).* No slack bits. With h_t = Σ_i p_{it} − D_t, the fine is

```latex
P_2^{ub} = \sum_t \left(-\alpha\, h_t + \beta\, h_t^2\right)
```

This rewards surplus up to a point and fines shortfall. It saves variables, but the fine is **not zero at feasible plans**: it varies with the surplus. That shifts the QUBO's lowest point away from the cheapest feasible plan. This is exactly the "uneven penalties on feasible solutions" Müller et al. report, and it predicts a non-zero penalty error even with well-chosen weights (hypothesis H5).

**The full score the solver minimises**, with x = all bits (u, b, e):

```latex
E(x) = \text{Cost}(x) + \lambda_1 P_1(x) + \lambda_2 P_2(x)
```

Every term is constant, linear or quadratic in the bits, so E is a QUBO.

## 5. How big must the fines be? (the key structural prediction)

A fine is *exact* when no rule-breaking plan can score lower than the best legal plan. Compare what cheating saves against what it costs.

- **Shortfall by one grid step.** Producing Δ less saves roughly b_max·Δ in fuel; the squared fine costs λ₂Δ². Cheating stops paying when

```latex
\lambda_2 \gtrsim \frac{b_{\max}}{\Delta} \;\propto\; \frac{2^B - 1}{P^{\max}-P^{\min}}
```

- **Switching a unit off entirely** saves about a_i + b_i P^min_i and creates a shortfall of at least P^min_i, giving λ₂ ≳ (a_i + b_i P^min_i)/(P^min_i)². This bound does not depend on B.
- **λ₁.** A unit that produces while nominally off (u = 0, some bits on) avoids its no-load cost and any start-up cost, while each lit bit costs λ₁. Cheating stops paying when λ₁ ≳ a_i + S_i. This bound does not depend on B.

**Consequence.** The one-step bound dominates once Δ is small, so the smallest exact λ₂ grows roughly like 2^B: **each extra bit roughly doubles the required demand penalty.** The squared term then contains coefficients of order λ₂Δ²·4^k next to fuel coefficients of order bΔ·2^k, so the spread of coefficient sizes in the QUBO widens as bits are added. Annealing-type solvers struggle with wide coefficient ranges (steep walls, narrow valleys). This gives a mechanism, not just a guess, for:

- **H2:** as B rises, discretization error falls but solver error rises, so they cross.
- **H3:** pushing λ₂ well past its threshold buys no penalty accuracy and makes the landscape steeper, so the total gap is smallest just above the threshold.

These bounds are first-order and per-step. Stage 5 should compute the exact minimum λ₂ numerically on reduced instances (smallest λ at which the exact ground state is feasible).

## 6. Where each error term is created, and how it is measured

| Term | Created at | Measured as | Computed by |
| --- | --- | --- | --- |
| Discretization | Step 1 (output on a grid of spacing Δ) | z_disc − z_cont | Two MILP solves to proven optimality |
| Penalty | Steps 3–4 (fines too weak, leftover slack mismatch, or unbalanced fines) | z_Q − z_disc, plus an infeasibility flag if the ground state breaks a rule | Exact ground state on reduced instances; on full instances, certified zero if no solver ever finds energy below z_disc (slack form only) |
| Solver | Search over E(x) | z_S − z_Q | Best feasible plan from SA, SB, QAOA over ≥ 10 seeds |

The three terms add up exactly to the total gap z_S − z_cont.

## 7. Size of the QUBO

Per unit-hour: 1 on/off bit + B output bits. Per hour: M slack bits for demand (slack form only). Total binaries:

```latex
n = N\,T\,(1 + B) + T\,M
```

Example: N = 10, T = 24, B = 4, M ≈ 8 gives 1,200 + 192 ≈ 1,400 binaries. For N = 20, about 2,600–2,800. A reduced instance with 3 units, 2 hours, 2 bits and 4 slack bits has 26 binaries, within statevector simulation for QAOA. Stage 5 confirms M from the actual demand range.

## 8. Checkpoint before moving on

Answer these without looking back. If any answer is shaky, that section is the one to reread.

1. Why can the encoded output never exceed P^max, and which rule still needs a fine?
2. Why does start-up cost need neither an extra variable nor a penalty?
3. What has to be true of the slack grid for the demand fine to be exactly zero at the best legal plan?
4. If you go from 4 to 8 bits, roughly what happens to the smallest exact λ₂, and why might that hurt the solver?
5. Why does unbalanced penalization produce penalty error even when its weights are well chosen?

## 9. Next

- **Stage 5 (Astra).** Load OR-LIB 10- and 20-unit data; confirm the cost form; compute n for B = 2…6; compute the exact minimum λ₁, λ₂ on reduced instances and compare with the Section 5 estimates; time one SA and one SB run at full size; find the largest reduced instance SCIP solves exactly.
- **Stage 6 (Codex).** Build the pipeline in this order: data loader → continuous and discretized MILP → QUBO builder (Sections 2–4) → solvers → decomposition table (Section 6).
