# QAOA Learning Guide for Insurance Product Bundling Optimization

**Yale Hackathon 2026 -- Quantum Optimization Track**

This guide provides a comprehensive reference for implementing the Quantum Approximate
Optimization Algorithm (QAOA) to solve a 0-1 Integer Linear Program (ILP) for P&C
insurance coverage bundling. It is designed to be read alongside our DQI (Decoded Quantum
Interferometry) implementation so the two approaches can be compared on the same problem.

---

## Table of Contents

1. [QAOA Algorithm Overview](#1-qaoa-algorithm-overview)
2. [QAOA for Combinatorial Optimization](#2-qaoa-for-combinatorial-optimization)
3. [QAOA Circuit Construction in Qiskit](#3-qaoa-circuit-construction-in-qiskit)
4. [Classical Optimization Loop](#4-classical-optimization-loop)
5. [QAOA vs DQI Comparison Framework](#5-qaoa-vs-dqi-comparison-framework)
6. [Implementation Notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. QAOA Algorithm Overview

### 1.1 Original Formulation (Farhi, Goldstone, Gutmann 2014)

The Quantum Approximate Optimization Algorithm was introduced by Farhi, Goldstone, and
Gutmann in 2014 [1]. The original paper frames QAOA as a gate-model quantum algorithm
for producing approximate solutions to combinatorial optimization problems. The key
insight is that a parameterized quantum circuit of depth proportional to an integer p
can encode a variational search over solution space, with quality improving as p
increases.

**Problem setup.** Given a combinatorial optimization problem defined by an objective
function C(z) over n-bit strings z in {0,1}^n, the goal is to find z that maximizes
(or minimizes) C(z). QAOA encodes C into a diagonal "cost Hamiltonian" C that acts on
n qubits, where computational basis states |z> are eigenstates with eigenvalues C(z).

**The QAOA circuit.** The algorithm proceeds as follows:

1. Prepare the uniform superposition state: |s> = |+>^n = H^{tensor n} |0>^n
2. Apply p layers of alternating unitaries:
   - Cost unitary: U_C(gamma_i) = exp(-i * gamma_i * C)
   - Mixer unitary: U_B(beta_i) = exp(-i * beta_i * B)
3. Measure in the computational basis.
4. Use a classical optimizer to update the 2p parameters (gamma_1,...,gamma_p,
   beta_1,...,beta_p) to maximize <C>.

The output state after p layers is:

|gamma, beta> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |s>

The standard mixer Hamiltonian B is the sum of Pauli-X operators on all qubits:
B = sum_{j=1}^{n} X_j. This generates rotations that mix between computational basis
states, playing a role analogous to the transverse field in quantum annealing.

**Approximation ratio.** For MaxCut on 3-regular graphs at p=1, Farhi et al. proved
that QAOA achieves a cut of at least 0.6924 times the optimal -- a nontrivial
guarantee that exceeds the trivial random assignment ratio of 0.5 [1].

### 1.2 Relationship to Quantum Adiabatic Computing

QAOA is deeply connected to quantum adiabatic computation (QAC). In QAC, one prepares
the ground state of a simple Hamiltonian B and slowly evolves the system under a
time-dependent Hamiltonian H(t) = (1 - t/T)*B + (t/T)*C. If the evolution is slow
enough (adiabatic), the system remains in the instantaneous ground state and ends in
the ground state of C -- the optimal solution.

QAOA can be viewed as a **Trotterized** version of the adiabatic algorithm. If we
discretize the adiabatic schedule into p steps, we get alternating applications of
exp(-i * delta_t * B) and exp(-i * delta_t * C), which is exactly the QAOA ansatz.
However, QAOA is strictly more powerful because:

- The parameters (gamma_i, beta_i) are freely optimized rather than constrained to
  follow a monotonic adiabatic schedule.
- Sack and Serbyn (2021) demonstrated that non-adiabatic parameter schedules can
  outperform the adiabatic limit at lower circuit depth [8].
- The variational freedom allows QAOA to find "shortcuts" through parameter space
  that the adiabatic path does not explore.

In the limit of large p with appropriately chosen parameters, QAOA recovers the
adiabatic algorithm. But at finite p, the variational optimization over parameters is
what gives QAOA its power.

### 1.3 When QAOA Works Well

QAOA is most effective in the following regimes:

- **Moderate-size problems (tens of qubits):** Where the optimization landscape is
  tractable and the circuit depth is manageable on near-term hardware.
- **Problems with local structure:** The cost Hamiltonian for problems like MaxCut
  involves only 2-local terms (ZZ interactions), leading to efficient circuit
  decomposition.
- **Low-to-moderate p:** At p = 1 to 5, QAOA already provides useful approximate
  solutions while keeping circuit depth within NISQ capabilities.
- **Problems where classical heuristics plateau:** QAOA can sometimes find solutions
  in regimes where greedy or local-search algorithms get stuck.

### 1.4 Limitations and Barriers

Understanding QAOA limitations is critical for honest benchmarking:

**Locality barrier and the Overlap Gap Property (OGP).** Farhi, Gamarnik, and Gutmann
(2020) showed that for certain random graph problems (e.g., Maximum Independent Set on
random graphs), QAOA at depth p < O(log n) cannot outperform a certain classical
threshold [5]. This is because low-depth QAOA produces locally correlated outputs,
and the Overlap Gap Property of near-optimal solutions in random instances creates an
obstruction. Marwaha, El Alaoui, and Montanari (2023) extended this to random CSPs,
showing that bounded-depth quantum circuits on bounded-degree architectures are
obstructed by the branching OGP [10].

**Concentration phenomena.** For random problem instances, QAOA objective values
concentrate around their expectation. This means that while QAOA can reliably achieve
a certain approximation ratio, it may be hard to beat that ratio without significantly
increasing p. The instance-to-instance variance of the QAOA output decreases as
problem size grows, which is both a strength (predictable performance) and a limitation
(hard to get lucky).

**Barren plateaus.** For general parameterized quantum circuits, the variance of the
cost function gradient can decrease exponentially with the number of qubits, making
optimization intractable. While QAOA with fixed p is somewhat protected from the worst
barren-plateau scenarios (because the circuit has specific structure), the issue becomes
relevant as p grows or as problem-specific structure is lost.

**Classical competition.** For many combinatorial optimization problems, sophisticated
classical algorithms (semidefinite programming relaxations, simulated annealing,
branch-and-bound) remain highly competitive. The Goemans-Williamson algorithm achieves
approximation ratio 0.878 for MaxCut, while p=1 QAOA only achieves 0.6924 on
3-regular graphs. Higher p is needed to approach or exceed classical performance, which
increases circuit depth and noise vulnerability.

**Parameter optimization overhead.** The classical outer loop requires many circuit
evaluations. For 2p parameters, each function evaluation requires many measurement
shots, and the optimizer may need hundreds or thousands of evaluations to converge.
This creates a significant wall-clock-time overhead.

---

## 2. QAOA for Combinatorial Optimization

### 2.1 From ILP to Cost Hamiltonian

Our insurance bundling problem is a 0-1 Integer Linear Program. The general form is:

```
minimize    c^T x
subject to  A x <= b
            x in {0, 1}^n
```

where x_i = 1 means coverage option i is included in the bundle. To encode this for
QAOA, we need two steps: (a) convert to an unconstrained binary optimization, and
(b) map to a qubit Hamiltonian.

**Step 1: Binary variable mapping.** The decision variables x_i in {0,1} map directly
to qubits. We use the standard substitution:

    x_i = (1 - Z_i) / 2

where Z_i is the Pauli-Z operator on qubit i. This maps x_i = 0 to Z_i eigenvalue +1
(state |0>) and x_i = 1 to Z_i eigenvalue -1 (state |1>).

**Step 2: Constructing the cost Hamiltonian.** Substituting the x_i -> Z_i mapping into
the objective function c^T x yields a Hamiltonian that is a linear combination of
Pauli-Z operators and ZZ products:

    C_obj = sum_i alpha_i Z_i + sum_{i<j} beta_{ij} Z_i Z_j + constant

For a purely linear objective c^T x, the cost Hamiltonian is:

    C_obj = sum_i (c_i / 2) * (I - Z_i)  =  (sum_i c_i / 2) * I  -  sum_i (c_i / 2) * Z_i

The constant term shifts the energy but does not affect which state is optimal.

### 2.2 QUBO and Ising Model Formulations

**QUBO (Quadratic Unconstrained Binary Optimization).** Any constrained 0-1 ILP can be
converted to an unconstrained problem by adding penalty terms for constraint violations.
The resulting formulation is:

    minimize  x^T Q x + q^T x

where Q is a matrix encoding both the objective and penalty terms. This is the QUBO
form.

**Ising model.** Substituting x_i = (1 - z_i) / 2, where z_i in {-1, +1}, transforms
the QUBO into an Ising model:

    H = sum_i h_i z_i + sum_{i<j} J_{ij} z_i z_j + constant

This maps directly to a diagonal Hamiltonian in the Pauli-Z basis:

    C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j

The computational basis states are eigenstates of C, and the eigenvalues are exactly
the Ising energies.

### 2.3 Constraint Handling Approaches

For our ILP problem, constraints are a central challenge. There are three main
approaches:

**Approach 1: Penalty terms (QUBO conversion).** Add quadratic penalty terms to the
objective for each constraint violation:

    C_total = C_obj + lambda * sum_k P_k

where P_k penalizes violation of constraint k and lambda is a penalty strength. For
an inequality constraint a^T x <= b, a common penalty is:

    P_k = max(0, a^T x - b)^2

In practice, this requires introducing slack variables to convert inequalities to
equalities, then penalizing: P_k = (a^T x + s_k - b)^2, where s_k is an integer
slack variable encoded in binary using ancilla qubits.

*Advantages:* Conceptually simple; standard QAOA mixer works; well-established.
*Disadvantages:* Choosing lambda is tricky (too small: constraints violated; too large:
optimization landscape distorted); slack variables increase qubit count; quadratic
penalties introduce higher-weight terms in the Hamiltonian.

**Approach 2: Feasibility filtering.** Run standard QAOA without explicit constraint
encoding, then post-select measurement outcomes that satisfy all constraints. This
works when the feasible region is a large fraction of the solution space.

*Advantages:* No additional qubits; simpler circuit.
*Disadvantages:* Exponentially poor sampling efficiency if the feasible region is
small; no guidance toward feasible solutions during optimization.

**Approach 3: Custom mixers (Quantum Alternating Operator Ansatz).** Hadfield et al.
(2019) generalized QAOA to the Quantum Alternating Operator Ansatz [4], which replaces
the standard transverse-field mixer with a mixer that preserves the feasible subspace.
If the initial state is feasible, and the mixer only transitions between feasible
states, then every measurement outcome is guaranteed feasible.

For example, for a cardinality constraint (sum x_i = k), one can use a mixer built
from partial SWAP gates: XY-mixers of the form (X_i X_j + Y_i Y_j) that swap
excitations between qubits without changing the total Hamming weight.

*Advantages:* Every sample is feasible; no penalty tuning; more efficient search.
*Disadvantages:* Mixer design is problem-specific; mixer circuits can be deeper; initial
feasible state preparation may be nontrivial.

**Recommendation for our ILP problem.** Start with Approach 1 (penalty terms) for
simplicity and rapid prototyping. If qubit count permits, Approach 3 with custom mixers
is the more principled solution. Approach 2 (post-selection) can supplement either
method as a secondary filter.

### 2.4 The Role of Circuit Depth p

The integer p controls the number of QAOA layers and is the primary knob for trading
quality against resources:

| Aspect                    | Low p (1-3)                    | High p (10+)                    |
|---------------------------|--------------------------------|---------------------------------|
| Circuit depth             | Shallow, NISQ-friendly         | Deep, error-prone               |
| Parameter count           | 2-6 parameters, easy to optimize | 20+ parameters, harder landscape |
| Solution quality          | Modest approximation ratios    | Approaches optimal               |
| Classical simulation cost | Often simulable classically    | Potentially intractable          |
| Noise sensitivity         | Low                            | High                             |

For our hackathon setting, p = 1 to 5 is the practical operating range. Increasing p
beyond the coherence limit of the hardware (or simulator budget) yields diminishing or
negative returns due to noise accumulation.

**Scaling intuition.** Each layer adds 2 free parameters but also O(m) gates where m
is the number of terms in the cost Hamiltonian. For a problem with n binary variables
and m constraint/objective terms, total circuit depth is approximately O(p * m).

---

## 3. QAOA Circuit Construction in Qiskit

This section describes how to build QAOA circuits using Qiskit >= 1.0. The patterns
here are for reference when implementing; they are not runnable code on their own.

### 3.1 Building the Cost Unitary exp(-i * gamma * C)

The cost Hamiltonian C is diagonal in the computational basis. For each term type:

**Single-qubit Z term:** h_i * Z_i contributes a phase gate Rz(2 * gamma * h_i) on
qubit i.

**Two-qubit ZZ term:** J_{ij} * Z_i Z_j is implemented as:
```python
# ZZ interaction: exp(-i * gamma * J * Z_i Z_j)
# Decomposition: CNOT(i,j) -> Rz(2*gamma*J, j) -> CNOT(i,j)
qc.cx(i, j)
qc.rz(2 * gamma * J_ij, j)
qc.cx(i, j)
```

**Complete cost unitary for a QUBO.** Given Ising coefficients h (linear) and J
(quadratic):

```python
from qiskit.circuit import QuantumCircuit, Parameter

def build_cost_layer(n_qubits, h_coeffs, J_coeffs, gamma):
    """
    Build exp(-i * gamma * C) where C = sum h_i Z_i + sum J_ij Z_i Z_j.

    Parameters:
        n_qubits: number of qubits
        h_coeffs: dict {i: h_i} for linear Z terms
        J_coeffs: dict {(i,j): J_ij} for quadratic ZZ terms
        gamma: Parameter or float for the cost angle
    Returns:
        QuantumCircuit implementing the cost unitary
    """
    qc = QuantumCircuit(n_qubits)

    # Single-qubit Z rotations
    for i, h_i in h_coeffs.items():
        qc.rz(2 * gamma * h_i, i)

    # Two-qubit ZZ interactions
    for (i, j), J_ij in J_coeffs.items():
        qc.cx(i, j)
        qc.rz(2 * gamma * J_ij, j)
        qc.cx(i, j)

    return qc
```

### 3.2 Building the Mixer Unitary exp(-i * beta * B)

The standard transverse-field mixer B = sum X_j is a product of single-qubit X
rotations (since the X operators on different qubits commute):

```python
def build_mixer_layer(n_qubits, beta):
    """
    Build exp(-i * beta * B) where B = sum X_j.
    Since [X_i, X_j] = 0, this factorizes into single-qubit rotations.

    Parameters:
        n_qubits: number of qubits
        beta: Parameter or float for the mixer angle
    Returns:
        QuantumCircuit implementing the mixer unitary
    """
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        qc.rx(2 * beta, j)
    return qc
```

**Custom XY-mixer for Hamming-weight preservation** (useful for cardinality constraints):

```python
def build_xy_mixer_layer(n_qubits, beta):
    """
    XY-mixer: exp(-i * beta * sum_{i<j} (X_i X_j + Y_i Y_j) / 2)
    Preserves Hamming weight of the state.
    Each (X_i X_j + Y_i Y_j)/2 is a partial SWAP.
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Implement exp(-i * beta * (XX + YY)/2)
            # Decomposition using CNOT and single-qubit gates:
            qc.cx(i, j)
            qc.ry(2 * beta, i)
            qc.rz(2 * beta, j)    # Note: exact decomposition depends
            qc.cx(i, j)           # on desired coupling topology
    return qc
```

Note: For the XY-mixer, the exact decomposition into native gates depends on which
qubit pairs you want to mix. A ring mixer (nearest-neighbor only) reduces gate count
compared to the all-to-all mixer shown above.

### 3.3 Full Parameterized QAOA Circuit

```python
from qiskit.circuit import QuantumCircuit, ParameterVector

def build_qaoa_circuit(n_qubits, h_coeffs, J_coeffs, p):
    """
    Build a complete p-layer QAOA circuit.

    Parameters:
        n_qubits: number of qubits
        h_coeffs: dict of linear Ising coefficients
        J_coeffs: dict of quadratic Ising coefficients
        p: number of QAOA layers
    Returns:
        QuantumCircuit with 2p parameters (gammas and betas)
    """
    gammas = ParameterVector('gamma', p)
    betas = ParameterVector('beta', p)

    qc = QuantumCircuit(n_qubits)

    # Initial state: uniform superposition
    qc.h(range(n_qubits))

    # p layers of cost + mixer
    for layer in range(p):
        # Cost unitary
        cost = build_cost_layer(n_qubits, h_coeffs, J_coeffs, gammas[layer])
        qc.compose(cost, inplace=True)

        # Mixer unitary
        mixer = build_mixer_layer(n_qubits, betas[layer])
        qc.compose(mixer, inplace=True)

    # Measurement
    qc.measure_all()

    return qc, gammas, betas
```

### 3.4 Qiskit-Specific Patterns and Best Practices

**Using Qiskit >= 1.0 primitives.** Qiskit 1.0 introduced a new execution model based
on primitives (Estimator and Sampler) rather than the legacy `backend.run()` interface.
For QAOA:

- Use `StatevectorEstimator` (from `qiskit.primitives`) for exact expectation values
  during development and testing on small instances.
- Use `StatevectorSampler` for exact sampling simulation.
- For noisy simulation or real hardware, use the `Estimator` or `Sampler` from
  `qiskit_ibm_runtime`.

**SparsePauliOp for the cost Hamiltonian.** Represent C using Qiskit's
`SparsePauliOp`:

```python
from qiskit.quantum_info import SparsePauliOp

# Example: C = 0.5 * Z0Z1 + 0.3 * Z0 - 0.2 * Z1
cost_op = SparsePauliOp.from_list([
    ("ZZ", 0.5),    # Z0 Z1 (rightmost qubit is qubit 0 in Qiskit)
    ("ZI", -0.2),   # Z1
    ("IZ", 0.3),    # Z0
])
```

Caution: Qiskit uses **little-endian** qubit ordering. The rightmost character in a
Pauli string corresponds to qubit 0. This is a common source of bugs.

**Qiskit built-in QAOA.** Qiskit's `qiskit_algorithms` library (now in the separate
`qiskit-algorithms` package) provides a `QAOA` class. However, for learning purposes
and maximum control, building the circuit manually (as shown above) is recommended.
The built-in class is useful for rapid prototyping:

```python
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

sampler = StatevectorSampler()
optimizer = COBYLA(maxiter=1000)

qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
result = qaoa.compute_minimum_eigenvalue(cost_op)
```

Note: The `qiskit_algorithms` package may have API changes. Always check the current
documentation for Qiskit >= 1.0 compatibility.

---

## 4. Classical Optimization Loop

### 4.1 Expectation Value Estimation

The quantity to optimize is the expectation value of the cost Hamiltonian:

    F(gamma, beta) = <gamma, beta| C |gamma, beta>

**From measurement samples.** Run the QAOA circuit with given parameters, collect N_shots
measurement outcomes z_1, ..., z_{N_shots}, and estimate:

    F_hat = (1 / N_shots) * sum_{k=1}^{N_shots} C(z_k)

where C(z_k) is the classical objective value for bitstring z_k.

**Statistical precision.** The standard error of the estimate scales as
sigma / sqrt(N_shots), where sigma is the standard deviation of C(z) over the output
distribution. For useful gradient estimates, one typically needs N_shots in the range
of 1,000 to 10,000 per evaluation.

**Estimator-based approach.** Using the Qiskit Estimator primitive, the expectation
value can be computed directly without manual post-processing of bitstrings. This is
cleaner and can leverage statevector simulation for exact values during development.

### 4.2 Optimizer Choices

The classical optimizer updates the 2p parameters to minimize (or maximize) the
expectation value. Common choices:

**COBYLA (Constrained Optimization BY Linear Approximations).**
- Gradient-free, derivative-free method.
- Works well for low-dimensional parameter spaces (small p).
- Robust to noise in function evaluations.
- Typical setting: maxiter = 500-1000 for p <= 5.
- Recommended as the default starting point.

**SPSA (Simultaneous Perturbation Stochastic Approximation).**
- Gradient-based but only requires 2 function evaluations per iteration
  (independent of parameter dimension).
- Specifically designed for noisy function evaluations.
- Well-suited for hardware execution where shots are limited.
- Requires tuning of learning rate and perturbation parameters.
- The `qiskit_algorithms.optimizers.SPSA` class provides calibration utilities.

**L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno, Bounded).**
- Quasi-Newton method requiring gradient information.
- Very efficient when gradients are available (e.g., via parameter-shift rule
  or finite differences on statevector simulator).
- Not recommended for noisy hardware evaluations due to sensitivity to noise.
- Best for noiseless statevector simulation during development.

**NFT (Nakanishi-Fujii-Todo).**
- Exploits the sinusoidal structure of QAOA's parameter landscape.
- Optimizes one parameter at a time using 3-point fitting of a sinusoid.
- Can be very efficient for QAOA specifically.

**Optimizer comparison summary:**

| Optimizer  | Gradient-free? | Noise-robust? | Best for           | Evaluations/iter |
|------------|:--------------:|:-------------:|--------------------|-----------------:|
| COBYLA     | Yes            | Moderate      | Small p, simulator | ~p+1             |
| SPSA       | No (stochastic)| Yes           | Hardware, noisy    | 2                |
| L-BFGS-B   | No             | No            | Statevector sim    | ~2p (gradients)  |
| NFT        | Yes            | Moderate      | QAOA-specific      | 3 per parameter  |

### 4.3 Parameter Initialization Strategies

Poor initialization is a leading cause of QAOA underperformance. Random initialization
frequently converges to suboptimal local minima [8].

**Linear ramp (TQA-inspired).** Sack and Serbyn (2021) showed that initializing QAOA
parameters based on a Trotterized Quantum Annealing (TQA) schedule gives robust
performance [8]:

```python
import numpy as np

def tqa_initialization(p, dt=0.5):
    """
    TQA-inspired initialization: linear ramp.
    gamma increases from 0 to dt, beta decreases from dt to 0.

    Parameters:
        p: number of QAOA layers
        dt: Trotter time step (typically 0.3 to 0.75)
    Returns:
        initial_gammas, initial_betas
    """
    gammas = np.array([(i + 0.5) * dt / p for i in range(p)])
    betas = np.array([(1.0 - (i + 0.5) / p) * dt for i in range(p)])
    return gammas, betas
```

This mimics an adiabatic schedule: early layers apply more mixer (large beta, small
gamma) and later layers apply more cost (large gamma, small beta).

**Fixed-angle strategies.** For specific problem classes like MaxCut on regular graphs,
there exist known good angles that transfer across instances. Wurtz and Love (2021)
computed optimal fixed angles for QAOA on MaxCut that work well as initializations.

**Interpolation from lower p.** Optimize at depth p, then initialize depth p+1 by
linearly interpolating the p parameters into p+1 parameters. This "parameter transfer"
strategy builds up depth incrementally and avoids optimizing many parameters from
scratch.

**Random multi-start.** Run the optimizer from multiple random starting points and keep
the best result. Simple but expensive: requires many independent optimization runs.

**Recommendation for our project.** Use TQA-inspired initialization with dt = 0.5 as
the default. If time permits, compare with interpolation from lower p.

### 4.4 Convergence Criteria and Monitoring

Track the following during optimization:

- **Objective value** F(gamma, beta) at each iteration -- should be decreasing
  (for minimization) or increasing (for maximization).
- **Parameter trajectory** -- plot gamma and beta values to detect oscillation or drift.
- **Approximation ratio** -- if the optimal value is known, compute
  F / F_optimal at each step.
- **Solution distribution** -- periodically examine the top-k most frequently measured
  bitstrings to see if the optimizer is concentrating probability on good solutions.

**Stopping criteria.** Typical choices:
- Maximum number of iterations (e.g., 500-1000).
- Objective value change below a threshold for consecutive iterations.
- Parameter change norm below a threshold.

### 4.5 Number of Shots and Statistical Considerations

The number of measurement shots per circuit evaluation affects:

1. **Accuracy of expectation value estimates:** More shots reduce variance.
2. **Total runtime:** More shots per evaluation multiplied by number of evaluations.
3. **Optimizer convergence:** Noisy estimates can slow or mislead the optimizer.

**Guidelines:**
- For statevector simulation: Use exact expectation values (no shot noise). This
  removes a major source of difficulty and is recommended during development.
- For shot-based simulation: Start with 1024 shots and increase to 4096-8192 if
  the optimizer struggles to converge.
- For hardware: Budget total shots across all optimizer iterations. With SPSA
  (2 evaluations per iteration) and 200 iterations, that is 400 circuit executions.
  At 4096 shots each, total cost is roughly 1.6 million shots.

---

## 5. QAOA vs DQI Comparison Framework

### 5.1 Fundamental Algorithmic Differences

| Aspect                  | QAOA                                    | DQI                                      |
|-------------------------|-----------------------------------------|------------------------------------------|
| Type                    | Variational / hybrid quantum-classical  | Interference-based / non-variational     |
| Core mechanism          | Parameterized cost/mixer alternation    | Quantum interference + classical decoding|
| Parameters              | 2p angles (gamma, beta)                 | No variational parameters                |
| Classical component     | Iterative parameter optimization loop   | Post-processing / decoding step          |
| Measurement strategy    | Computational basis measurement         | Structured measurement + classical decode|
| Theoretical foundation  | Adiabatic theorem / variational principle | Quantum signal processing / interferometry |

**QAOA** is a variational quantum eigensolver (VQE) variant specialized for
combinatorial optimization. It iteratively adjusts quantum circuit parameters using
classical feedback to improve solution quality. The quantum computer proposes candidate
solutions; the classical computer evaluates and steers.

**DQI (Decoded Quantum Interferometry)** takes a fundamentally different approach. It
uses quantum interference to amplify the signal of good solutions, then applies a
classical decoding step to extract the answer. There is no variational optimization
loop -- the circuit structure is fixed by the problem, and the classical post-processing
does the heavy lifting of extracting the solution from interference patterns.

### 5.2 Circuit Depth Comparison

**QAOA:** Circuit depth is O(p * m) where m is the number of terms in the cost
Hamiltonian and p is the number of layers. For a QUBO on n variables with dense
quadratic terms, m = O(n^2), giving depth O(p * n^2). With hardware-native gate
decomposition and routing overhead, the practical depth can be 3-10x higher.

**DQI:** Circuit depth depends on the encoding strategy but is typically fixed (not
iteratively increasing). The depth is determined by the problem encoding, not a
variational parameter. For certain problem structures, DQI can achieve shorter circuits
because it does not require multiple alternating layers.

### 5.3 Parameter Optimization Overhead

This is a key differentiator:

**QAOA** requires a classical optimization loop that typically needs:
- Hundreds to thousands of circuit evaluations to converge.
- Multiple random restarts to avoid local minima.
- Careful tuning of optimizer hyperparameters and initialization.
- The total number of circuit executions scales as
  (optimization iterations) x (shots per evaluation) x (restarts).

**DQI** has no variational parameters and thus no optimization overhead. The circuit is
executed a fixed number of times, and the classical decoder processes the results.
This is a significant practical advantage: the total runtime is predictable and there
is no risk of convergence failure.

### 5.4 Scalability Considerations

**QAOA scalability challenges:**
- Parameter optimization landscape becomes increasingly complex with n and p.
- Noise accumulation limits practical circuit depth on real hardware.
- The optimal p may need to grow with n for asymptotic advantage, but
  near-term hardware constrains p to small values.
- For dense QUBOs (like a fully-connected ILP relaxation), the cost layer requires
  O(n^2) two-qubit gates per layer.

**DQI scalability characteristics:**
- No optimization loop means predictable scaling of quantum resources.
- The classical decoding step may have its own computational cost that scales
  with problem size.
- Amenable to error mitigation techniques due to structured circuit form.

### 5.5 Known Theoretical Limitations

**QAOA limitations:**
- Locality barrier: Low-depth QAOA is provably limited on certain random instances [5].
- The Overlap Gap Property obstructs not just QAOA but all bounded-depth quantum
  circuits on bounded-degree architectures [10].
- Concentration bounds show that QAOA's performance converges to a fixed ratio as
  problem size grows, which for some problems is below classical solver performance.
- For MAX-k-SAT, Boulebnane and Montanaro (2022) showed that QAOA can match or
  exceed classical solvers at sufficient depth (around p=14 for 8-SAT) [9].

**DQI limitations:**
- Being newer, theoretical guarantees are still being developed.
- The classical decoding step's computational complexity may itself be a bottleneck.
- Performance depends on the structure of the problem's interference pattern.

**For our hackathon comparison:** We should benchmark both algorithms on the same ILP
instances, measuring: (1) solution quality (approximation ratio), (2) total quantum
resources (circuit depth, gate count, qubit count), (3) total wall-clock time
including classical processing, and (4) scaling behavior as problem size increases.

---

## 6. Implementation Notes

### 6.1 Practical Tips for Qiskit >= 1.0

**Package structure.** As of Qiskit 1.0+, the ecosystem is modular:
- `qiskit` (core): circuits, transpiler, quantum info.
- `qiskit-algorithms`: QAOA, VQE, optimizers (install separately via pip).
- `qiskit-aer`: high-performance simulators (statevector, qasm, density matrix).
- `qiskit-ibm-runtime`: IBM hardware access and cloud primitives.

**Transpilation.** Always transpile circuits before execution:
```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
transpiled_qc = pm.run(qc)
```
Use optimization_level=2 or 3 for production runs; level 1 for quick iteration.

**Parameter binding.** Bind parameters efficiently using Qiskit's parameter binding:
```python
# Given a parameterized circuit qc with ParameterVectors gammas, betas:
param_values = {gammas[i]: gamma_vals[i] for i in range(p)}
param_values.update({betas[i]: beta_vals[i] for i in range(p)})
bound_qc = qc.assign_parameters(param_values)
```

**Qubit ordering.** Qiskit uses little-endian ordering: qubit 0 is the rightmost
(least significant) bit in a measurement string. If your measurement result is
"01101", then qubit 0 is in state 1, qubit 1 is in state 0, qubit 2 is in state 1,
etc. Always verify your mapping with a small test case.

### 6.2 Simulation Strategies

**Statevector simulation (small instances, n <= 25).**
Best for development and debugging. Provides exact expectation values and full
state information. No shot noise.

```python
from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()
job = estimator.run([(qc, cost_op, param_values)])
result = job.result()
energy = result[0].data.evs  # exact expectation value
```

**Qasm / shot-based simulation (medium instances, n <= 30).**
Simulates the measurement process with shot noise. More realistic for hardware
comparison but slower than statevector for expectation values.

```python
from qiskit.primitives import StatevectorSampler

sampler = StatevectorSampler()
job = sampler.run([(qc, param_values)], shots=4096)
result = job.result()
counts = result[0].data.meas.get_counts()
```

**Aer GPU simulation (large instances, n <= 40).**
Qiskit Aer supports GPU-accelerated statevector simulation for larger problems.

**Matrix Product State (MPS) simulation.** For circuits with limited entanglement
(low p), MPS-based simulators can handle larger qubit counts by exploiting the
low-entanglement structure. Available in Qiskit Aer.

### 6.3 How to Handle ILP Constraints in QAOA

For the insurance bundling ILP, the typical constraint types and their QAOA treatments
are:

**Budget constraint (sum c_i x_i <= B):**
Penalty approach -- add lambda * (sum c_i x_i - B)^2 to the QUBO, requiring
O(log B) ancilla qubits for slack variable encoding.

**Mutual exclusion (x_i + x_j <= 1):**
Penalty approach -- add lambda * x_i * x_j to the QUBO. This only requires a
single ZZ term, no ancilla qubits. This is particularly efficient.

**At-least-one constraints (sum_{i in S} x_i >= 1):**
Penalty approach -- add lambda * product_{i in S} (1 - x_i) which penalizes the
all-zero assignment on S. For small sets, this is tractable; for large sets, the
product has exponentially many terms and approximations are needed.

**Cardinality constraint (sum x_i = k):**
Custom mixer approach -- use an XY-mixer that preserves Hamming weight. Initialize
in a Dicke state |D_n^k> (equal superposition of all n-choose-k bitstrings with
exactly k ones). The XY-mixer then only explores states with Hamming weight k.

**General strategy for our problem:**
1. Classify each ILP constraint by type.
2. Use penalty terms for simple linear constraints (budget, mutual exclusion).
3. Consider custom mixers for hard cardinality constraints if they dominate the
   constraint structure.
4. Start with modest penalty weights (lambda ~ 2-5x the largest objective
   coefficient) and increase if constraints are frequently violated.
5. Monitor constraint satisfaction rate in measurement outcomes to tune penalties.

---

## 7. References

### Foundational QAOA Papers

[1] E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization
    Algorithm," arXiv:1411.4028 (2014).
    https://arxiv.org/abs/1411.4028

[2] E. Farhi, J. Goldstone, S. Gutmann, "Quantum Algorithms by Adiabatic Evolution,"
    arXiv:quant-ph/0001106 (2000).
    https://arxiv.org/abs/quant-ph/0001106

[3] E. Farhi, A.W. Harrow, "Quantum Supremacy through the Quantum Approximate
    Optimization Algorithm," arXiv:1602.07674 (2016).
    https://arxiv.org/abs/1602.07674

### QAOA Variants and Extensions

[4] S. Hadfield, Z. Wang, B. O'Gorman, E.G. Rieffel, D. Venturelli, R. Biswas,
    "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating
    Operator Ansatz," Algorithms 12(2):34 (2019). arXiv:1709.03489.
    https://arxiv.org/abs/1709.03489

[5] E. Farhi, D. Gamarnik, S. Gutmann, "The Quantum Approximate Optimization
    Algorithm Needs to See the Whole Graph: A Typical Case," arXiv:2004.09002 (2020).
    https://arxiv.org/abs/2004.09002

[6] V. Vijendran et al., "An Expressive Ansatz for Low-Depth Quantum Approximate
    Optimisation (XQAOA)," Quantum Science and Technology 9, 025010 (2024).
    arXiv:2302.04479.
    https://arxiv.org/abs/2302.04479

### Comprehensive Reviews

[7] S. Blekos, D. Brand, A. Ceschini, C.H. Chou, R.H. Li, K. Mantri, O. Moussa,
    "A Review on Quantum Approximate Optimization Algorithm and its Variants,"
    Physics Reports 1068 (2024). arXiv:2306.09198.
    https://arxiv.org/abs/2306.09198

### Parameter Initialization and Optimization

[8] S.H. Sack, M. Serbyn, "Quantum Annealing Initialization of the Quantum
    Approximate Optimization Algorithm," Quantum 5, 491 (2021). arXiv:2101.05742.
    https://arxiv.org/abs/2101.05742

### QAOA for Constraint Satisfaction and Optimization

[9] S. Boulebnane, A. Montanaro, "Solving Boolean Satisfiability Problems with the
    Quantum Approximate Optimization Algorithm," arXiv:2208.06909 (2022).
    https://arxiv.org/abs/2208.06909

[10] K. Marwaha, A. El Alaoui, A. Montanari, "An Optimal Classical Algorithm for
     CSPs Obstructed by the OGP and Low-Depth QAOA," arXiv:2310.01563 (2023).
     https://arxiv.org/abs/2310.01563

### QAOA Circuit Cutting and Scalability

[11] M. Medvidovic, G. Carleo, "Circuit Cutting for QAOA,"
     Quantum 7, 934 (2023). arXiv:2207.14734.
     https://arxiv.org/abs/2207.14734

### Quantum Optimization for Binary Problems (QUBO/GAS)

[12] A. Gilliam, S. Woerner, C. Gonciulea, "Grover Adaptive Search for Constrained
     Polynomial Binary Optimization," Quantum 5, 428 (2021). arXiv:1912.04088.
     https://arxiv.org/abs/1912.04088

### DQI (Decoded Quantum Interferometry)

[13] BCG X, "Decoded Quantum Interferometry."
     https://bcg-x-official.github.io/dqi/

---

## Appendix A: Quick-Start Checklist for QAOA on an ILP

1. **Formulate the ILP** -- define decision variables, objective, and constraints.
2. **Convert to QUBO** -- substitute x_i = (1 - Z_i)/2, add penalty terms for
   constraints, collect Ising coefficients h_i and J_{ij}.
3. **Choose p** -- start with p=1 for debugging, increase to p=3-5 for better results.
4. **Build the circuit** -- cost layer from Ising coefficients, standard X-mixer,
   p layers, measurement.
5. **Initialize parameters** -- use TQA-inspired linear ramp (Section 4.3).
6. **Run optimization** -- COBYLA optimizer, 1000 iterations, statevector simulation.
7. **Analyze results** -- examine top bitstrings, check constraint satisfaction,
   compute approximation ratio.
8. **Compare with DQI** -- run same instances, compare solution quality, circuit
   depth, and total runtime.

## Appendix B: QUBO Conversion Worked Example

Consider a tiny insurance bundling problem with 3 coverage options:

```
Objective: maximize  3*x1 + 5*x2 + 2*x3  (total coverage value)
Constraints:
    2*x1 + 3*x2 + x3 <= 4   (budget constraint)
    x1 + x2 <= 1             (mutual exclusion: can't have both option 1 and 2)
```

**Step 1: Convert to minimization.** Minimize -3*x1 - 5*x2 - 2*x3.

**Step 2: Add penalty for budget constraint.**
Need slack: 2*x1 + 3*x2 + x3 + s = 4, where s in {0,1,2,3}.
Encode s in binary: s = s0 + 2*s1 (using 2 ancilla qubits).
Penalty: lambda1 * (2*x1 + 3*x2 + x3 + s0 + 2*s1 - 4)^2.

**Step 3: Add penalty for mutual exclusion.**
Penalty: lambda2 * x1 * x2.

**Step 4: Expand and collect terms** to get the QUBO matrix Q.

**Step 5: Substitute x_i = (1-Z_i)/2** to get the Ising Hamiltonian.

This yields a 5-qubit problem (3 decision + 2 slack) that can be directly encoded
in a QAOA circuit. The penalty weights lambda1, lambda2 should be chosen large enough
to enforce constraints (typically 2-5x the largest objective coefficient).
