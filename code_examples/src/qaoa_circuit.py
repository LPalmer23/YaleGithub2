"""QAOA circuit construction and optimization for insurance bundling ILP.

Implements the Quantum Approximate Optimization Algorithm (QAOA) for solving
the P&C insurance product bundling problem formulated as a 0-1 ILP.

The approach:
1. Encode the ILP objective + penalty for constraint violations as a diagonal
   cost Hamiltonian.
2. Build a parameterized QAOA circuit with alternating cost and mixer layers.
3. Optimize the variational parameters using a classical optimizer.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from src.insurance_model import BundlingProblem, build_ilp, get_ilp_matrices
from src.utils import bitstring_to_array, simulate_circuit


# ---------------------------------------------------------------------------
# Helper: extract constraint senses from the PuLP model
# ---------------------------------------------------------------------------

def get_ilp_matrices_with_senses(
    problem: BundlingProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract ILP matrices and constraint senses from a BundlingProblem.

    Returns:
        Tuple of (c, A, b, senses) where senses[k] is one of '<=', '>=', '=='.
    """
    model, x_vars = build_ilp(problem)
    c, A, b = get_ilp_matrices(problem)

    # PuLP constraint sense: -1 = <=, 0 = ==, 1 = >=
    sense_map = {-1: "<=", 0: "==", 1: ">="}
    senses = []
    for constraint in model.constraints.values():
        senses.append(sense_map.get(constraint.sense, "<="))

    return c, A, b, senses


# ---------------------------------------------------------------------------
# 1. Cost Hamiltonian diagonal
# ---------------------------------------------------------------------------

def cost_hamiltonian_diagonal(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
    penalty: float = 10.0,
) -> np.ndarray:
    """Build the diagonal of the cost Hamiltonian for the ILP.

    For each computational basis state |z>, the diagonal entry is:

        H(z) = c^T z  -  penalty * (number of violated constraints)

    where c^T z is the objective value (to maximize) and violations are penalized.

    Args:
        c: Cost vector of length n.
        A: Constraint matrix of shape (m, n).
        b: Right-hand side vector of length m.
        senses: List of constraint senses ('<=', '>=', '==').
        penalty: Penalty weight for each violated constraint.

    Returns:
        Real diagonal vector of length 2^n.
    """
    n = len(c)
    num_states = 2**n
    diag = np.zeros(num_states)

    for idx in range(num_states):
        # Convert integer index to binary vector (standard qubit ordering)
        x = np.array([(idx >> bit) & 1 for bit in range(n)], dtype=float)

        # Objective value
        obj = float(c @ x)

        # Count constraint violations
        ax = A @ x
        violations = 0
        for k, sense in enumerate(senses):
            if sense == "<=" and ax[k] > b[k] + 1e-10:
                violations += 1
            elif sense == ">=" and ax[k] < b[k] - 1e-10:
                violations += 1
            elif sense == "==" and abs(ax[k] - b[k]) > 1e-10:
                violations += 1

        diag[idx] = obj - penalty * violations

    return diag


# ---------------------------------------------------------------------------
# 2. Build QAOA circuit
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
    gammas: list[float] | np.ndarray,
    betas: list[float] | np.ndarray,
    penalty: float = 10.0,
) -> QuantumCircuit:
    """Build a QAOA circuit for the insurance bundling ILP.

    Uses the diagonal unitary approach for the cost layer, which is exact
    for small qubit counts (suitable for the toy problem with <= ~20 qubits).

    The circuit structure for p layers is:
        |0>^n  ->  H^n  ->  [U_C(gamma_k) U_B(beta_k)]_{k=1..p}  ->  measure

    Args:
        c: Cost vector of length n.
        A: Constraint matrix of shape (m, n).
        b: Right-hand side of constraints, length m.
        senses: Constraint senses ('<=', '>=', '==').
        gammas: Cost-layer angles, one per QAOA layer.
        betas: Mixer-layer angles, one per QAOA layer.
        penalty: Penalty weight for constraint violations.

    Returns:
        QuantumCircuit with measurement gates.
    """
    n = len(c)
    p = len(gammas)
    assert len(betas) == p, "gammas and betas must have the same length"

    # Precompute the diagonal cost Hamiltonian
    diag = cost_hamiltonian_diagonal(c, A, b, senses, penalty)

    qc = QuantumCircuit(n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for layer in range(p):
        # --- Cost unitary: exp(-i * gamma * C) ---
        # Apply as a diagonal unitary matrix via UnitaryGate
        phases = -gammas[layer] * diag
        diag_unitary = np.diag(np.exp(1j * phases))
        from qiskit.circuit.library import UnitaryGate
        qc.append(UnitaryGate(diag_unitary, label="U_C"), range(n))

        # --- Mixer unitary: exp(-i * beta * B) where B = sum_j X_j ---
        # Factorizes into single-qubit RX rotations
        for j in range(n):
            qc.rx(2 * betas[layer], j)

    # Add measurements
    qc.measure_all()

    return qc


# ---------------------------------------------------------------------------
# 3. Evaluate QAOA measurement results
# ---------------------------------------------------------------------------

def evaluate_qaoa(
    counts: dict[str, int],
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
    penalty: float = 10.0,
) -> dict:
    """Evaluate QAOA measurement results.

    Computes the expected objective value from measurement counts,
    filtering for feasible solutions and identifying the best one.

    Args:
        counts: Measurement counts dict {bitstring: count}.
        c: Cost vector.
        A: Constraint matrix.
        b: Right-hand side vector.
        senses: Constraint senses.
        penalty: Penalty weight (not used in evaluation, included for API consistency).

    Returns:
        Dict with keys:
            mean_objective: Mean objective over all feasible solutions (weighted by counts).
            best_objective: Best feasible objective value found.
            best_bitstring: Bitstring achieving best_objective.
            feasibility_rate: Fraction of shots that produced feasible solutions.
    """
    total_shots = sum(counts.values())

    feasible_obj_sum = 0.0
    feasible_count = 0
    best_objective = -np.inf
    best_bitstring = None

    for bitstring, count in counts.items():
        x = bitstring_to_array(bitstring)
        # Pad or truncate to match cost vector length
        if len(x) < len(c):
            x = np.concatenate([x, np.zeros(len(c) - len(x))])
        x = x[: len(c)]

        # Check feasibility
        ax = A @ x
        feasible = True
        for k, sense in enumerate(senses):
            if sense == "<=" and ax[k] > b[k] + 1e-10:
                feasible = False
                break
            elif sense == ">=" and ax[k] < b[k] - 1e-10:
                feasible = False
                break
            elif sense == "==" and abs(ax[k] - b[k]) > 1e-10:
                feasible = False
                break

        if feasible:
            obj = float(c @ x)
            feasible_obj_sum += obj * count
            feasible_count += count
            if obj > best_objective:
                best_objective = obj
                best_bitstring = bitstring

    feasibility_rate = feasible_count / total_shots if total_shots > 0 else 0.0
    mean_objective = feasible_obj_sum / feasible_count if feasible_count > 0 else 0.0

    if best_bitstring is None:
        best_objective = 0.0

    return {
        "mean_objective": mean_objective,
        "best_objective": best_objective,
        "best_bitstring": best_bitstring,
        "feasibility_rate": feasibility_rate,
    }


# ---------------------------------------------------------------------------
# 4. Full QAOA optimization loop
# ---------------------------------------------------------------------------

def run_qaoa(
    problem: BundlingProblem,
    p: int = 1,
    max_iterations: int = 200,
    shots: int = 1024,
    optimizer: str = "COBYLA",
    penalty: float = 10.0,
    seed: int | None = 42,
) -> dict:
    """Run the full QAOA optimization loop for a BundlingProblem.

    Extracts the ILP matrices, builds parameterized QAOA circuits, and
    uses scipy.optimize.minimize to find optimal variational parameters.

    Args:
        problem: Insurance bundling problem instance.
        p: Number of QAOA layers.
        max_iterations: Maximum optimizer iterations.
        shots: Number of measurement shots per circuit evaluation.
        optimizer: Scipy optimizer name ('COBYLA', 'Nelder-Mead', 'Powell').
        penalty: Penalty weight for constraint violations.
        seed: Random seed for reproducibility (None for no seeding).

    Returns:
        Dict with keys:
            optimal_params: Best [gammas..., betas...] found.
            optimal_value: Best (negative) expectation value from optimizer.
            convergence_history: List of objective values during optimization.
            best_solution: Best feasible bitstring found across all evaluations.
            best_objective: Objective value of best_solution.
    """
    c, A, b, senses = get_ilp_matrices_with_senses(problem)
    n = len(c)

    # Precompute diagonal for statevector-based expectation value
    diag = cost_hamiltonian_diagonal(c, A, b, senses, penalty)

    convergence_history: list[float] = []
    best_solution_overall: str | None = None
    best_obj_overall: float = -np.inf

    def objective(params: np.ndarray) -> float:
        """Objective function for the classical optimizer.

        We negate the cost because scipy minimizes, but we want to maximize.
        Uses statevector simulation for exact expectation values (efficient
        for small qubit counts).
        """
        nonlocal best_solution_overall, best_obj_overall

        gammas = params[:p]
        betas = params[p:]

        # Use Statevector directly for exact expectation (fast, no Aer needed)
        sv = Statevector.from_int(0, 2**n)
        qc_sv = QuantumCircuit(n)
        qc_sv.h(range(n))
        for layer in range(p):
            phases = -gammas[layer] * diag
            diag_unitary = np.diag(np.exp(1j * phases))
            from qiskit.circuit.library import UnitaryGate
            qc_sv.append(UnitaryGate(diag_unitary), range(n))
            for j in range(n):
                qc_sv.rx(2 * betas[layer], j)
        sv = sv.evolve(qc_sv)
        probs = sv.probabilities()
        expectation = float(np.dot(probs, diag))

        convergence_history.append(-expectation)

        # Track best feasible solution from probability distribution
        best_idx = np.argmax(probs * diag)
        best_x = np.array([(best_idx >> bit) & 1 for bit in range(n)], dtype=float)

        # Check feasibility of the most likely good state
        ax = A @ best_x
        is_feasible = True
        for k, sense in enumerate(senses):
            if sense == "<=" and ax[k] > b[k] + 1e-10:
                is_feasible = False
                break
            elif sense == ">=" and ax[k] < b[k] - 1e-10:
                is_feasible = False
                break
            elif sense == "==" and abs(ax[k] - b[k]) > 1e-10:
                is_feasible = False
                break

        if is_feasible:
            obj = float(c @ best_x)
            if obj > best_obj_overall:
                best_obj_overall = obj
                # Convert to Qiskit bitstring (reversed bit order)
                best_solution_overall = "".join(
                    str(int(best_x[i])) for i in reversed(range(n))
                )

        # Negate because scipy minimizes
        return -expectation

    # Initial parameters
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * p)

    result = minimize(
        objective,
        x0,
        method=optimizer,
        options={"maxiter": max_iterations},
    )

    # Always sample the optimized circuit: variational tracking is approximate;
    # measurement over the full distribution finds high-quality feasible states.
    gammas_opt = result.x[:p]
    betas_opt = result.x[p:]
    qc_final = build_qaoa_circuit(c, A, b, senses, gammas_opt, betas_opt, penalty)
    counts_final = simulate_circuit(qc_final, shots=shots)
    eval_result = evaluate_qaoa(counts_final, c, A, b, senses, penalty)
    if eval_result["best_objective"] > best_obj_overall:
        best_obj_overall = eval_result["best_objective"]
        best_solution_overall = eval_result["best_bitstring"]

    return {
        "optimal_params": result.x,
        "optimal_value": result.fun,
        "convergence_history": convergence_history,
        "best_solution": best_solution_overall,
        "best_objective": best_obj_overall,
        "final_counts": counts_final,
        "final_feasibility_rate": eval_result["feasibility_rate"],
        "final_mean_objective_feasible": eval_result["mean_objective"],
    }
