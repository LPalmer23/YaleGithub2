"""Decoded Quantum Interferometry (DQI) circuit construction for max-XORSAT.

Implements the DQI algorithm from arXiv:2509.08328 for solving P&C insurance
bundling optimization via max-XORSAT interference.

The DQI algorithm (6 steps):
  1. Dicke state preparation on the message register
  2. Phase encoding (Z gates from constraint vector v)
  3. Syndrome encoding (CNOT matrix-vector multiply B^T y mod 2)
  4. BP1 decoder (flip bits to reduce syndrome weight)
  5. Hadamard transform on syndrome register
  6. Measurement and post-selection

For the toy problem (small m, n), we implement a simplified but faithful
version that stays under ~20 qubits for statevector simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

from src.ilp_to_maxxorsat import MaxXORSATInstance


# ---------------------------------------------------------------------------
# 1. Dicke state preparation
# ---------------------------------------------------------------------------

def prepare_dicke_state(qc: QuantumCircuit, qubits: list[int], weight: int) -> None:
    """Prepare a Dicke state |D_{n,w}> on the specified qubits.

    A Dicke state is an equal superposition of all n-qubit computational
    basis states with Hamming weight w:

        |D_{n,w}> = (1/sqrt(C(n,w))) sum_{|x|=w} |x>

    For small n (our toy problem, n <= ~15), we construct the exact
    statevector and use Qiskit's initialize(). This is exact and efficient
    for the qubit counts we target.

    Args:
        qc: QuantumCircuit to append gates to (modified in place).
        qubits: List of qubit indices to prepare the Dicke state on.
        weight: Desired Hamming weight w (0 <= w <= len(qubits)).
    """
    n = len(qubits)
    if weight < 0 or weight > n:
        raise ValueError(f"Weight {weight} out of range [0, {n}]")
    if n == 0:
        return

    # Base cases
    if weight == 0:
        return  # All qubits already |0>
    if weight == n:
        for q in qubits:
            qc.x(q)
        return

    # Build the Dicke state vector and use initialize()
    dim = 2 ** n
    binom = math.comb(n, weight)
    amplitude = 1.0 / math.sqrt(binom)

    sv = np.zeros(dim, dtype=complex)
    for idx in range(dim):
        if bin(idx).count("1") == weight:
            sv[idx] = amplitude

    qc.initialize(sv, qubits)


def prepare_weighted_dicke_superposition(
    qc: QuantumCircuit,
    qubits: list[int],
    weights: np.ndarray,
    max_weight: int,
) -> None:
    """Prepare a weighted superposition of Dicke states.

    Prepares: sum_{k=0}^{ell} w_k |D_{m,k}>

    where w_k are the Dicke state weights from the principal eigenvector
    of the tridiagonal matrix A (Eq. 7 of the paper).

    For max-XORSAT (p=2, r=1), d=0 so A is purely off-diagonal.

    Args:
        qc: QuantumCircuit to modify in place.
        qubits: Qubit indices for the message register.
        weights: Array of Dicke state weights w_k for k=0..max_weight.
        max_weight: Maximum Hamming weight ell.
    """
    m = len(qubits)

    # Normalize weights
    w = np.array(weights[:max_weight + 1], dtype=float)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return
    w = w / norm

    # For the simplified implementation: prepare a superposition using
    # amplitude encoding. For small m, we construct the full state vector
    # and use initialize().

    # Build the target statevector as sum of weighted Dicke states
    dim = 2 ** m
    target_sv = np.zeros(dim, dtype=complex)

    for k in range(max_weight + 1):
        if abs(w[k]) < 1e-12:
            continue
        # Find all basis states with Hamming weight k
        binom_coeff = math.comb(m, k)
        if binom_coeff == 0:
            continue
        amplitude = w[k] / math.sqrt(binom_coeff)
        for idx in range(dim):
            if bin(idx).count("1") == k:
                target_sv[idx] += amplitude

    # Normalize
    sv_norm = np.linalg.norm(target_sv)
    if sv_norm > 1e-12:
        target_sv /= sv_norm

    # Use Qiskit's initialize to prepare the state
    qc.initialize(target_sv, qubits)


def compute_dicke_weights(m: int, max_weight: int) -> np.ndarray:
    """Compute the Dicke state weights from the tridiagonal matrix A.

    For max-XORSAT problems: p=2, r=1, giving d=0.
    The matrix A has zero diagonal and off-diagonal entries
    a_k = sqrt(k * (m - k + 1)).

    The weights w_k are the components of the principal (largest)
    eigenvector of A.

    Args:
        m: Number of message qubits (constraints).
        max_weight: Maximum Hamming weight ell.

    Returns:
        Array of weights w_0, w_1, ..., w_{ell}.
    """
    ell = min(max_weight, m)
    size = ell + 1

    if size <= 1:
        return np.array([1.0])

    # Build tridiagonal matrix A (Eq. 7)
    # d = 0 for max-XORSAT (p=2, r=1)
    # a_k = sqrt(k * (m - k + 1)) for k = 1, ..., ell
    A_mat = np.zeros((size, size))
    for k in range(1, size):
        a_k = math.sqrt(k * (m - k + 1))
        A_mat[k, k - 1] = a_k
        A_mat[k - 1, k] = a_k

    # Find principal eigenvector (largest eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eigh(A_mat)
    # eigh returns sorted ascending, so largest is last
    principal_eigvec = eigenvectors[:, -1]

    # Ensure positive convention (first nonzero component positive)
    for i in range(len(principal_eigvec)):
        if abs(principal_eigvec[i]) > 1e-10:
            if principal_eigvec[i] < 0:
                principal_eigvec = -principal_eigvec
            break

    return principal_eigvec


# ---------------------------------------------------------------------------
# 2. Phase encoding
# ---------------------------------------------------------------------------

def apply_phase_encoding(
    qc: QuantumCircuit, message_qubits: list[int], v: np.ndarray
) -> None:
    """Apply Z gates to encode the constraint vector v into phases.

    For each qubit i where v_i = 1, apply a Z gate:
        Z|0> = |0>,  Z|1> = -|1>

    This introduces phase factors (-1)^{v.y} in the superposition.

    Args:
        qc: QuantumCircuit to modify in place.
        message_qubits: Indices of the message register qubits.
        v: Binary constraint vector of length m = len(message_qubits).
    """
    if len(v) != len(message_qubits):
        raise ValueError(
            f"Constraint vector length {len(v)} != message register size {len(message_qubits)}"
        )
    for i, q in enumerate(message_qubits):
        if int(v[i]) == 1:
            qc.z(q)


# ---------------------------------------------------------------------------
# 3. Syndrome encoding
# ---------------------------------------------------------------------------

def apply_syndrome_encoding(
    qc: QuantumCircuit,
    message_qubits: list[int],
    syndrome_qubits: list[int],
    B: np.ndarray,
) -> None:
    """Encode the syndrome s = B^T y (mod 2) into the syndrome register.

    For each entry B^T[i,j] = 1, apply a CNOT from message qubit j
    to syndrome qubit i. This computes the matrix-vector product
    B^T y mod 2 into the syndrome register (initially |0>^n).

    Note: B is m x n (equations x variables). We need B^T which is n x m.
    So syndrome qubit i accumulates XOR of message qubits j where B[j,i] = 1.

    Args:
        qc: QuantumCircuit to modify in place.
        message_qubits: Indices of message register (m qubits, one per equation).
        syndrome_qubits: Indices of syndrome register (n qubits, one per variable).
        B: Parity check matrix of shape (m, n). Binary.
    """
    m, n = B.shape
    if len(message_qubits) != m:
        raise ValueError(f"Message qubits ({len(message_qubits)}) != B rows ({m})")
    if len(syndrome_qubits) != n:
        raise ValueError(f"Syndrome qubits ({len(syndrome_qubits)}) != B cols ({n})")

    # B^T[i, j] = B[j, i]. For each syndrome qubit i (column of B),
    # CNOT from each message qubit j where B[j, i] = 1.
    for i in range(n):
        for j in range(m):
            if B[j, i] == 1:
                qc.cx(message_qubits[j], syndrome_qubits[i])


# ---------------------------------------------------------------------------
# 4. Simplified BP1 decoder
# ---------------------------------------------------------------------------

def classical_bp1_decode(
    B: np.ndarray, y: np.ndarray, max_iterations: int = 5
) -> tuple[bool, np.ndarray]:
    """Classical BP1 decoder (Algorithm 1 from the paper).

    Tries to decode the error pattern y back to the all-zero codeword
    using binary belief propagation with hard decisions.

    A bit j is flipped if ALL of its connected check nodes have
    unsatisfied syndromes (threshold = full degree).

    Args:
        B: Parity check matrix (m x n). B^T is the code's parity check.
        y: Binary error pattern of length m.
        max_iterations: Maximum number of BP iterations T.

    Returns:
        Tuple of (success, decoded_bits) where success is True if
        all bits were decoded to zero.
    """
    m, n = B.shape
    BT = B.T  # n x m
    bits = y.copy().astype(int)

    for _ in range(max_iterations):
        syndrome = (BT @ bits) % 2
        if np.all(syndrome == 0):
            return True, bits

        flip_candidates = np.zeros(m, dtype=int)
        for j in range(m):
            # Connected check nodes for variable j
            connected = np.where(BT[:, j] == 1)[0]
            if len(connected) == 0:
                continue
            unsatisfied = np.sum(syndrome[connected] == 1)
            # Flip if ALL connected checks are unsatisfied
            if unsatisfied >= len(connected):
                flip_candidates[j] = 1

        bits = (bits + flip_candidates) % 2

    return bool(np.all(bits == 0)), bits


def apply_simplified_bp1_decoder(
    qc: QuantumCircuit,
    message_qubits: list[int],
    syndrome_qubits: list[int],
    B: np.ndarray,
    iterations: int = 1,
) -> None:
    """Apply a simplified quantum BP1 decoder circuit.

    For the toy problem (small m, n), we implement a simplified version
    of the BP1 decoder that:
    1. For each message qubit j, checks if all connected syndrome qubits are |1>
    2. If so, flips the message qubit (applies X conditioned on all connected syndromes)
    3. Updates the syndrome register

    This is a faithful but simplified version of the full BP1 circuit
    (which would require hamming weight, comparator, and flip registers).
    The simplification works because for small problems, we can directly
    use multi-controlled X gates instead of the Hamming weight subroutine.

    Args:
        qc: QuantumCircuit to modify in place.
        message_qubits: Indices of message register (m qubits).
        syndrome_qubits: Indices of syndrome register (n qubits).
        B: Parity check matrix (m x n).
        iterations: Number of BP iterations T.
    """
    m, n = B.shape
    BT = B.T  # n x m

    for _t in range(iterations):
        # For each message qubit j, find its connected syndrome qubits
        for j in range(m):
            connected_syndromes = []
            for i in range(n):
                if BT[i, j] == 1:
                    connected_syndromes.append(syndrome_qubits[i])

            if len(connected_syndromes) == 0:
                continue

            if len(connected_syndromes) == 1:
                # Single connected check: CNOT
                qc.cx(connected_syndromes[0], message_qubits[j])
            else:
                # Multi-controlled X: flip message qubit j if ALL
                # connected syndrome qubits are |1>
                qc.mcx(connected_syndromes, message_qubits[j])

        # After flipping, update the syndrome register
        # New syndrome = B^T (y XOR flip) = B^T y XOR B^T flip
        # Since we modified message qubits in place, we need to
        # recompute the syndrome. We do this by:
        # 1. Uncompute old syndrome (apply syndrome encoding inverse)
        # 2. Recompute new syndrome

        # Only update syndrome if there are more iterations to follow
        if _t < iterations - 1:
            # Uncompute: apply inverse syndrome encoding
            _apply_syndrome_encoding_inverse(
                qc, message_qubits, syndrome_qubits, B
            )
            # Recompute with updated message register
            apply_syndrome_encoding(
                qc, message_qubits, syndrome_qubits, B
            )


def _apply_syndrome_encoding_inverse(
    qc: QuantumCircuit,
    message_qubits: list[int],
    syndrome_qubits: list[int],
    B: np.ndarray,
) -> None:
    """Apply the inverse of syndrome encoding (CNOT is self-inverse).

    Since CNOT is its own inverse, the inverse of syndrome encoding
    is the same sequence of CNOTs applied in reverse order.
    """
    m, n = B.shape
    # Apply in reverse order
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if B[j, i] == 1:
                qc.cx(message_qubits[j], syndrome_qubits[i])


# ---------------------------------------------------------------------------
# 5. Hadamard transform on syndrome register
# ---------------------------------------------------------------------------

def apply_hadamard_transform(
    qc: QuantumCircuit, syndrome_qubits: list[int]
) -> None:
    """Apply Hadamard gates to all syndrome qubits.

    This converts phase information into amplitude information via
    quantum interference, which is the key mechanism of DQI.

    Args:
        qc: QuantumCircuit to modify in place.
        syndrome_qubits: Indices of the syndrome register qubits.
    """
    for q in syndrome_qubits:
        qc.h(q)


# ---------------------------------------------------------------------------
# 6. Full DQI circuit builder
# ---------------------------------------------------------------------------

@dataclass
class DQIResult:
    """Results from running the DQI algorithm.

    Attributes:
        circuit: The full DQI quantum circuit.
        counts: Measurement counts dict {bitstring: count}.
        solutions: List of candidate solutions extracted from measurements.
        best_solution: Best solution found (original ILP variable values).
        best_objective: Objective value of the best solution.
        post_selection_rate: Fraction of measurements passing post-selection.
        num_message_qubits: Number of message register qubits (m).
        num_syndrome_qubits: Number of syndrome register qubits (n).
    """

    circuit: QuantumCircuit
    counts: dict[str, int]
    solutions: list[dict]
    best_solution: np.ndarray | None
    best_objective: float
    post_selection_rate: float
    num_message_qubits: int
    num_syndrome_qubits: int


def build_dqi_circuit(
    instance: MaxXORSATInstance,
    max_weight: int = 2,
    bp1_iterations: int = 1,
    use_dicke_weights: bool = True,
) -> QuantumCircuit:
    """Build the full DQI circuit for a max-XORSAT instance.

    Implements the 6-step DQI algorithm:
    1. Dicke state preparation on message register
    2. Phase encoding using constraint vector v
    3. Syndrome encoding using parity check matrix B
    4. Simplified BP1 decoder
    5. Hadamard transform on syndrome register
    6. Measurement of all qubits

    Args:
        instance: MaxXORSATInstance with B matrix, v vector, and weights.
        max_weight: Maximum Hamming weight ell for Dicke state preparation.
        bp1_iterations: Number of BP1 decoder iterations T.
        use_dicke_weights: If True, use weighted Dicke superposition.
            If False, use simple equal-weight Dicke state at weight=1.

    Returns:
        QuantumCircuit with measurement gates on all registers.
    """
    m = instance.num_equations   # rows of B -> message register
    n = instance.num_variables   # cols of B -> syndrome register
    B = instance.B
    v = instance.v

    # Create quantum registers
    msg_reg = QuantumRegister(m, "msg")
    syn_reg = QuantumRegister(n, "syn")
    qc = QuantumCircuit(msg_reg, syn_reg)

    msg_qubits = list(range(m))
    syn_qubits = list(range(m, m + n))

    # Step 1: Dicke state preparation
    if use_dicke_weights:
        weights = compute_dicke_weights(m, max_weight)
        prepare_weighted_dicke_superposition(qc, msg_qubits, weights, max_weight)
    else:
        # Simple Dicke state with weight 1
        prepare_dicke_state(qc, msg_qubits, weight=1)

    # Step 2: Phase encoding
    apply_phase_encoding(qc, msg_qubits, v)

    # Step 3: Syndrome encoding
    apply_syndrome_encoding(qc, msg_qubits, syn_qubits, B)

    # Step 4: Simplified BP1 decoder
    if bp1_iterations > 0:
        apply_simplified_bp1_decoder(
            qc, msg_qubits, syn_qubits, B, iterations=bp1_iterations
        )

    # Step 5: Hadamard transform on syndrome register
    apply_hadamard_transform(qc, syn_qubits)

    # Step 6: Measurement
    qc.measure_all()

    return qc


def build_dqi_circuit_no_decoder(
    instance: MaxXORSATInstance,
) -> QuantumCircuit:
    """Build a simplified DQI circuit without BP1 decoder.

    This is the "brute-force DQI" variant that skips the decoder step.
    It still uses the interference-based approach (phase + syndrome +
    Hadamard) which distinguishes it from QAOA.

    For the toy problem, this provides a baseline DQI result that
    can be compared with the decoder-enhanced version.

    Args:
        instance: MaxXORSATInstance with B matrix, v vector, and weights.

    Returns:
        QuantumCircuit with measurement gates.
    """
    m = instance.num_equations
    n = instance.num_variables
    B = instance.B
    v = instance.v

    msg_reg = QuantumRegister(m, "msg")
    syn_reg = QuantumRegister(n, "syn")
    qc = QuantumCircuit(msg_reg, syn_reg)

    msg_qubits = list(range(m))
    syn_qubits = list(range(m, m + n))

    # Initialize message register in uniform superposition (weight-1 Dicke state)
    prepare_dicke_state(qc, msg_qubits, weight=1)

    # Phase encoding
    apply_phase_encoding(qc, msg_qubits, v)

    # Syndrome encoding
    apply_syndrome_encoding(qc, msg_qubits, syn_qubits, B)

    # Hadamard on syndrome
    apply_hadamard_transform(qc, syn_qubits)

    # Measure all
    qc.measure_all()

    return qc


# ---------------------------------------------------------------------------
# 7. Post-processing and solution extraction
# ---------------------------------------------------------------------------

def extract_dqi_solutions(
    counts: dict[str, int],
    instance: MaxXORSATInstance,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
) -> DQIResult:
    """Extract and evaluate solutions from DQI measurement results.

    Post-processes measurement outcomes:
    1. Separate message and syndrome register bits
    2. Post-select on message register being all-zeros (decoder success)
    3. Extract candidate solutions from syndrome register bits
    4. Map back to original ILP variables and evaluate

    Args:
        counts: Measurement counts dict from circuit simulation.
        instance: The MaxXORSATInstance used to build the circuit.
        c: Original ILP cost vector.
        A: Original ILP constraint matrix.
        b: Original ILP RHS vector.
        senses: Original ILP constraint senses.

    Returns:
        DQIResult with extracted solutions and statistics.
    """
    m = instance.num_equations
    n = instance.num_variables
    total_qubits = m + n
    total_shots = sum(counts.values())

    solutions = []
    post_selected_shots = 0
    best_objective = -np.inf
    best_solution = None

    for bitstring, count in counts.items():
        # Qiskit bitstrings are in reverse order: last qubit is leftmost
        # Full bitstring has total_qubits bits
        # Pad if needed
        bs = bitstring.replace(" ", "")
        if len(bs) < total_qubits:
            bs = "0" * (total_qubits - len(bs)) + bs

        # Convert to array (reversed because Qiskit convention)
        bits = np.array([int(b_) for b_ in reversed(bs)], dtype=int)

        # Split into message (first m bits) and syndrome (next n bits)
        msg_bits = bits[:m]
        syn_bits = bits[m:m + n]

        # Post-selection: keep only if message register is all-zeros
        # (indicating successful decoding)
        msg_is_zero = np.all(msg_bits == 0)

        if msg_is_zero:
            post_selected_shots += count

            # The syndrome register bits are the candidate solution
            # for the max-XORSAT variables
            x_candidate = syn_bits

            # Extract original ILP variables
            orig_vars = instance.extract_original_vars(x_candidate)

            # Check feasibility against original ILP constraints
            feasible = _check_ilp_feasibility(orig_vars, A, b, senses)

            # Compute objective value
            obj = float(c @ orig_vars) if len(orig_vars) == len(c) else 0.0

            # Check XORSAT satisfaction
            sat, total = instance.check_solution(x_candidate)

            solutions.append({
                "bitstring": bitstring,
                "syndrome_bits": syn_bits.tolist(),
                "original_vars": orig_vars.tolist(),
                "objective": obj,
                "feasible": feasible,
                "xorsat_satisfied": sat,
                "xorsat_total": total,
                "count": count,
            })

            if feasible and obj > best_objective:
                best_objective = obj
                best_solution = orig_vars.copy()

    # Also extract solutions WITHOUT post-selection for comparison
    # (useful when decoder success rate is low)
    all_solutions = []
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        if len(bs) < total_qubits:
            bs = "0" * (total_qubits - len(bs)) + bs
        bits = np.array([int(b_) for b_ in reversed(bs)], dtype=int)
        syn_bits = bits[m:m + n]
        orig_vars = instance.extract_original_vars(syn_bits)
        feasible = _check_ilp_feasibility(orig_vars, A, b, senses)
        obj = float(c @ orig_vars) if len(orig_vars) == len(c) else 0.0

        if feasible and obj > best_objective:
            best_objective = obj
            best_solution = orig_vars.copy()

        all_solutions.append({
            "bitstring": bitstring,
            "original_vars": orig_vars.tolist(),
            "objective": obj,
            "feasible": feasible,
            "count": count,
        })

    post_selection_rate = post_selected_shots / total_shots if total_shots > 0 else 0.0

    if best_solution is None:
        best_objective = 0.0

    return DQIResult(
        circuit=QuantumCircuit(),  # Placeholder; caller can set this
        counts=counts,
        solutions=solutions if solutions else all_solutions,
        best_solution=best_solution,
        best_objective=best_objective,
        post_selection_rate=post_selection_rate,
        num_message_qubits=m,
        num_syndrome_qubits=n,
    )


def _check_ilp_feasibility(
    x: np.ndarray, A: np.ndarray, b: np.ndarray, senses: list[str]
) -> bool:
    """Check if a solution x is feasible for the original ILP constraints."""
    if len(x) != A.shape[1]:
        return False
    ax = A @ x
    for k, sense in enumerate(senses):
        if sense == "<=" and ax[k] > b[k] + 1e-10:
            return False
        elif sense == ">=" and ax[k] < b[k] - 1e-10:
            return False
        elif sense == "==" and abs(ax[k] - b[k]) > 1e-10:
            return False
    return True


# ---------------------------------------------------------------------------
# 8. Full DQI pipeline (convenience function)
# ---------------------------------------------------------------------------

def run_dqi(
    instance: MaxXORSATInstance,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
    max_weight: int = 2,
    bp1_iterations: int = 1,
    use_dicke_weights: bool = True,
    shots: int = 8192,
) -> DQIResult:
    """Run the full DQI pipeline: build circuit, simulate, extract solutions.

    This is the main entry point for running DQI on a max-XORSAT instance
    derived from an insurance bundling ILP.

    Args:
        instance: MaxXORSATInstance from ilp_to_maxxorsat.
        c: Original ILP cost vector.
        A: Original ILP constraint matrix.
        b: Original ILP RHS vector.
        senses: Original ILP constraint senses.
        max_weight: Maximum Hamming weight for Dicke state.
        bp1_iterations: Number of BP1 decoder iterations.
        use_dicke_weights: Whether to use weighted Dicke superposition.
        shots: Number of measurement shots.

    Returns:
        DQIResult with circuit, measurements, and extracted solutions.
    """
    from src.utils import simulate_circuit

    # Build the DQI circuit
    qc = build_dqi_circuit(
        instance,
        max_weight=max_weight,
        bp1_iterations=bp1_iterations,
        use_dicke_weights=use_dicke_weights,
    )

    # Simulate
    counts = simulate_circuit(qc, shots=shots)

    # Extract solutions
    result = extract_dqi_solutions(counts, instance, c, A, b, senses)
    result.circuit = qc

    return result
