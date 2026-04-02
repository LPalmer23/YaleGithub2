"""Shared utilities for DQI vs QAOA insurance optimization project."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def simulate_circuit(qc: QuantumCircuit, shots: int = 8192, method: str = "automatic") -> dict:
    """Run a quantum circuit on the Aer simulator and return counts.

    Args:
        qc: Qiskit QuantumCircuit to simulate.
        shots: Number of measurement shots.
        method: Simulation method ('statevector', 'qasm', 'automatic').

    Returns:
        Dictionary mapping bitstrings to counts.
    """
    simulator = AerSimulator(method=method)
    result = simulator.run(qc, shots=shots).result()
    return result.get_counts()


def statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    """Get the statevector from a circuit (without measurement gates).

    Args:
        qc: QuantumCircuit without measurements.

    Returns:
        Complex numpy array of statevector amplitudes.
    """
    simulator = AerSimulator(method="statevector")
    qc_copy = qc.copy()
    qc_copy.save_statevector()
    result = simulator.run(qc_copy).result()
    return np.array(result.get_statevector())


def bitstring_to_array(bitstring: str) -> np.ndarray:
    """Convert a Qiskit bitstring to a numpy binary array.

    Qiskit returns bitstrings in reverse qubit order (least significant first).
    This function returns an array in standard order [q0, q1, ...].

    Args:
        bitstring: String of '0' and '1' characters.

    Returns:
        Numpy array of 0s and 1s.
    """
    return np.array([int(b) for b in reversed(bitstring)])


def objective_value_from_bitstring(bitstring: str, cost_vector: np.ndarray) -> float:
    """Compute the objective value c^T x for a given bitstring solution.

    Args:
        bitstring: Solution bitstring from quantum measurement.
        cost_vector: Objective coefficients c.

    Returns:
        Scalar objective value.
    """
    x = bitstring_to_array(bitstring)
    return float(cost_vector @ x[:len(cost_vector)])


def top_solutions(counts: dict, cost_vector: np.ndarray, k: int = 5) -> list[dict]:
    """Extract the top-k solutions by objective value from measurement counts.

    Args:
        counts: Measurement counts dict {bitstring: count}.
        cost_vector: Objective coefficients c.
        k: Number of top solutions to return.

    Returns:
        List of dicts with 'bitstring', 'objective', 'count', 'probability'.
    """
    total_shots = sum(counts.values())
    solutions = []
    for bitstring, count in counts.items():
        obj = objective_value_from_bitstring(bitstring, cost_vector)
        solutions.append({
            "bitstring": bitstring,
            "objective": obj,
            "count": count,
            "probability": count / total_shots,
        })
    solutions.sort(key=lambda s: s["objective"], reverse=True)
    return solutions[:k]
