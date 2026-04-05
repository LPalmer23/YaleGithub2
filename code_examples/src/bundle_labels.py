"""Human-readable bundle labels from ILP / Qiskit bitstrings."""

from __future__ import annotations

import numpy as np

from src.insurance_model import BundlingProblem
from src.utils import bitstring_to_array


def coverage_display_name(name: str) -> str:
    """Turn CSV name like ``auto_liability_basic`` into a short title."""
    return name.replace("_", " ").strip().title()


def humanize_selected_coverages(names: list[str], *, max_items: int = 8) -> str:
    if not names:
        return "—"
    parts = [coverage_display_name(n) for n in names[:max_items]]
    if len(names) > max_items:
        parts.append(f"+{len(names) - max_items} more")
    return " + ".join(parts)


def format_packages_assignment(
    problem: BundlingProblem,
    packages: list[list[str]],
) -> str:
    """Format classical ``solve_ilp`` ``packages`` list as one readable line."""
    lines = []
    for m, cov_names in enumerate(packages):
        if problem.package_names and m < len(problem.package_names):
            pkg = problem.package_names[m]
        else:
            pkg = f"Package {m + 1}"
        lines.append(f"{pkg}: {humanize_selected_coverages(cov_names)}")
    return " │ ".join(lines)


def flat_solution_to_packages(problem: BundlingProblem, x: np.ndarray) -> list[list[str]]:
    """Build per-package coverage names from flat vector (order ``m*N+i``)."""
    N, M = problem.N, problem.M
    x = np.asarray(x).astype(float).reshape(-1)
    x = x[: N * M]
    out: list[list[str]] = []
    for m in range(M):
        chosen = [problem.coverages[i].name for i in range(N) if x[m * N + i] > 0.5]
        out.append(chosen)
    return out


def format_bundle_from_vector(problem: BundlingProblem, x: np.ndarray) -> str:
    """Readable multi-package label from a flat binary solution vector."""
    return format_packages_assignment(problem, flat_solution_to_packages(problem, x))


def bitstring_to_flat_vector(bitstring: str, n: int) -> np.ndarray:
    """Qiskit bitstring → flat ``x`` consistent with ``get_ilp_matrices`` ordering."""
    x = bitstring_to_array(bitstring).astype(float)
    if len(x) < n:
        x = np.concatenate([x, np.zeros(n - len(x))])
    return x[:n]


def format_bundle_from_bitstring(problem: BundlingProblem, bitstring: str) -> str:
    n = problem.N * problem.M
    x = bitstring_to_flat_vector(bitstring, n)
    return format_bundle_from_vector(problem, x)


def fallback_labels_for_indices(n: int, prefix: str = "Product") -> list[str]:
    """Generic labels when coverages are unavailable (``Product 0``, …)."""
    return [f"{prefix} {i}" for i in range(n)]


def pick_overall_best_method(
    classical: float,
    qaoa: float,
    dqi: float,
    *,
    tol: float = 1e-6,
) -> tuple[float, list[str]]:
    """Return (best_value, list of methods tied for best)."""
    best = max(classical, qaoa, dqi)
    names: list[str] = []
    if abs(classical - best) <= tol:
        names.append("classical")
    if abs(qaoa - best) <= tol:
        names.append("QAOA")
    if abs(dqi - best) <= tol:
        names.append("DQI")
    return best, names
