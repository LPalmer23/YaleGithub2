"""Matplotlib figures for quantum benchmark artifacts (no Jupyter dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.bundle_labels import format_bundle_from_bitstring, format_packages_assignment
from src.insurance_model import BundlingProblem, build_ilp, get_ilp_matrices
from src.utils import bitstring_to_array


class _BarChartMetrics(Protocol):
    classical_optimal: float
    qaoa_best: float
    dqi_best: float
    n_vars: int


def _finish_figure(fig: Any, path: Path | None, *, dpi: int, show: bool) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def _ilp_feasibility_rows(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[str],
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    total = sum(counts.values())
    rows = []
    for bs, cnt in counts.items():
        x = bitstring_to_array(bs)
        if len(x) < len(c):
            x = np.concatenate([x, np.zeros(len(c) - len(x))])
        x = x[: len(c)]
        axv = A @ x
        ok = True
        for k, s in enumerate(senses):
            if s == "<=" and axv[k] > b[k] + 1e-9:
                ok = False
                break
            if s == ">=" and axv[k] < b[k] - 1e-9:
                ok = False
                break
            if s == "==" and abs(axv[k] - b[k]) > 1e-9:
                ok = False
                break
        if not ok:
            continue
        rows.append({"bitstring": bs, "p": cnt / total, "objective": float(c @ x)})
    return rows


def feasible_qaoa_sample_rows(
    problem: BundlingProblem,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """QAOA measurement counts restricted to ILP-feasible bitstrings (with probabilities)."""
    c, A, b = get_ilp_matrices(problem)
    model, _ = build_ilp(problem)
    sense_map = {-1: "<=", 0: "==", 1: ">="}
    senses = [sense_map[int(c_.sense)] for c_ in model.constraints.values()]
    return _ilp_feasibility_rows(c, A, b, senses, counts)


def chart_top_bundles_data(
    problem: BundlingProblem,
    counts: dict[str, int],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Chart.js-ready rows: label, objective, probability, is_best (within top-k QAOA feasible)."""
    rows = feasible_qaoa_sample_rows(problem, counts)
    if not rows:
        return []
    best_by_bs: dict[str, dict[str, Any]] = {}
    for r in rows:
        bs = r["bitstring"]
        if bs not in best_by_bs or r["objective"] > best_by_bs[bs]["objective"]:
            best_by_bs[bs] = r
    unique = list(best_by_bs.values())
    unique.sort(key=lambda r: r["objective"], reverse=True)
    top = unique[:top_k]
    best_o = top[0]["objective"]
    return [
        {
            "label": format_bundle_from_bitstring(problem, r["bitstring"]),
            "objective": r["objective"],
            "probability": r["p"],
            "is_best": abs(r["objective"] - best_o) < 1e-9,
        }
        for r in top
    ]


def save_objectives_bar_chart(
    row: _BarChartMetrics,
    path: Path | None,
    *,
    dpi: int = 120,
    show: bool = False,
    chart_title: str | None = None,
) -> Path | None:
    labels = ["Classical\n(optimal)", "QAOA\n(best sampled)", "DQI\n(best feasible)"]
    values = [row.classical_optimal, row.qaoa_best, row.dqi_best]
    colors = ["#2e7d32", "#1565c0", "#6a1b9a"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(
        row.classical_optimal,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="Classical optimum",
    )
    ax.set_ylabel("Objective (contribution margin)")
    ax.set_title(
        chart_title or f"LTM subsample n={row.n_vars} — Qiskit Aer simulation"
    )
    for b, v in zip(bars, values):
        ratio = v / row.classical_optimal if row.classical_optimal else 0.0
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.1f}\n({ratio:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.legend(loc="lower right")
    plt.tight_layout()
    _finish_figure(fig, path, dpi=dpi, show=show)
    return path


def save_qaoa_convergence_plot(
    convergence: list[float],
    path: Path | None,
    *,
    dpi: int = 120,
    show: bool = False,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(convergence, color="#1565c0", linewidth=1)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("-⟨H⟩ (minimizer objective)")
    ax.set_title("QAOA convergence (statevector expectation per step)")
    plt.tight_layout()
    _finish_figure(fig, path, dpi=dpi, show=show)
    return path


def save_qaoa_top_feasible_horizontal(
    problem: BundlingProblem,
    counts: dict[str, int],
    path: Path | None,
    *,
    top_k: int = 12,
    dpi: int = 120,
    show: bool = False,
) -> Path | None:
    """Empirical probability of feasible outcomes; Y labels are readable bundles."""
    c, A, b = get_ilp_matrices(problem)
    model, _ = build_ilp(problem)
    sense_map = {-1: "<=", 0: "==", 1: ">="}
    senses = [sense_map[int(c_.sense)] for c_ in model.constraints.values()]
    rows = _ilp_feasibility_rows(c, A, b, senses, counts)
    if not rows:
        return None
    rows.sort(key=lambda r: r["objective"], reverse=True)
    top = rows[:top_k]
    # Ascending objective so best sits at top of chart
    top = list(reversed(top))
    y_labels = []
    for i, r in enumerate(top):
        label = format_bundle_from_bitstring(problem, r["bitstring"])
        # Disambiguate duplicate bundle text
        dup = sum(
            1
            for r2 in top
            if format_bundle_from_bitstring(problem, r2["bitstring"]) == label
        )
        if dup > 1:
            label = f"{label}  (p={r['p']:.3f})"
        y_labels.append(label)

    probs = [r["p"] for r in top]
    objectives = [r["objective"] for r in top]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    bars = ax.barh(y_labels, probs, color="#1565c0")
    ax.set_xlabel("Empirical probability (feasible shots)")
    ax.set_title("QAOA: top feasible outcomes (readable bundles)")
    # Objective value at end of each bar
    xmax = max(probs) if probs else 1.0
    for bar, obj in zip(bars, objectives):
        ax.text(
            bar.get_width() + 0.002 * xmax,
            bar.get_y() + bar.get_height() / 2,
            f"{obj:.1f}",
            va="center",
            fontsize=9,
            color="#333",
        )
    ax.set_xlim(0, xmax * 1.35 if xmax > 0 else 1.0)
    plt.tight_layout()
    _finish_figure(fig, path, dpi=dpi, show=show)
    return path


def save_top_recommended_bundles_chart(
    problem: BundlingProblem,
    counts: dict[str, int],
    classical_optimal: float,
    classical_packages: list[list[str]] | None,
    path: Path | None,
    *,
    top_k: int = 10,
    dpi: int = 120,
    show: bool = False,
    chart_title: str | None = None,
) -> Path | None:
    """Horizontal bars: objective on X, human-readable bundle on Y; best QAOA bundle highlighted."""
    c, A, b = get_ilp_matrices(problem)
    model, _ = build_ilp(problem)
    sense_map = {-1: "<=", 0: "==", 1: ">="}
    senses = [sense_map[int(c_.sense)] for c_ in model.constraints.values()]
    rows = _ilp_feasibility_rows(c, A, b, senses, counts)
    if not rows:
        return None

    # Deduplicate by bitstring; keep highest probability entry
    best_by_bs: dict[str, dict[str, Any]] = {}
    for r in rows:
        bs = r["bitstring"]
        if bs not in best_by_bs or r["objective"] > best_by_bs[bs]["objective"]:
            best_by_bs[bs] = r
    unique = list(best_by_bs.values())
    unique.sort(key=lambda r: r["objective"])
    top = unique[-top_k:]

    y_labels = [format_bundle_from_bitstring(problem, r["bitstring"]) for r in top]
    xs = [r["objective"] for r in top]
    best_qaoa_obj = max(xs) if xs else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(top))))
    colors = []
    for obj in xs:
        colors.append("#ffb300" if abs(obj - best_qaoa_obj) < 1e-9 else "#3949ab")

    ax.barh(y_labels, xs, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Objective (contribution margin)")
    ax.set_title(chart_title or "Top recommended bundles")
    ax.axvline(
        classical_optimal,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="Classical optimum",
    )
    span = max(xs) - min(xs) if len(set(xs)) > 1 else max(max(xs), classical_optimal, 1.0) * 0.05
    margin = 0.02 * span if span > 0 else 1.0
    for patch, obj in zip(ax.patches, xs):
        ax.text(
            obj + margin,
            patch.get_y() + patch.get_height() / 2,
            f"{obj:.1f}",
            va="center",
            fontsize=9,
        )
    ax.legend(loc="lower right")
    plt.tight_layout()
    if classical_packages:
        fig.text(
            0.02,
            0.02,
            "Classical: " + format_packages_assignment(problem, classical_packages)[:200],
            fontsize=8,
            color="#555",
        )
    _finish_figure(fig, path, dpi=dpi, show=show)
    return path
