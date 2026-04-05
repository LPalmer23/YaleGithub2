"""Qiskit (Aer) benchmark harness: QAOA and DQI vs classical baseline.

Use this instead of vendor-specific emulators when collecting comparison data.

Entry points:
  - ``run_benchmark_on_problem`` — low-level row + details dict.
  - ``analyze_problem_from_notebook`` — one-call analysis + optional artifacts under ``code_examples/data/``.

FastAPI: serialize results with ``notebook_analysis_to_jsonable`` (plain dict, Path → str).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark_viz import (
    chart_top_bundles_data,
    save_objectives_bar_chart,
    save_qaoa_convergence_plot,
    save_qaoa_top_feasible_horizontal,
    save_top_recommended_bundles_chart,
)
from src.bundle_labels import (
    format_bundle_from_bitstring,
    format_bundle_from_vector,
    format_packages_assignment,
    pick_overall_best_method,
)
from src.dqi_circuit import DQIResult, run_dqi
from src.insurance_model import BundlingProblem, build_ilp, get_ilp_matrices, solve_ilp
from src.ilp_to_maxxorsat import MaxXORSATInstance, ilp_to_maxxorsat
from src.qaoa_circuit import run_qaoa


@dataclass
class QuantumBenchmarkRow:
    """One row of benchmark results for a single instance."""

    n_vars: int
    classical_optimal: float
    classical_time_s: float
    qaoa_best: float
    qaoa_time_s: float
    qaoa_approx_ratio: float
    qaoa_feasibility_rate: float
    dqi_best: float
    dqi_time_s: float
    dqi_approx_ratio: float
    dqi_post_selection_rate: float
    xorsat_qubits: int


@dataclass
class NotebookAnalysisResult:
    """Structured result for notebooks, CLIs, or HTTP responses (after JSON conversion)."""

    classical_optimal: float
    qaoa_best_objective: float
    qaoa_feasibility_rate: float
    dqi_best_objective: float
    dqi_post_selection_rate: float
    benchmark_json_path: str | None = None
    plot_paths: dict[str, str] = field(default_factory=dict)
    n_vars: int = 0
    xorsat_qubits: int = 0
    classical_bundle_label: str = ""
    qaoa_best_bundle_label: str = ""
    dqi_best_bundle_label: str = ""
    overall_best_objective: float = 0.0
    overall_best_methods: list[str] = field(default_factory=list)
    overall_best_bundle_summary: str = ""
    summary_confidence: str = ""


def resolve_code_examples_root(start: Path | None = None) -> Path:
    """Find the ``code_examples`` directory (the folder that contains ``src/``).

    Walks upward from ``start`` or the current working directory.
    """
    cur = (start or Path.cwd()).resolve()
    marker = cur / "src" / "insurance_model.py"
    if marker.is_file():
        return cur
    for _ in range(12):
        cur = cur.parent
        marker = cur / "src" / "insurance_model.py"
        if marker.is_file():
            return cur
        if cur.parent == cur:
            break
    raise FileNotFoundError(
        "Could not locate code_examples root (expected parent path containing src/insurance_model.py). "
        f"Started from {(start or Path.cwd()).resolve()}"
    )


def resolve_ltm_data_dir(code_examples_root: Path | None = None) -> Path:
    """Locate ``YQH26_data`` under repo ``LTM/`` or ``docs/data/`` (sibling of ``code_examples``)."""
    root = (code_examples_root or resolve_code_examples_root()).resolve()
    repo_root = root.parent
    candidates = [
        repo_root / "LTM" / "YQH26_data",
        repo_root / "docs" / "data" / "YQH26_data",
    ]
    for p in candidates:
        if (p / "instance_coverages.csv").is_file():
            return p.resolve()
    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"YQH26_data not found. Tried: {tried}")


def default_data_dir(code_examples_root: Path | None = None) -> Path:
    """``code_examples/data`` — JSON and PNG outputs."""
    root = code_examples_root or resolve_code_examples_root()
    d = (root / "data").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _senses_pulp_to_str(model: Any) -> tuple[list[int], list[str]]:
    sense_map_chr = {-1: "<=", 0: "==", 1: ">="}
    senses_int: list[int] = []
    senses_str: list[str] = []
    for c_ in model.constraints.values():
        senses_int.append(int(c_.sense))
        senses_str.append(sense_map_chr[int(c_.sense)])
    return senses_int, senses_str


def build_maxxorsat_for_problem(problem: BundlingProblem) -> MaxXORSATInstance:
    c, A, b = get_ilp_matrices(problem)
    model, _ = build_ilp(problem)
    senses_int, _ = _senses_pulp_to_str(model)
    return ilp_to_maxxorsat(c, A, b, senses_int)


def run_benchmark_on_problem(
    problem: BundlingProblem,
    *,
    qaoa_p: int = 2,
    qaoa_maxiter: int = 150,
    qaoa_shots: int = 8192,
    dqi_shots: int = 8192,
    dqi_max_weight: int = 2,
    dqi_bp1_iterations: int = 1,
    seed: int | None = 42,
) -> tuple[QuantumBenchmarkRow, dict[str, Any]]:
    """Run classical solve, QAOA, and DQI on the same BundlingProblem.

    Returns:
        QuantumBenchmarkRow for tabular use, and a detail dict (counts, history, etc.).
    """
    n_vars = problem.N * problem.M
    c, A, b = get_ilp_matrices(problem)
    model, _ = build_ilp(problem)
    senses_int, senses_str = _senses_pulp_to_str(model)

    t0 = time.perf_counter()
    classical = solve_ilp(problem)
    classical_time = time.perf_counter() - t0
    opt = float(classical["objective"])

    instance = ilp_to_maxxorsat(c, A, b, senses_int)
    q_total = instance.num_equations + instance.num_variables

    t0 = time.perf_counter()
    qaoa_out = run_qaoa(
        problem,
        p=qaoa_p,
        max_iterations=qaoa_maxiter,
        shots=qaoa_shots,
        seed=seed,
    )
    qaoa_time = time.perf_counter() - t0
    qaoa_best = float(qaoa_out["best_objective"])
    qaoa_feas = float(qaoa_out["final_feasibility_rate"])

    t0 = time.perf_counter()
    dqi_result: DQIResult = run_dqi(
        instance,
        c,
        A,
        b,
        senses_str,
        max_weight=dqi_max_weight,
        bp1_iterations=dqi_bp1_iterations,
        shots=dqi_shots,
    )
    dqi_time = time.perf_counter() - t0
    dqi_best = float(dqi_result.best_objective)

    row = QuantumBenchmarkRow(
        n_vars=n_vars,
        classical_optimal=opt,
        classical_time_s=classical_time,
        qaoa_best=qaoa_best,
        qaoa_time_s=qaoa_time,
        qaoa_approx_ratio=qaoa_best / opt if opt > 0 else 0.0,
        qaoa_feasibility_rate=qaoa_feas,
        dqi_best=dqi_best,
        dqi_time_s=dqi_time,
        dqi_approx_ratio=dqi_best / opt if opt > 0 else 0.0,
        dqi_post_selection_rate=float(dqi_result.post_selection_rate),
        xorsat_qubits=q_total,
    )

    details: dict[str, Any] = {
        "qaoa_convergence": qaoa_out["convergence_history"],
        "qaoa_final_counts": qaoa_out["final_counts"],
        "dqi_counts": dqi_result.counts,
        "maxxorsat_shape": tuple(instance.B.shape),
        "classical_packages": classical["packages"],
        "qaoa_best_bitstring": qaoa_out.get("best_solution"),
        "dqi_best_solution": None
        if dqi_result.best_solution is None
        else np.asarray(dqi_result.best_solution).tolist(),
    }

    return row, details


def save_benchmark_json(
    row: QuantumBenchmarkRow,
    path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Serialize benchmark row (and optional extra metadata) to JSON."""
    payload = asdict(row)
    if extra:
        payload["_extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_benchmark_artifacts(
    problem: BundlingProblem,
    row: QuantumBenchmarkRow,
    details: dict[str, Any],
    data_dir: Path,
    *,
    benchmark_basename: str = "qiskit_benchmark.json",
    seed: int | None = None,
    ltm_dir: str | None = None,
    save_json: bool = True,
    save_plot_files: bool = True,
    show_plots: bool = False,
    dpi: int = 120,
) -> tuple[Path | None, dict[str, Path | None]]:
    """Write JSON and/or PNG plots. With ``show_plots=True``, each figure is also shown (e.g. Jupyter)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path: Path | None = None
    if save_json:
        json_path = data_dir / benchmark_basename
        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        if ltm_dir is not None:
            extra["ltm_dir"] = ltm_dir
        save_benchmark_json(row, json_path, extra=extra or None)

    def _p(name: str) -> Path | None:
        return (data_dir / name) if save_plot_files else None

    plot_paths: dict[str, Path | None] = {}
    plot_paths["objectives"] = save_objectives_bar_chart(
        row, _p("qiskit_benchmark_objectives.png"), dpi=dpi, show=show_plots
    )
    plot_paths["qaoa_convergence"] = save_qaoa_convergence_plot(
        details["qaoa_convergence"], _p("qiskit_qaoa_convergence.png"), dpi=dpi, show=show_plots
    )
    top_path = save_qaoa_top_feasible_horizontal(
        problem,
        details["qaoa_final_counts"],
        _p("qiskit_qaoa_top_feasible.png"),
        dpi=dpi,
        show=show_plots,
    )
    if top_path is not None:
        plot_paths["qaoa_top_feasible"] = top_path

    classical_pkg = details.get("classical_packages")
    bundles_path = save_top_recommended_bundles_chart(
        problem,
        details["qaoa_final_counts"],
        row.classical_optimal,
        classical_pkg,
        _p("qiskit_top_recommended_bundles.png"),
        dpi=dpi,
        show=show_plots,
    )
    if bundles_path is not None:
        plot_paths["top_recommended_bundles"] = bundles_path

    return json_path, plot_paths


def _summarize_bundles(
    problem: BundlingProblem,
    row: QuantumBenchmarkRow,
    details: dict[str, Any],
) -> tuple[str, str, str, float, list[str], str, str]:
    classical_lbl = format_packages_assignment(
        problem, details["classical_packages"]
    )
    qbs = details.get("qaoa_best_bitstring")
    qaoa_lbl = format_bundle_from_bitstring(problem, qbs) if qbs else ""
    dqi_vec = details.get("dqi_best_solution")
    if dqi_vec is not None:
        dqi_lbl = format_bundle_from_vector(
            problem, np.array(dqi_vec, dtype=float)
        )
    else:
        dqi_lbl = ""
    best_obj, methods = pick_overall_best_method(
        row.classical_optimal, row.qaoa_best, row.dqi_best
    )
    if len(methods) == 1:
        m0 = methods[0]
        if m0 == "classical":
            bundle_sum = classical_lbl
        elif m0 == "QAOA":
            bundle_sum = qaoa_lbl or "(no QAOA bitstring decoded)"
        else:
            bundle_sum = dqi_lbl or "(no DQI solution vector)"
    else:
        bundle_sum = (
            "Objective tie across methods — compare classical, QAOA, and DQI bundle lines below."
        )
    confidence = (
        f"Classical: exact ILP optimum (PuLP/CBC). "
        f"QAOA: {row.qaoa_feasibility_rate:.1%} of final shots were ILP-feasible. "
        f"DQI: {row.dqi_post_selection_rate:.1%} post-selection (message register all-zero)."
    )
    return classical_lbl, qaoa_lbl, dqi_lbl, best_obj, methods, bundle_sum, confidence


def notebook_result_from_run(
    problem: BundlingProblem,
    row: QuantumBenchmarkRow,
    details: dict[str, Any],
    *,
    benchmark_json_path: str | None = None,
    plot_paths: dict[str, str] | None = None,
) -> NotebookAnalysisResult:
    """Build ``NotebookAnalysisResult`` from an existing benchmark run (no re-simulation)."""
    (
        classical_lbl,
        qaoa_lbl,
        dqi_lbl,
        best_obj,
        best_methods,
        bundle_sum,
        confidence,
    ) = _summarize_bundles(problem, row, details)
    return NotebookAnalysisResult(
        classical_optimal=row.classical_optimal,
        qaoa_best_objective=row.qaoa_best,
        qaoa_feasibility_rate=row.qaoa_feasibility_rate,
        dqi_best_objective=row.dqi_best,
        dqi_post_selection_rate=row.dqi_post_selection_rate,
        benchmark_json_path=benchmark_json_path,
        plot_paths=dict(plot_paths or {}),
        n_vars=row.n_vars,
        xorsat_qubits=row.xorsat_qubits,
        classical_bundle_label=classical_lbl,
        qaoa_best_bundle_label=qaoa_lbl,
        dqi_best_bundle_label=dqi_lbl,
        overall_best_objective=best_obj,
        overall_best_methods=best_methods,
        overall_best_bundle_summary=bundle_sum,
        summary_confidence=confidence,
    )


def plain_english_interpretation(result: NotebookAnalysisResult) -> str:
    """Short demo copy comparing classical, QAOA, and DQI on this run."""
    c = result.classical_optimal
    q = result.qaoa_best_objective
    d = result.dqi_best_objective
    qf = result.qaoa_feasibility_rate
    df = result.dqi_post_selection_rate
    scale = max(abs(c), 1.0)
    tol = max(1e-6, 0.005 * scale)
    methods = result.overall_best_methods

    sentences: list[str] = []
    if abs(c - q) < tol and abs(c - d) < tol:
        sentences.append(
            "Every method agrees on the same portfolio value for this scenario—your recommended "
            "bundles line up on contribution margin."
        )
    else:
        sentences.append(
            "The classical optimizer still certifies the strongest margin; treat it as the "
            "scorecard while quantum methods catch up with more depth or shots."
        )

    if qf < 0.02 and df < 0.02:
        sentences.append(
            "Sampling was thin—feasible hits were rare for both quantum routes, so treat the "
            "confidence badges as directional, not final."
        )
    elif df >= qf + 0.08:
        sentences.append(
            f"DQI’s post-selection rate (~{df:.0%}) outpaced QAOA’s feasible-shot rate "
            f"(~{qf:.0%}), so DQI acted like the steadier recommender on this draw."
        )
    elif qf >= df + 0.08:
        sentences.append(
            f"QAOA landed on feasible bundles more often (~{qf:.0%}) than DQI’s decoder "
            f"pattern (~{df:.0%}), so the variational sampler looked surer-footed here."
        )
    else:
        sentences.append(
            "Quantum sampling reliability was in the same ballpark for QAOA and DQI."
        )

    if len(methods) == 3:
        sentences.append(
            "All three solvers tied on the headline objective—great news for cross-checking."
        )
    elif methods == ["classical"]:
        sentences.append(
            "Classical alone sits on the overall best margin this run; quantum outputs are "
            "useful for exploration and storytelling."
        )
    return " ".join(sentences)


def run_judge_demo(
    problem: BundlingProblem,
    *,
    persist_plots: bool = True,
    code_examples_root: Path | None = None,
    **bench_kw: Any,
) -> dict[str, Any]:
    """Full JSON for the interactive judge UI: metrics, interpretation, charts, optional PNG URLs."""
    row, details = run_benchmark_on_problem(problem, **bench_kw)
    result = notebook_result_from_run(problem, row, details)
    root = code_examples_root or resolve_code_examples_root()
    plot_urls: dict[str, str] = {}
    if persist_plots:
        web_plots = root / "web" / "static" / "plots"
        web_plots.mkdir(parents=True, exist_ok=True)
        save_objectives_bar_chart(
            row,
            web_plots / "demo_objectives.png",
            show=False,
            chart_title="Contribution margin by approach",
        )
        save_top_recommended_bundles_chart(
            problem,
            details["qaoa_final_counts"],
            row.classical_optimal,
            details["classical_packages"],
            web_plots / "demo_top_bundles.png",
            top_k=10,
            show=False,
            chart_title="Top recommended bundles (QAOA feasible samples)",
        )
        plot_urls["objectives"] = "/static/plots/demo_objectives.png"
        plot_urls["top_bundles"] = "/static/plots/demo_top_bundles.png"

    chart_top = chart_top_bundles_data(
        problem, details["qaoa_final_counts"], top_k=8
    )

    return {
        "overall_best_bundle": result.overall_best_bundle_summary,
        "overall_best_objective": result.overall_best_objective,
        "overall_best_methods": result.overall_best_methods,
        "classical_bundle": result.classical_bundle_label,
        "classical_objective": result.classical_optimal,
        "classical_confidence": "Exact optimum from the ILP (PuLP/CBC).",
        "qaoa_bundle": result.qaoa_best_bundle_label or None,
        "qaoa_objective": result.qaoa_best_objective,
        "qaoa_feasibility_rate": result.qaoa_feasibility_rate,
        "qaoa_confidence": (
            f"{result.qaoa_feasibility_rate:.1%} of post-optimization shots were ILP-feasible."
        ),
        "dqi_bundle": result.dqi_best_bundle_label or None,
        "dqi_objective": result.dqi_best_objective,
        "dqi_post_selection_rate": result.dqi_post_selection_rate,
        "dqi_confidence": (
            f"{result.dqi_post_selection_rate:.1%} of shots showed the decoder success pattern."
        ),
        "interpretation": plain_english_interpretation(result),
        "chart_objectives": {
            "labels": ["Classical optimum", "QAOA best sampled", "DQI best feasible"],
            "values": [row.classical_optimal, row.qaoa_best, row.dqi_best],
        },
        "chart_top_bundles": chart_top,
        "plot_urls": plot_urls,
        "summary_confidence": result.summary_confidence,
        "n_vars": row.n_vars,
        "xorsat_qubits": row.xorsat_qubits,
    }


# Alias for HTTP layers that mirror ``analyze_problem_for_api`` naming.
analyze_problem_for_judge_demo = run_judge_demo


def print_benchmark_summary(result: NotebookAnalysisResult) -> None:
    """Human-readable summary for notebooks or logs."""
    print("=== Benchmark summary ===")
    print(f"Overall best objective: {result.overall_best_objective:.4f}")
    print(f"Best method(s): {', '.join(result.overall_best_methods)}")
    print(f"Note: {result.overall_best_bundle_summary}")
    print()
    print(f"Classical optimum: {result.classical_optimal:.4f}")
    print(f"  {result.classical_bundle_label}")
    print(f"QAOA best sampled: {result.qaoa_best_objective:.4f}")
    print(f"  {result.qaoa_best_bundle_label}")
    print(f"DQI best: {result.dqi_best_objective:.4f}")
    print(
        f"  {result.dqi_best_bundle_label or '(no decoded bundle — check DQI circuit / post-selection)'}"
    )
    print()
    print(result.summary_confidence)
    print()


def analyze_problem_for_api(
    problem: BundlingProblem,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return JSON-serializable benchmark results for a FastAPI route (or any HTTP handler).

    Passes keyword arguments through to ``analyze_problem_from_notebook``. For stateless
    APIs, use ``save_outputs=False`` unless the server should persist artifacts to disk.
    """
    result = analyze_problem_from_notebook(problem, **kwargs)
    return notebook_analysis_to_jsonable(result)


def notebook_analysis_to_jsonable(result: NotebookAnalysisResult) -> dict[str, Any]:
    """Convert analysis result to a JSON-serializable dict (e.g. FastAPI ``JSONResponse``)."""
    d: dict[str, Any] = {
        "classical_optimal": result.classical_optimal,
        "qaoa_best_objective": result.qaoa_best_objective,
        "qaoa_feasibility_rate": result.qaoa_feasibility_rate,
        "dqi_best_objective": result.dqi_best_objective,
        "dqi_post_selection_rate": result.dqi_post_selection_rate,
        "benchmark_json_path": result.benchmark_json_path,
        "plot_paths": dict(result.plot_paths),
        "n_vars": result.n_vars,
        "xorsat_qubits": result.xorsat_qubits,
        "classical_bundle_label": result.classical_bundle_label,
        "qaoa_best_bundle_label": result.qaoa_best_bundle_label,
        "dqi_best_bundle_label": result.dqi_best_bundle_label,
        "overall_best_objective": result.overall_best_objective,
        "overall_best_methods": list(result.overall_best_methods),
        "overall_best_bundle_summary": result.overall_best_bundle_summary,
        "summary_confidence": result.summary_confidence,
    }
    return d


def analyze_problem_from_notebook(
    problem: BundlingProblem,
    *,
    save_outputs: bool = True,
    show_plots: bool = False,
    code_examples_root: Path | None = None,
    data_dir: Path | None = None,
    benchmark_basename: str | None = None,
    seed: int | None = 42,
    ltm_dir: str | None = None,
    qaoa_p: int = 2,
    qaoa_maxiter: int = 150,
    qaoa_shots: int = 8192,
    dqi_shots: int = 8192,
    dqi_max_weight: int = 2,
    dqi_bp1_iterations: int = 1,
) -> NotebookAnalysisResult:
    """Run classical / QAOA / DQI and optionally save JSON + plots under ``code_examples/data/``.

    ``code_examples_root`` defaults to discovery from ``Path.cwd()`` so the notebook can run
    from either ``code_examples/`` or ``code_examples/notebooks/``.

    For a future FastAPI route, reuse this function and return
    ``notebook_analysis_to_jsonable(result)``.
    """
    root = code_examples_root or resolve_code_examples_root()
    out_dir = data_dir or default_data_dir(root)
    n_vars = problem.N * problem.M
    base_name = benchmark_basename or f"qiskit_benchmark_n{n_vars}.json"

    row, details = run_benchmark_on_problem(
        problem,
        qaoa_p=qaoa_p,
        qaoa_maxiter=qaoa_maxiter,
        qaoa_shots=qaoa_shots,
        dqi_shots=dqi_shots,
        dqi_max_weight=dqi_max_weight,
        dqi_bp1_iterations=dqi_bp1_iterations,
        seed=seed,
    )

    json_path_str: str | None = None
    plot_strs: dict[str, str] = {}

    if save_outputs or show_plots:
        json_path, plot_paths = save_benchmark_artifacts(
            problem,
            row,
            details,
            out_dir,
            benchmark_basename=base_name,
            seed=seed,
            ltm_dir=ltm_dir,
            save_json=save_outputs,
            save_plot_files=save_outputs,
            show_plots=show_plots,
        )
        if json_path is not None:
            json_path_str = str(json_path.resolve())
        plot_strs = {
            k: str(v.resolve())
            for k, v in plot_paths.items()
            if v is not None
        }

    return notebook_result_from_run(
        problem,
        row,
        details,
        benchmark_json_path=json_path_str,
        plot_paths=plot_strs,
    )
