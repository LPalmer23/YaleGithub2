"""Interactive bundle-judge demo: FastAPI + static UI.

Run from ``code_examples/``:

    pip install fastapi uvicorn
    uvicorn web.app:app --reload --app-dir .

Or:

    cd code_examples && uvicorn web.app:app --reload --app-dir .
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]

# code_examples/ on path (parent of web/)
_CODE_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_CODE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_CODE_EXAMPLES))

from src.insurance_model import load_ltm_instance, subsample_problem  # noqa: E402
from src.quantum_benchmark import (  # noqa: E402
    analyze_problem_for_judge_demo,
    resolve_ltm_data_dir,
    run_benchmark_size_sweep,
)

WEB_ROOT = Path(__file__).resolve().parent
STATIC_DIR = WEB_ROOT / "static"
PLOTS_DIR = STATIC_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_MAX_VARS = 21

app = FastAPI(title="Bundle recommendation judge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _cached_ltm():
    return load_ltm_instance(resolve_ltm_data_dir(_CODE_EXAMPLES))


class OptimizeBody(BaseModel):
    n_coverages: int = Field(5, ge=3, le=20)
    n_packages: int = Field(2, ge=1, le=10)
    package_start: int = Field(0, ge=0)
    seed: int | None = 42


class BenchmarkSweepBody(BaseModel):
    """Vary coverage count with fixed packages; keeps total variables within the demo cap."""

    package_start: int = Field(0, ge=0)
    n_packages: int = Field(2, ge=1, le=10)
    n_coverages_min: int = Field(3, ge=3, le=20)
    n_coverages_max: int = Field(7, ge=3, le=20)
    seed: int | None = 42


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def api_config() -> dict:
    ltm = _cached_ltm()
    names = ltm.package_names or [f"Package {i}" for i in range(ltm.M)]
    segments = []
    for i, name in enumerate(names):
        segments.append(
            {
                "package_start": i,
                "label": f"{name} (first segment in window)",
            }
        )
    return {
        "segments": segments,
        "package_names": names,
        "max_coverages": ltm.N,
        "max_packages": ltm.M,
        "defaults": {"n_coverages": 5, "n_packages": 2, "package_start": 0},
        "demo_max_product_vars": DEMO_MAX_VARS,
        "hint": "Choose which customer segment starts the package window; size sets how many consecutive bundles you optimize together.",
    }


@app.post("/api/optimize")
def api_optimize(body: OptimizeBody) -> dict:
    product = body.n_coverages * body.n_packages
    if product > DEMO_MAX_VARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Coverages × packages = {product} exceeds demo limit {DEMO_MAX_VARS}. "
                "Lower one of the sliders so the quantum simulation stays responsive."
            ),
        )
    ltm = _cached_ltm()
    if body.package_start >= ltm.M:
        raise HTTPException(status_code=400, detail="Segment index out of range.")
    try:
        problem = subsample_problem(
            ltm,
            body.n_coverages,
            body.n_packages,
            package_start=body.package_start,
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return analyze_problem_for_judge_demo(
            problem,
            persist_plots=True,
            code_examples_root=_CODE_EXAMPLES,
            seed=body.seed,
            qaoa_p=2,
            qaoa_maxiter=100,
            qaoa_shots=4096,
            dqi_shots=4096,
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"Optimization failed: {e!s}"
        ) from e


@app.post("/api/benchmark_sweep")
def api_benchmark_sweep(body: BenchmarkSweepBody) -> dict:
    """Benchmark classical, QAOA, and DQI for each coverage count in ``[min, max]`` (inclusive)."""
    ltm = _cached_ltm()
    if body.package_start >= ltm.M:
        raise HTTPException(status_code=400, detail="Segment index out of range.")

    lo, hi = sorted((body.n_coverages_min, body.n_coverages_max))
    lo = max(3, lo)
    hi = min(ltm.N, hi)
    n_pkg = body.n_packages
    if n_pkg > ltm.M:
        raise HTTPException(status_code=400, detail="n_packages exceeds model package count.")

    size_points: list[tuple[int, int]] = [(n, n_pkg) for n in range(lo, hi + 1)]
    size_points = [
        (n, p) for n, p in size_points if n * p <= DEMO_MAX_VARS
    ]
    if not size_points:
        raise HTTPException(
            status_code=400,
            detail=(
                "No sweep points fit under the demo variable cap. "
                "Lower packages or narrow the coverage range."
            ),
        )

    try:
        return run_benchmark_size_sweep(
            ltm,
            size_points=size_points,
            package_start=body.package_start,
            demo_max_vars=DEMO_MAX_VARS,
            seed=body.seed,
            qaoa_p=2,
            qaoa_maxiter=60,
            qaoa_shots=2048,
            dqi_shots=2048,
            dqi_max_weight=2,
            dqi_bp1_iterations=1,
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"Benchmark sweep failed: {e!s}"
        ) from e


_AGENT_SYSTEM = (
    "You are a concise assistant explaining a quantum optimization demo comparing classical, "
    "QAOA, and DQI methods. Be clear, intuitive, and insightful."
)


@app.post("/api/agent")
def agent_chat(body: dict) -> dict[str, str]:
    message = str(body.get("message", "") or "").strip()
    if not message:
        return {"reply": "Please enter a question first."}

    if OpenAI is None:
        return {
            "reply": (
                "The assistant needs the `openai` package (`pip install openai`). "
                "The optimization demo works without it."
            ),
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "reply": (
                "AI assistant is not configured. Set the OPENAI_API_KEY environment variable to "
                "enable answers. Everything else on this page runs normally."
            ),
        }

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _AGENT_SYSTEM},
                {"role": "user", "content": message},
            ],
        )
        choice = completion.choices[0]
        reply_text = (choice.message.content or "").strip()
        if not reply_text:
            reply_text = "(Empty response from model.)"
        return {"reply": reply_text}
    except Exception as e:  # pragma: no cover
        return {
            "reply": (
                "Sorry — the assistant could not complete that request right now. "
                f"Details: {e!s}"
            ),
        }


@app.get("/")
def serve_ui():
    return FileResponse(WEB_ROOT / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
