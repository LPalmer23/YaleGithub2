"""Interactive bundle-judge demo: FastAPI + static UI.

Run from ``code_examples/``:

    pip install fastapi uvicorn
    uvicorn web.app:app --reload --app-dir .

Or:

    cd code_examples && uvicorn web.app:app --reload --app-dir .
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# code_examples/ on path (parent of web/)
_CODE_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_CODE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_CODE_EXAMPLES))

from src.insurance_model import load_ltm_instance, subsample_problem  # noqa: E402
from src.quantum_benchmark import (  # noqa: E402
    analyze_problem_for_judge_demo,
    resolve_ltm_data_dir,
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


@app.get("/")
def serve_ui():
    return FileResponse(WEB_ROOT / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
