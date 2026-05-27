from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas import FigureList

router = APIRouter(prefix="/api/report", tags=["report"])

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "04_benchmark" / "resultados" / "figuras"
_REPORT_GENERATOR = _PROJECT_ROOT / "04_benchmark" / "generate_figures_sober.py"

_ALLOWED_EXTENSIONS = {".png", ".csv"}


@router.get("/figures", response_model=FigureList)
def list_figures() -> FigureList:
    if not _FIGURES_DIR.exists():
        return FigureList(figures=[])
    figures = [
        f.name for f in sorted(_FIGURES_DIR.iterdir())
        if f.suffix in _ALLOWED_EXTENSIONS
    ]
    return FigureList(figures=figures)


@router.get("/figures/{filename}")
def get_figure(filename: str):
    # Sanitize: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido.")
    path = _FIGURES_DIR / filename
    if not path.exists() or path.suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=404, detail=f"Figura '{filename}' no encontrada.")
    media_type = "image/png" if path.suffix == ".png" else "text/csv"
    return FileResponse(
        str(path),
        media_type=media_type,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@router.post("/generate")
def generate_report() -> dict:
    csv_path = _PROJECT_ROOT / "04_benchmark" / "resultados" / "results_metrics.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No hay resultados CSV. Ejecuta el benchmark primero.")

    result = subprocess.run(
        [sys.executable, str(_REPORT_GENERATOR)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT / "04_benchmark"),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Error generando figuras: {result.stderr[:500]}")

    figures = []
    if _FIGURES_DIR.exists():
        figures = [f.name for f in sorted(_FIGURES_DIR.iterdir()) if f.suffix == ".png"]
    return {"status": "ok", "figures": figures}


@router.get("/export/csv")
def export_csv():
    csv_path = _PROJECT_ROOT / "04_benchmark" / "resultados" / "results_metrics.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay resultados CSV. Ejecuta el benchmark primero.",
        )
    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename="results_metrics.csv",
    )


@router.get("/export/summary")
def export_summary():
    summary_path = _PROJECT_ROOT / "04_benchmark" / "resultados" / "results_summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay resumen de resultados.",
        )
    return FileResponse(
        str(summary_path),
        media_type="application/json",
        filename="results_summary.json",
    )
