"""HTTP routes: health, endpoint catalogue, prediction, explanation, batch screening, benchmark."""

import csv
import io
import json
from typing import Literal

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from app.core.cheminfo import molecular_properties
from app.core.config import MODEL_DIR, TARGETS
from app.core.report import build_csv, summarize
from app.ml.predictor import Predictor


router = APIRouter()
MAX_BATCH_FILE_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5000
INVALID_SMILES = "The supplied text is not a valid SMILES structure."


class SmilesRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=1000)


class MapRequest(SmilesRequest):
    target: str = Field(default="NR-AhR", max_length=40)


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _model_unavailable(error: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


def _invalid(smiles: str, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"valid": False, "smiles": smiles, "error": str(error)})


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    return _predictor(request).health()


@router.get("/targets")
def targets(request: Request) -> dict[str, object]:
    """The 12 Tox21 endpoints with group, description, base rate and held-out scaffold-split metrics."""
    return {"targets": _predictor(request).targets_info()}


@router.get("/biology")
def biology(request: Request) -> dict[str, object]:
    """What each assay measures (protein/gene/function) and the pathway groups used for roll-up scores."""
    return _predictor(request).biology()


@router.get("/benchmark", response_model=None)
def benchmark() -> JSONResponse | dict[str, object]:
    """Model comparison produced by `python -m app.ml.evaluate` (random vs scaffold split)."""
    path = MODEL_DIR / "benchmark.json"
    if not path.is_file():
        return JSONResponse(status_code=404, content={"detail": "Benchmark not generated yet. Run: python -m app.ml.evaluate"})
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/calibration", response_model=None)
def calibration(request: Request) -> dict[str, object]:
    """Per-endpoint calibration quality (Brier and ECE, raw vs Platt-calibrated) on the held-out scaffold test set."""
    rows = []
    for target, meta in _predictor(request).metadata.items():
        m = meta["metrics"]
        rows.append({
            "target": target, "brier_raw": m["brier_raw"], "brier_calibrated": m["brier_calibrated"],
            "ece_raw": m["ece_raw"], "ece_calibrated": m["ece_calibrated"], "reliability": m["reliability_calibrated"],
        })
    return {"endpoints": rows}


@router.post("/predict", response_model=None)
def predict(payload: SmilesRequest, request: Request) -> JSONResponse | dict[str, object]:
    try:
        return _predictor(request).predict(payload.smiles.strip())
    except ValueError as error:
        return _invalid(payload.smiles, error)
    except RuntimeError as error:
        return _model_unavailable(error)


@router.post("/explain", response_model=None)
def explain(payload: SmilesRequest, request: Request) -> JSONResponse | dict[str, object]:
    try:
        return _predictor(request).explain(payload.smiles.strip())
    except ValueError as error:
        return _invalid(payload.smiles, error)
    except RuntimeError as error:
        return _model_unavailable(error)


@router.post("/explain/map")
def explain_map(payload: MapRequest, request: Request) -> Response:
    """SVG with atom-level SHAP contours for one endpoint (red raises the score, blue lowers it)."""
    try:
        svg = _predictor(request).explain_map(payload.smiles.strip(), payload.target)
    except ValueError as error:
        return _invalid(payload.smiles, error)
    except RuntimeError as error:
        return _model_unavailable(error)
    return Response(content=svg, media_type="image/svg+xml")


@router.post("/predict/batch", response_model=None)
async def predict_batch(request: Request, file: UploadFile = File(...), format: Literal["csv", "json"] = Query("csv")) -> Response:
    """Screen a CSV of structures. `format=csv` downloads the results; `format=json` adds a risk-alert summary."""
    content = await file.read(MAX_BATCH_FILE_BYTES + 1)
    await file.close()
    if len(content) > MAX_BATCH_FILE_BYTES:
        return JSONResponse(status_code=413, content={"detail": "CSV upload exceeds the 5 MB limit."})

    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        smiles_column = next((n for n in (reader.fieldnames or []) if n and n.strip().lower() == "smiles"), None)
        if smiles_column is None:
            return JSONResponse(status_code=400, content={"detail": "CSV must contain a 'smiles' column."})
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        return JSONResponse(status_code=400, content={"detail": f"Could not parse CSV: {error}"})

    if not rows:
        return JSONResponse(status_code=400, content={"detail": "CSV contains no data rows."})
    if len(rows) > MAX_BATCH_ROWS:
        return JSONResponse(status_code=413, content={"detail": f"CSV exceeds the {MAX_BATCH_ROWS}-row limit."})

    smiles_values = [(row.get(smiles_column) or "").strip() for row in rows]
    try:
        predictions = _predictor(request).predict_batch(smiles_values)
    except RuntimeError as error:
        return _model_unavailable(error)

    report = summarize(smiles_values, predictions)
    results_csv = build_csv(smiles_values, predictions)
    if format == "json":
        return JSONResponse(content={"summary": report, "csv": results_csv})
    return StreamingResponse(
        iter([results_csv]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=tox21_batch_results.csv",
            "X-Input-Rows": str(report["total"]), "X-Valid-Rows": str(report["valid"]),
            "X-Invalid-Rows": str(report["invalid"]), "X-Alert-Rows": str(report["compounds_with_alert"]),
        },
    )


@router.post("/properties", response_model=None)
def properties(payload: SmilesRequest) -> JSONResponse | dict[str, object]:
    """Molecular descriptors and Lipinski Rule of Five assessment (no trained models needed)."""
    molecule = Chem.MolFromSmiles(payload.smiles.strip())
    if molecule is None:
        return JSONResponse(status_code=422, content={"valid": False, "error": INVALID_SMILES})
    return {
        "valid": True, "smiles": payload.smiles, "canonical_smiles": Chem.MolToSmiles(molecule),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule), "properties": molecular_properties(molecule),
    }


@router.post("/structure")
def structure(payload: SmilesRequest) -> Response:
    """Render a validated SMILES structure as an RDKit SVG depiction."""
    molecule = Chem.MolFromSmiles(payload.smiles.strip())
    if molecule is None:
        return JSONResponse(status_code=422, content={"valid": False, "error": INVALID_SMILES})
    drawer = rdMolDraw2D.MolDraw2DSVG(480, 300)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return Response(content=drawer.GetDrawingText(), media_type="image/svg+xml")
