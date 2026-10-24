import pytest
from fastapi.testclient import TestClient

from app.core.config import MODEL_DIR, TARGETS
from app.main import app

pytestmark = pytest.mark.skipif(
    not all((MODEL_DIR / f"{slug}.ubj").is_file() for slug in TARGETS.values()),
    reason="Trained models not found. Run: python -m app.ml.train",
)
TCDD = "Clc1cc2Oc3cc(Cl)c(Cl)cc3Oc2cc1Cl"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_targets(client):
    assert client.get("/api/health").json()["models_loaded"] is True
    catalogue = client.get("/api/targets").json()["targets"]
    assert [t["name"] for t in catalogue] == list(TARGETS)
    assert {t["group"] for t in catalogue} == {"Nuclear receptor", "Stress response"}


def test_predict_includes_pathway_layer_and_biology_endpoint(client):
    body = client.post("/api/predict", json={"smiles": TCDD}).json()
    assert len(body["pathways"]) == 7
    ahr = next(p for p in body["pathways"] if p["slug"] == "xenobiotic")
    assert ahr["score"] == body["targets"]["NR-AhR"]["probability"] and ahr["driving_assay"] == "NR-AhR"
    assert ahr["held_out_roc_auc"] is not None
    biology = client.get("/api/biology").json()
    assert len(biology["endpoints"]) == 12 and len(biology["pathways"]) == 7


def test_predict_returns_all_endpoints_with_domain(client):
    body = client.post("/api/predict", json={"smiles": TCDD}).json()
    assert set(body["targets"]) == set(TARGETS)
    for prediction in body["targets"].values():
        assert 0 <= prediction["probability"] <= 1 and prediction["risk_level"] in {"Low", "Medium", "High"}
    assert body["domain"]["status"] in {"In domain", "Borderline", "Out of domain"}
    assert body["targets"]["NR-AhR"]["probability"] > body["targets"]["NR-AR"]["probability"]


@pytest.mark.parametrize("path", ["/api/predict", "/api/explain", "/api/explain/map", "/api/structure", "/api/properties"])
def test_invalid_smiles_gets_422(client, path):
    assert client.post(path, json={"smiles": "not a smiles"}).status_code == 422


def test_empty_and_oversized_smiles_rejected(client):
    assert client.post("/api/predict", json={"smiles": ""}).status_code == 422
    assert client.post("/api/predict", json={"smiles": "C" * 1001}).status_code == 422


def test_explain_separates_atom_mappable_and_global_features(client):
    body = client.post("/api/explain", json={"smiles": TCDD}).json()["targets"]["NR-AhR"]
    kinds = {f["kind"] for f in body["positive_features"] + body["negative_features"]}
    assert kinds <= {"fingerprint_bit", "maccs_key", "descriptor"}
    assert 0 <= body["fingerprint_share"] <= 1


def test_atom_map_is_small_svg_and_validates_target(client):
    response = client.post("/api/explain/map", json={"smiles": TCDD, "target": "NR-AhR"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("image/svg+xml")
    assert len(response.text) < 100_000
    assert client.post("/api/explain/map", json={"smiles": TCDD, "target": "nope"}).status_code == 422


def test_batch_json_summary_and_validation(client):
    csv_text = f"SMILES,name\nCCO,e\n{TCDD},t\nnotvalid,x\n"
    body = client.post("/api/predict/batch?format=json", files={"file": ("a.csv", csv_text, "text/csv")}).json()
    assert (body["summary"]["total"], body["summary"]["valid"], body["summary"]["invalid"]) == (3, 2, 1)
    assert "nr_ahr_probability" in body["csv"].splitlines()[0] and "top_pathway" in body["csv"].splitlines()[0]
    assert len(body["summary"]["pathways"]) == 7


def test_batch_csv_download_headers(client):
    response = client.post("/api/predict/batch", files={"file": ("a.csv", "smiles\nCCO\n", "text/csv")})
    assert response.status_code == 200 and response.headers["x-valid-rows"] == "1"
    assert "attachment" in response.headers["content-disposition"]


def test_batch_rejects_bad_files(client):
    assert client.post("/api/predict/batch", files={"file": ("a.csv", "a,b\n1,2\n", "text/csv")}).status_code == 400
    assert client.post("/api/predict/batch", files={"file": ("a.csv", "smiles\n", "text/csv")}).status_code == 400
    too_many = "smiles\n" + "CCO\n" * 5001
    assert client.post("/api/predict/batch", files={"file": ("a.csv", too_many, "text/csv")}).status_code == 413
    assert client.post("/api/predict/batch", files={"file": ("a.csv", b"\xff\xfe\x00", "text/csv")}).status_code == 400


def test_benchmark_and_calibration_endpoints(client):
    assert client.get("/api/calibration").json()["endpoints"]
    assert client.get("/api/benchmark").status_code in {200, 404}
