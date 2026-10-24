import csv
import io
import json

import numpy as np

from app.core.config import TARGETS
from app.core.report import BASE_COLUMNS, TARGET_COLUMNS, build_csv, csv_safe, summarize
from app.ml import train as train_module


def fake_prediction(probability_by_target, valid=True):
    if not valid:
        return {"valid": False, "smiles": "bad", "error": "invalid"}
    bands = lambda p: "Low" if p < 0.2 else "Medium" if p < 0.5 else "High"
    return {
        "valid": True, "canonical_smiles": "C", "molecular_formula": "CH4",
        "properties": {"molecular_weight_g_mol": 16.0, "log_p": 0.6, "hydrogen_bond_donors": 0, "hydrogen_bond_acceptors": 0,
                       "tpsa_angstrom2": 0.0, "lipinski_pass": True, "lipinski_violations": []},
        "domain": {"max_tanimoto": 0.8, "status": "In domain"},
        "targets": {t: {"probability": probability_by_target.get(t, 0.05), "risk_level": bands(probability_by_target.get(t, 0.05))} for t in TARGETS},
    }


def test_csv_safe_blocks_formula_injection():
    assert csv_safe("=1+1") == "'=1+1"
    assert csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert csv_safe("CCO") == "CCO"


def test_build_csv_and_summary():
    smiles = ["CCO", "bad", "=evil"]
    predictions = [fake_prediction({"NR-AhR": 0.9}), fake_prediction({}, valid=False), fake_prediction({"SR-p53": 0.3})]
    rows = list(csv.DictReader(io.StringIO(build_csv(smiles, predictions))))
    assert list(rows[0].keys()) == BASE_COLUMNS + TARGET_COLUMNS
    assert rows[0]["risk_alert"] == "ALERT" and rows[0]["high_risk_endpoints"] == "NR-AhR"
    assert rows[1]["valid"] == "False" and rows[1]["error"] == "invalid"
    assert rows[2]["smiles"] == "'=evil" and rows[2]["risk_alert"] == ""
    summary = summarize(smiles, predictions)
    assert (summary["total"], summary["valid"], summary["invalid"], summary["compounds_with_alert"]) == (3, 2, 1, 1)
    assert summary["endpoints"]["NR-AhR"]["High"] == 1 and summary["endpoints"]["SR-p53"]["Medium"] == 1
    assert summary["top_alerts"][0]["top_endpoint"] == "NR-AhR"


def test_train_target_exports_portable_artifacts(tmp_path):
    rng = np.random.RandomState(0)
    features = rng.rand(900, 40).astype(np.float32)
    labels = (features[:, 0] + 0.3 * rng.randn(900) > 0.8).astype(np.float32)
    labels[rng.rand(900) < 0.1] = np.nan  # unmeasured values must be ignored
    split = (np.arange(0, 700), np.arange(700, 800), np.arange(800, 900))
    meta = train_module.train_target("NR-AhR", features, labels, split, output_dir=tmp_path)
    assert (tmp_path / "nr_ahr.ubj").is_file()
    stored = json.loads((tmp_path / "nr_ahr.json").read_text())
    assert stored["calibration"].keys() == {"slope", "intercept"}
    assert 0 < stored["base_rate"] < 1
    assert meta["metrics"]["roc_auc"] > 0.7
    assert meta["metrics"]["test_samples"] <= 100
