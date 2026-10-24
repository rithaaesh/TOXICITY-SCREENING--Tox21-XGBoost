import json

import numpy as np
import pytest

from app.core.biology import ENDPOINT_BIOLOGY, PATHWAYS, biology_catalogue, pathway_genes, pathway_members, pathway_scores, risk_band
from app.core.config import MODEL_DIR, RISK_BANDS, TARGETS
from app.ml.pathway_eval import pathway_labels


def fake_targets(overrides):
    return {t: {"probability": overrides.get(t, 0.05)} for t in TARGETS}


def test_every_assay_maps_to_exactly_one_pathway_and_back():
    assert set(ENDPOINT_BIOLOGY) == set(TARGETS)
    members = [t for slug in PATHWAYS for t in pathway_members(slug)]
    assert sorted(members) == sorted(TARGETS) and len(members) == len(set(members))
    assert [t for t in TARGETS if t in pathway_members("endocrine")] == pathway_members("endocrine")


def test_known_biology():
    assert ENDPOINT_BIOLOGY["NR-AhR"]["gene"] == "AHR"
    assert ENDPOINT_BIOLOGY["SR-p53"]["gene"] == "TP53"
    assert ENDPOINT_BIOLOGY["SR-MMP"]["gene"] is None
    assert pathway_genes("genotoxic") == ["ATAD5", "TP53"]
    assert set(pathway_members("endocrine")) == {"NR-AR", "NR-AR-LBD", "NR-ER", "NR-ER-LBD", "NR-Aromatase"}


def test_pathway_score_is_max_over_members_and_names_the_driver():
    rows = {r["slug"]: r for r in pathway_scores(fake_targets({"NR-ER": 0.3, "NR-AR": 0.6, "SR-p53": 0.9}))}
    assert rows["endocrine"]["score"] == pytest.approx(0.6) and rows["endocrine"]["driving_assay"] == "NR-AR"
    assert rows["endocrine"]["risk_level"] == "High" and rows["endocrine"]["n_high_assays"] == 1
    assert rows["genotoxic"]["driving_assay"] == "SR-p53" and rows["genotoxic"]["score"] == pytest.approx(0.9)
    assert rows["mitochondrial"]["risk_level"] == "Low"
    assert len(rows) == len(PATHWAYS)


def test_risk_band_boundaries_match_config():
    assert risk_band(RISK_BANDS[0] - 1e-9) == "Low" and risk_band(RISK_BANDS[0]) == "Medium" and risk_band(RISK_BANDS[1]) == "High"


def test_pathway_metrics_enrich_scores():
    rows = {r["slug"]: r for r in pathway_scores(fake_targets({"NR-AhR": 0.5}), {"xenobiotic": {"base_rate": 0.1, "roc_auc": 0.85}})}
    assert rows["xenobiotic"]["enrichment"] == pytest.approx(5.0) and rows["xenobiotic"]["held_out_roc_auc"] == 0.85
    assert rows["endocrine"]["enrichment"] is None


def test_pathway_labels_require_full_measurement():
    labels = np.array([[1, np.nan], [0, 0], [np.nan, 0], [0, 1], [np.nan, np.nan]], dtype=float)
    label, usable = pathway_labels(labels, [0, 1])
    assert usable.tolist() == [False, True, False, True, False]  # partial measurements are excluded, even positives
    assert label[usable].tolist() == [0, 1]


def test_catalogue_shape():
    catalogue = biology_catalogue()
    assert len(catalogue["endpoints"]) == 12 and len(catalogue["pathways"]) == 7
    assert [e["assay"] for e in catalogue["endpoints"]] == list(TARGETS)


@pytest.mark.skipif(not (MODEL_DIR / "pathway_metrics.json").is_file(), reason="run python -m app.ml.pathway_eval")
def test_stored_pathway_metrics_are_sane():
    metrics = json.loads((MODEL_DIR / "pathway_metrics.json").read_text())
    assert set(metrics) == set(PATHWAYS)
    for entry in metrics.values():
        assert 0.5 < entry["roc_auc"] <= 1.0 and 0 < entry["base_rate"] < 1 and entry["test_positives"] >= 10
