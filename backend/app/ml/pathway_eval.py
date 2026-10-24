"""Validate the pathway layer on the held-out scaffold test fold (same split as train.py, seed 0).

Pathway label (per molecule), defined only for molecules measured in EVERY member assay:
  1  if any member assay is active, 0 otherwise.
Other molecules are excluded. (Counting a molecule as positive on a partial measurement while requiring full
measurement for negatives would inflate the base rate of multi-assay pathways.)
Pathway score = max calibrated probability over member assays, exactly as served by the API.
Writes pathway_metrics.json with each pathway's base rate (training fold), test ROC-AUC and PR-AUC.

Run:  python -m app.ml.pathway_eval
"""

import json
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from app.core.biology import PATHWAYS, pathway_members
from app.core.cheminfo import scaffold_split
from app.core.config import MODEL_DIR
from app.ml.data import Dataset, load_dataset
from app.ml.predictor import Predictor

LOGGER = logging.getLogger(__name__)


def pathway_labels(labels: np.ndarray, columns: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (label, usable-mask) for one pathway across all molecules."""
    block = labels[:, columns]
    measured = np.isfinite(block)
    any_active = np.nanmax(np.where(measured, block, -1), axis=1) == 1
    return any_active.astype(int), measured.all(axis=1)


def evaluate_pathways(dataset: Dataset | None = None, predictor: Predictor | None = None) -> dict[str, dict]:
    dataset = dataset or load_dataset()
    predictor = predictor or Predictor()
    predictor._require_models()
    train, _, test = scaffold_split(dataset.scaffolds, seed=0)
    probabilities = predictor._probabilities(dataset.combined()[test])
    report: dict[str, dict] = {}
    for slug, info in PATHWAYS.items():
        members = pathway_members(slug)
        columns = [dataset.targets.index(t) for t in members]
        label, usable = pathway_labels(dataset.labels, columns)
        score = np.max([probabilities[t]["calibrated"] for t in members], axis=0)
        keep = usable[test]
        y_test, s_test = label[test][keep], score[keep]
        entry = {
            "name": info["name"], "assays": members,
            "base_rate": float(label[train][usable[train]].mean()),
            "test_molecules": int(keep.sum()), "test_positives": int(y_test.sum()),
            "roc_auc": float(roc_auc_score(y_test, s_test)), "pr_auc": float(average_precision_score(y_test, s_test)),
        }
        report[slug] = entry
        LOGGER.info("%-38s base rate %.3f | test n=%d (pos %d) | ROC-AUC %.3f PR-AUC %.3f", info["name"], entry["base_rate"],
                    entry["test_molecules"], entry["test_positives"], entry["roc_auc"], entry["pr_auc"])
    (MODEL_DIR / "pathway_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate_pathways()
