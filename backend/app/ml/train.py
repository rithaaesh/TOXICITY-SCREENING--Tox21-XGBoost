"""Train one calibrated XGBoost model per Tox21 endpoint and export portable artifacts.

Features: 2048 ECFP bits + 167 MACCS keys + RDKit 2D descriptors. The benchmark (evaluate.py) showed this feature
set beats ECFP alone by ~0.06 macro ROC-AUC under scaffold splits, far outside seed-to-seed noise.

Evaluation of the deployed models uses the balanced scaffold split (80/10/10, seed 0), the same
protocol as `evaluate.py`'s "scaffold" rows, so training metrics and benchmark seed 0 are comparable:
  train  -> fit XGBoost (early stopping uses the validation fold)
  valid  -> early stopping, Platt calibration, and the F1 threshold
  test   -> every reported metric (never used for any choice)
No class re-weighting is applied, so raw scores are already on a probability-like scale; Platt scaling
then corrects residual miscalibration under scaffold shift. Artifacts per endpoint:
  <slug>.ubj   native XGBoost model (portable across xgboost versions)
  <slug>.json  calibration coefficients, base rate, and test metrics
plus reference_fingerprints.npz (training molecules, for the applicability domain).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from xgboost import XGBClassifier

from app.core.cheminfo import scaffold_split
from app.core.config import MODEL_DIR, TARGETS
from app.ml.data import load_dataset
from app.ml.evaluate import best_f1_threshold


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def reliability(labels: np.ndarray, probabilities: np.ndarray, bins: int = 5) -> tuple[list[dict], float]:
    """Equal-frequency reliability table and expected calibration error (ECE)."""
    order = np.argsort(probabilities)
    rows, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        predicted, observed = float(probabilities[chunk].mean()), float(labels[chunk].mean())
        rows.append({"mean_predicted": predicted, "observed_rate": observed, "n": int(len(chunk))})
        ece += len(chunk) / len(order) * abs(predicted - observed)
    return rows, float(ece)


def train_target(
    target: str, features: np.ndarray, labels: np.ndarray, split: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path = MODEL_DIR,
) -> dict:
    """Train, calibrate, evaluate, and export one endpoint."""
    measured = np.isfinite(labels)
    train, valid, test = (idx[measured[idx]] for idx in split)
    y = np.where(measured, labels, 0).astype(int)
    y_train, y_valid, y_test = y[train], y[valid], y[test]
    if min(int(y_train.sum()), int((1 - y_train).sum())) < 5 or int(y_valid.sum()) < 3:
        raise ValueError(f"Not enough positive and negative observations to train {target}.")

    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.5, reg_lambda=1.0,
        tree_method="hist", n_jobs=-1, random_state=42, eval_metric="logloss", early_stopping_rounds=30,
    )
    model.fit(features[train], y_train, eval_set=[(features[valid], y_valid)], verbose=False)

    raw_valid = model.predict_proba(features[valid])[:, 1]
    raw_test = model.predict_proba(features[test])[:, 1]
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(logit(raw_valid).reshape(-1, 1), y_valid)
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    calibrated_valid = sigmoid(slope * logit(raw_valid) + intercept)
    calibrated_test = sigmoid(slope * logit(raw_test) + intercept)

    threshold = best_f1_threshold(y_valid, calibrated_valid)
    predicted = (calibrated_test >= threshold).astype(int)
    table, ece_calibrated = reliability(y_test, calibrated_test)
    _, ece_raw = reliability(y_test, raw_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, calibrated_test)),
        "pr_auc": float(average_precision_score(y_test, calibrated_test)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "f1_threshold": float(threshold),
        "brier_raw": float(brier_score_loss(y_test, raw_test)),
        "brier_calibrated": float(brier_score_loss(y_test, calibrated_test)),
        "ece_raw": ece_raw,
        "ece_calibrated": ece_calibrated,
        "reliability_calibrated": table,
        "confusion_matrix_at_threshold": [
            [int(((predicted == 0) & (y_test == 0)).sum()), int(((predicted == 1) & (y_test == 0)).sum())],
            [int(((predicted == 0) & (y_test == 1)).sum()), int(((predicted == 1) & (y_test == 1)).sum())],
        ],
        "train_samples": int(len(train)), "valid_samples": int(len(valid)), "test_samples": int(len(test)),
        "positive_train_samples": int(y_train.sum()), "positive_test_samples": int(y_test.sum()),
        "best_iteration": int(model.best_iteration),
    }
    metadata = {
        "target": target, "slug": TARGETS[target], "base_rate": float(y_train.mean()),
        "features": "ecfp2048+maccs167+rdkit_descriptors", "calibration": {"slope": slope, "intercept": intercept},
        "metrics": metrics, "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / f"{TARGETS[target]}.ubj")
    (output_dir / f"{TARGETS[target]}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info(
        "%-14s test ROC-AUC %.3f PR-AUC %.3f | Brier %.4f -> %.4f | ECE %.3f -> %.3f",
        target, metrics["roc_auc"], metrics["pr_auc"], metrics["brier_raw"], metrics["brier_calibrated"], ece_raw, ece_calibrated,
    )
    return metadata


def train(force_download: bool = False) -> dict[str, dict]:
    """Run the complete 12-endpoint training workflow."""
    dataset = load_dataset(force_download)
    split = scaffold_split(dataset.scaffolds, seed=0)
    LOGGER.info("Scaffold split: %d train / %d valid / %d test molecules", *(len(part) for part in split))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_DIR / "reference_fingerprints.npz", bits=np.packbits(dataset.ecfp[split[0]], axis=1), width=dataset.ecfp.shape[1])
    features = dataset.combined()
    reports = {}
    for column, target in enumerate(dataset.targets):
        reports[target] = train_target(target, features, dataset.labels[:, column], split)
    (MODEL_DIR / "training_metrics.json").write_text(
        json.dumps({t: r["metrics"] | {"base_rate": r["base_rate"]} for t, r in reports.items()}, indent=2), encoding="utf-8",
    )
    from app.ml.pathway_eval import evaluate_pathways  # imported late: needs the freshly written models

    evaluate_pathways(dataset)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-download", action="store_true", help="Download a fresh copy of the Tox21 CSV.")
    train(force_download=parser.parse_args().force_download)


if __name__ == "__main__":
    main()
