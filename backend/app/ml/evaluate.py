"""Benchmark several models under random and scaffold splits, across all 12 Tox21 endpoints.

Protocol
  * Molecule-level 80/10/10 train/valid/test splits, shared by all endpoints and models.
  * "random": random permutation. "scaffold": Bemis-Murcko groups never cross partitions (balanced,
    seeded). "scaffold_strict": the deterministic DeepChem-style split (largest groups -> train, so the
    rarest scaffolds form the test fold); it has no randomness, so it is run once.
  * Each (split, seed) is repeated; summaries are mean +/- sd over seeds.
  * Per endpoint, only molecules with a measured label are used (no imputation).
  * XGBoost uses the validation fold for early stopping. For every model, the F1/balanced-accuracy
    threshold is chosen on the validation fold only; the test fold is never used for any choice.
  * Results are appended to a JSONL cache so an interrupted run resumes where it stopped.

Run:  python -m app.ml.evaluate --seeds 3
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from app.core.cheminfo import max_tanimoto, random_split, scaffold_split
from app.core.config import DATA_DIR, MODEL_DIR, TARGETS
from app.ml.data import Dataset, load_dataset


LOGGER = logging.getLogger(__name__)
RUNS_PATH = DATA_DIR / "benchmark_runs.jsonl"
BENCHMARK_PATH = MODEL_DIR / "benchmark.json"
AD_BINS = ((0.0, 0.3, "<0.30"), (0.3, 0.5, "0.30-0.50"), (0.5, 1.01, ">=0.50"))

LR_NAME = "Logistic regression (ECFP)"
RF_NAME = "Random forest (ECFP)"
XGB_NAME = "XGBoost (ECFP)"
XGB_COMBINED_NAME = "XGBoost (ECFP+MACCS+descriptors)"
SPLIT_NAMES = ("random", "scaffold", "scaffold_strict")
MODEL_ORDER = [LR_NAME, RF_NAME, XGB_NAME, XGB_COMBINED_NAME]
METRICS = ("roc_auc", "pr_auc", "f1", "balanced_accuracy")


def make_xgb(seed: int, colsample: float, has_validation: bool) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500 if has_validation else 200, learning_rate=0.05, max_depth=5, subsample=0.85,
        colsample_bytree=colsample, reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=seed,
        eval_metric="logloss", early_stopping_rounds=30 if has_validation else None,
    )


def fit_predict(name: str, data: Dataset, features: dict[str, np.ndarray], split: tuple, y: np.ndarray, seed: int):
    """Fit one model on the train fold; return (valid_probs, test_probs) for the labeled rows."""
    train, valid, test = split
    feature_key = "combined" if name == XGB_COMBINED_NAME else "ecfp"
    matrix = features[feature_key]
    x_train, x_valid, x_test = matrix[train], matrix[valid], matrix[test]
    y_train, y_valid = y[train], y[valid]
    has_validation = int(y_valid.sum()) >= 3 and int((1 - y_valid).sum()) >= 3
    if name == LR_NAME:
        model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=200)
        model.fit(x_train, y_train)
    elif name == RF_NAME:
        model = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced_subsample", n_jobs=1, random_state=seed,
        )
        model.fit(x_train, y_train)
    else:
        model = make_xgb(seed, 0.5 if feature_key == "combined" else 0.8, has_validation)
        if has_validation:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        else:
            model.fit(x_train, y_train, verbose=False)
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def best_f1_threshold(y_valid: np.ndarray, p_valid: np.ndarray) -> float:
    """Threshold maximizing F1 on the validation fold (0.5 if validation has no positives)."""
    if int(y_valid.sum()) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_valid, p_valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5


def score(y_valid, p_valid, y_test, p_test) -> dict[str, float | None]:
    if len(np.unique(y_test)) < 2:
        return {metric: None for metric in METRICS}
    threshold = best_f1_threshold(y_valid, p_valid)
    predicted = (p_test >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
    }


def ad_bins(y_test: np.ndarray, p_test: np.ndarray, similarity: np.ndarray) -> list[dict]:
    """ROC-AUC of one fit within bins of max-Tanimoto similarity to the training partition."""
    rows = []
    for low, high, label in AD_BINS:
        selected = (similarity >= low) & (similarity < high)
        labels = y_test[selected]
        usable = int(labels.sum()) >= 3 and int((1 - labels).sum()) >= 3
        rows.append({
            "bin": label, "n": int(selected.sum()), "positives": int(labels.sum()),
            "roc_auc": float(roc_auc_score(labels, p_test[selected])) if usable else None,
        })
    return rows


def load_done() -> set[tuple]:
    done = set()
    if RUNS_PATH.exists():
        for line in RUNS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["split"], record["seed"], record["model"], record["task"]))
    return done


def run(seeds: int, tasks: list[str], models: list[str]) -> None:
    data = load_dataset()
    features = {"ecfp": data.ecfp, "combined": data.combined()}
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    started = time.time()
    with RUNS_PATH.open("a", encoding="utf-8") as sink:
        for split_name in SPLIT_NAMES:
            for seed in range(1 if split_name == "scaffold_strict" else seeds):
                if split_name == "random":
                    split = random_split(len(data.smiles), seed=seed)
                elif split_name == "scaffold":
                    split = scaffold_split(data.scaffolds, seed=seed)
                else:
                    split = scaffold_split(data.scaffolds, seed=None)
                train, valid, test = split
                similarity = None
                for task in tasks:
                    column = data.targets.index(task)
                    labels = data.labels[:, column]
                    measured = np.isfinite(labels)
                    # Restrict each fold to molecules with a measured label for this endpoint.
                    fold = tuple(idx[measured[idx]] for idx in split)
                    y = np.where(measured, labels, 0).astype(int)
                    for model_name in models:
                        key = (split_name, seed, model_name, task)
                        if key in done:
                            continue
                        p_valid, p_test = fit_predict(model_name, data, features, fold, y, seed)
                        record = {
                            "split": split_name, "seed": seed, "model": model_name, "task": task,
                            "n_train": int(len(fold[0])), "n_test": int(len(fold[2])),
                            "test_positives": int(y[fold[2]].sum()),
                            **score(y[fold[1]], p_valid, y[fold[2]], p_test),
                        }
                        if split_name == "scaffold" and model_name == XGB_NAME:
                            if similarity is None:
                                similarity = max_tanimoto(data.ecfp[test], data.ecfp[train])
                                similarity_index = {int(i): s for i, s in zip(test, similarity)}
                            sims = np.array([similarity_index[int(i)] for i in fold[2]])
                            record["ad_bins"] = ad_bins(y[fold[2]], p_test, sims)
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
                        done.add(key)
                    LOGGER.info("%s seed %d %s done (%.0fs elapsed)", split_name, seed, task, time.time() - started)


def aggregate(seeds: int, tasks: list[str], models: list[str]) -> dict:
    """Collapse the JSONL run log into the dashboard/README benchmark report."""
    records = [json.loads(line) for line in RUNS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["seed"] < seeds and r["task"] in tasks and r["model"] in models]
    data = load_dataset()

    summary, per_task = [], {task: {} for task in tasks}
    for model in models:
        for split_name in SPLIT_NAMES:
            subset = [r for r in records if r["model"] == model and r["split"] == split_name]
            if not subset:
                continue
            row = {"model": model, "split": split_name, "n_runs": len(subset)}
            for metric in METRICS:
                # Macro-average over endpoints within each seed, then mean/sd across seeds.
                seed_means = []
                for seed in range(seeds):
                    values = [r[metric] for r in subset if r["seed"] == seed and r[metric] is not None]
                    if values:
                        seed_means.append(float(np.mean(values)))
                row[f"{metric}_mean"] = float(np.mean(seed_means))
                row[f"{metric}_sd"] = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            summary.append(row)
            for task in tasks:
                task_rows = [r for r in subset if r["task"] == task]
                entry = {}
                for metric in METRICS:
                    values = [r[metric] for r in task_rows if r[metric] is not None]
                    entry[metric] = float(np.mean(values)) if values else None
                    entry[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                per_task[task].setdefault(model, {})[split_name] = entry

    ad_rows = []
    for _, _, label in AD_BINS:
        bins = [b for r in records if "ad_bins" in r for b in r["ad_bins"] if b["bin"] == label]
        aucs = [b["roc_auc"] for b in bins if b["roc_auc"] is not None]
        ad_rows.append({
            "bin": label, "mean_test_molecules_per_endpoint": float(np.mean([b["n"] for b in bins])) if bins else 0.0,
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "endpoint_runs_with_enough_data": len(aucs),
        })

    base_rates = {
        task: float(np.nanmean(data.labels[:, data.targets.index(task)])) for task in tasks
    }
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seeds": seeds, "endpoints": len(tasks), "molecules": len(data.smiles),
            "models": models,
            "protocol": [
                "80/10/10 molecule-level split shared by all endpoints; only measured labels are used per endpoint.",
                "Random split: random permutation. Scaffold split: Bemis-Murcko groups never cross partitions (balanced, seeded). Strict scaffold: deterministic DeepChem-style split where the rarest scaffolds form the test fold (single run).",
                "Summary values are macro-averages over endpoints per seed, then mean and sd (n-1) over seeds.",
                "F1 and balanced accuracy use a threshold chosen on the validation fold only; XGBoost early-stops on the validation fold.",
                "Acyclic molecules share one empty scaffold, so every scaffold test fold contains only ring-containing molecules.",
            ],
        },
        "summary": summary,
        "per_task": per_task,
        "base_rates": base_rates,
        "ad_analysis": ad_rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tasks", default=",".join(TARGETS), help="Comma-separated endpoint subset.")
    parser.add_argument("--models", default="all", help="Comma-separated subset of model names, or 'all'.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip training; rebuild benchmark.json from the run log.")
    arguments = parser.parse_args()
    tasks = [t.strip() for t in arguments.tasks.split(",") if t.strip()]
    models = MODEL_ORDER if arguments.models == "all" else [m.strip() for m in arguments.models.split(",")]
    if not arguments.aggregate_only:
        run(arguments.seeds, tasks, models)
    report = aggregate(arguments.seeds, tasks, models)
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", BENCHMARK_PATH)


if __name__ == "__main__":
    main()
