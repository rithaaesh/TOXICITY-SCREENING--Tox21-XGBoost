"""Model loading, calibrated inference, applicability domain, and SHAP explanations."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from xgboost import XGBClassifier

from app.core.biology import biology_catalogue, pathway_scores
from app.core.cheminfo import combined_features, domain_label, feature_names, fingerprint_smiles, max_tanimoto, molecular_properties
from app.core.config import FINGERPRINT_BITS, MODEL_DIR, RISK_BANDS, TARGET_INFO, TARGETS


LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


class Predictor:
    """Load trained artifacts and provide calibrated predictions and explanations."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.models: dict[str, XGBClassifier] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.explainers: dict[str, shap.TreeExplainer] = {}
        self.reference: np.ndarray | None = None
        self.pathway_metrics: dict[str, dict[str, Any]] = {}
        self.feature_names = feature_names()
        self.reload_models()

    def reload_models(self) -> None:
        """Load available endpoint artifacts without preventing API startup."""
        self.models.clear()
        self.metadata.clear()
        self.explainers.clear()
        self.reference = None
        self.pathway_metrics = {}
        metrics_path = self.model_dir / "pathway_metrics.json"
        if metrics_path.is_file():
            self.pathway_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for target, slug in TARGETS.items():
            model_path, meta_path = self.model_dir / f"{slug}.ubj", self.model_dir / f"{slug}.json"
            if not (model_path.is_file() and meta_path.is_file()):
                continue
            try:
                model = XGBClassifier()
                model.load_model(model_path)
                self.models[target] = model
                self.metadata[target] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.exception("Could not load model artifact %s", model_path)
        reference_path = self.model_dir / "reference_fingerprints.npz"
        if reference_path.is_file():
            stored = np.load(reference_path)
            width = int(stored["width"])
            self.reference = np.unpackbits(stored["bits"], axis=1)[:, :width].astype(np.float32)

    # ---- status ------------------------------------------------------------------------

    def health(self) -> dict[str, object]:
        loaded = [target for target in TARGETS if target in self.models]
        missing = [target for target in TARGETS if target not in self.models]
        return {
            "status": "ready" if not missing else "models_missing",
            "models_loaded": not missing,
            "loaded_targets": loaded,
            "missing_targets": missing,
            "applicability_domain_ready": self.reference is not None,
        }

    def targets_info(self) -> list[dict[str, object]]:
        """Endpoint catalogue with base rates and held-out scaffold-split metrics."""
        rows = []
        for target, slug in TARGETS.items():
            group, description = TARGET_INFO[target]
            meta = self.metadata.get(target, {})
            metrics = meta.get("metrics", {})
            rows.append({
                "name": target, "slug": slug, "group": group, "description": description,
                "available": target in self.models, "base_rate": meta.get("base_rate"),
                "scaffold_test_roc_auc": metrics.get("roc_auc"), "scaffold_test_pr_auc": metrics.get("pr_auc"),
            })
        return rows

    def biology(self) -> dict[str, object]:
        """Assay-to-protein map and pathway definitions, with held-out pathway metrics when available."""
        return biology_catalogue(self.pathway_metrics)

    @staticmethod
    def risk_level(probability: float) -> str:
        if probability < RISK_BANDS[0]:
            return "Low"
        if probability < RISK_BANDS[1]:
            return "Medium"
        return "High"

    def _require_models(self) -> None:
        missing = [target for target in TARGETS if target not in self.models]
        if missing:
            raise RuntimeError(f"Models are not trained for: {', '.join(missing)}")

    # ---- prediction --------------------------------------------------------------------

    def _probabilities(self, features: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        output = {}
        for target in TARGETS:
            raw = np.asarray(self.models[target].predict_proba(features)[:, 1], dtype=float)
            calibration = self.metadata[target]["calibration"]
            calibrated = 1.0 / (1.0 + np.exp(-(calibration["slope"] * _logit(raw) + calibration["intercept"])))
            output[target] = {"raw": raw, "calibrated": calibrated}
        return output

    def _similarity(self, features: np.ndarray) -> np.ndarray:
        if self.reference is None:
            return np.full(len(features), np.nan, dtype=np.float32)
        return max_tanimoto(features[:, :FINGERPRINT_BITS], self.reference)

    def predict(self, smiles: str) -> dict[str, object]:
        """Predict calibrated endpoint probabilities for one valid SMILES string."""
        vector, molecule, _ = fingerprint_smiles(smiles)
        self._require_models()
        features = combined_features(molecule, vector).reshape(1, -1)
        return self._result(smiles, molecule, self._probabilities(features), self._similarity(features), 0)

    def predict_batch(self, smiles_values: list[str]) -> list[dict[str, object]]:
        """Predict many structures with one batched model call per endpoint."""
        self._require_models()
        results: list[dict[str, object] | None] = [None] * len(smiles_values)
        valid_indices: list[int] = []
        vectors: list[np.ndarray] = []
        molecules: list[Any] = []
        for index, smiles in enumerate(smiles_values):
            try:
                vector, molecule, _ = fingerprint_smiles(smiles)
            except ValueError as error:
                results[index] = {"valid": False, "smiles": smiles, "error": str(error)}
                continue
            valid_indices.append(index)
            vectors.append(vector)
            molecules.append(molecule)
        if vectors:
            features = np.asarray([combined_features(m, v) for m, v in zip(molecules, vectors)], dtype=np.float32)
            probabilities = self._probabilities(features)
            similarity = self._similarity(features)
            for row, index in enumerate(valid_indices):
                results[index] = self._result(smiles_values[index], molecules[row], probabilities, similarity, row)
        return [result for result in results if result is not None]

    def _result(self, smiles: str, molecule: Any, probabilities: dict, similarity: np.ndarray, row: int) -> dict[str, object]:
        targets: dict[str, dict[str, object]] = {}
        for target in TARGETS:
            probability = float(probabilities[target]["calibrated"][row])
            base_rate = float(self.metadata[target]["base_rate"])
            targets[target] = {
                "probability": probability,
                "raw_score": float(probabilities[target]["raw"][row]),
                "risk_level": self.risk_level(probability),
                "base_rate": base_rate,
                "enrichment": probability / base_rate if base_rate > 0 else None,
            }
        similarity_value = float(similarity[row])
        domain = None if np.isnan(similarity_value) else {"max_tanimoto": similarity_value, "status": domain_label(similarity_value)}
        return {
            "valid": True,
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(molecule),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
            "properties": molecular_properties(molecule),
            "domain": domain,
            "targets": targets,
            "pathways": pathway_scores(targets, self.pathway_metrics),
            "interpretation": "Exploratory model scores, not a validated safety or clinical determination.",
        }

    # ---- explanations ------------------------------------------------------------------

    def _explainer(self, target: str) -> shap.TreeExplainer:
        if target not in self.explainers:
            self.explainers[target] = shap.TreeExplainer(self.models[target])
        return self.explainers[target]

    def _shap_vector(self, target: str, features: np.ndarray) -> np.ndarray:
        values = np.asarray(self._explainer(target).shap_values(features))
        if values.ndim == 3:  # some shap/xgboost combinations return (rows, features, classes)
            values = values[..., -1]
        return values[0].astype(float, copy=False)

    @staticmethod
    def _environment_atoms(molecule: Any, atom_index: int, radius: int) -> list[int]:
        bonds = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, atom_index) if radius > 0 else ()
        atoms = {atom_index}
        for bond_index in bonds:
            bond = molecule.GetBondWithIdx(bond_index)
            atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        return sorted(atoms)

    def _fragment_smiles(self, molecule: Any, atom_index: int, radius: int) -> str:
        bonds = list(Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, atom_index)) if radius > 0 else []
        atoms = self._environment_atoms(molecule, atom_index, radius)
        return Chem.MolFragmentToSmiles(molecule, atomsToUse=atoms, bondsToUse=bonds, canonical=True)

    def atom_weights(self, molecule: Any, bit_info: dict[int, list[tuple[int, int]]], values: np.ndarray) -> np.ndarray:
        """Distribute each present bit's SHAP value over the atoms of its circular environments.

        A bit with k occurrences gives value/k to each occurrence, shared equally by that occurrence's
        atoms. Bits that are absent from the fingerprint carry SHAP mass that cannot be placed on atoms,
        so the weights sum to the contribution of the present bits only.
        """
        weights = np.zeros(molecule.GetNumAtoms(), dtype=float)
        for bit, occurrences in bit_info.items():
            if values[bit] == 0:
                continue
            share = values[bit] / len(occurrences)
            for atom_index, radius in occurrences:
                atoms = self._environment_atoms(molecule, atom_index, radius)
                weights[atoms] += share / len(atoms)
        return weights

    def explain(self, smiles: str, limit: int = 8) -> dict[str, object]:
        """Largest signed SHAP contributions for every endpoint.

        Features are fingerprint bits (mapped to structure fragments), MACCS keys, or named RDKit
        descriptors. Only fingerprint bits can be localized to atoms (see `explain_map`).
        """
        vector, molecule, bit_info = fingerprint_smiles(smiles)
        self._require_models()
        features = combined_features(molecule, vector).reshape(1, -1)
        probabilities = self._probabilities(features)
        explanations: dict[str, dict[str, object]] = {}
        for target in TARGETS:
            values = self._shap_vector(target, features)
            positive = np.flatnonzero(values > 0)
            negative = np.flatnonzero(values < 0)
            positive = positive[np.argsort(values[positive])[::-1]][:limit]
            negative = negative[np.argsort(values[negative])][:limit]

            def describe(index: int) -> dict[str, object]:
                name = self.feature_names[index]
                kind = "fingerprint_bit" if index < FINGERPRINT_BITS else ("maccs_key" if name.startswith("MACCS") else "descriptor")
                fragments: list[str] = []
                if kind == "fingerprint_bit":
                    for atom_index, radius in bit_info.get(int(index), [])[:3]:
                        fragment = self._fragment_smiles(molecule, atom_index, radius)
                        if fragment and fragment not in fragments:
                            fragments.append(fragment)
                return {
                    "feature_index": int(index), "name": name, "kind": kind,
                    "value": float(features[0, index]), "shap_value": float(values[index]),
                    "direction": "increases" if values[index] > 0 else "decreases", "fragments": fragments,
                }

            total = float(np.abs(values).sum())
            probability = float(probabilities[target]["calibrated"][0])
            explanations[target] = {
                "probability": probability,
                "risk_level": self.risk_level(probability),
                "fingerprint_share": float(np.abs(values[:FINGERPRINT_BITS]).sum() / total) if total else 0.0,
                "positive_features": [describe(int(i)) for i in positive],
                "negative_features": [describe(int(i)) for i in negative],
            }
        return {
            "valid": True,
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(molecule),
            "targets": explanations,
            "explanation_note": "SHAP values (log-odds of the uncalibrated model) describe model contributions, not causal biological mechanisms.",
        }

    def explain_map(self, smiles: str, target: str, size: tuple[int, int] = (480, 360)) -> str:
        """SVG of the molecule with atoms shaded by SHAP weight (red raises the score, blue lowers it).

        RDKit's SimilarityMaps contour output was ~1.4 MB per molecule, so the same per-atom weights are
        drawn as colored atom highlights instead (a few KB). Only fingerprint-bit SHAP can be placed on atoms;
        descriptor and MACCS contributions are global and are not shown on the structure. Shade intensity scales with |weight| relative
        to the molecule's largest weight; atoms below 10% of it are left unshaded.
        """
        if target not in TARGETS:
            raise ValueError(f"Unknown endpoint: {target}")
        vector, molecule, bit_info = fingerprint_smiles(smiles)
        self._require_models()
        features = combined_features(molecule, vector).reshape(1, -1)
        weights = self.atom_weights(molecule, bit_info, self._shap_vector(target, features)[:FINGERPRINT_BITS])
        peak = float(np.abs(weights).max()) if len(weights) else 0.0
        colors: dict[int, tuple[float, float, float]] = {}
        radii: dict[int, float] = {}
        for atom_index, weight in enumerate(weights):
            if peak == 0 or abs(weight) < 0.10 * peak:
                continue
            strength = 0.25 + 0.75 * abs(weight) / peak
            base = (0.90, 0.35, 0.25) if weight > 0 else (0.22, 0.45, 0.75)
            colors[atom_index] = tuple(1.0 - strength * (1.0 - channel) for channel in base)
            radii[atom_index] = 0.30 + 0.20 * abs(weight) / peak
        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        drawer.DrawMolecule(molecule, highlightAtoms=list(colors), highlightAtomColors=colors, highlightAtomRadii=radii)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
