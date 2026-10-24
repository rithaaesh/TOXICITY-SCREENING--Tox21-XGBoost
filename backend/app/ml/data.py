"""Tox21 dataset loading and cached feature construction shared by training and evaluation."""

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.cheminfo import descriptor_vector, fingerprint_molecule, maccs_vector, murcko_scaffold
from app.core.config import DATA_DIR, TARGETS, TOX21_URL
from rdkit import Chem


LOGGER = logging.getLogger(__name__)
DATASET_PATH = DATA_DIR / "tox21.csv.gz"
CACHE_PATH = DATA_DIR / "features_cache.npz"


@dataclass
class Dataset:
    """Valid molecules with aligned labels (NaN = not measured) and feature matrices."""

    smiles: list[str]
    targets: list[str]
    labels: np.ndarray      # (n_molecules, n_targets) float32, NaN where unmeasured
    ecfp: np.ndarray        # (n, 2048) uint8
    maccs: np.ndarray       # (n, 167) uint8
    descriptors: np.ndarray  # (n, ~217) float32
    scaffolds: list[str]
    skipped: int

    def combined(self) -> np.ndarray:
        """ECFP + MACCS + RDKit descriptors as one float32 matrix."""
        return np.hstack([self.ecfp, self.maccs, self.descriptors]).astype(np.float32)


def download_dataset(force: bool = False) -> Path:
    """Download the public MoleculeNet Tox21 CSV when it is not cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not DATASET_PATH.exists():
        temporary_path = DATASET_PATH.with_suffix(DATASET_PATH.suffix + ".part")
        LOGGER.info("Downloading Tox21 data from %s", TOX21_URL)
        urllib.request.urlretrieve(TOX21_URL, temporary_path)
        temporary_path.replace(DATASET_PATH)
    return DATASET_PATH


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"smiles", *TARGETS}.difference(frame.columns)
    if missing:
        raise ValueError(f"Tox21 data is missing required columns: {sorted(missing)}")
    return frame


def load_dataset(force_download: bool = False, use_cache: bool = True) -> Dataset:
    """Build (or load from cache) features for every parseable molecule."""
    path = download_dataset(force_download)
    targets = list(TARGETS)
    if use_cache and CACHE_PATH.exists() and CACHE_PATH.stat().st_mtime >= path.stat().st_mtime:
        cached = np.load(CACHE_PATH, allow_pickle=False)
        return Dataset(
            smiles=cached["smiles"].tolist(), targets=targets, labels=cached["labels"], ecfp=cached["ecfp"],
            maccs=cached["maccs"], descriptors=cached["descriptors"], scaffolds=cached["scaffolds"].tolist(),
            skipped=int(cached["skipped"]),
        )

    frame = load_frame(path)
    rows: list[int] = []
    ecfp, maccs, descriptors = [], [], []
    for index, smiles in frame["smiles"].items():
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if molecule is None:
            continue
        try:
            fingerprint = fingerprint_molecule(molecule)[0]
            maccs_bits = maccs_vector(molecule)
            descriptor_values = descriptor_vector(molecule)
        except Exception:  # rare RDKit failures on exotic structures: skip the molecule
            continue
        rows.append(index)
        ecfp.append(fingerprint)
        maccs.append(maccs_bits)
        descriptors.append(descriptor_values)
    valid = frame.loc[rows]
    smiles_list = valid["smiles"].tolist()
    labels = np.column_stack([pd.to_numeric(valid[t], errors="coerce").to_numpy() for t in targets]).astype(np.float32)
    dataset = Dataset(
        smiles=smiles_list, targets=targets, labels=labels, ecfp=np.asarray(ecfp, dtype=np.uint8),
        maccs=np.asarray(maccs, dtype=np.uint8), descriptors=np.asarray(descriptors, dtype=np.float32),
        scaffolds=[murcko_scaffold(s) for s in smiles_list], skipped=len(frame) - len(rows),
    )
    if use_cache:
        np.savez_compressed(
            CACHE_PATH, smiles=np.array(dataset.smiles), labels=dataset.labels, ecfp=dataset.ecfp, maccs=dataset.maccs,
            descriptors=dataset.descriptors, scaffolds=np.array(dataset.scaffolds), skipped=dataset.skipped,
        )
    LOGGER.info("Prepared %d valid molecules; skipped %d", len(smiles_list), dataset.skipped)
    return dataset
