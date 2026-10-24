"""Application paths, endpoint definitions, and environment-backed settings."""

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("TOX21_DATA_DIR", PROJECT_DIR / "data"))
MODEL_DIR = Path(os.getenv("TOX21_MODEL_DIR", APP_DIR / "ml" / "models"))
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2

# Endpoint name -> artifact slug. Order follows the original Tox21 challenge.
TARGETS = {
    "NR-AR": "nr_ar",
    "NR-AR-LBD": "nr_ar_lbd",
    "NR-AhR": "nr_ahr",
    "NR-Aromatase": "nr_aromatase",
    "NR-ER": "nr_er",
    "NR-ER-LBD": "nr_er_lbd",
    "NR-PPAR-gamma": "nr_ppar_gamma",
    "SR-ARE": "sr_are",
    "SR-ATAD5": "sr_atad5",
    "SR-HSE": "sr_hse",
    "SR-MMP": "sr_mmp",
    "SR-p53": "sr_p53",
}

TARGET_INFO = {
    "NR-AR": ("Nuclear receptor", "Androgen receptor (full length)"),
    "NR-AR-LBD": ("Nuclear receptor", "Androgen receptor ligand-binding domain"),
    "NR-AhR": ("Nuclear receptor", "Aryl hydrocarbon receptor"),
    "NR-Aromatase": ("Nuclear receptor", "Aromatase inhibition"),
    "NR-ER": ("Nuclear receptor", "Estrogen receptor alpha (full length)"),
    "NR-ER-LBD": ("Nuclear receptor", "Estrogen receptor alpha ligand-binding domain"),
    "NR-PPAR-gamma": ("Nuclear receptor", "Peroxisome proliferator-activated receptor gamma"),
    "SR-ARE": ("Stress response", "Antioxidant response element (oxidative stress)"),
    "SR-ATAD5": ("Stress response", "ATAD5 genotoxicity / DNA damage"),
    "SR-HSE": ("Stress response", "Heat shock factor response"),
    "SR-MMP": ("Stress response", "Mitochondrial membrane potential"),
    "SR-p53": ("Stress response", "p53 DNA damage response"),
}

# Display bands on the *calibrated* probability. Not clinical or regulatory thresholds.
RISK_BANDS = (0.20, 0.50)

# Applicability domain: max Tanimoto similarity to any training molecule.
AD_IN_DOMAIN = 0.50
AD_BORDERLINE = 0.30


def cors_origins() -> list[str]:
    """Return configured browser origins, defaulting to local development."""
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
