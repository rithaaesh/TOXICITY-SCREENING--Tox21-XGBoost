"""SMILES validation, fingerprints, descriptors, scaffold splitting, and applicability domain."""

from typing import Any, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.core.config import AD_BORDERLINE, AD_IN_DOMAIN, FINGERPRINT_BITS, FINGERPRINT_RADIUS


_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)


def parse_smiles(smiles: str) -> Any:
    """Parse a SMILES string or raise ValueError with a user-facing message."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError("The supplied text is not a valid SMILES structure.")
    return molecule


def fingerprint_molecule(molecule: Any, with_bit_info: bool = False) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    """Return the Morgan bit vector and, optionally, bit -> [(atom, radius)] occurrences."""
    if not with_bit_info:
        return _GENERATOR.GetFingerprintAsNumPy(molecule).astype(np.uint8, copy=False), {}
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    vector = _GENERATOR.GetFingerprintAsNumPy(molecule, additionalOutput=additional).astype(np.uint8, copy=False)
    bit_info = {int(bit): [(int(a), int(r)) for a, r in occurrences] for bit, occurrences in additional.GetBitInfoMap().items()}
    return vector, bit_info


def fingerprint_smiles(smiles: str) -> tuple[np.ndarray, Any, dict[int, list[tuple[int, int]]]]:
    """Return a Morgan bit vector, parsed molecule, and bit-to-atom mapping."""
    molecule = parse_smiles(smiles)
    vector, bit_info = fingerprint_molecule(molecule, with_bit_info=True)
    return vector, molecule, bit_info


def molecular_properties(molecule: Any) -> dict[str, object]:
    """Calculate core descriptors and Lipinski Rule of Five violations."""
    molecular_weight = float(Descriptors.MolWt(molecule))
    log_p = float(Crippen.MolLogP(molecule))
    donors = int(Lipinski.NumHDonors(molecule))
    acceptors = int(Lipinski.NumHAcceptors(molecule))
    tpsa = float(rdMolDescriptors.CalcTPSA(molecule))

    violations = []
    if molecular_weight > 500:
        violations.append("molecular_weight_over_500")
    if log_p > 5:
        violations.append("log_p_over_5")
    if donors > 5:
        violations.append("hydrogen_bond_donors_over_5")
    if acceptors > 10:
        violations.append("hydrogen_bond_acceptors_over_10")

    return {
        "molecular_weight_g_mol": molecular_weight,
        "log_p": log_p,
        "hydrogen_bond_donors": donors,
        "hydrogen_bond_acceptors": acceptors,
        "tpsa_angstrom2": tpsa,
        "lipinski_pass": not violations,
        "lipinski_violations": violations,
    }


# --- Richer feature sets (used by the benchmark only) -------------------------------------

def maccs_vector(molecule: Any) -> np.ndarray:
    fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    vector = np.zeros((fingerprint.GetNumBits(),), dtype=np.uint8)
    for bit in fingerprint.GetOnBits():
        vector[bit] = 1
    return vector


def descriptor_vector(molecule: Any) -> np.ndarray:
    """All RDKit 2D descriptors as float32, with non-finite values clipped to finite numbers."""
    values = np.array(list(Descriptors.CalcMolDescriptors(molecule).values()), dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(values, -1e6, 1e6).astype(np.float32)


def descriptor_names() -> list[str]:
    """RDKit 2D descriptor names, in the order produced by `descriptor_vector`."""
    return [name for name, _ in Descriptors._descList]


def combined_features(molecule: Any, fingerprint: np.ndarray) -> np.ndarray:
    """Model input: 2048 ECFP bits + 167 MACCS keys + RDKit descriptors, as float32."""
    return np.concatenate([fingerprint, maccs_vector(molecule), descriptor_vector(molecule)]).astype(np.float32)


def feature_names() -> list[str]:
    """Human-readable name for every column of `combined_features`."""
    return [f"ECFP bit {i}" for i in range(FINGERPRINT_BITS)] + [f"MACCS key {i}" for i in range(167)] + descriptor_names()


# --- Scaffold splitting --------------------------------------------------------------------

def murcko_scaffold(smiles: str) -> str:
    """Generic-free Bemis-Murcko scaffold SMILES; '' for acyclic molecules or parse failures."""
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:  # RDKit can fail on exotic valences; treat as scaffold-less
        return ""


def scaffold_split(
    scaffolds: Iterable[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group molecules by scaffold and assign whole groups to train/valid/test.

    seed=None gives the deterministic DeepChem-style split (largest scaffold groups fill train
    first). With an integer seed, groups are shuffled (Chemprop's "scaffold_balanced" scheme):
    groups too large to sit inside valid/test go to train first, the rest are shuffled. No scaffold
    ever appears in more than one partition, which is the property the benchmark depends on.
    """
    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(index)
    total = sum(len(members) for members in groups.values())
    train_cut, valid_cut = fractions[0] * total, (fractions[0] + fractions[1]) * total
    ordered = sorted(groups.values(), key=len, reverse=True)
    if seed is not None:
        rng = np.random.RandomState(seed)
        big = [g for g in ordered if len(g) > fractions[1] * total / 2]
        small = [g for g in ordered if len(g) <= fractions[1] * total / 2]
        rng.shuffle(small)
        ordered = big + small

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for members in ordered:
        if len(train) + len(members) <= train_cut:
            train.extend(members)
        elif len(train) + len(valid) + len(members) <= valid_cut:
            valid.extend(members)
        else:
            test.extend(members)
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def random_split(
    count: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain random molecule-level split, kept as the optimistic comparison for the scaffold split."""
    order = np.random.RandomState(seed).permutation(count)
    train_end = int(fractions[0] * count)
    valid_end = int((fractions[0] + fractions[1]) * count)
    return np.sort(order[:train_end]), np.sort(order[train_end:valid_end]), np.sort(order[valid_end:])


# --- Applicability domain ------------------------------------------------------------------

def max_tanimoto(queries: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Max Tanimoto similarity of each query bit-vector to any reference bit-vector."""
    queries = np.atleast_2d(queries).astype(np.float32)
    reference = reference.astype(np.float32)
    reference_counts = reference.sum(axis=1)
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        intersection = block @ reference.T
        union = block.sum(axis=1, keepdims=True) + reference_counts[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union > 0, intersection / union, 0.0)
        result[start:start + chunk] = similarity.max(axis=1)
    return result


def domain_label(similarity: float) -> str:
    """Human-readable applicability-domain status for a max-Tanimoto similarity."""
    if similarity >= AD_IN_DOMAIN:
        return "In domain"
    if similarity >= AD_BORDERLINE:
        return "Borderline"
    return "Out of domain"
