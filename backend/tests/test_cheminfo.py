import numpy as np
import pytest

from app.core.cheminfo import (
    combined_features, descriptor_names, domain_label, feature_names, fingerprint_smiles, max_tanimoto,
    molecular_properties, murcko_scaffold, parse_smiles, random_split, scaffold_split,
)
from app.core.config import AD_BORDERLINE, AD_IN_DOMAIN, FINGERPRINT_BITS


@pytest.mark.parametrize("bad", ["", "   ", "not a smiles", "C1CC", None, 123])
def test_parse_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        parse_smiles(bad)


def test_fingerprint_shape_and_bit_info():
    vector, molecule, bit_info = fingerprint_smiles("c1ccccc1")
    assert vector.shape == (FINGERPRINT_BITS,) and vector.dtype == np.uint8
    assert set(bit_info) == set(np.flatnonzero(vector).tolist())
    assert molecule.GetNumAtoms() == 6


def test_feature_layout_matches_vector():
    vector, molecule, _ = fingerprint_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    features = combined_features(molecule, vector)
    assert features.shape == (len(feature_names()),)
    assert np.isfinite(features).all()
    assert len(descriptor_names()) == len(features) - FINGERPRINT_BITS - 167


def test_lipinski_flags_large_lipophilic_molecule():
    aspirin = molecular_properties(parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
    assert aspirin["lipinski_pass"]
    greasy = molecular_properties(parse_smiles("C" * 60))
    assert not greasy["lipinski_pass"] and "log_p_over_5" in greasy["lipinski_violations"]


def test_murcko_scaffold():
    assert murcko_scaffold("Cc1ccccc1") == murcko_scaffold("CCc1ccccc1") == "c1ccccc1"
    assert murcko_scaffold("CCCC") == ""
    assert murcko_scaffold("garbage") == ""


@pytest.mark.parametrize("seed", [None, 0, 1, 7])
def test_scaffold_split_keeps_groups_intact(seed):
    rng = np.random.RandomState(3)
    scaffolds = [f"s{rng.randint(0, 120)}" for _ in range(1000)]
    train, valid, test = scaffold_split(scaffolds, seed=seed)
    parts = [set(scaffolds[i] for i in part) for part in (train, valid, test)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert len(train) + len(valid) + len(test) == 1000
    assert len(train) >= 700 and len(test) > 0


def test_scaffold_split_is_reproducible_and_seed_sensitive():
    scaffolds = [f"s{i % 90}" for i in range(900)]
    first, second = scaffold_split(scaffolds, seed=1), scaffold_split(scaffolds, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(scaffold_split(scaffolds, seed=1)[2], scaffold_split(scaffolds, seed=2)[2])


def test_random_split_partitions_everything():
    parts = random_split(500, seed=4)
    assert sorted(np.concatenate(parts).tolist()) == list(range(500))


def test_max_tanimoto_and_domain_labels():
    reference = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    queries = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    similarity = max_tanimoto(queries, reference)
    assert similarity[0] == pytest.approx(1.0) and similarity[1] == pytest.approx(0.5) and similarity[2] == 0.0
    assert domain_label(AD_IN_DOMAIN) == "In domain"
    assert domain_label(AD_BORDERLINE) == "Borderline"
    assert domain_label(AD_BORDERLINE - 0.01) == "Out of domain"
