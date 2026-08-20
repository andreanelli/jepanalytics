import numpy as np

from jepanalytics.chemistry import (
    FORMULA_FEATURES,
    formula_feature_vector,
    formula_matching_features,
    parse_molecular_formula,
)


def test_formula_parser_and_feature_vector():
    assert parse_molecular_formula("C8H9Cl2NO") == {
        "C": 8,
        "H": 9,
        "Cl": 2,
        "N": 1,
        "O": 1,
    }
    vector = formula_feature_vector("C8H9Cl2NO")
    values = dict(zip(FORMULA_FEATURES, vector, strict=True))
    assert values["C"] == 8
    assert values["Cl"] == 2
    assert values["other"] == 0
    assert values["total_atoms"] == 21
    assert values["heavy_atoms"] == 12


def test_formula_features_collect_uncommon_elements_as_other():
    vector = formula_feature_vector("C2H6Na")
    values = dict(zip(FORMULA_FEATURES, vector, strict=True))
    assert values["other"] == 1
    assert np.isfinite(vector).all()


def test_formula_matching_features_recover_mass_from_train_normalization():
    manifest = {
        "features": ["H", "C", "O", "total_atoms"],
        "train_mean": [0.0, 0.0, 0.0, 0.0],
        "train_std": [1.0, 1.0, 1.0, 1.0],
    }
    counts = np.asarray([[6.0, 2.0, 1.0, 9.0]], dtype=np.float32)
    targets = np.log1p(counts)
    matching = formula_matching_features(targets, manifest)
    expected_mass = 6 * 1.00794 + 2 * 12.0107 + 15.9994
    assert matching.shape == (1, 5)
    assert np.isclose(matching[0, -1], expected_mass / 100.0, rtol=1e-5)
