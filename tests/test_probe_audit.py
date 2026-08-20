import numpy as np
import pytest

from jepanalytics.probe_audit import (
    ProbeSpec,
    compose_representation,
    coverage_controlled_molecule_indices,
    fit_validation_probe,
    per_label_metrics,
    predict_probe,
    subset_encoded_molecules,
)


def test_compose_representation_slices_before_concatenating():
    encoded = {
        "general": np.arange(30, dtype=np.float32).reshape(5, 6),
        "aligned": np.arange(20, dtype=np.float32).reshape(5, 4),
        "patch_pool": np.arange(30, dtype=np.float32).reshape(5, 6) + 100,
    }
    indices = np.asarray([1, 4])

    combined = compose_representation(encoded, "general_aligned", indices)

    assert combined.shape == (2, 10)
    np.testing.assert_array_equal(combined[:, :6], encoded["general"][indices])
    np.testing.assert_array_equal(combined[:, 6:], encoded["aligned"][indices])
    with pytest.raises(ValueError, match="does not contain"):
        compose_representation({"general": encoded["general"]}, "patch_pool")


def test_coverage_sampling_meets_feasible_minima_and_keeps_budget():
    molecules = np.repeat(np.arange(20), 3)
    molecule_labels = np.zeros((20, 4), dtype=np.float32)
    molecule_labels[:, 0] = 1
    molecule_labels[:8, 1] = 1
    molecule_labels[8:14, 2] = 1
    molecule_labels[14:18, 3] = 1
    labels = np.repeat(molecule_labels, 3, axis=0)

    selected, diagnostics = coverage_controlled_molecule_indices(
        molecules,
        labels,
        fraction=0.5,
        seed=17,
        minimum_positives=3,
    )

    chosen_molecules = np.unique(molecules[selected])
    assert chosen_molecules.size == 10
    assert diagnostics["zero_positive_labels"] == 0
    assert diagnostics["labels_below_minimum"] == 0
    assert min(diagnostics["positive_counts"]) >= 3


def _linearly_separable_problem():
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(80, 12)).astype(np.float32)
    validation_x = rng.normal(size=(40, 12)).astype(np.float32)
    test_x = rng.normal(size=(30, 12)).astype(np.float32)
    weights = rng.normal(size=(12, 3))
    train_y = (train_x @ weights > 0).astype(np.float32)
    validation_y = (validation_x @ weights > 0).astype(np.float32)
    test_y = (test_x @ weights > 0).astype(np.float32)
    return train_x, train_y, validation_x, validation_y, test_x, test_y


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_validation_probe_fits_and_reports_per_label_metrics(kind):
    train_x, train_y, validation_x, validation_y, test_x, test_y = (
        _linearly_separable_problem()
    )

    fitted = fit_validation_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        kind=kind,
        spec=ProbeSpec(0.01, 1e-4),
        epochs=30,
        seed=17,
        device="cpu",
        evaluation_interval=10,
        patience=3,
    )
    probability = predict_probe(fitted, test_x, device="cpu")
    reports = per_label_metrics(test_y, probability)

    assert probability.shape == test_y.shape
    assert 0 <= fitted.validation_auprc <= 1
    if kind == "linear":
        # L-BFGS reports the iteration count it converged at.
        assert fitted.best_epoch >= 1
    else:
        assert fitted.best_epoch in {10, 20, 30}
    assert len(reports) == 3
    assert all(np.isfinite(report["normalized_lift"]) for report in reports)


def test_linear_probe_converges_on_separable_data():
    train_x, train_y, validation_x, validation_y, test_x, test_y = (
        _linearly_separable_problem()
    )
    fitted = fit_validation_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        kind="linear",
        spec=ProbeSpec(1.0, 0.01),
        epochs=500,
        seed=17,
        device="cpu",
    )
    probability = predict_probe(fitted, test_x, device="cpu")
    from jepanalytics.metrics import macro_auprc

    # A converged logistic probe must nearly solve a linearly separable task;
    # the previous under-fitted first-order loop did not reliably reach this.
    assert macro_auprc(test_y, probability) > 0.95
    assert fitted.validation_auprc > 0.95


def test_encoded_molecule_subset_keeps_complete_groups():
    encoded = {
        "molecule_index": np.repeat(np.arange(10), 3),
        "general": np.arange(60).reshape(30, 2),
    }
    subset = subset_encoded_molecules(encoded, 4, 19)
    assert np.unique(subset["molecule_index"]).size == 4
    assert subset["general"].shape == (12, 2)
