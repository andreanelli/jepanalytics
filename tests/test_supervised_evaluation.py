import numpy as np
import torch

from jepanalytics.baselines import CoordinateAwareSpectrumCNN
from jepanalytics.data import CanonicalSpectraDataset, build_synthetic_store
from jepanalytics.supervised_evaluation import (
    CNNFitSpec,
    _family_local_indices,
    _selected_family_indices,
    fit_coordinate_cnn,
)


def test_coordinate_aware_cnn_output_shape():
    model = CoordinateAwareSpectrumCNN(7, channels=8, coordinate_bins=4)
    assert model(torch.randn(3, 256)).shape == (3, 7)


def test_supervised_cnn_tiny_fit(tmp_path):
    root = tmp_path / "store"
    build_synthetic_store(root, n_molecules=40, n_bins=256, n_labels=4, seed=9)
    train = CanonicalSpectraDataset(root, "train")
    validation = CanonicalSpectraDataset(root, "validation")
    family = 0
    train_family = _family_local_indices(train, family)
    validation_family = _family_local_indices(validation, family)
    selected = _selected_family_indices(train, train_family, 0.5, 3)
    fitted = fit_coordinate_cnn(
        train,
        selected,
        validation,
        validation_family,
        spec=CNNFitSpec(1e-3, 1e-3),
        epochs=1,
        batch_size=8,
        seed=3,
        device="cpu",
        evaluation_interval=1,
        patience=1,
    )
    assert np.isfinite(fitted.validation_auprc)
    assert fitted.best_epoch == 1
