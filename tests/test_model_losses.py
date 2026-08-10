import numpy as np
import torch

from jepanalytics.losses import (
    alignment_diagnostics,
    covariance_regularization,
    embedding_diagnostics,
    group_centroid_regularization,
    masked_latent_loss,
    modality_centroid_alignment_loss,
    multi_positive_alignment_loss,
    multiview_alignment_diagnostics,
    pairwise_multiview_alignment_diagnostics,
    symmetric_alignment_loss,
    symmetric_multi_positive_alignment_loss,
)
from jepanalytics.masking import mixed_patch_mask
from jepanalytics.model import EncoderConfig, UniversalSpectrumEncoder
from jepanalytics.signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal


def tiny_config():
    return EncoderConfig(
        n_bins=64,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        mlp_ratio=2,
        aligned_dim=16,
    )


def batch(batch_size=3):
    return {
        "intensity": torch.randn(batch_size, 64),
        "continuous_metadata": torch.randn(batch_size, 10),
        "axis_type": torch.tensor([0, 1, 2][:batch_size]),
        "axis_unit": torch.tensor([0, 1, 2][:batch_size]),
        "acquisition": torch.tensor([0, 1, 3][:batch_size]),
    }


def test_encoder_returns_all_three_representations():
    model = UniversalSpectrumEncoder(tiny_config())
    output = model(**batch())
    assert output.general.shape == (3, 32)
    assert output.aligned.shape == (3, 16)
    assert output.patches.shape == (3, 8, 32)
    assert torch.allclose(output.aligned.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_public_encode_accepts_validated_signal():
    model = UniversalSpectrumEncoder(tiny_config())
    signal = SpectralSignal(
        coordinate=np.linspace(400, 4000, 50),
        intensity=np.sin(np.linspace(0, 6, 50)) ** 2,
        axis_type=AxisType.WAVENUMBER,
        axis_unit=AxisUnit.INVERSE_CENTIMETER,
        acquisition=AcquisitionFamily.IR,
        molecule_id="m",
        source="test",
        source_id="r",
    )
    output = model.encode(signal)
    assert output.general.shape == (1, 32)


def test_mixed_mask_has_requested_size_and_masks_peaks():
    scores = torch.zeros(2, 12)
    scores[:, 7] = 100
    mask = mixed_patch_mask(scores, 0.5, generator=torch.Generator().manual_seed(2))
    assert torch.all(mask.sum(dim=1) == 6)
    assert torch.all(mask[:, 7])


def test_losses_and_collapse_diagnostics_are_finite():
    predicted = torch.randn(2, 4, 8)
    target = predicted + 0.05 * torch.randn_like(predicted)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    assert masked_latent_loss(predicted, target, mask) < 0.1
    aligned = torch.eye(4)
    assert symmetric_alignment_loss(aligned, aligned) < symmetric_alignment_loss(
        aligned, aligned.flip(0)
    )
    healthy = embedding_diagnostics(torch.randn(32, 16), min_effective_rank=4)
    collapsed = embedding_diagnostics(torch.ones(32, 16), min_effective_rank=4)
    assert not healthy.collapsed
    assert collapsed.collapsed


def test_covariance_regularization_detects_collinear_nonconstant_features():
    generator = torch.Generator().manual_seed(17)
    independent = torch.randn(128, 16, generator=generator, requires_grad=True)
    shared = torch.randn(128, 1, generator=generator)
    collinear = shared.repeat(1, 16).requires_grad_()

    independent_loss = covariance_regularization(independent)
    collinear_loss = covariance_regularization(collinear)

    assert collinear.std(dim=0).mean() > 0.1
    assert collinear_loss > 10 * independent_loss
    collinear_loss.backward()
    assert torch.isfinite(collinear.grad).all()


def test_multi_positive_alignment_treats_replicates_as_positives():
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    first_molecule = torch.tensor([0, 0, 1])
    second_molecule = torch.tensor([0, 1, 1])
    good = symmetric_multi_positive_alignment_loss(
        first, second, first_molecule, second_molecule
    )
    bad = symmetric_multi_positive_alignment_loss(
        first.flip(0), second, first_molecule, second_molecule
    )
    assert good < bad
    diagnostics = alignment_diagnostics(
        first, second, first_molecule, second_molecule
    )
    assert diagnostics["alignment_positives_per_anchor"] > 1
    assert diagnostics["alignment_cosine_margin"] > 0


def test_group_centroid_regularization_detects_common_direction():
    collapsed = torch.tensor([[1.0, 0.0]]).repeat(8, 1)
    spread = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    ).repeat(2, 1)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert group_centroid_regularization(collapsed, groups) > 5 * group_centroid_regularization(
        spread, groups
    )


def test_multiview_alignment_uses_every_other_view_as_a_positive():
    embedding = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    molecule = torch.tensor([0, 0, 0, 1, 1, 1])
    assert multi_positive_alignment_loss(embedding, molecule) < 1e-3
    diagnostics = multiview_alignment_diagnostics(embedding, molecule)
    assert diagnostics["alignment_batch_top1"] == 1.0
    assert diagnostics["alignment_positives_per_anchor"] == 2.0
    acquisition = torch.tensor([0, 1, 2, 0, 1, 2])
    pairwise = pairwise_multiview_alignment_diagnostics(
        embedding, molecule, acquisition
    )
    assert pairwise["alignment_pair_0_1_recall_at_1"] == 1.0
    assert pairwise["alignment_pair_0_2_cosine_margin"] > 0


def test_modality_centroid_loss_detects_technique_prototypes():
    acquisition = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    mixed = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    assert modality_centroid_alignment_loss(
        separated, acquisition
    ) > modality_centroid_alignment_loss(mixed, acquisition)


def test_default_encoder_parameter_count_matches_preregistered_range():
    model = UniversalSpectrumEncoder()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 20_000_000 <= count <= 25_000_000
