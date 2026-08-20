import numpy as np
import torch

from jepanalytics.losses import (
    alignment_diagnostics,
    covariance_regularization,
    embedding_diagnostics,
    group_centroid_regularization,
    hard_negative_prototype_alignment_loss,
    latent_vector_loss,
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


def test_hard_negative_prototypes_use_leave_one_view_out_targets():
    molecule = torch.tensor([10, 10, 10, 20, 20, 20, 30, 30, 30])
    base = torch.eye(3).repeat_interleave(3, dim=0)
    target = base + 0.01 * torch.randn_like(base)
    query = base.clone().requires_grad_(True)
    formula = torch.tensor(
        [[0.0, 0.0]] * 3 + [[0.1, 0.0]] * 3 + [[3.0, 3.0]] * 3
    )
    loss, diagnostics = hard_negative_prototype_alignment_loss(
        query,
        target,
        molecule,
        formula,
        temperature=0.1,
        hard_negatives=1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert query.grad is not None
    assert diagnostics["prototype_batch_top1"] == 1.0
    assert diagnostics["prototype_cosine_margin"] > 0.9
    assert diagnostics["prototype_negatives_per_anchor"] == 1.0


def test_all_tokenizers_preserve_the_fixed_token_budget():
    for tokenizer_type in (
        "linear_patch",
        "overlap_conv",
        "multiscale_conv",
        "hybrid_peak_multiscale",
    ):
        config = EncoderConfig(
            n_bins=96,
            patch_size=8,
            hidden_dim=36,
            depth=1,
            heads=4,
            mlp_ratio=2,
            aligned_dim=16,
            tokenizer_type=tokenizer_type,
            overlap_kernel_size=15,
            multiscale_kernel_sizes=(5, 9, 15),
            hybrid_peak_tokens=3,
            peak_window_size=5,
            peak_suppression_size=5,
        )
        model = UniversalSpectrumEncoder(config)
        tokenized = model.tokenize(torch.randn(2, 96))
        assert tokenized.tokens.shape == (2, 12, 36)
        assert tokenized.normalized_coordinate.shape == (2, 12)
        output = model(
            intensity=torch.randn(2, 96),
            continuous_metadata=torch.randn(2, 10),
            axis_type=torch.tensor([0, 1]),
            axis_unit=torch.tensor([0, 1]),
            acquisition=torch.tensor([0, 1]),
        )
        assert output.patches.shape == (2, 12, 36)


def test_hybrid_peak_tokens_retain_exact_sorted_coordinates():
    config = EncoderConfig(
        n_bins=96,
        patch_size=8,
        hidden_dim=36,
        depth=1,
        heads=4,
        aligned_dim=16,
        tokenizer_type="hybrid_peak_multiscale",
        multiscale_kernel_sizes=(5, 9, 15),
        hybrid_peak_tokens=2,
        peak_window_size=5,
        peak_suppression_size=5,
    )
    model = UniversalSpectrumEncoder(config)
    intensity = torch.zeros(1, 96)
    intensity[0, 10] = 2.0
    intensity[0, 70] = 3.0
    tokenized = model.tokenize(intensity)
    expected = torch.tensor([(10.5 / 96), (70.5 / 96)])
    assert torch.allclose(tokenized.normalized_coordinate[0, -2:], expected)
    assert torch.equal(tokenized.token_type[0, -2:], torch.ones(2, dtype=torch.long))


def test_full_hybrid_resamples_128_dense_positions_to_96_plus_32_peaks():
    model = UniversalSpectrumEncoder(
        EncoderConfig(tokenizer_type="hybrid_peak_multiscale")
    )
    tokenized = model.tokenize(torch.randn(2, 4096))
    assert tokenized.tokens.shape == (2, 128, 384)
    assert torch.all(tokenized.token_type[:, :96] == 0)
    assert torch.all(tokenized.token_type[:, 96:] == 1)


def test_masked_tokens_keep_distinct_position_and_coordinate_features():
    model = UniversalSpectrumEncoder(tiny_config())
    captured = {}

    def capture_input(_module, args):
        captured["sequence"] = args[0].detach().clone()

    handle = model.transformer.register_forward_pre_hook(capture_input)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[:, 1] = True
    mask[:, 5] = True
    model(**batch(), patch_mask=mask)
    handle.remove()
    sequence = captured["sequence"]
    assert not torch.allclose(sequence[:, 3], sequence[:, 7])


def test_overlap_tokenizer_cannot_read_raw_values_inside_masked_region():
    config = EncoderConfig(
        n_bins=96,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        aligned_dim=16,
        tokenizer_type="overlap_conv",
        overlap_kernel_size=15,
    )
    model = UniversalSpectrumEncoder(config)
    intensity = torch.zeros(1, 96)
    intensity[0, 20] = 10.0
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, 2] = True
    masked = model.tokenize(intensity, patch_mask=mask).tokens
    zero = model.tokenize(torch.zeros_like(intensity), patch_mask=mask).tokens
    assert torch.allclose(masked, zero)


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
    # Every patch carries signal here, so the requested ratio applies to the
    # full row and the strongest patch is still selected by the peak component.
    scores = torch.rand(2, 12) + 0.5
    scores[:, 7] = 100
    mask = mixed_patch_mask(scores, 0.5, generator=torch.Generator().manual_seed(2))
    assert torch.all(mask.sum(dim=1) == 6)
    assert torch.all(mask[:, 7])


def test_mixed_mask_ratio_applies_to_informative_patches_when_sparse():
    scores = torch.zeros(2, 12)
    scores[:, 7] = 100
    occupancy_aware = mixed_patch_mask(
        scores, 0.5, generator=torch.Generator().manual_seed(2)
    )
    legacy = mixed_patch_mask(
        scores,
        0.5,
        generator=torch.Generator().manual_seed(2),
        occupancy_aware=False,
    )
    # Only the one occupied patch is a valid target; masking half of all twelve
    # would make the objective solvable by predicting an empty patch.
    assert torch.all(occupancy_aware.sum(dim=1) == 1)
    assert torch.all(occupancy_aware[:, 7])
    assert torch.all(legacy.sum(dim=1) == 6)


def test_losses_and_collapse_diagnostics_are_finite():
    predicted = torch.randn(2, 4, 8)
    target = predicted + 0.05 * torch.randn_like(predicted)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    assert masked_latent_loss(predicted, target, mask) < 0.1
    assert latent_vector_loss(predicted[:, 0], target[:, 0]) < 0.1
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


def test_tokenizer_variants_are_parameter_matched_within_five_percent():
    counts = []
    for tokenizer_type in (
        "linear_patch",
        "overlap_conv",
        "multiscale_conv",
        "hybrid_peak_multiscale",
    ):
        model = UniversalSpectrumEncoder(
            EncoderConfig(tokenizer_type=tokenizer_type)
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert max(counts) / min(counts) < 1.05
