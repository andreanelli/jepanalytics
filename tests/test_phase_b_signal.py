import numpy as np
import torch

from jepanalytics.augment import AugmentationConfig, SpectralAugmenter
from jepanalytics.masking import informative_patches, mixed_patch_mask
from jepanalytics.preprocessing import (
    DEFAULT_MS_MZ_RANGE,
    SignalProcessor,
    _axis_scale,
    resample_peak_preserving,
)
from jepanalytics.signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal


def _ms_signal(peaks, precursor):
    coordinate = np.asarray([m for m, _ in peaks], dtype=np.float64)
    intensity = np.asarray([i for _, i in peaks], dtype=np.float64)
    return SpectralSignal(
        coordinate=coordinate,
        intensity=intensity,
        axis_type=AxisType.MASS_TO_CHARGE,
        axis_unit=AxisUnit.MZ,
        acquisition=AcquisitionFamily.MSMS_POSITIVE,
        molecule_id="m",
        source="test",
        source_id="s",
        metadata={"representation": "peak_list", "precursor_mz": precursor},
    )


def test_absolute_mz_grid_puts_equal_mz_in_equal_bin():
    processor = SignalProcessor(n_bins=4096)
    light = processor(_ms_signal([(50.0, 1.0), (150.0, 0.5)], precursor=200.0))
    heavy = processor(_ms_signal([(50.0, 1.0), (150.0, 0.5), (600.0, 0.2)], precursor=700.0))

    # The same fragment mass must land in the same bin regardless of precursor.
    assert np.argmax(light.intensity) == np.argmax(heavy.intensity)
    low, high = DEFAULT_MS_MZ_RANGE
    expected = round((50.0 - low) / (high - low) * (4096 - 1))
    assert int(np.argmax(light.intensity)) == expected


def test_precursor_metadata_excluded_by_default():
    processor = SignalProcessor(n_bins=512)
    excluded = processor(_ms_signal([(50.0, 1.0), (150.0, 0.5)], precursor=321.0))
    included = SignalProcessor(n_bins=512, include_precursor_metadata=True)(
        _ms_signal([(50.0, 1.0), (150.0, 0.5)], precursor=321.0)
    )

    assert excluded.continuous_metadata[5] == 0.0
    assert included.continuous_metadata[5] > 0.0
    # The coordinate window no longer varies with the molecule, so the axis
    # metadata that feeds every content token is precursor independent.
    other = processor(_ms_signal([(50.0, 1.0), (150.0, 0.5)], precursor=900.0))
    np.testing.assert_allclose(
        excluded.continuous_metadata[:3], other.continuous_metadata[:3]
    )


def test_out_of_range_peaks_are_dropped_and_counted():
    processor = SignalProcessor(n_bins=512)
    processed = processor(
        _ms_signal([(50.0, 1.0), (1500.0, 9.0)], precursor=1500.0)
    )
    assert processed.out_of_range_fraction == 0.5
    assert np.isfinite(processed.intensity).all()


def test_proton_axis_scale_is_not_the_carbon_constant():
    proton = _axis_scale(AxisType.CHEMICAL_SHIFT, AcquisitionFamily.H1_NMR)
    carbon = _axis_scale(AxisType.CHEMICAL_SHIFT, AcquisitionFamily.C13_NMR)
    assert proton == 12.0
    assert carbon == 250.0
    # A -2..10 ppm proton window must not normalize to a near-constant zero.
    assert 10.0 / proton > 0.5


def test_dense_downsampling_preserves_a_narrow_line():
    coordinate = np.linspace(0.0, 100.0, 10_000)
    intensity = np.zeros_like(coordinate)
    intensity[5000] = 1.0  # a single-sample NMR line

    _, resampled = resample_peak_preserving(coordinate, intensity, 4096)
    assert resampled.max() == 1.0


def test_sparse_augmentation_keeps_sparsity_and_dense_gets_baseline():
    augmenter = SpectralAugmenter(AugmentationConfig(missing_span_probability=0.0))
    intensity = torch.zeros(2, 256)
    intensity[:, ::32] = 1.0
    output = augmenter(
        intensity, acquisition=torch.tensor([int(AcquisitionFamily.MSMS_POSITIVE), 0])
    )

    sparse_row, dense_row = output[0], output[1]
    # No baseline or Gaussian noise is added to a centroided spectrum, so bins
    # that held no peak stay exactly empty.
    assert float((sparse_row != 0).float().mean()) < 0.05
    assert float((dense_row != 0).float().mean()) > 0.5


def test_augmentation_is_drawn_per_sample():
    augmenter = SpectralAugmenter()
    intensity = torch.rand(16, 256).repeat_interleave(1, dim=0)
    intensity[:] = intensity[0]
    output = augmenter(intensity, acquisition=torch.zeros(16, dtype=torch.long))
    # Identical inputs must not receive an identical batch-wide perturbation.
    assert not torch.allclose(output[0], output[1])


def test_occupancy_aware_mask_targets_informative_patches():
    patch_intensity = torch.zeros(1, 128)
    patch_intensity[0, ::16] = 1.0
    mask = mixed_patch_mask(patch_intensity, 0.5)
    informative = informative_patches(patch_intensity)

    assert int(mask.sum()) >= 1
    # Every hidden patch carries signal, so the predictor cannot satisfy the
    # objective by emitting an empty patch.
    assert bool((mask & ~informative).sum() == 0)
    assert int(mask.sum()) < int((~informative).sum())


def test_occupancy_aware_mask_matches_legacy_on_dense_rows():
    patch_intensity = torch.rand(4, 128) + 0.5
    assert int(mixed_patch_mask(patch_intensity, 0.5).sum(dim=1)[0]) == 64
