import numpy as np
import pytest

from jepanalytics.preprocessing import SignalProcessor, resample_peak_preserving, robust_scale
from jepanalytics.signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal


def make_ir(coordinate, intensity):
    return SpectralSignal(
        coordinate=np.asarray(coordinate),
        intensity=np.asarray(intensity),
        axis_type=AxisType.WAVENUMBER,
        axis_unit=AxisUnit.INVERSE_CENTIMETER,
        acquisition=AcquisitionFamily.IR,
        molecule_id="molecule",
        source="test",
        source_id="record",
    )


def test_canonicalizes_descending_axis_and_duplicate_coordinates():
    signal = make_ir([4.0, 3.0, 3.0, 2.0, 1.0], [1.0, 2.0, 5.0, 3.0, 4.0])
    canonical = signal.canonical()
    assert np.all(np.diff(canonical.coordinate) > 0)
    assert canonical.original_descending
    assert canonical.intensity[canonical.coordinate.tolist().index(3.0)] == 5.0


def test_rejects_incompatible_axis_unit():
    with pytest.raises(ValueError, match="incompatible"):
        SpectralSignal(
            coordinate=np.arange(3),
            intensity=np.arange(3),
            axis_type=AxisType.CHEMICAL_SHIFT,
            axis_unit=AxisUnit.MZ,
            acquisition=AcquisitionFamily.H1_NMR,
            molecule_id="m",
            source="s",
            source_id="r",
        )


def test_dense_resampling_preserves_peak_location_height_and_area():
    x = np.linspace(400.0, 4000.0, 1800)
    y = np.exp(-0.5 * ((x - 1720.0) / 18.0) ** 2)
    target, values = resample_peak_preserving(x, y, 4096)
    step = target[1] - target[0]
    assert abs(target[np.argmax(values)] - 1720.0) <= step
    assert np.max(values) == pytest.approx(1.0, rel=0.01)
    assert np.trapezoid(values, target) == pytest.approx(np.trapezoid(y, x), rel=0.01)


def test_sparse_resampling_preserves_centroided_peak_and_physical_range():
    target, values = resample_peak_preserving(
        np.array([100.0, 250.0, 510.0]),
        np.array([0.3, 1.0, 0.6]),
        1024,
        peak_list=True,
        target_range=(0.0, 600.0),
    )
    assert target[0] == 0.0 and target[-1] == 600.0
    assert values.max() == 1.0
    assert target[np.argmax(values)] == pytest.approx(250.0, abs=target[1] - target[0])


def test_sparse_robust_scale_does_not_collapse_to_clip_value():
    values = np.zeros(4096)
    values[[100, 200, 300]] = [0.2, 0.5, 1.0]
    scaled, stats = robust_scale(values)
    assert stats.scale > 0.1
    assert 0.9 <= scaled.max() <= 1.2
    assert scaled[100] < scaled[200] < scaled[300]


def test_processor_retains_preprocessing_and_acquisition_metadata():
    signal = make_ir(np.linspace(400, 4000, 50), np.linspace(0, 1, 50))
    processed = SignalProcessor(n_bins=128)(signal)
    assert processed.intensity.shape == (128,)
    assert processed.continuous_metadata.shape == (10,)
    assert processed.axis_type == int(AxisType.WAVENUMBER)
    assert processed.acquisition == int(AcquisitionFamily.IR)

