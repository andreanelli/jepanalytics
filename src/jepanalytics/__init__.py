"""JEPAnalytics public API."""

from .model import EmbeddingBundle, EncoderConfig, UniversalSpectrumEncoder
from .signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal

__all__ = [
    "AcquisitionFamily",
    "AxisType",
    "AxisUnit",
    "EmbeddingBundle",
    "EncoderConfig",
    "SpectralSignal",
    "UniversalSpectrumEncoder",
]

__version__ = "0.1.0"

