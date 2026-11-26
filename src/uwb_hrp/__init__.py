"""IEEE Std 802.15.4-2024 HRP UWB PHY frame and baseband waveform generator."""

from importlib.metadata import PackageNotFoundError, version

from uwb_hrp.config import HRPConfig, HRPFieldIndices
from uwb_hrp.fec import (
    bits_lsb_to_bytes,
    bytes_to_bits_lsb,
    compute_fcs16,
    compute_fcs16_bits,
    compute_fcs32,
    compute_fcs32_bits,
)
from uwb_hrp.generator import HRPWaveformResult, field_indices, generate_chips, generate_waveform
from uwb_hrp.pulse import (
    ButterworthPulse,
    PulseComplianceReport,
    PulseConfig,
    PulseShaper,
    RRCPulse,
    butterworth_pulse,
    rrc_pulse,
)

try:
    __version__ = version("uwb-hrp")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "ButterworthPulse",
    "HRPConfig",
    "HRPFieldIndices",
    "HRPWaveformResult",
    "PulseComplianceReport",
    "PulseConfig",
    "PulseShaper",
    "RRCPulse",
    "__version__",
    "bits_lsb_to_bytes",
    "butterworth_pulse",
    "bytes_to_bits_lsb",
    "compute_fcs16",
    "compute_fcs16_bits",
    "compute_fcs32",
    "compute_fcs32_bits",
    "field_indices",
    "generate_chips",
    "generate_waveform",
    "rrc_pulse",
]
