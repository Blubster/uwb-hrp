"""IEEE Std 802.15.4-2024 HRP UWB PHY frame and baseband waveform generator."""

from importlib.metadata import PackageNotFoundError, version

from uwb_hrp.config import HRPConfig, HRPFieldIndices
from uwb_hrp.export import (
    ARBExporter,
    MarkerTrack,
    NormalizationMode,
    PowerReference,
    PreparedARBWaveform,
    RSWaveformExporter,
    RSWaveformPowerMetrics,
    WaveformMarkers,
    WaveformPowerMetrics,
    build_rs_markers,
    build_waveform_markers,
    compute_power_metrics,
    compute_rs_power_metrics,
    prepare_arb_waveform,
    save_rs_waveform,
    to_rs_waveform,
)
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
    "ARBExporter",
    "ButterworthPulse",
    "HRPConfig",
    "HRPFieldIndices",
    "HRPWaveformResult",
    "MarkerTrack",
    "NormalizationMode",
    "PowerReference",
    "PreparedARBWaveform",
    "PulseComplianceReport",
    "PulseConfig",
    "PulseShaper",
    "RRCPulse",
    "RSWaveformExporter",
    "RSWaveformPowerMetrics",
    "WaveformMarkers",
    "WaveformPowerMetrics",
    "__version__",
    "bits_lsb_to_bytes",
    "build_rs_markers",
    "build_waveform_markers",
    "butterworth_pulse",
    "bytes_to_bits_lsb",
    "compute_fcs16",
    "compute_fcs16_bits",
    "compute_fcs32",
    "compute_fcs32_bits",
    "compute_power_metrics",
    "compute_rs_power_metrics",
    "field_indices",
    "generate_chips",
    "generate_waveform",
    "prepare_arb_waveform",
    "rrc_pulse",
    "save_rs_waveform",
    "to_rs_waveform",
]
