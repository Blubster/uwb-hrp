"""Extensible ARB waveform export layer (vendor-neutral core and instrument backends).

Provides vendor-neutral active-power normalization, semantic marker tracks, and
the [ARBExporter][uwb_hrp.export.ARBExporter] abstract base class alongside the
built-in Rohde & Schwarz `.wv` exporter
([RSWaveformExporter][uwb_hrp.export.RSWaveformExporter]).
"""

from uwb_hrp.export.base import (
    ARBExporter,
    MarkerTrack,
    NormalizationMode,
    PowerReference,
    PreparedARBWaveform,
    WaveformMarkers,
    WaveformPowerMetrics,
    build_waveform_markers,
    compute_power_metrics,
    prepare_arb_waveform,
)
from uwb_hrp.export.rs import (
    RS_INT16_FULL_SCALE,
    RSWaveformExporter,
    RSWaveformPowerMetrics,
    build_rs_markers,
    compute_rs_power_metrics,
    save_rs_waveform,
    to_rs_waveform,
)

__all__ = [
    "RS_INT16_FULL_SCALE",
    "ARBExporter",
    "MarkerTrack",
    "NormalizationMode",
    "PowerReference",
    "PreparedARBWaveform",
    "RSWaveformExporter",
    "RSWaveformPowerMetrics",
    "WaveformMarkers",
    "WaveformPowerMetrics",
    "build_rs_markers",
    "build_waveform_markers",
    "compute_power_metrics",
    "compute_rs_power_metrics",
    "prepare_arb_waveform",
    "save_rs_waveform",
    "to_rs_waveform",
]
