"""Vendor-agnostic ARB waveform normalization, active-power metrics, and markers.

This module provides the shared DSP and metadata layer for exporting
:class:`uwb_hrp.HRPWaveformResult` objects to arbitrary waveform generator (ARB)
file formats (Rohde & Schwarz ``.wv``, LitePoint ``.iqvsg``, Keysight ``.wfm``,
etc.) without duplicating active-power or marker-timing logic.

To add a new instrument vendor, subclass :class:`ARBExporter` and implement
:meth:`ARBExporter.save` using :meth:`ARBExporter.prepare`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from uwb_hrp.config import HRPConfig
from uwb_hrp.generator import HRPWaveformResult

PowerReference = Literal["active", "shr", "sync", "sts", "phr", "psdu", "all"]
NormalizationMode = Literal["peak", "rms"]

_VALID_POWER_REFERENCES: frozenset[str] = frozenset(
    {"active", "shr", "sync", "sts", "phr", "psdu", "all"}
)
_VALID_NORMALIZATION_MODES: frozenset[str] = frozenset({"peak", "rms"})
_DEFAULT_MIN_MARKER_PULSE_SAMPLES: int = 16


@dataclass(frozen=True)
class WaveformPowerMetrics:
    """Vendor-neutral power and full-scale level-offset metrics for an ARB waveform.

    All offset values are expressed in dB below unity full scale (``1.0``), matching
    standard ARB conventions (positive values when the signal is below full scale):

    - ``rms_offset_db = -10 * log10(mean(|z_ref|^2))``
    - ``peak_offset_db = -20 * log10(max(|z|))``
    - ``crest_factor_db = rms_offset_db - peak_offset_db`` (PAPR relative to
      ``power_reference``)

    Attributes:
        rms_offset_db: RMS level offset in dB below unity full scale computed over
            ``power_reference`` (default ``"active"``, excluding ``idle_time``).
        peak_offset_db: Peak level offset in dB below unity full scale across the
            entire waveform (``0.0`` when ``normalization="peak"`` with zero backoff).
        crest_factor_db: Peak-to-average power ratio (``rms_offset_db - peak_offset_db``)
            in dB relative to ``power_reference``.
        total_rms_offset_db: RMS level offset in dB across the entire waveform,
            including any inter-packet ``idle_time`` zeros.
        duty_cycle: Active packet sample fraction ``active_samples / total_samples``
            in ``(0, 1]``.
        idle_dilution_db: Difference ``total_rms_offset_db - rms_offset_db`` in dB.
            Equals ``-10 * log10(duty_cycle)`` when ``power_reference == "active"``.
        power_reference: Waveform region used to compute ``rms_offset_db``.
        normalization: Scaling convention used (``"peak"`` or ``"rms"``).
        active_samples_per_packet: Number of active (non-idle) samples per packet.
        total_samples: Total number of I/Q samples in the exported waveform.
    """

    rms_offset_db: float
    peak_offset_db: float
    crest_factor_db: float
    total_rms_offset_db: float
    duty_cycle: float
    idle_dilution_db: float
    power_reference: PowerReference
    normalization: NormalizationMode
    active_samples_per_packet: int
    total_samples: int


@dataclass(frozen=True)
class MarkerTrack:
    """Single sample-synchronous binary marker track as half-open ``[start, stop)`` intervals.

    Provides vendor-neutral storage with direct converters to both transition-list
    formats (e.g. Rohde & Schwarz ``MARKER LIST``) and per-sample boolean masks
    (e.g. Keysight / LitePoint / plotting).
    """

    name: str
    intervals: tuple[tuple[int, int], ...]
    total_samples: int

    def __post_init__(self) -> None:
        if self.total_samples < 1:
            raise ValueError(f"total_samples must be >= 1, got {self.total_samples}.")
        prev_stop = -1
        for start, stop in self.intervals:
            if not (0 <= start < stop <= self.total_samples):
                raise ValueError(
                    f"Marker track {self.name!r} interval ({start}, {stop}) is invalid "
                    f"for total_samples={self.total_samples}."
                )
            if start < prev_stop:
                raise ValueError(
                    f"Marker track {self.name!r} intervals must be sorted and non-overlapping."
                )
            prev_stop = stop

    @classmethod
    def from_intervals(
        cls,
        name: str,
        intervals: Sequence[tuple[int, int]],
        total_samples: int,
    ) -> MarkerTrack:
        """Create a ``MarkerTrack``, validating and coalescing contiguous intervals."""
        if total_samples < 1:
            raise ValueError(f"total_samples must be >= 1, got {total_samples}.")
        merged: list[tuple[int, int]] = []
        for raw_start, raw_stop in intervals:
            start = int(raw_start)
            stop = int(raw_stop)
            if not (0 <= start < stop <= total_samples):
                raise ValueError(
                    f"Marker track {name!r} interval ({start}, {stop}) is invalid "
                    f"for total_samples={total_samples}."
                )
            if merged and start < merged[-1][1]:
                raise ValueError(
                    f"Marker track {name!r} intervals must be sorted and non-overlapping."
                )
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], stop)
            else:
                merged.append((start, stop))
        return cls(name=name, intervals=tuple(merged), total_samples=total_samples)

    def to_transitions(self) -> list[list[int]]:
        """Convert to canonical ``[[sample_index, state], ...]`` transitions starting at 0."""
        if not self.intervals:
            return [[0, 0]]
        transitions: list[list[int]] = []
        if self.intervals[0][0] > 0:
            transitions.append([0, 0])
        for start, stop in self.intervals:
            transitions.append([start, 1])
            if stop < self.total_samples:
                transitions.append([stop, 0])
        return transitions

    def to_mask(self) -> np.ndarray:
        """Convert to a 1D boolean array of length ``total_samples``."""
        mask = np.zeros(self.total_samples, dtype=np.bool_)
        for start, stop in self.intervals:
            mask[start:stop] = True
        return mask


@dataclass(frozen=True)
class WaveformMarkers:
    """Vendor-neutral semantic marker tracks for an HRP UWB waveform.

    Attributes:
        active_burst: HIGH during each active UWB packet, LOW during ``idle_time``
            (ideal for RF Blanking, ALC Hold, and burst-gated power measurement).
        rmarker: Rising-edge trigger pulse at the exact IEEE 802.15.4-2024 RMARKER
            instant (first chip after SFD) of every packet.
        packet_start: Rising-edge trigger pulse at sample 0 of every packet.
        sts_gate: HIGH during the STS field (``None`` when STS is disabled / SP0).
        total_samples: Total number of samples in the waveform.
    """

    active_burst: MarkerTrack
    rmarker: MarkerTrack | None
    packet_start: MarkerTrack | None
    sts_gate: MarkerTrack | None
    total_samples: int

    def to_bitmask(self) -> np.ndarray:
        """Pack markers into a 1D ``uint8`` per-sample bitmask (bits 0..3).

        Bit assignment:
            - Bit 0 (``0x01``): ``active_burst``
            - Bit 1 (``0x02``): ``rmarker``
            - Bit 2 (``0x04``): ``packet_start``
            - Bit 3 (``0x08``): ``sts_gate``
        """
        bits = np.zeros(self.total_samples, dtype=np.uint8)
        for bit_idx, track in enumerate(
            (self.active_burst, self.rmarker, self.packet_start, self.sts_gate)
        ):
            if track is None:
                continue
            flag = np.uint8(1 << bit_idx)
            for start, stop in track.intervals:
                bits[start:stop] |= flag
        return bits


@dataclass(frozen=True)
class PreparedARBWaveform:
    """Normalized I/Q samples, power metrics, and semantic markers ready for ARB export."""

    iq: np.ndarray
    sample_rate: float
    metrics: WaveformPowerMetrics
    markers: WaveformMarkers


def _clean_zero(value: float) -> float:
    """Normalize ``-0.0`` to ``0.0`` for clean formatting."""
    return 0.0 if value == 0.0 else value


def _packet_geometry(result: HRPWaveformResult) -> tuple[int, int, int, int]:
    """Return ``(total_samples, packet_stride, active_samples_per_packet, spp)``."""
    if result.waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform array, got shape {result.waveform.shape}.")
    total_samples = len(result.waveform)
    if total_samples == 0:
        raise ValueError("Cannot export an empty waveform.")
    if result.num_packets < 1:
        raise ValueError(f"num_packets must be >= 1, got {result.num_packets}.")
    if result.idle_samples < 0:
        raise ValueError(f"idle_samples must be >= 0, got {result.idle_samples}.")
    if total_samples % result.num_packets != 0:
        raise ValueError(
            f"Waveform length {total_samples} is not divisible by num_packets={result.num_packets}."
        )

    stride = total_samples // result.num_packets
    active_samples = stride - result.idle_samples
    if active_samples <= 0:
        raise ValueError(
            f"Invalid packet geometry: stride={stride}, idle_samples={result.idle_samples} "
            f"leaves {active_samples} active samples per packet."
        )

    total_chips = result.indices.total_chips
    if total_chips <= 0 or active_samples % total_chips != 0:
        raise ValueError(
            f"Active packet length ({active_samples} samples) is not a positive integer "
            f"multiple of indices.total_chips ({total_chips})."
        )

    spp = active_samples // total_chips
    return total_samples, stride, active_samples, spp


def _extract_reference_samples(
    iq: np.ndarray,
    result: HRPWaveformResult,
    power_reference: PowerReference,
    *,
    stride: int,
    active_samples: int,
    spp: int,
) -> np.ndarray:
    """Slice ``iq`` to the sample region selected by ``power_reference``."""
    if power_reference not in _VALID_POWER_REFERENCES:
        valid = ", ".join(sorted(_VALID_POWER_REFERENCES))
        raise ValueError(
            f"Unsupported power_reference {power_reference!r}; expected one of: {valid}."
        )

    if power_reference == "all":
        return iq

    packets = iq.reshape(result.num_packets, stride)
    if power_reference == "active":
        return packets[:, :active_samples]
    if power_reference == "shr":
        return packets[:, result.indices.sync.start * spp : result.indices.sfd.stop * spp]
    if power_reference == "sync":
        return packets[:, result.indices.sync.start * spp : result.indices.sync.stop * spp]
    if power_reference == "sts":
        sts_slice = result.indices.sts1 if result.indices.sts1 is not None else result.indices.sts2
        if sts_slice is None:
            raise ValueError("Cannot use power_reference='sts' for a frame without an STS field.")
        return packets[:, sts_slice.start * spp : sts_slice.stop * spp]
    if power_reference == "phr":
        if result.indices.phr is None:
            raise ValueError(
                "Cannot use power_reference='phr' for a frame without a PHR field "
                "(e.g. STS packet configuration 3)."
            )
        return packets[:, result.indices.phr.start * spp : result.indices.phr.stop * spp]
    if power_reference == "psdu":
        if result.indices.psdu is None:
            raise ValueError(
                "Cannot use power_reference='psdu' for a frame without a PSDU field "
                "(e.g. zero-length PSDU or STS packet configuration 3)."
            )
        return packets[:, result.indices.psdu.start * spp : result.indices.psdu.stop * spp]

    raise AssertionError(f"Unhandled power_reference: {power_reference!r}")  # pragma: no cover


def _normalize_and_measure(
    result: HRPWaveformResult,
    *,
    power_reference: PowerReference = "active",
    normalization: NormalizationMode = "peak",
    digital_backoff_db: float = 0.0,
) -> tuple[np.ndarray, WaveformPowerMetrics]:
    """Normalize ``result.waveform`` and compute vendor-neutral ARB power metrics in float64."""
    if normalization not in _VALID_NORMALIZATION_MODES:
        valid = ", ".join(sorted(_VALID_NORMALIZATION_MODES))
        raise ValueError(f"Unsupported normalization {normalization!r}; expected one of: {valid}.")
    if not math.isfinite(digital_backoff_db) or digital_backoff_db < 0.0:
        raise ValueError(
            f"digital_backoff_db must be a finite value >= 0.0, got {digital_backoff_db}."
        )

    total_samples, stride, active_samples, spp = _packet_geometry(result)

    iq = np.asarray(result.waveform, dtype=np.complex128)
    if not np.all(np.isfinite(iq)):
        raise ValueError("Waveform contains NaN or infinite values.")

    peak_amp = float(np.max(np.abs(iq)))
    if peak_amp == 0.0:
        raise ValueError("Cannot normalize an all-zero waveform.")

    raw_ref = _extract_reference_samples(
        iq,
        result,
        power_reference,
        stride=stride,
        active_samples=active_samples,
        spp=spp,
    )
    raw_ref_mean_power = float(np.mean(np.abs(raw_ref) ** 2))
    if raw_ref_mean_power <= 0.0:
        raise ValueError(f"Selected power_reference {power_reference!r} has zero power.")

    target_level = 10.0 ** (-digital_backoff_db / 20.0)
    if normalization == "peak":
        scale = target_level / peak_amp
    else:
        scale = target_level / math.sqrt(raw_ref_mean_power)

    iq_norm = iq * scale
    ref_mean_power = raw_ref_mean_power * (scale**2)
    norm_peak = peak_amp * scale
    total_mean_power = float(np.mean(np.abs(iq_norm) ** 2))

    rms_offset_db = (
        _clean_zero(float(digital_backoff_db))
        if normalization == "rms"
        else _clean_zero(-10.0 * math.log10(ref_mean_power))
    )
    peak_offset_db = (
        _clean_zero(float(digital_backoff_db))
        if normalization == "peak"
        else _clean_zero(-20.0 * math.log10(norm_peak))
    )
    crest_factor_db = _clean_zero(rms_offset_db - peak_offset_db)
    total_rms_offset_db = _clean_zero(-10.0 * math.log10(total_mean_power))
    duty_cycle = active_samples / stride
    idle_dilution_db = _clean_zero(total_rms_offset_db - rms_offset_db)

    metrics = WaveformPowerMetrics(
        rms_offset_db=rms_offset_db,
        peak_offset_db=peak_offset_db,
        crest_factor_db=crest_factor_db,
        total_rms_offset_db=total_rms_offset_db,
        duty_cycle=duty_cycle,
        idle_dilution_db=idle_dilution_db,
        power_reference=power_reference,
        normalization=normalization,
        active_samples_per_packet=active_samples,
        total_samples=total_samples,
    )
    return iq_norm, metrics


def compute_power_metrics(
    result: HRPWaveformResult,
    *,
    power_reference: PowerReference = "active",
    normalization: NormalizationMode = "peak",
    digital_backoff_db: float = 0.0,
) -> WaveformPowerMetrics:
    """Compute vendor-neutral active-power and full-scale level-offset metrics.

    Args:
        result: Waveform result returned by :func:`uwb_hrp.generate_waveform`.
        power_reference: Waveform region used to define the reference RMS power:
            - ``"active"`` *(default)*: All active packet samples, excluding any
              inter-packet ``idle_time`` zeros so idle time never dilutes the power.
            - ``"shr"``: SYNC + SFD header samples.
            - ``"sync"``: Preamble SYNC samples only.
            - ``"sts"``: STS field samples (SP1/SP2/SP3).
            - ``"phr"``: PHR field samples.
            - ``"psdu"``: PSDU field samples.
            - ``"all"``: Entire waveform including ``idle_time`` zeros.
        normalization: ``"peak"`` *(default)* normalizes the global peak to full scale
            (``1.0`` minus ``digital_backoff_db``); ``"rms"`` normalizes the reference
            region RMS to ``1.0`` minus ``digital_backoff_db``.
        digital_backoff_db: Optional digital backoff in dB (default ``0.0``).

    Returns:
        :class:`WaveformPowerMetrics` containing exact ``float64`` level offsets,
        crest factor, duty cycle, and idle dilution.
    """
    _, metrics = _normalize_and_measure(
        result,
        power_reference=power_reference,
        normalization=normalization,
        digital_backoff_db=digital_backoff_db,
    )
    return metrics


def build_waveform_markers(
    result: HRPWaveformResult,
    *,
    pulse_samples: int | None = None,
) -> WaveformMarkers:
    """Build vendor-neutral semantic marker tracks for an HRP UWB waveform.

    Args:
        result: Waveform result returned by :func:`uwb_hrp.generate_waveform`.
        pulse_samples: Optional pulse width in samples for trigger markers
            (``rmarker`` and ``packet_start``). Defaults to ``max(spp, 16)`` clamped
            to packet bounds.

    Returns:
        :class:`WaveformMarkers` containing ``active_burst``, ``rmarker``,
        ``packet_start``, and (when STS is enabled) ``sts_gate``.
    """
    if pulse_samples is not None and pulse_samples < 1:
        raise ValueError(f"pulse_samples must be >= 1, got {pulse_samples}.")

    total_samples, stride, active_samples, spp = _packet_geometry(result)
    default_pw = max(spp, _DEFAULT_MIN_MARKER_PULSE_SAMPLES)

    # 1. Active packet / burst gate (coalesces to [(0, total_samples)] when idle_samples == 0)
    burst_intervals = [
        (pkt_idx * stride, pkt_idx * stride + active_samples)
        for pkt_idx in range(result.num_packets)
    ]
    active_burst = MarkerTrack.from_intervals("active_burst", burst_intervals, total_samples)

    # 2. RMARKER rising-edge trigger pulse (exact first chip after SFD)
    rmarker: MarkerTrack | None = None
    rmarker_offset = result.indices.rmarker_chip * spp
    if 0 <= rmarker_offset < active_samples:
        max_rm_pw = active_samples - rmarker_offset
        rm_pw = (
            min(default_pw, max(1, max_rm_pw // 2))
            if pulse_samples is None
            else min(pulse_samples, max_rm_pw)
        )
        rm_intervals = [
            (
                pkt_idx * stride + rmarker_offset,
                min(pkt_idx * stride + rmarker_offset + rm_pw, (pkt_idx + 1) * stride),
            )
            for pkt_idx in range(result.num_packets)
        ]
        rmarker = MarkerTrack.from_intervals("rmarker", rm_intervals, total_samples)

    # 3. Packet start / frame trigger pulse
    packet_start: MarkerTrack | None = None
    if stride > 1:
        start_pw = (
            min(default_pw, max(1, active_samples // 2))
            if pulse_samples is None
            else min(pulse_samples, stride - 1)
        )
        start_intervals = [
            (pkt_idx * stride, pkt_idx * stride + start_pw) for pkt_idx in range(result.num_packets)
        ]
        packet_start = MarkerTrack.from_intervals("packet_start", start_intervals, total_samples)

    # 4. STS gate (SP1 / SP2 / SP3 only)
    sts_gate: MarkerTrack | None = None
    sts_slice = result.indices.sts1 if result.indices.sts1 is not None else result.indices.sts2
    if sts_slice is not None:
        sts_start = sts_slice.start * spp
        sts_stop = sts_slice.stop * spp
        sts_intervals = [
            (pkt_idx * stride + sts_start, pkt_idx * stride + sts_stop)
            for pkt_idx in range(result.num_packets)
        ]
        sts_gate = MarkerTrack.from_intervals("sts_gate", sts_intervals, total_samples)

    return WaveformMarkers(
        active_burst=active_burst,
        rmarker=rmarker,
        packet_start=packet_start,
        sts_gate=sts_gate,
        total_samples=total_samples,
    )


def prepare_arb_waveform(
    result: HRPWaveformResult,
    *,
    power_reference: PowerReference = "active",
    normalization: NormalizationMode = "peak",
    digital_backoff_db: float = 0.0,
    marker_pulse_samples: int | None = None,
) -> PreparedARBWaveform:
    """Normalize an :class:`HRPWaveformResult` and compute metrics + semantic markers."""
    iq_norm, metrics = _normalize_and_measure(
        result,
        power_reference=power_reference,
        normalization=normalization,
        digital_backoff_db=digital_backoff_db,
    )
    markers = build_waveform_markers(result, pulse_samples=marker_pulse_samples)
    return PreparedARBWaveform(
        iq=iq_norm,
        sample_rate=float(result.sample_rate),
        metrics=metrics,
        markers=markers,
    )


@dataclass(frozen=True)
class ARBExporter(ABC):
    """Abstract base class for instrument-specific ARB waveform exporters.

    Subclasses (such as :class:`uwb_hrp.RSWaveformExporter`, or custom exporters for
    LitePoint, Keysight, NI, etc.) inherit shared active-power normalization and
    semantic marker generation via :meth:`prepare` and implement :meth:`save`.

    Attributes:
        power_reference: Waveform region used for reference RMS power calculation
            (default ``"active"``, excluding inter-packet ``idle_time``).
        digital_backoff_db: Optional digital backoff in dB (default ``0.0``).
        include_markers: Whether to emit sample-synchronous markers (default ``True``).
        marker_pulse_samples: Optional pulse width in samples for trigger markers.
    """

    power_reference: PowerReference = "active"
    digital_backoff_db: float = 0.0
    include_markers: bool = True
    marker_pulse_samples: int | None = None

    @property
    def normalization_mode(self) -> NormalizationMode:
        """Scaling convention expected by the target instrument (default ``"peak"``)."""
        return "peak"

    def prepare(self, result: HRPWaveformResult) -> PreparedARBWaveform:
        """Normalize ``result`` and build vendor-neutral metrics and markers."""
        return prepare_arb_waveform(
            result,
            power_reference=self.power_reference,
            normalization=self.normalization_mode,
            digital_backoff_db=self.digital_backoff_db,
            marker_pulse_samples=self.marker_pulse_samples,
        )

    @abstractmethod
    def save(
        self,
        result: HRPWaveformResult,
        path: str | Path,
        *,
        config: HRPConfig | None = None,
    ) -> WaveformPowerMetrics:
        """Export ``result`` to ``path`` and return its power metrics."""
