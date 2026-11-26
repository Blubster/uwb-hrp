"""Rohde & Schwarz `.wv` (`SMU-WV`) ARB waveform exporter.

Builds on the vendor-neutral [uwb_hrp.export.base][] normalization and marker
layer and serializes `.wv` files via the optional `RsWaveform` package
(`pip install "uwb-hrp[rs]"`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uwb_hrp.config import HRPConfig
from uwb_hrp.export.base import (
    ARBExporter,
    PowerReference,
    WaveformPowerMetrics,
    build_waveform_markers,
    compute_power_metrics,
)
from uwb_hrp.generator import HRPWaveformResult

RS_INT16_FULL_SCALE: float = float(np.iinfo(np.int16).max)  # 32767.0

# Backwards-compatible alias
RSWaveformPowerMetrics = WaveformPowerMetrics


def _normalize_rs_marker_list(
    transitions: Sequence[tuple[int, int]],
    total_samples: int,
    *,
    marker_num: int,
) -> list[list[int]]:
    """Validate and canonicalize a custom Rohde & Schwarz `MARKER LIST` table.

    Args:
        transitions: Sequence of `(sample_index, state)` pairs with `state` in
            `{0, 1}` and strictly increasing sample indices.
        total_samples: Total sample count of the waveform.
        marker_num: Rohde & Schwarz marker channel number (`1..4`).

    Returns:
        Canonical transition list `[[sample_index, state], ...]` starting at
        sample 0 with redundant consecutive states removed.

    Raises:
        ValueError: If `marker_num` is out of `1..4`, `transitions` is empty,
            or any sample index/state is invalid or non-increasing.
    """
    if marker_num not in (1, 2, 3, 4):
        raise ValueError(f"Marker number must be in 1..4, got {marker_num}.")
    if len(transitions) == 0:
        raise ValueError(f"Marker {marker_num} transition list must not be empty.")

    prev_pos = -1
    cleaned: list[list[int]] = []
    for raw_pos, raw_val in transitions:
        pos = int(raw_pos)
        val = int(raw_val)
        if not (0 <= pos < total_samples):
            raise ValueError(
                f"Marker {marker_num} position {pos} out of range [0, {total_samples - 1}]."
            )
        if val not in (0, 1):
            raise ValueError(f"Marker {marker_num} state must be 0 or 1, got {val}.")
        if pos <= prev_pos:
            raise ValueError(
                f"Marker {marker_num} positions must be strictly increasing "
                f"(got {pos} after {prev_pos})."
            )
        prev_pos = pos
        if not cleaned and pos > 0:
            cleaned.append([0, 0])
        if cleaned and cleaned[-1][1] == val:
            continue
        cleaned.append([pos, val])

    return cleaned


def _validate_header_text(name: str, value: str) -> None:
    """Reject braces that would corrupt R&S `{TAG:...}` ASCII header framing."""
    if "{" in value or "}" in value:
        raise ValueError(f"{name} must not contain '{{' or '}}' characters, got {value!r}.")


def _default_comment(config: HRPConfig | None) -> str:
    """Generate a default `{COMMENT:...}` string matching RsWaveform's parser regex."""
    if config is None:
        return "IEEE 802.15.4 HRP UWB waveform"
    return f"IEEE 802.15.4 HRP UWB {config.mode} CH{config.channel} SP{config.sts_packet_config}"


def _import_rs_waveform() -> Any:
    """Lazily import the optional `RsWaveform` package with an actionable error."""
    try:
        import RsWaveform
    except ImportError as exc:
        raise ImportError(
            "The optional 'RsWaveform' package is required to create or save Rohde & Schwarz "
            "waveform files. Install it with: pip install 'uwb-hrp[rs]' (or 'uv sync --extra rs')."
        ) from exc
    return RsWaveform


def build_rs_markers(
    result: HRPWaveformResult,
    *,
    pulse_samples: int | None = None,
) -> dict[str, list[list[int]]]:
    """Build standard Rohde & Schwarz `MARKER LIST 1..4` tables from semantic markers.

    Maps [build_waveform_markers][uwb_hrp.export.build_waveform_markers] onto
    R&S marker channels:

    - **Marker 1 (`marker_list_1`):** Active packet / burst gate (RF Blanking / ALC Hold).
    - **Marker 2 (`marker_list_2`):** RMARKER rising-edge trigger pulse.
    - **Marker 3 (`marker_list_3`):** Packet start / frame trigger pulse.
    - **Marker 4 (`marker_list_4`):** STS gate (included only for SP1/SP2/SP3).

    Args:
        result: Waveform result returned by
            [generate_waveform][uwb_hrp.generate_waveform].
        pulse_samples: Optional pulse width in samples for trigger markers
            (Markers 2 and 3).

    Returns:
        Dictionary keyed by `"marker_list_1"`..`"marker_list_4"` containing
        canonical `[[sample_index, state], ...]` transition lists.
    """
    wm = build_waveform_markers(result, pulse_samples=pulse_samples)
    out: dict[str, list[list[int]]] = {
        "marker_list_1": wm.active_burst.to_transitions(),
    }
    if wm.rmarker is not None:
        out["marker_list_2"] = wm.rmarker.to_transitions()
    if wm.packet_start is not None:
        out["marker_list_3"] = wm.packet_start.to_transitions()
    if wm.sts_gate is not None:
        out["marker_list_4"] = wm.sts_gate.to_transitions()
    return out


def compute_rs_power_metrics(
    result: HRPWaveformResult,
    *,
    power_reference: PowerReference = "active",
    digital_backoff_db: float = 0.0,
) -> WaveformPowerMetrics:
    """Compute peak-normalized R&S `{LEVEL OFFS}` power metrics without `RsWaveform`.

    Args:
        result: Waveform result returned by
            [generate_waveform][uwb_hrp.generate_waveform].
        power_reference: Waveform region used for reference RMS power
            calculation (default `"active"`, excluding `idle_time`).
        digital_backoff_db: Optional peak backoff below full scale in dB.

    Returns:
        [WaveformPowerMetrics][uwb_hrp.export.WaveformPowerMetrics] with
        `normalization="peak"`.
    """
    return compute_power_metrics(
        result,
        power_reference=power_reference,
        normalization="peak",
        digital_backoff_db=digital_backoff_db,
    )


@dataclass(frozen=True)
class RSWaveformExporter(ARBExporter):
    """Rohde & Schwarz `SMU-WV` (`.wv`) ARB waveform exporter.

    Peak-normalizes I/Q samples symmetrically into 16-bit full scale `[-32767, +32767]`
    and populates `{LEVEL OFFS: rms_offset_db, peak_offset_db}` in `float64` from
    `power_reference` (default `"active"`, excluding `idle_time`).

    Attributes:
        power_reference: Waveform region used for `{LEVEL OFFS}` RMS offset
            calculation (default `"active"`, excluding `idle_time`).
        digital_backoff_db: Optional digital peak backoff below full scale in dB.
        include_markers: If `True` (default) and `markers` is `None`,
            automatically populate Markers 1-4 and set `{CONTROL LENGTH}`.
        marker_pulse_samples: Optional pulse width in samples for trigger markers.
        markers: Optional custom marker map `{marker_num: [(sample_idx, state), ...]}`
            for markers `1..4` (overrides `include_markers`).
        comment: Optional ASCII header comment (must not contain `{` or `}`).
        copyright: ASCII header copyright tag (default `"uwb_hrp"`).
        reflevel_dbm: Optional `{REFLEVEL}` tag in dBm (leave `None` for
            standard R&S ARB playback where output power is set on the instrument).
    """

    markers: Mapping[int, Sequence[tuple[int, int]]] | None = None
    comment: str | None = None
    copyright: str = "uwb_hrp"
    reflevel_dbm: float | None = None

    def _build_rs_waveform(
        self,
        result: HRPWaveformResult,
        *,
        config: HRPConfig | None = None,
    ) -> tuple[Any, WaveformPowerMetrics]:
        """Prepare normalized I/Q, markers, and an `RsWaveform` instance."""
        resolved_comment = _default_comment(config) if self.comment is None else self.comment
        _validate_header_text("comment", resolved_comment)
        _validate_header_text("copyright", self.copyright)
        if self.reflevel_dbm is not None and not math.isfinite(self.reflevel_dbm):
            raise ValueError(f"reflevel_dbm must be finite, got {self.reflevel_dbm}.")

        prepared = self.prepare(result)
        metrics = prepared.metrics

        resolved_markers: dict[str, list[list[int]]] = {}
        if self.markers is not None:
            for marker_num, transitions in sorted(self.markers.items()):
                resolved_markers[f"marker_list_{marker_num}"] = _normalize_rs_marker_list(
                    transitions,
                    metrics.total_samples,
                    marker_num=marker_num,
                )
        elif self.include_markers:
            wm = prepared.markers
            resolved_markers["marker_list_1"] = wm.active_burst.to_transitions()
            if wm.rmarker is not None:
                resolved_markers["marker_list_2"] = wm.rmarker.to_transitions()
            if wm.packet_start is not None:
                resolved_markers["marker_list_3"] = wm.packet_start.to_transitions()
            if wm.sts_gate is not None:
                resolved_markers["marker_list_4"] = wm.sts_gate.to_transitions()

        rs_mod = _import_rs_waveform()
        wv = rs_mod.RsWaveform()
        wv.data[0] = prepared.iq
        wv.meta[0].type = "SMU-WV"
        wv.meta[0].copyright = self.copyright
        wv.meta[0].comment = resolved_comment
        wv.meta[0].clock = prepared.sample_rate
        wv.meta[0].samples = metrics.total_samples
        wv.meta[0].rms = metrics.rms_offset_db
        wv.meta[0].peak = metrics.peak_offset_db
        if self.reflevel_dbm is not None:
            wv.meta[0].reflevel = float(self.reflevel_dbm)
        if resolved_markers:
            wv.meta[0].control_length = metrics.total_samples
            wv.meta[0].marker = resolved_markers

        return wv, metrics

    def to_rs_waveform(
        self,
        result: HRPWaveformResult,
        *,
        config: HRPConfig | None = None,
    ) -> Any:
        """Convert `result` into a configured `RsWaveform.RsWaveform` instance.

        Args:
            result: Waveform result returned by
                [generate_waveform][uwb_hrp.generate_waveform].
            config: Optional [HRPConfig][uwb_hrp.HRPConfig] used to generate
                a descriptive default `{COMMENT}` header when `self.comment` is
                `None`.

        Returns:
            Populated `RsWaveform.RsWaveform` object ready for inspection or
            saving.

        Raises:
            ImportError: If the optional `RsWaveform` extra is not installed.
            ValueError: If header tags, markers, or waveform values are invalid.
        """
        wv, _ = self._build_rs_waveform(result, config=config)
        return wv

    def save(
        self,
        result: HRPWaveformResult,
        path: str | Path,
        *,
        config: HRPConfig | None = None,
    ) -> WaveformPowerMetrics:
        """Write `result` to a Rohde & Schwarz `.wv` file and return power metrics.

        Args:
            result: Waveform result returned by
                [generate_waveform][uwb_hrp.generate_waveform].
            path: Destination `.wv` file path.
            config: Optional [HRPConfig][uwb_hrp.HRPConfig] used for default
                `{COMMENT}` header text.

        Returns:
            Exact `float64` [WaveformPowerMetrics][uwb_hrp.export.WaveformPowerMetrics]
            written into `{LEVEL OFFS}`.

        Raises:
            ImportError: If the optional `RsWaveform` extra is not installed.
            ValueError: If header tags, markers, or waveform values are invalid.
        """
        wv, metrics = self._build_rs_waveform(result, config=config)
        wv.save(Path(path), scale=RS_INT16_FULL_SCALE)
        return metrics


def to_rs_waveform(
    result: HRPWaveformResult,
    *,
    config: HRPConfig | None = None,
    power_reference: PowerReference = "active",
    digital_backoff_db: float = 0.0,
    include_markers: bool = True,
    markers: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    marker_pulse_samples: int | None = None,
    comment: str | None = None,
    copyright: str = "uwb_hrp",
    reflevel_dbm: float | None = None,
) -> Any:
    """Convert an [HRPWaveformResult][uwb_hrp.HRPWaveformResult] to `RsWaveform.RsWaveform`.

    Convenience wrapper around [RSWaveformExporter][uwb_hrp.RSWaveformExporter].

    Args:
        result: Waveform result returned by
            [generate_waveform][uwb_hrp.generate_waveform].
        config: Optional [HRPConfig][uwb_hrp.HRPConfig] for default `{COMMENT}` text.
        power_reference: Waveform region used for `{LEVEL OFFS}` RMS calculation
            (default `"active"`, excluding `idle_time`).
        digital_backoff_db: Optional peak backoff below full scale in dB.
        include_markers: Whether to emit default Markers 1-4 when `markers` is
            `None` (default `True`).
        markers: Optional custom marker map `{marker_num: [(sample_idx, state), ...]}`
            for markers `1..4`.
        marker_pulse_samples: Optional pulse width in samples for trigger markers.
        comment: Optional ASCII `{COMMENT}` header string (no `{` or `}`).
        copyright: ASCII `{COPYRIGHT}` header string (default `"uwb_hrp"`).
        reflevel_dbm: Optional `{REFLEVEL}` tag in dBm.

    Returns:
        Configured `RsWaveform.RsWaveform` object.

    Raises:
        ImportError: If the optional `RsWaveform` extra is not installed.
        ValueError: If header tags, markers, or waveform parameters are invalid.
    """
    exporter = RSWaveformExporter(
        power_reference=power_reference,
        digital_backoff_db=digital_backoff_db,
        include_markers=include_markers,
        marker_pulse_samples=marker_pulse_samples,
        markers=markers,
        comment=comment,
        copyright=copyright,
        reflevel_dbm=reflevel_dbm,
    )
    return exporter.to_rs_waveform(result, config=config)


def save_rs_waveform(
    result: HRPWaveformResult,
    path: str | Path,
    *,
    config: HRPConfig | None = None,
    power_reference: PowerReference = "active",
    digital_backoff_db: float = 0.0,
    include_markers: bool = True,
    markers: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    marker_pulse_samples: int | None = None,
    comment: str | None = None,
    copyright: str = "uwb_hrp",
    reflevel_dbm: float | None = None,
) -> WaveformPowerMetrics:
    """Export an [HRPWaveformResult][uwb_hrp.HRPWaveformResult] to a Rohde & Schwarz `.wv` file.

    Convenience wrapper around [RSWaveformExporter][uwb_hrp.RSWaveformExporter].

    Args:
        result: Waveform result returned by
            [generate_waveform][uwb_hrp.generate_waveform].
        path: Destination `.wv` file path.
        config: Optional [HRPConfig][uwb_hrp.HRPConfig] for default `{COMMENT}` text.
        power_reference: Waveform region used for `{LEVEL OFFS}` RMS calculation
            (default `"active"`, excluding `idle_time`).
        digital_backoff_db: Optional peak backoff below full scale in dB.
        include_markers: Whether to emit default Markers 1-4 when `markers` is
            `None` (default `True`).
        markers: Optional custom marker map `{marker_num: [(sample_idx, state), ...]}`
            for markers `1..4`.
        marker_pulse_samples: Optional pulse width in samples for trigger markers.
        comment: Optional ASCII `{COMMENT}` header string (no `{` or `}`).
        copyright: ASCII `{COPYRIGHT}` header string (default `"uwb_hrp"`).
        reflevel_dbm: Optional `{REFLEVEL}` tag in dBm.

    Returns:
        Exact `float64` [WaveformPowerMetrics][uwb_hrp.export.WaveformPowerMetrics]
        written into `{LEVEL OFFS}`.

    Raises:
        ImportError: If the optional `RsWaveform` extra is not installed.
        ValueError: If header tags, markers, or waveform parameters are invalid.
    """
    exporter = RSWaveformExporter(
        power_reference=power_reference,
        digital_backoff_db=digital_backoff_db,
        include_markers=include_markers,
        marker_pulse_samples=marker_pulse_samples,
        markers=markers,
        comment=comment,
        copyright=copyright,
        reflevel_dbm=reflevel_dbm,
    )
    return exporter.save(result, path, config=config)
