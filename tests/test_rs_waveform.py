"""Tests for Rohde & Schwarz .wv waveform export, active power normalization, and markers."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from uwb_hrp import (
    ARBExporter,
    ButterworthPulse,
    HRPConfig,
    HRPWaveformResult,
    MarkerTrack,
    NormalizationMode,
    PowerReference,
    RRCPulse,
    RSWaveformExporter,
    WaveformPowerMetrics,
    build_rs_markers,
    build_waveform_markers,
    compute_power_metrics,
    compute_rs_power_metrics,
    generate_waveform,
    prepare_arb_waveform,
    save_rs_waveform,
    to_rs_waveform,
)
from uwb_hrp.export.rs import RS_INT16_FULL_SCALE


def _parse_wv_raw(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    """Parse ASCII header tags and raw little-endian int16 I/Q samples from a .wv file."""
    raw = path.read_bytes()
    m_wf = re.search(rb"\{WAVEFORM-(\d+):#(.*)$", raw, flags=re.DOTALL)
    assert m_wf is not None, "Missing {WAVEFORM-...:#...} tag"
    byte_count = int(m_wf.group(1))
    payload_plus_close = m_wf.group(2)
    iq_bytes = payload_plus_close[: byte_count - 1]
    assert payload_plus_close[byte_count - 1 : byte_count] == b"}"

    header_bytes = raw[: m_wf.start()]
    tags: dict[str, str] = {}
    for match in re.finditer(rb"\{([A-Z0-9 _-]+):([^}]*)\}", header_bytes):
        key = match.group(1).decode("ascii")
        if key.startswith("EMPTYTAG-"):
            continue
        tags[key] = match.group(2).decode("ascii").strip()

    interleaved = np.frombuffer(iq_bytes, dtype="<i2")
    assert len(interleaved) % 2 == 0
    return tags, interleaved[0::2], interleaved[1::2]


def test_active_rms_invariant_to_idle_time_bprf_sp2() -> None:
    cfg = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=9,
        preamble_duration=64,
        sfd_number=2,
        psdu_length=20,
        sts_packet_config=2,
    )
    payload = b"\xba\xbe" * 10
    pulse = ButterworthPulse(samples_per_pulse=4)

    res_0 = generate_waveform(payload, cfg, pulse=pulse, idle_time=0.0)
    res_10us = generate_waveform(payload, cfg, pulse=pulse, idle_time=10e-6)
    res_500us = generate_waveform(payload, cfg, pulse=pulse, idle_time=500e-6)

    m_0 = compute_rs_power_metrics(res_0)
    m_10 = compute_rs_power_metrics(res_10us)
    m_500 = compute_rs_power_metrics(res_500us)

    # Active RMS offset, peak offset, and crest factor must be identical regardless of idle_time
    assert m_0.rms_offset_db == pytest.approx(m_10.rms_offset_db, abs=1e-12)
    assert m_0.rms_offset_db == pytest.approx(m_500.rms_offset_db, abs=1e-12)
    assert m_0.peak_offset_db == 0.0
    assert m_10.peak_offset_db == 0.0
    assert m_500.peak_offset_db == 0.0
    assert m_0.crest_factor_db == pytest.approx(m_500.crest_factor_db, abs=1e-12)

    # Duty cycle and avoided idle dilution match exact analytical formula -10*log10(duty_cycle)
    assert m_0.duty_cycle == pytest.approx(1.0, abs=1e-12)
    assert m_0.idle_dilution_db == pytest.approx(0.0, abs=1e-12)
    assert m_500.duty_cycle < 0.3
    assert m_500.idle_dilution_db == pytest.approx(-10.0 * math.log10(m_500.duty_cycle), abs=1e-12)
    assert m_500.total_rms_offset_db == pytest.approx(
        m_500.rms_offset_db + m_500.idle_dilution_db, abs=1e-12
    )


def test_multipacket_active_rms_invariant_to_idle_time() -> None:
    cfg = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=124.8,
        preamble_duration=32,
        sfd_number=2,
        psdu_length=16,
        sts_packet_config=1,
        sts_segment_length=32,
    )
    payloads = [b"\x11" * 16, b"\x22" * 16, b"\x33" * 16]
    pulse = RRCPulse(samples_per_pulse=4, rolloff=0.5)

    res_0 = generate_waveform(payloads, cfg, pulse=pulse, num_packets=3, idle_time=0.0)
    res_idle = generate_waveform(payloads, cfg, pulse=pulse, num_packets=3, idle_time=50e-6)

    m_0 = compute_rs_power_metrics(res_0)
    m_idle = compute_rs_power_metrics(res_idle)

    assert m_0.rms_offset_db == pytest.approx(m_idle.rms_offset_db, abs=1e-12)
    assert m_idle.active_samples_per_packet == m_0.active_samples_per_packet
    assert m_idle.total_samples == 3 * (m_0.active_samples_per_packet + res_idle.idle_samples)


def test_digital_backoff_and_power_references() -> None:
    cfg = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=249.6,
        preamble_duration=32,
        sfd_number=2,
        psdu_length=20,
        sts_packet_config=1,
        sts_segment_length=32,
    )
    res = generate_waveform(b"\xaa" * 20, cfg, pulse=ButterworthPulse(4), idle_time=20e-6)

    m_ref = compute_rs_power_metrics(res, power_reference="active", digital_backoff_db=0.0)
    m_bo = compute_rs_power_metrics(res, power_reference="active", digital_backoff_db=3.5)

    # Backoff shifts both RMS and peak offsets equally while crest factor stays invariant
    assert m_bo.peak_offset_db == pytest.approx(3.5, abs=1e-12)
    assert m_bo.rms_offset_db == pytest.approx(m_ref.rms_offset_db + 3.5, abs=1e-12)
    assert m_bo.crest_factor_db == pytest.approx(m_ref.crest_factor_db, abs=1e-12)

    # In HPRF 249.6 MHz, PSDU (249.6 MHz) has higher mean power (~3 dB lower rms_offset) than SHR
    ref_modes: tuple[PowerReference, ...] = ("active", "shr", "sync", "sts", "phr", "psdu", "all")
    by_mode = {mode: compute_rs_power_metrics(res, power_reference=mode) for mode in ref_modes}
    assert by_mode["psdu"].rms_offset_db < by_mode["shr"].rms_offset_db - 2.0
    assert by_mode["all"].rms_offset_db == pytest.approx(m_ref.total_rms_offset_db, abs=1e-12)
    assert by_mode["all"].idle_dilution_db == pytest.approx(0.0, abs=1e-12)


def test_power_reference_validation_errors() -> None:
    cfg_sp0 = HRPConfig(mode="BPRF", channel=9, psdu_length=10, sts_packet_config=0)
    res_sp0 = generate_waveform(b"\x55" * 10, cfg_sp0, pulse=ButterworthPulse(2))
    with pytest.raises(ValueError, match="without an STS field"):
        compute_rs_power_metrics(res_sp0, power_reference="sts")

    cfg_sp3 = HRPConfig(mode="BPRF", channel=9, sts_packet_config=3)
    res_sp3 = generate_waveform(None, cfg_sp3, pulse=ButterworthPulse(2))
    with pytest.raises(ValueError, match="without a PHR field"):
        compute_rs_power_metrics(res_sp3, power_reference="phr")
    with pytest.raises(ValueError, match="without a PSDU field"):
        compute_rs_power_metrics(res_sp3, power_reference="psdu")

    with pytest.raises(ValueError, match="Unsupported power_reference"):
        compute_rs_power_metrics(res_sp0, power_reference="invalid")  # type: ignore[arg-type]


def test_build_rs_markers_single_and_multipacket() -> None:
    cfg_sp0 = HRPConfig(mode="BPRF", channel=9, preamble_duration=16, psdu_length=10)
    res_sp0_no_idle = generate_waveform(b"\x12" * 10, cfg_sp0, pulse=ButterworthPulse(4))
    markers_sp0 = build_rs_markers(res_sp0_no_idle)
    assert set(markers_sp0.keys()) == {"marker_list_1", "marker_list_2", "marker_list_3"}
    assert markers_sp0["marker_list_1"] == [[0, 1]]
    assert markers_sp0["marker_list_2"][0] == [0, 0]
    assert markers_sp0["marker_list_2"][1] == [res_sp0_no_idle.indices.rmarker_chip * 4, 1]
    assert markers_sp0["marker_list_3"][0] == [0, 1]

    cfg_sp2 = HRPConfig(
        mode="BPRF",
        channel=9,
        preamble_duration=16,
        psdu_length=10,
        sts_packet_config=2,
    )
    res_sp2 = generate_waveform(
        [b"\x12" * 10, b"\x34" * 10],
        cfg_sp2,
        pulse=ButterworthPulse(4),
        num_packets=2,
        idle_time=10e-6,
    )
    markers_sp2 = build_rs_markers(res_sp2, pulse_samples=8)
    assert set(markers_sp2.keys()) == {
        "marker_list_1",
        "marker_list_2",
        "marker_list_3",
        "marker_list_4",
    }

    active = res_sp2.indices.total_chips * 4
    stride = active + res_sp2.idle_samples
    assert markers_sp2["marker_list_1"] == [
        [0, 1],
        [active, 0],
        [stride, 1],
        [stride + active, 0],
    ]
    rm = res_sp2.indices.rmarker_chip * 4
    assert markers_sp2["marker_list_2"] == [
        [0, 0],
        [rm, 1],
        [rm + 8, 0],
        [stride + rm, 1],
        [stride + rm + 8, 0],
    ]
    assert markers_sp2["marker_list_3"] == [
        [0, 1],
        [8, 0],
        [stride, 1],
        [stride + 8, 0],
    ]
    assert res_sp2.indices.sts2 is not None
    sts_s = res_sp2.indices.sts2.start * 4
    sts_e = res_sp2.indices.sts2.stop * 4
    assert markers_sp2["marker_list_4"] == [
        [0, 0],
        [sts_s, 1],
        [sts_e, 0],
        [stride + sts_s, 1],
        [stride + sts_e, 0],
    ]


def test_save_rs_waveform_binary_and_rswaveform_roundtrip(tmp_path: Path) -> None:
    RsWaveform = pytest.importorskip("RsWaveform")

    cfg = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=9,
        preamble_duration=64,
        sfd_number=2,
        psdu_length=20,
        sts_packet_config=2,
    )
    res = generate_waveform(
        b"\xba\xbe" * 10,
        cfg,
        pulse=ButterworthPulse(samples_per_pulse=4),
        idle_time=100e-6,
    )

    out_path = tmp_path / "bprf_sp2.wv"
    metrics = save_rs_waveform(res, out_path, config=cfg)

    # 1. Direct binary & ASCII header verification (exact 16-bit precision)
    tags, i_int16, q_int16 = _parse_wv_raw(out_path)
    assert tags["TYPE"] == "SMU-WV"
    assert tags["COPYRIGHT"] == "uwb_hrp"
    assert tags["COMMENT"] == "IEEE 802.15.4 HRP UWB BPRF CH9 SP2"
    assert float(tags["CLOCK"]) == pytest.approx(res.sample_rate)
    assert int(tags["SAMPLES"]) == len(res.waveform)
    assert int(tags["CONTROL LENGTH"]) == len(res.waveform)
    assert "MARKER LIST 1" in tags
    assert "MARKER LIST 2" in tags
    assert "MARKER LIST 3" in tags
    assert "MARKER LIST 4" in tags
    assert "REFLEVEL" not in tags

    hdr_rms, hdr_peak = (float(x) for x in tags["LEVEL OFFS"].split(","))
    assert hdr_rms == pytest.approx(metrics.rms_offset_db, abs=1e-6)
    assert hdr_peak == pytest.approx(0.0, abs=1e-6)

    # Symmetric 16-bit full scale: max |I| == 32767, never -32768, Q == 0 everywhere
    assert len(i_int16) == len(res.waveform)
    assert int(np.max(np.abs(i_int16.astype(np.int32)))) == int(RS_INT16_FULL_SCALE)
    assert int(np.min(i_int16)) >= -int(RS_INT16_FULL_SCALE)
    assert np.all(q_int16 == 0)

    # Idle tail is exact zero
    active_len = metrics.active_samples_per_packet
    assert np.all(i_int16[active_len:] == 0)

    # Simulate R&S instrument playback at RF Level = -50.0 dBm:
    # P_active_dBm = RF_level + rms_offset + 10*log10(mean(|z_int16_active / 32767|^2))
    rf_level_dbm = -50.0
    z_active_norm = i_int16[:active_len].astype(np.float64) / RS_INT16_FULL_SCALE
    actual_active_power_dbm = (
        rf_level_dbm + hdr_rms + 10.0 * math.log10(float(np.mean(z_active_norm**2)))
    )
    assert actual_active_power_dbm == pytest.approx(rf_level_dbm, abs=1e-3)

    # 2. Reload via RsWaveform itself and verify all metadata round-trips cleanly
    loaded = RsWaveform.RsWaveform(file=out_path)
    assert loaded.meta[0].clock == pytest.approx(res.sample_rate)
    assert loaded.meta[0].samples == len(res.waveform)
    assert loaded.meta[0].rms == pytest.approx(metrics.rms_offset_db, abs=1e-6)
    assert loaded.meta[0].peak == pytest.approx(0.0, abs=1e-6)
    assert loaded.meta[0].comment == "IEEE 802.15.4 HRP UWB BPRF CH9 SP2"
    assert loaded.meta[0].control_length == len(res.waveform)
    assert set(loaded.meta[0].marker.keys()) == {
        "marker_list_1",
        "marker_list_2",
        "marker_list_3",
        "marker_list_4",
    }


def test_to_rs_waveform_options_and_custom_markers(tmp_path: Path) -> None:
    pytest.importorskip("RsWaveform")

    cfg = HRPConfig(mode="802.15.4a", channel=5, mean_prf=15.6, preamble_duration=16, psdu_length=8)
    res = generate_waveform(b"\x01" * 8, cfg, pulse=ButterworthPulse(2), idle_time=5e-6)

    # Without markers and with explicit reflevel
    wv_plain = to_rs_waveform(
        res,
        include_markers=False,
        comment="Custom comment 1.0",
        copyright="Lab Team",
        reflevel_dbm=-50.0,
    )
    assert wv_plain.meta[0].control_length is None
    assert wv_plain.meta[0].marker == {}
    assert wv_plain.meta[0].reflevel == pytest.approx(-50.0)
    assert wv_plain.meta[0].comment == "Custom comment 1.0"
    assert wv_plain.meta[0].copyright == "Lab Team"

    # With custom markers and digital backoff
    out = tmp_path / "custom.wv"
    metrics = save_rs_waveform(
        res,
        out,
        digital_backoff_db=6.0,
        markers={1: [(10, 1), (50, 0)]},
    )
    tags, i_int16, _ = _parse_wv_raw(out)
    assert tags["MARKER LIST 1"] == "0:0;10:1;50:0"
    assert "MARKER LIST 2" not in tags
    expected_peak_int = round(RS_INT16_FULL_SCALE * 10.0 ** (-6.0 / 20.0))
    assert abs(int(np.max(np.abs(i_int16.astype(np.int32)))) - expected_peak_int) <= 1
    assert metrics.peak_offset_db == pytest.approx(6.0, abs=1e-12)


def test_missing_rswaveform_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = HRPConfig(mode="BPRF", channel=9, preamble_duration=16, psdu_length=8)
    res = generate_waveform(b"\xaa" * 8, cfg, pulse=ButterworthPulse(2))

    monkeypatch.setitem(sys.modules, "RsWaveform", None)

    # Pure-Python helpers must still work without RsWaveform installed
    metrics = compute_rs_power_metrics(res)
    assert metrics.rms_offset_db > 0.0
    markers = build_rs_markers(res)
    assert "marker_list_1" in markers

    with pytest.raises(ImportError, match=r"uwb-hrp\[rs\]"):
        to_rs_waveform(res)


def test_rs_waveform_input_validation_errors() -> None:
    cfg = HRPConfig(mode="BPRF", channel=9, preamble_duration=16, psdu_length=8)
    res = generate_waveform(b"\xaa" * 8, cfg, pulse=ButterworthPulse(2))

    with pytest.raises(ValueError, match="digital_backoff_db"):
        compute_rs_power_metrics(res, digital_backoff_db=-1.0)

    with pytest.raises(ValueError, match="all-zero"):
        compute_rs_power_metrics(replace(res, waveform=np.zeros_like(res.waveform)))

    bad_nan = res.waveform.copy()
    bad_nan[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        compute_rs_power_metrics(replace(res, waveform=bad_nan))

    with pytest.raises(ValueError, match="empty waveform"):
        compute_rs_power_metrics(replace(res, waveform=np.empty(0, dtype=np.float64)))

    with pytest.raises(ValueError, match="1D waveform"):
        compute_rs_power_metrics(replace(res, waveform=np.zeros((2, 4), dtype=np.float64)))

    with pytest.raises(ValueError, match="pulse_samples"):
        build_rs_markers(res, pulse_samples=0)

    with pytest.raises(ValueError, match="must not contain"):
        to_rs_waveform(res, comment="bad {brace}")

    with pytest.raises(ValueError, match="Marker number"):
        to_rs_waveform(res, markers={5: [(0, 1)]})

    with pytest.raises(ValueError, match="strictly increasing"):
        to_rs_waveform(res, markers={1: [(10, 1), (10, 0)]})

    with pytest.raises(ValueError, match="out of range"):
        to_rs_waveform(res, markers={1: [(len(res.waveform), 1)]})


def test_vendor_neutral_rms_normalization_and_markers() -> None:
    cfg = HRPConfig(
        mode="BPRF",
        channel=9,
        preamble_duration=16,
        psdu_length=10,
        sts_packet_config=1,
    )
    res = generate_waveform(b"\x5a" * 10, cfg, pulse=ButterworthPulse(4), idle_time=20e-6)

    prep_peak = prepare_arb_waveform(res, normalization="peak")
    prep_rms = prepare_arb_waveform(res, normalization="rms")

    # Peak-normalized: max(|iq|) == 1.0, peak_offset_db == 0.0
    assert float(np.max(np.abs(prep_peak.iq))) == pytest.approx(1.0, abs=1e-12)
    assert prep_peak.metrics.peak_offset_db == 0.0
    assert prep_peak.metrics.normalization == "peak"

    # RMS-normalized: active RMS == 1.0, rms_offset_db == 0.0
    active_len = prep_rms.metrics.active_samples_per_packet
    active_rms = math.sqrt(float(np.mean(np.abs(prep_rms.iq[:active_len]) ** 2)))
    assert active_rms == pytest.approx(1.0, abs=1e-12)
    assert prep_rms.metrics.rms_offset_db == 0.0
    assert prep_rms.metrics.normalization == "rms"
    assert prep_rms.metrics.peak_offset_db == pytest.approx(
        -prep_peak.metrics.crest_factor_db, abs=1e-12
    )
    # Crest factor (PAPR) is invariant across normalization modes
    assert prep_rms.metrics.crest_factor_db == pytest.approx(
        prep_peak.metrics.crest_factor_db, abs=1e-12
    )

    # Semantic markers: bitmask bits 0..3 match individual boolean masks
    wm = build_waveform_markers(res, pulse_samples=8)
    bitmask = wm.to_bitmask()
    assert bitmask.dtype == np.uint8
    assert len(bitmask) == len(res.waveform)
    assert np.array_equal((bitmask & 0x01) != 0, wm.active_burst.to_mask())
    assert wm.rmarker is not None
    assert np.array_equal((bitmask & 0x02) != 0, wm.rmarker.to_mask())
    assert wm.packet_start is not None
    assert np.array_equal((bitmask & 0x04) != 0, wm.packet_start.to_mask())
    assert wm.sts_gate is not None
    assert np.array_equal((bitmask & 0x08) != 0, wm.sts_gate.to_mask())

    # MarkerTrack coalesces contiguous intervals cleanly
    coalesced = MarkerTrack.from_intervals("test", [(0, 10), (10, 20), (30, 40)], 50)
    assert coalesced.intervals == ((0, 20), (30, 40))
    assert coalesced.to_transitions() == [[0, 1], [20, 0], [30, 1], [40, 0]]
    empty_track = MarkerTrack.from_intervals("empty", [], 10)
    assert empty_track.to_transitions() == [[0, 0]]
    assert not np.any(empty_track.to_mask())

    with pytest.raises(ValueError, match="Unsupported normalization"):
        compute_power_metrics(res, normalization="invalid")  # type: ignore[arg-type]


def test_custom_arb_exporter_subclass_and_rs_exporter_class(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class MockRMSVendorExporter(ARBExporter):
        """Example third-party exporter (e.g. LitePoint-style RMS-normalized binary)."""

        @property
        def normalization_mode(self) -> NormalizationMode:
            return "rms"

        def save(
            self,
            result: HRPWaveformResult,
            path: str | Path,
            *,
            config: HRPConfig | None = None,
        ) -> WaveformPowerMetrics:
            del config
            prepared = self.prepare(result)
            np.savez(
                path,
                iq=prepared.iq,
                sample_rate=prepared.sample_rate,
                markers=prepared.markers.to_bitmask(),
            )
            return prepared.metrics

    cfg = HRPConfig(mode="BPRF", channel=9, preamble_duration=16, psdu_length=10)
    res = generate_waveform(b"\x3c" * 10, cfg, pulse=ButterworthPulse(2), idle_time=15e-6)

    vendor_out = tmp_path / "mock_vendor.npz"
    vendor_metrics = MockRMSVendorExporter().save(res, vendor_out, config=cfg)
    assert vendor_metrics.normalization == "rms"
    assert vendor_metrics.rms_offset_db == 0.0
    loaded_npz = np.load(vendor_out)
    assert float(loaded_npz["sample_rate"]) == pytest.approx(res.sample_rate)
    assert len(loaded_npz["iq"]) == len(res.waveform)
    assert loaded_npz["markers"].dtype == np.uint8

    # RSWaveformExporter class API directly
    pytest.importorskip("RsWaveform")
    rs_out = tmp_path / "class_api.wv"
    rs_metrics = RSWaveformExporter(digital_backoff_db=1.5).save(res, rs_out, config=cfg)
    assert rs_metrics.normalization == "peak"
    assert rs_metrics.peak_offset_db == pytest.approx(1.5, abs=1e-12)
    assert rs_metrics.crest_factor_db == pytest.approx(vendor_metrics.crest_factor_db, abs=1e-12)
