"""Export IEEE 802.15.4-2024 HRP UWB waveforms to Rohde & Schwarz .wv files."""

from __future__ import annotations

import tempfile
from pathlib import Path

from uwb_hrp import (
    ButterworthPulse,
    HRPConfig,
    build_rs_markers,
    generate_waveform,
    save_rs_waveform,
)


def main() -> None:
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

    res_no_idle = generate_waveform(payload, cfg, pulse=pulse, idle_time=0.0)
    res_with_idle = generate_waveform(payload, cfg, pulse=pulse, idle_time=100e-6)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_no_idle = Path(tmpdir) / "bprf_sp2_no_idle.wv"
        out_with_idle = Path(tmpdir) / "bprf_sp2_100us_idle.wv"

        m_no_idle = save_rs_waveform(res_no_idle, out_no_idle, config=cfg)
        m_with_idle = save_rs_waveform(res_with_idle, out_with_idle, config=cfg)

        print("=== Rohde & Schwarz .wv Export (BPRF SP2, CH9) ===")
        print(
            f"No idle   -> LEVEL OFFS=({m_no_idle.rms_offset_db:.4f}, "
            f"{m_no_idle.peak_offset_db:.4f}) dB | duty={m_no_idle.duty_cycle:.3f} | "
            f"size={out_no_idle.stat().st_size:,} bytes"
        )
        print(
            f"100us idle-> LEVEL OFFS=({m_with_idle.rms_offset_db:.4f}, "
            f"{m_with_idle.peak_offset_db:.4f}) dB | duty={m_with_idle.duty_cycle:.3f} | "
            f"avoided idle dilution={m_with_idle.idle_dilution_db:.4f} dB | "
            f"size={out_with_idle.stat().st_size:,} bytes"
        )

        # Verify active burst power when instrument RF level is set to -50.0 dBm
        target_rf_dbm = -50.0
        pep_dbm = target_rf_dbm + m_with_idle.crest_factor_db
        print(
            f"At RF Level={target_rf_dbm:.1f} dBm: active burst RMS={target_rf_dbm:.2f} dBm, "
            f"PEP={pep_dbm:.2f} dBm (Crest Factor={m_with_idle.crest_factor_db:.2f} dB)"
        )

        markers = build_rs_markers(res_with_idle)
        print("Auto-generated markers:")
        for key, transitions in markers.items():
            print(f"  {key}: {transitions}")


if __name__ == "__main__":
    main()
