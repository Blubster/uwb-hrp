"""HPRF frame examples covering K=3 / K=7, multi-segment SP2 STS, and extended PSDU length."""

from __future__ import annotations

from uwb_hrp import HRPConfig, RRCPulse, generate_chips, generate_waveform


def main() -> None:
    # 1. HPRF 124.8 MHz (K=3), SP2 with 2 STS segments and an extra STS gap
    psdu_short = b"\xba\xbe" * 10
    cfg_sp2 = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=124.8,
        code_index=25,
        preamble_duration=64,
        sfd_number=2,
        psdu_length=len(psdu_short),
        constraint_length=3,
        sts_packet_config=2,
        num_sts_segments=2,
        sts_segment_length=64,
        extra_sts_gap=16,  # 16 * 4 = 64 extra chips between PSDU and STS
        extra_sts_gap_index=1,
    )
    res_sp2 = generate_waveform(psdu_short, cfg_sp2, pulse=RRCPulse(samples_per_pulse=4))
    print("=== HPRF 124.8 MHz (K=3, SP2, 2 STS segments, extra gap) ===")
    print(f"Data rate: {cfg_sp2.data_rate} Mb/s, PHR rate: {cfg_sp2.phr_data_rate} Mb/s")
    print(f"Chips: {len(res_sp2.chips)}, Waveform samples: {len(res_sp2.waveform)}")
    print(f"SRMARKER chips: {res_sp2.indices.srmarker_chips}")
    print(f"1-based field ranges: {res_sp2.indices.to_1based_ranges()}\n")

    # 2. HPRF 249.6 MHz (K=7, 31.2 Mb/s) with an extended 2048-byte PSDU (Table 16-16)
    psdu_long = b"\xab" * 2048
    cfg_k7 = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=249.6,
        code_index=27,
        preamble_duration=32,
        sfd_number=3,
        psdu_length=len(psdu_long),
        constraint_length=7,
        sts_packet_config=0,
    )
    chips_k7 = generate_chips(psdu_long, cfg_k7)
    print("=== HPRF 249.6 MHz (K=7, 31.2 Mb/s, 2048-byte PSDU) ===")
    print(f"PSDU size extension mode: {cfg_k7.psdu_size_extension}")
    print(f"Chips: {len(chips_k7)} (duration: {len(chips_k7) / cfg_k7.peak_prf * 1e6:.2f} us)\n")

    # 3. HPRF SP3 (ranging-only, no PHR/PSDU)
    cfg_sp3 = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=124.8,
        code_index=29,
        preamble_duration=64,
        sfd_number=2,
        sts_packet_config=3,
        num_sts_segments=1,
        sts_segment_length=128,
    )
    chips_sp3 = generate_chips(None, cfg_sp3)
    print("=== HPRF SP3 (STS-only ranging frame) ===")
    print(f"Chips: {len(chips_sp3)}")


if __name__ == "__main__":
    main()
