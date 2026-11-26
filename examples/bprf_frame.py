"""BPRF frame generation example (SP0 and SP1 with STS, FCS-16, and trailing idle time)."""

from __future__ import annotations

from uwb_hrp import (
    ButterworthPulse,
    HRPConfig,
    compute_fcs16,
    generate_waveform,
)


def main() -> None:
    # Build a 20-byte PSDU with an IEEE 802.15.4 Section 7.2.11 16-bit FCS
    mac_payload = b"\xba\xbe" * 9
    psdu = mac_payload + compute_fcs16(mac_payload)

    # 1. BPRF SP0 (no STS) on channel 9 with trailing idle time
    cfg_sp0 = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=9,
        preamble_duration=64,
        sfd_number=2,
        phr_data_rate=0.85,
        psdu_length=len(psdu),
        sts_packet_config=0,
    )
    res_sp0 = generate_waveform(
        psdu,
        cfg_sp0,
        pulse=ButterworthPulse(samples_per_pulse=4),
        idle_time=1e-6,
    )
    print("=== BPRF SP0 (with 1 us trailing idle time) ===")
    print(f"Channel {cfg_sp0.channel} ({cfg_sp0.center_frequency / 1e6:.1f} MHz)")
    print(f"Chips: {len(res_sp0.chips)}, Waveform samples: {len(res_sp0.waveform)}")
    print(f"Sample rate: {res_sp0.sample_rate / 1e6:.2f} MHz, Idle samples: {res_sp0.idle_samples}")
    print(f"RMARKER chip: {res_sp0.indices.rmarker_chip}")
    print(f"1-based field ranges: {res_sp0.indices.to_1based_ranges()}\n")

    # 2. BPRF SP1 (STS between SFD and PHR) with high-rate PHR (6.81 Mb/s)
    cfg_sp1 = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=10,
        preamble_duration=64,
        sfd_number=2,
        phr_data_rate=6.81,
        psdu_length=len(psdu),
        ranging=True,
        sts_packet_config=1,
        num_sts_segments=1,
        sts_segment_length=64,
    )
    res_sp1 = generate_waveform(psdu, cfg_sp1)
    print("=== BPRF SP1 (STS + 6.81 Mb/s PHR) ===")
    print(f"Chips: {len(res_sp1.chips)}, Waveform samples: {len(res_sp1.waveform)}")
    print(f"RMARKER chip: {res_sp1.indices.rmarker_chip}")
    print(f"SRMARKER chips: {res_sp1.indices.srmarker_chips}")
    print(f"1-based field ranges: {res_sp1.indices.to_1based_ranges()}")


if __name__ == "__main__":
    main()
