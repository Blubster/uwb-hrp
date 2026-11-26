"""Multi-packet burst generation with per-packet idle time and STS counter advancement."""

from __future__ import annotations

import numpy as np

from uwb_hrp import ButterworthPulse, HRPConfig, generate_waveform


def main() -> None:
    cfg = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=9,
        preamble_duration=64,
        sfd_number=2,
        psdu_length=20,
        sts_packet_config=1,
        num_sts_segments=1,
        sts_segment_length=64,
    )

    payloads = [b"\xba\xbe" * 10, b"\xca\xfe" * 10, b"\xde\xad" * 10]
    pulse = ButterworthPulse(samples_per_pulse=4)

    res = generate_waveform(
        payloads,
        cfg,
        pulse=pulse,
        num_packets=len(payloads),
        idle_time=2e-6,
    )

    print("=== 3-Packet BPRF SP1 Burst ===")
    print(f"Chips array shape: {res.chips.shape} (chips_per_packet x num_packets)")
    print(f"Waveform samples:  {len(res.waveform)} @ {res.sample_rate / 1e6:.2f} MHz")
    print(f"Idle per packet:   {res.idle_samples} samples (2.0 us)")
    assert res.next_sts_v is not None
    print(f"Next STS V:        {res.next_sts_v.hex()}")

    sts_slice = res.indices.sts1
    assert sts_slice is not None
    sts_unique = not np.array_equal(res.chips[sts_slice, 0], res.chips[sts_slice, 1])
    print(f"STS differs between packets 0 and 1 (counter advanced): {sts_unique}")


if __name__ == "__main__":
    main()
