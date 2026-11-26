"""Legacy IEEE 802.15.4a HRP UWB examples across 3.9, 15.6, and 62.4 MHz mean PRFs."""

from __future__ import annotations

from uwb_hrp import ButterworthPulse, HRPConfig, generate_waveform


def main() -> None:
    payload = b"\x12\x34\x56\x78" * 5  # 20 bytes

    configs = [
        (
            "3.9 MHz PRF / 0.11 Mb/s (long 64-symbol SFD, 4.03 MHz preamble PRF)",
            HRPConfig(
                mode="802.15.4a",
                channel=1,
                mean_prf=3.9,
                preamble_mean_prf=4.03,
                data_rate=0.11,
                code_index=1,
                preamble_duration=16,
                sfd_number="long",
                psdu_length=len(payload),
            ),
            ButterworthPulse(samples_per_pulse=4),
        ),
        (
            "15.6 MHz PRF / 27.24 Mb/s (Viterbi rate 1, wideband channel 4)",
            HRPConfig(
                mode="802.15.4a",
                channel=4,
                mean_prf=15.6,
                preamble_mean_prf=16.1,
                data_rate=27.24,
                code_index=7,
                preamble_duration=64,
                psdu_length=len(payload),
            ),
            ButterworthPulse(samples_per_pulse=4),  # Wideband BW=1331.2 MHz needs spp >= 3
        ),
        (
            "62.4 MHz PRF / 6.81 Mb/s (length-127 preamble code)",
            HRPConfig(
                mode="802.15.4a",
                channel=9,
                mean_prf=62.4,
                data_rate=6.81,
                code_index=9,
                preamble_duration=64,
                psdu_length=len(payload),
            ),
            ButterworthPulse(samples_per_pulse=4),
        ),
    ]

    for label, cfg, pulse in configs:
        res = generate_waveform(payload, cfg, pulse=pulse)
        duration_us = len(res.chips) / cfg.peak_prf * 1e6
        print(f"=== {label} ===")
        print(
            f"Channel {cfg.channel} (BW={cfg.bandwidth / 1e6:.2f} MHz), "
            f"code={cfg.code_index}, SFD={cfg.sfd_number!r}"
        )
        print(
            f"Chips: {len(res.chips)} ({duration_us:.2f} us), "
            f"Waveform samples: {len(res.waveform)} @ {res.sample_rate / 1e6:.2f} MHz\n"
        )


if __name__ == "__main__":
    main()
