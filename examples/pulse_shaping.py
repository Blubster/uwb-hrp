"""Pulse shaping comparison, Section 16.4.4 compliance checks, and custom PulseConfig subclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uwb_hrp import (
    ButterworthPulse,
    HRPConfig,
    PulseConfig,
    PulseShaper,
    RRCPulse,
    generate_waveform,
)
from uwb_hrp.tables import ChannelSpec


@dataclass(frozen=True)
class GaussianPulse(PulseConfig):
    """Example user-defined pulse shape extending PulseConfig."""

    sigma_chips: float = 0.35
    span_chips: int = 8

    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        del channel_spec
        half = (self.span_chips * self.samples_per_pulse) // 2
        t_chips = np.arange(-half, half + 1, dtype=np.float64) / self.samples_per_pulse
        taps = np.exp(-0.5 * (t_chips / self.sigma_chips) ** 2)
        return taps / float(np.linalg.norm(taps))


def main() -> None:
    cfg = HRPConfig(mode="BPRF", channel=9, preamble_duration=64, psdu_length=20)
    payload = b"\xba\xbe" * 10

    pulses: list[PulseConfig] = [
        ButterworthPulse(samples_per_pulse=4, order=4),
        RRCPulse(samples_per_pulse=4, rolloff=0.5),
        RRCPulse(samples_per_pulse=8, rolloff=0.45),
        GaussianPulse(samples_per_pulse=4, sigma_chips=0.35),
    ]

    for pulse_cfg in pulses:
        shaper = PulseShaper.for_hrp_config(cfg, pulse_cfg)
        report = shaper.verify_compliance()
        res = generate_waveform(payload, cfg, pulse=shaper)
        print(
            f"{type(pulse_cfg).__name__:18s} (spp={pulse_cfg.samples_per_pulse}): "
            f"compliant={report.compliant!s:5s} | peak={report.peak_correlation:.4f} | "
            f"Tw={report.main_lobe_width_ns:.3f} ns (min {report.required_tw_ns} ns) | "
            f"sidelobe={report.max_sidelobe:.4f} | samples={len(res.waveform)}"
        )


if __name__ == "__main__":
    main()
