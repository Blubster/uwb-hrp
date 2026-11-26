# Pulse Shaping & Compliance

`uwb-hrp` decouples baseband pulse shaping from PHY chip generation via [PulseConfig][uwb_hrp.PulseConfig] and [PulseShaper][uwb_hrp.PulseShaper]. Every pulse configuration automatically adapts to the target channel's bandwidth and timing parameters (`Tp`, `Tw`) from Tables 16-27 and 16-28.

## Built-In Pulse Shapes

### `ButterworthPulse` (Default)

[ButterworthPulse][uwb_hrp.ButterworthPulse] implements a causal digital baseband lowpass Butterworth IIR filter (default `order=4`, `samples_per_pulse=4`), which is the complex-baseband equivalent of the 8th-order RF bandpass Butterworth filter in IEEE Std 802.15.4-2024 Figure 16-23.

- Applies direct recursive IIR filtering (`scipy.signal.lfilter`) during waveform shaping to avoid impulse truncation error.
- Cutoff frequency is set to half the double-sided 3 dB RF channel bandwidth (`bandwidth_hz / 2`), e.g. `249.6 MHz` for narrowband channels (`499.2 MHz` RF bandwidth).

### `RRCPulse`

[RRCPulse][uwb_hrp.RRCPulse] implements the closed-form Section 16.4.4 Root Raised Cosine pulse \(r(t)\) with analytical handling of the removable singularities at \(t = 0\) and \(t = \pm T_p / (4\beta)\), normalized to unit energy:

\[
r(t) = \frac{4\beta}{\pi\sqrt{T_p}} \cdot \frac{\cos\!\left((1+\beta)\pi\frac{t}{T_p}\right) + \frac{\sin\!\left((1-\beta)\pi\frac{t}{T_p}\right)}{4\beta\frac{t}{T_p}}}{1 - \left(4\beta\frac{t}{T_p}\right)^2}
\]

- Default roll-off \(\beta = 0.5\) (reference pulse) or \(\beta = 0.45\).
- Default span `8` chip periods (covering \(\pm 4 T_c\), exceeding the \(\pm 3 T_c\) minimum).
- Peak-aligned via `fftconvolve` so sample `k * samples_per_pulse` aligns with chip `k`'s main peak.

!!! tip "Oversampling on wideband channels"
    Wideband channels (`4`, `7`, `11`, `15`) have RF bandwidths up to `1354.97 MHz` (`Tp = 0.74 ns`). Use `samples_per_pulse >= 4` (or `8` for crisp time-domain inspection and full PSD stopband resolution above `0.8 / Tp`).

## Automated Sections 16.4.4 & 16.4.5 Compliance Verification

Call [PulseShaper.verify_compliance][uwb_hrp.PulseShaper.verify_compliance] to check any built-in or custom pulse against IEEE Std 802.15.4-2024 transmit pulse requirements:

```python
from uwb_hrp import HRPConfig, PulseShaper, RRCPulse, generate_waveform

cfg = HRPConfig(mode="BPRF", channel=9, psdu_length=20)
shaper = PulseShaper.for_hrp_config(cfg, RRCPulse(samples_per_pulse=8, rolloff=0.45))

report = shaper.verify_compliance()
print(report)
assert report.compliant

result = generate_waveform(b"\xba\xbe" * 10, cfg, pulse=shaper)
```

[PulseComplianceReport][uwb_hrp.PulseComplianceReport] evaluates three checks:

1. **Section 16.4.4 Cross-Correlation (mandatory):** Bandlimited-interpolated cross-correlation \(|\phi(\tau)|\) against the reference RRC (\(\beta = 0.5\)) must have continuous main-lobe width at \(|\phi(\tau)| \ge 0.8\) of at least \(T_w\) (`0.5 ns` narrowband, `0.2 ns` wideband) and maximum sidelobe peak \(\le 0.3\).
2. **Section 16.4.5 Transmit PSD Mask (mandatory):** Relative PSD of the actual discrete-time pulse taps must stay `< -10 dBr` for \(0.65/T_p < |f| < 0.8/T_p\) and `< -18 dBr` for \(|f| > 0.8/T_p\).
3. **Figure 16-24 Minimum-Precursor Mask (informational):** Time-domain amplitude envelope check relative to the first main peak (`min_precursor_mask_compliant`).

## Writing a Custom `PulseConfig` Subclass

To plug in any custom pulse shape (e.g., Gaussian, raised cosine, or measured hardware impulse response), subclass [PulseConfig][uwb_hrp.PulseConfig] as a frozen dataclass and implement `build_taps`:

```python
from dataclasses import dataclass

import numpy as np

from uwb_hrp import HRPConfig, PulseConfig, PulseShaper, generate_waveform
from uwb_hrp.tables import PEAK_PRF_HZ, ChannelSpec


@dataclass(frozen=True)
class GaussianPulse(PulseConfig):
    """Custom zero-phase Gaussian baseband pulse."""

    sigma_tp: float = 0.32
    span_chips: int = 8

    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        fs = PEAK_PRF_HZ * self.samples_per_pulse
        tp_s = channel_spec.pulse_duration_tp_ns * 1e-9
        half = (self.span_chips * self.samples_per_pulse) // 2
        t = np.arange(-half, half + 1, dtype=np.float64) / fs
        sigma = self.sigma_tp * tp_s
        taps = np.exp(-0.5 * (t / sigma) ** 2)
        return taps / np.linalg.norm(taps)


cfg = HRPConfig(mode="BPRF", channel=9, psdu_length=20)
shaper = PulseShaper.for_hrp_config(cfg, GaussianPulse(samples_per_pulse=8))
print("Compliant:", shaper.verify_compliance().compliant)
res = generate_waveform(b"\xaa" * 20, cfg, pulse=shaper)
```

If your custom pulse uses recursive IIR filtering instead of FIR convolution, override [PulseConfig.build_filter][uwb_hrp.PulseConfig.build_filter] as well (see [ButterworthPulse][uwb_hrp.ButterworthPulse] for a reference implementation).
