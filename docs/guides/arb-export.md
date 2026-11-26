# ARB Waveform Export

The `uwb_hrp.export` package provides a two-layer architecture for arbitrary waveform generator (ARB) playback on RF signal generators:

1. **Vendor-neutral core (`uwb_hrp.export.base`):** Active-only power normalization, exact `float64` level-offset metrics, semantic marker tracks, and the [ARBExporter][uwb_hrp.export.ARBExporter] abstract base class. Requires no vendor SDK.
2. **Instrument backends:** Built-in Rohde & Schwarz `.wv` (`SMU-WV`) exporter ([RSWaveformExporter][uwb_hrp.RSWaveformExporter], optional `uwb-hrp[rs]` extra), easily extended to Keysight, LitePoint, NI, or custom formats.

## Why Active-Power Normalization Matters

UWB packets are commonly played back in a loop with substantial silent `idle_time` between bursts (for example, a 200 µs packet followed by 9.8 ms of silence = 2% duty cycle). If an ARB computes RMS over the *entire* waveform including zeros, the reported RMS drops by \(-10\log_{10}(\text{duty\_cycle}) \approx 17\text{ dB}\) (`idle_dilution_db`), causing the signal generator to overdrive the active burst by 17 dB!

`uwb-hrp` solves this by computing `rms_offset_db` over `power_reference="active"` by default (all active packet samples, excluding `idle_time` zeros):

\[
\text{rms\_offset\_db} = -10\log_{10}\!\left(\operatorname{mean}\!\left(|z_{\text{ref}}|^2\right)\right),
\qquad
\text{peak\_offset\_db} = -20\log_{10}\!\left(\max |z|\right)
\]

\[
\text{crest\_factor\_db} = \text{rms\_offset\_db} - \text{peak\_offset\_db},
\qquad
\text{idle\_dilution\_db} = \text{total\_rms\_offset\_db} - \text{rms\_offset\_db}
\]

Supported `power_reference` regions:

| `power_reference` | Samples Included in RMS Reference |
| --- | --- |
| `"active"` *(default)* | All active packet samples (excludes `idle_time`) |
| `"shr"` | `SYNC` + `SFD` header |
| `"sync"` | Preamble `SYNC` only |
| `"sts"` | STS field (`SP1`/`SP2`/`SP3`) |
| `"phr"` | PHR field (`SP0`/`SP1`/`SP2`) |
| `"psdu"` | PSDU field (non-empty `SP0`/`SP1`/`SP2`) |
| `"all"` | Entire waveform including `idle_time` zeros |

## Semantic Marker Tracks

[build_waveform_markers][uwb_hrp.export.build_waveform_markers] builds up to four sample-synchronous [MarkerTrack][uwb_hrp.export.MarkerTrack] objects:

| Track | R&S Channel | Bitmask (`to_bitmask()`) | Semantics |
| --- | --- | --- | --- |
| `active_burst` | Marker 1 | Bit 0 (`0x01`) | HIGH during active UWB packets, LOW during `idle_time` (route to RF Blanking / ALC Hold) |
| `rmarker` | Marker 2 | Bit 1 (`0x02`) | Rising-edge trigger pulse at exact IEEE 802.15.4-2024 `RMARKER` (first chip after SFD) |
| `packet_start` | Marker 3 | Bit 2 (`0x04`) | Rising-edge trigger pulse at sample 0 of every packet |
| `sts_gate` | Marker 4 | Bit 3 (`0x08`) | HIGH during the STS field (`SP1`/`SP2`/`SP3` only; `None` for `SP0`) |

Every [MarkerTrack][uwb_hrp.export.MarkerTrack] converts directly to either transition tables (`.to_transitions()`) or per-sample boolean arrays (`.to_mask()`).

## Rohde & Schwarz `.wv` Export

Install the optional `rs` extra (`uv sync --extra rs` or `pip install "uwb-hrp[rs]"`), then export directly with [save_rs_waveform][uwb_hrp.save_rs_waveform] or [RSWaveformExporter][uwb_hrp.RSWaveformExporter]:

```python
from uwb_hrp import ButterworthPulse, HRPConfig, generate_waveform, save_rs_waveform

cfg = HRPConfig(mode="BPRF", channel=9, sfd_number=2, psdu_length=20, sts_packet_config=2)
res = generate_waveform(b"\xba\xbe" * 10, cfg, pulse=ButterworthPulse(4), idle_time=100e-6)

metrics = save_rs_waveform(res, "bprf_sp2.wv", config=cfg)
print(f"LEVEL OFFS : RMS={metrics.rms_offset_db:.3f} dB, Peak={metrics.peak_offset_db:.3f} dB")
print(f"Duty cycle : {metrics.duty_cycle * 100:.2f}% (idle dilution: {metrics.idle_dilution_db:.2f} dB)")
```

!!! tip "Instrument setup (R&S SMW200A / SMBV100B)"
    Because `{LEVEL OFFS}` reflects active burst power only, setting **RF Level = -50 dBm** on the generator delivers **-50 dBm active burst power** regardless of `idle_time`. Route **Marker 1** (`active_burst`) to **RF Blanking / ALC Hold**, and **Marker 2** (`rmarker`) to your oscilloscope or UWB analyzer trigger input.

## Writing a Custom Instrument Exporter

Subclass [ARBExporter][uwb_hrp.export.ARBExporter] to target any other ARB format (Keysight, LitePoint, NI, raw binary I/Q) while reusing all active-power normalization and marker timing:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uwb_hrp import ARBExporter, HRPConfig, HRPWaveformResult, WaveformPowerMetrics


@dataclass(frozen=True)
class BinaryIQExporter(ARBExporter):
    """Example custom exporter writing interleaved float32 I/Q + uint8 marker bitmask."""

    def save(
        self,
        result: HRPWaveformResult,
        path: str | Path,
        *,
        config: HRPConfig | None = None,
    ) -> WaveformPowerMetrics:
        del config
        prepared = self.prepare(result)
        iq_f32 = prepared.iq.astype(np.complex64)
        out_path = Path(path)
        iq_f32.tofile(out_path)
        prepared.markers.to_bitmask().tofile(out_path.with_suffix(".markers.u8"))
        return prepared.metrics
```
