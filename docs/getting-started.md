# Getting Started

## Requirements & Installation

`uwb-hrp` requires **Python 3.12+** and depends on `numpy`, `scipy`, and `cryptography`.

=== "uv"

    ```bash
    # Core package
    uv sync

    # With optional Rohde & Schwarz .wv export (RsWaveform)
    uv sync --extra rs
    ```

=== "pip"

    ```bash
    # Core package
    pip install -e .

    # With optional Rohde & Schwarz .wv export (RsWaveform)
    pip install -e ".[rs]"
    ```

## Generate Your First Frame

PHY configuration ([HRPConfig][uwb_hrp.HRPConfig]) is intentionally decoupled from baseband pulse shaping ([PulseConfig][uwb_hrp.PulseConfig] / [PulseShaper][uwb_hrp.PulseShaper]):

```python
from uwb_hrp import HRPConfig, compute_fcs16, generate_chips, generate_waveform

# Build an 18-byte MAC payload + 2-byte IEEE 802.15.4 Section 7.2.11 FCS-16
mac_body = b"\x41\x88\x01\xca\xfe\xff\xff\x01\x00" + b"\xab" * 9
psdu = mac_body + compute_fcs16(mac_body)  # 20 bytes

cfg = HRPConfig(
    mode="HPRF",
    channel=9,
    mean_prf=124.8,
    code_index=25,
    preamble_duration=64,
    sfd_number=2,
    psdu_length=len(psdu),
    sts_packet_config=1,
    num_sts_segments=1,
    sts_segment_length=64,
)

# 1. Ternary chip sequence {-1, 0, +1} at 499.2 MHz
chips = generate_chips(psdu, cfg)

# 2. Pulse-shaped baseband waveform (defaults to ButterworthPulse @ 4 samples/pulse)
result = generate_waveform(psdu, cfg, idle_time=1e-6)
print(f"Sample rate : {result.sample_rate / 1e6:.2f} MHz")
print(f"Samples     : {len(result.waveform)} (idle: {result.idle_samples})")
print(f"RMARKER chip: {result.indices.rmarker_chip}")
print(f"1-based map : {result.indices.to_1based_ranges()}")
```

## Multi-Packet Bursts & Continuous STS Streaming

Pass a list of payloads (or a flat byte/bit stream) together with `num_packets` and `idle_time` to synthesize a multi-packet ARB burst. By default, `advance_sts_counter=True` increments the 32-bit AES-128 DRBG `VCounter` across packets per IEEE Std 802.15.4-2024 Section 16.2.9.2.

To stream subsequent bursts continuously without repeating STS sequences, pass `result.next_sts_v` into the next configuration via `dataclasses.replace`:

```python
from dataclasses import replace

from uwb_hrp import ButterworthPulse, HRPConfig, generate_waveform

cfg = HRPConfig(
    mode="BPRF",
    channel=9,
    code_index=9,
    preamble_duration=64,
    sfd_number=2,
    psdu_length=20,
    sts_packet_config=1,
)

burst_1 = generate_waveform(
    [b"\xba\xbe" * 10, b"\xca\xfe" * 10],
    cfg,
    pulse=ButterworthPulse(samples_per_pulse=4),
    num_packets=2,
    idle_time=10e-6,
)

# Continue seamlessly from burst_1's ending DRBG VCounter
cfg_next = replace(cfg, sts_v=burst_1.next_sts_v)
burst_2 = generate_waveform(b"\xde\xad" * 10, cfg_next, idle_time=10e-6)
```

## Runnable Examples

The repository includes six self-contained example scripts under `examples/`:

| Script | What It Demonstrates |
| --- | --- |
| `examples/bprf_frame.py` | BPRF `SP0`/`SP1` frames, `compute_fcs16()`, field indices, single-packet `idle_time` |
| `examples/hprf_frame.py` | HPRF `124.8`/`249.6 MHz`, `K=3`/`K=7`, multi-segment `SP2` STS + extra gap, 2048-byte PSDU, `SP3` |
| `examples/legacy_802154a.py` | Legacy `802.15.4a` across `3.9`/`15.6`/`62.4 MHz` PRFs, long SFD, wideband channel `4` |
| `examples/pulse_shaping.py` | `ButterworthPulse` vs. `RRCPulse`, §16.4.4/§16.4.5 compliance checks, custom `PulseConfig` subclass |
| `examples/multipacket_burst.py` | Multi-packet burst with per-packet `idle_time` and `advance_sts_counter=True` |
| `examples/rs_waveform_export.py` | Rohde & Schwarz `.wv` export with active-power normalization and Markers 1–4 |

Run any example with:

```bash
uv run python examples/bprf_frame.py
```
