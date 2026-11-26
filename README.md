# uwb-hrp

An **IEEE Std 802.15.4-2024 (Clause 16)** compliant High-Rate Pulse Repetition Frequency (HRP) UWB PHY frame and baseband waveform generator in Python.

## Features

- **All HRP UWB PHY operating modes**:
  - **Legacy (`802.15.4a`)**: Mean PRFs `3.90`, `15.60`, and `62.40 MHz` across all 12 Table 16-4 data rate / PRF combinations (`0.11`, `0.85`, `1.70`, `6.81`, `27.24 Mb/s`), selectable preamble mean PRF (`4.03` / `16.1 MHz`), short & long (64-symbol) SFD, and BPM-BPSK modulation (Viterbi rate 1/2 and rate 1).
  - **BPRF (`62.40 MHz`)**: `6.81 Mb/s` data rate, `0.85` / `6.81 Mb/s` PHR rates, SFD selectors `0`–`4` (`0` and `2` mandatory), and STS packet configurations `0`–`3`.
  - **HPRF (`124.8` / `249.6 MHz`)**: Constraint length `K = 3` (`RS6(63,55)` + rate-1/2 convolutional coding) and `K = 7` (`133_8, 171_8` convolutional coding), SFD selectors `0`–`4`, PSR `16`–`256`, PSDU lengths up to `4095` bytes (Table 16-16 length extension), STS packet configurations `0`–`3` with `1`–`4` segments (`16`–`256` × 512 chips), and configurable SP2 extra STS gap (`0`–`127` × 4 chips).
- **All 16 UWB channels (0–15)** including wideband channels (`4`, `7`, `11`, `15`) and **all 32 ternary preamble codes** (Tables 16-7, 16-8, 16-9).
- **Cryptographic STS (Section 16.2.9 / Annex G)**: AES-128 CTR DRBG with optional per-packet `VCounter` advancement across multi-packet sequences.
- **Decoupled, extensible baseband pulse shaping (`PulseConfig` / `PulseShaper`)**:
  - `ButterworthPulse`: Causal 4th-order baseband Butterworth IIR filter (baseband equivalent of Figure 16-23's 8th-order RF bandpass filter).
  - `RRCPulse`: Closed-form Root Raised Cosine (`rolloff = 0.5` reference or `0.45`).
  - Subclass `PulseConfig` to plug in custom pulse shapes without changing `PulseShaper` or `generate_waveform`.
  - Built-in Section 16.4.4 cross-correlation compliance verification (`PulseShaper.verify_compliance()`).
- **Single- and multi-packet generation** with per-packet `idle_time`, exact chip-level field slices (`field_indices`), RMARKER / SRMARKER offsets, and Section 7.2.11 FCS-16 / FCS-32 helpers.
- **Extensible ARB waveform export (`uwb_hrp.export`)**:
  - Vendor-neutral `ARBExporter` base class, active-only power normalization (`peak` or `rms`, excluding `idle_time`), and semantic marker tracks (`active_burst`, `rmarker`, `packet_start`, `sts_gate`) convertible to transition lists or per-sample bitmasks — subclass `ARBExporter` to add new instrument vendors (e.g. LitePoint, Keysight).
  - Built-in **Rohde & Schwarz `.wv` exporter** (`RSWaveformExporter` / `save_rs_waveform`, optional `uwb-hrp[rs]` extra using [`RsWaveform`](https://github.com/Rohde-Schwarz/RsWaveform)): peak-normalized 16-bit I/Q export with exact `float64` active-power `{LEVEL OFFS}` and Markers 1–4.
- **Verified against all 93 official IEEE P802.15-20-0002-01-004z test vectors** chip-for-chip, plus coherent BPM-BPSK round-trip demodulation across all legacy PRF modes.

## Installation

Requires Python 3.12+. Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
# With optional Rohde & Schwarz .wv waveform export support:
uv sync --extra rs
```

Or with `pip`:

```bash
pip install -e .
# With optional Rohde & Schwarz .wv waveform export support:
pip install -e ".[rs]"
```

## Quick Start

### Generate ternary chips and baseband waveform

```python
from uwb_hrp import HRPConfig, generate_chips, generate_waveform

# Configure an HPRF frame with STS packet configuration 1
cfg = HRPConfig(
    mode="HPRF",
    channel=9,
    mean_prf=124.8,
    code_index=25,
    preamble_duration=64,
    sfd_number=2,
    psdu_length=20,
    sts_packet_config=1,
    num_sts_segments=1,
    sts_segment_length=64,
)

payload = b"\xba\xbe" * 10

# 1. Chip sequence {-1, 0, +1} at 499.2 MHz
chips = generate_chips(payload, cfg)

# 2. Pulse-shaped baseband waveform (defaults to ButterworthPulse @ 4 samples/pulse)
#    Optional idle_time appends silent zero samples after the packet.
result = generate_waveform(payload, cfg, idle_time=1e-6)
print(f"Sample rate: {result.sample_rate / 1e6:.2f} MHz, samples: {len(result.waveform)}")
print(f"RMARKER chip: {result.indices.rmarker_chip}, field ranges: {result.indices.to_1based_ranges()}")
```

### Custom pulse shaping and Section 16.4.4 compliance check

```python
from uwb_hrp import HRPConfig, PulseShaper, RRCPulse, generate_waveform

cfg = HRPConfig(mode="BPRF", channel=9, psdu_length=20)
pulse_cfg = RRCPulse(samples_per_pulse=8, rolloff=0.45)

shaper = PulseShaper.for_hrp_config(cfg, pulse_cfg)
report = shaper.verify_compliance()
assert report.compliant

result = generate_waveform(b"\xba\xbe" * 10, cfg, pulse=shaper)
```

### Multi-packet waveform with idle time and STS counter advancement

```python
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

res = generate_waveform(
    [b"\xba\xbe" * 10, b"\xca\xfe" * 10],
    cfg,
    pulse=ButterworthPulse(samples_per_pulse=4),
    num_packets=2,
    idle_time=1e-6,
)
# res.next_sts_v gives the next 16-byte DRBG V value for streaming subsequent frames
```

### Export to Rohde & Schwarz `.wv` format (active-power normalized)

Requires the optional `rs` extra (`uv sync --extra rs` / `pip install -e ".[rs]"`):

```python
from uwb_hrp import ButterworthPulse, HRPConfig, generate_waveform, save_rs_waveform

cfg = HRPConfig(mode="BPRF", channel=9, sfd_number=2, psdu_length=20, sts_packet_config=2)
res = generate_waveform(b"\xba\xbe" * 10, cfg, pulse=ButterworthPulse(4), idle_time=100e-6)

# Peak-normalizes I/Q to [-32767, +32767] and computes {LEVEL OFFS} over active
# packet samples only (excluding idle_time), plus auto-generates Markers 1-4.
metrics = save_rs_waveform(res, "bprf_sp2.wv", config=cfg)
print(f"LEVEL OFFS: RMS={metrics.rms_offset_db:.3f} dB, Peak={metrics.peak_offset_db:.3f} dB")
# On an R&S SMW200A / SMBV100B, setting RF Level = -50 dBm now delivers -50 dBm
# active burst power regardless of idle_time. Route Marker 1 to RF Blanking / ALC Hold.
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `HRPConfig` | Validated PHY-layer frame configuration (`802.15.4a`, `BPRF`, `HPRF`) |
| `HRPFieldIndices` | 0-based chip slices (`sync`, `sfd`, `sts1`, `phr`, `psdu`, `sts2`), `rmarker_chip`, `srmarker_chips` |
| `PulseConfig` | Abstract base class for baseband pulse-shaping configurations |
| `ButterworthPulse` | Causal baseband Butterworth pulse config (`order=4`, `samples_per_pulse=4`) |
| `RRCPulse` | Root Raised Cosine pulse config (`rolloff=0.5`, `samples_per_pulse=4`) |
| `PulseShaper` | Reusable pulse shaper with `.shape(chips)` and `.verify_compliance()` |
| `PulseComplianceReport` | Section 16.4.4 main-lobe width & sidelobe compliance report |
| `HRPWaveformResult` | Returned by `generate_waveform()` with `waveform`, `chips`, `indices`, `sample_rate`, `pulse_taps` |
| `ARBExporter` | Abstract base class for instrument ARB exporters (shared active-power normalization & markers) |
| `WaveformPowerMetrics` | Vendor-neutral active/total level offsets, crest factor, duty cycle, and idle dilution metrics |
| `MarkerTrack` / `WaveformMarkers` | Semantic marker intervals (`active_burst`, `rmarker`, `packet_start`, `sts_gate`) with `.to_transitions()`, `.to_mask()`, `.to_bitmask()` |
| `PreparedARBWaveform` | Normalized I/Q, metrics, and markers returned by `prepare_arb_waveform()` / `ARBExporter.prepare()` |
| `RSWaveformExporter` | Rohde & Schwarz `.wv` exporter subclass of `ARBExporter` (`uwb-hrp[rs]`) |
| `generate_chips()` | Generate 499.2 MHz ternary chip array (1D for 1 packet, 2D `(chips, num_packets)` for multi-packet) |
| `generate_waveform()` | Generate baseband pulse-shaped waveform for one or more packets with optional `idle_time` |
| `field_indices()` | Compute `HRPFieldIndices` without generating chips |
| `compute_power_metrics()` / `prepare_arb_waveform()` | Vendor-neutral active-power metrics and ARB normalization (no vendor SDK required) |
| `build_waveform_markers()` / `build_rs_markers()` | Build vendor-neutral semantic markers or R&S `MARKER LIST 1..4` tables |
| `to_rs_waveform()` / `save_rs_waveform()` | Convert/export `HRPWaveformResult` to Rohde & Schwarz `.wv` (`uwb-hrp[rs]`) |
| `compute_fcs16()` / `compute_fcs32()` | IEEE 802.15.4 Section 7.2.11 16-bit and 32-bit FCS helpers |

## Examples

Ready-to-run scripts in [`examples/`](examples/) demonstrate the main workflows:

| Script | Highlights |
| --- | --- |
| [`examples/bprf_frame.py`](examples/bprf_frame.py) | BPRF `SP0`/`SP1` frames, `compute_fcs16()`, field indices, single-packet `idle_time` |
| [`examples/hprf_frame.py`](examples/hprf_frame.py) | HPRF `124.8`/`249.6 MHz`, `K=3`/`K=7`, multi-segment `SP2` STS + extra gap, 2048-byte PSDU, `SP3` |
| [`examples/legacy_802154a.py`](examples/legacy_802154a.py) | Legacy `802.15.4a` across `3.9`/`15.6`/`62.4 MHz` PRFs, long SFD, wideband channel `4` |
| [`examples/pulse_shaping.py`](examples/pulse_shaping.py) | `ButterworthPulse` vs `RRCPulse`, §16.4.4 compliance checks, custom `PulseConfig` subclass |
| [`examples/multipacket_burst.py`](examples/multipacket_burst.py) | Multi-packet burst with per-packet `idle_time` and `advance_sts_counter=True` |
| [`examples/rs_waveform_export.py`](examples/rs_waveform_export.py) | Rohde & Schwarz `.wv` export with active-power normalization (`idle_time` invariant) and Markers 1–4 |

Run any example with:

```bash
uv run python examples/bprf_frame.py
```

## Development & Testing

```bash
uv sync --dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src tests examples
uv run pytest --cov=uwb_hrp
```

### IEEE P802.15.4z Test Vectors

The `tests/ieee_test_vectors/` directory contains the 93 official HRP UWB PHY test vectors from IEEE document **P802.15-20-0002-01-004z** (released publicly by IEEE P802.15; see [tests/ieee_test_vectors/readme.txt](tests/ieee_test_vectors/readme.txt)). Note that Draft D08 (April 2020) omitted the trailing 512 silent zero chips at the end of SP2/SP3 frames; `HRPConfig.sts_trailing_gap` defaults to `True` per IEEE Std 802.15.4-2024 Figures 16-10 and 16-12 and can be set to `False` when reproducing Draft D08 vectors.

## License

This project is licensed under the [MIT License](LICENSE).
