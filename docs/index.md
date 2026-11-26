# uwb-hrp

An **IEEE Std 802.15.4-2024 (Clause 16)** compliant High-Rate Pulse Repetition Frequency (HRP) UWB physical-layer frame and baseband waveform generator in Python.

`uwb-hrp` generates bit-exact 499.2 MHz ternary chip sequences (`{-1, 0, +1}`), applies extensible baseband pulse shaping with built-in Sections 16.4.4 and 16.4.5 compliance verification, and exports active-power-normalized arbitrary waveform generator (ARB) files with sample-synchronous markers.

## Highlights

- **Complete Clause 16 PHY coverage:** Legacy `802.15.4a` (`3.9`, `15.6`, `62.4 MHz`), `BPRF` (`62.4 MHz`), and `HPRF` (`124.8` / `249.6 MHz`, `K = 3` and `K = 7`), all 16 UWB channels (`0..15`), and all 32 ternary preamble codes.
- **Cryptographic STS (Section 16.2.9 / Annex G):** AES-128 CTR DRBG supporting `SP0`–`SP3`, `1`–`4` segments (`16`–`256` × 512 chips), SP2 `extra_sts_gap`, and automatic `VCounter` advancement across multi-packet sequences.
- **Extensible baseband pulse shaping:** Built-in causal `ButterworthPulse` (Figure 16-23) and closed-form `RRCPulse` (Section 16.4.4), plus a `PulseConfig` base class for custom pulse shapes and automated compliance reporting (`PulseShaper.verify_compliance()`).
- **Instrument-ready ARB export:** Vendor-neutral `ARBExporter` base class with active-only power normalization (unaffected by `idle_time`), semantic markers (`active_burst`, `rmarker`, `packet_start`, `sts_gate`), and a built-in Rohde & Schwarz `.wv` backend (`uwb-hrp[rs]`).
- **Standard-verified:** Bit-exact chip-for-chip agreement against all **93 official IEEE P802.15-20-0002-01-004z test vectors** plus coherent BPM-BPSK round-trip demodulation across all legacy PRF modes.

## Quick Install

=== "uv"

    ```bash
    uv sync
    # With optional Rohde & Schwarz .wv waveform export:
    uv sync --extra rs
    ```

=== "pip"

    ```bash
    pip install -e .
    # With optional Rohde & Schwarz .wv waveform export:
    pip install -e ".[rs]"
    ```

## Where to Go Next

- **[Getting Started](getting-started.md):** Generate your first BPRF/HPRF chips, baseband waveforms, and multi-packet streams.
- **[PHY Modes & Configuration](guides/phy-modes.md):** Complete parameter matrices for `802.15.4a`, `BPRF`, and `HPRF`.
- **[Frame Structure & STS](guides/frame-structure.md):** Packet diagrams (`SP0`–`SP3`), `RMARKER`/`SRMARKER` chip indexing, and STS DRBG streaming.
- **[Pulse Shaping & Compliance](guides/pulse-shaping.md):** Butterworth vs. RRC shaping, §16.4.4 / §16.4.5 compliance checks, and custom `PulseConfig` subclasses.
- **[ARB Waveform Export](guides/arb-export.md):** Active-power `{LEVEL OFFS}` math, R&S `.wv` export, and writing custom instrument backends.
- **[Standards & Verification](standards-compliance.md):** IEEE Std 802.15.4-2024 traceability table and test vector verification details.
- **[API Reference](api/index.md):** Full typed reference generated directly from source docstrings.

!!! note "Acknowledgments"
    This project—including the PHY implementation, pulse shaping, ARB export layer, test suite, and documentation—was entirely created with **Gemini 4 Neon**.
