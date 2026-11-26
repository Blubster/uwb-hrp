# Standards Compliance & Verification

`uwb-hrp` implements **IEEE Std 802.15.4-2024 Clause 16** (HRP UWB PHY) and **Section 7.2.11** (MAC FCS-16/32 helpers).

## IEEE Std 802.15.4-2024 Traceability Matrix

| Standard Reference | Scope | Implementation |
| --- | --- | --- |
| **Section 7.2.11** | 2-octet ITU-T CRC-16 (`G16`) and 4-octet ANSI X3.66 CRC-32 (`G32`) FCS | `uwb_hrp.fec`: [compute_fcs16][uwb_hrp.compute_fcs16], [compute_fcs32][uwb_hrp.compute_fcs32] |
| **Figures 16-1, 16-3, 16-9 to 16-12** | PPDU structure (`SP0`–`SP3`), `RMARKER`, `SRMARKER`, trailing gap | `uwb_hrp.generator`: [field_indices][uwb_hrp.field_indices], [generate_chips][uwb_hrp.generate_chips] |
| **Tables 16-2 & 16-3** | Viterbi rate-1/2 and rate-1 BPM-BPSK PHR/PSDU framing | `uwb_hrp.modulation`: [encode_and_modulate_phr_psdu][uwb_hrp.modulation.encode_and_modulate_phr_psdu] |
| **Tables 16-4 to 16-6, 16-10** | Mean PRFs, data rates, BPM-BPSK symbol timing, preamble spreading `delta_L` | `uwb_hrp.tables`, `uwb_hrp.config`: [HRPConfig][uwb_hrp.HRPConfig] |
| **Tables 16-7, 16-8, 16-9** | All 32 ternary preamble codes (lengths 31, 127, 91) & channel assignments | `uwb_hrp.tables`: `PREAMBLE_CODES`, `uwb_hrp.shr` |
| **Section 16.2.6.3, Table 16-11** | SFD sequences `0`–`4` and legacy 64-symbol `"long"` SFD | `uwb_hrp.tables`: `SFD_SEQUENCES`, `uwb_hrp.shr`: [generate_sfd][uwb_hrp.shr.generate_sfd] |
| **Sections 16.2.7.1–16.2.7.3, Figures 16-7 & 16-8, Tables 16-12 to 16-16** | Legacy/BPRF and HPRF 19-bit PHR formatting, SECDED parity, PSDU length extension | `uwb_hrp.phr`: [build_phr_bits][uwb_hrp.phr.build_phr_bits], [compute_secded][uwb_hrp.phr.compute_secded] |
| **Section 16.2.9, Annex G, Tables 16-17 to 16-19** | AES-128 CTR DRBG STS, multi-segment STS, SP2 `extra_sts_gap`, `VCounter` advancement | `uwb_hrp.sts`: [sts_drbg_blocks][uwb_hrp.sts.sts_drbg_blocks], [generate_sts][uwb_hrp.sts.generate_sts] |
| **Sections 16.3.1 & 16.3.2** | BPM-BPSK time hopping, Note 3 (`n_cpb < m`), 15-bit PRBS scrambler (`1 + D^14 + D^15`) | `uwb_hrp.scrambler`, `uwb_hrp.modulation` |
| **Section 16.3.3.2** | Systematic `RS6(63, 55)` outer coding over `GF(2^6)` (`1 + x + x^6`) | `uwb_hrp.fec`: [rs_encode][uwb_hrp.fec.rs_encode] |
| **Section 16.3.3.3** | `K=3` (`010_2, 101_2`) and `K=7` (`133_8, 171_8`) convolutional encoders | `uwb_hrp.fec`: [conv_encode_k3][uwb_hrp.fec.conv_encode_k3], [conv_encode_k7][uwb_hrp.fec.conv_encode_k7] |
| **Section 16.3.4, Tables 16-21 to 16-26** | HPRF BPSK modulation (`124.8` / `249.6 MHz`, `K=3` / `K=7`) | `uwb_hrp.modulation` |
| **Sections 16.4.4 & 16.4.5, Figures 16-23 & 16-24, Tables 16-27 & 16-28** | Channel plan (0–15), reference RRC pulse, Butterworth shaping, cross-correlation, PSD mask, precursor mask | `uwb_hrp.pulse`: [PulseShaper][uwb_hrp.PulseShaper], [ButterworthPulse][uwb_hrp.ButterworthPulse], [RRCPulse][uwb_hrp.RRCPulse] |

## Verification Suite

The automated test suite under `tests/` validates the generator from multiple independent angles:

1. **All 93 Official IEEE P802.15-20-0002-01-004z Test Vectors (`tests/test_ieee_vectors.py`):**
   - Verified chip-for-chip (`np.array_equal`) and field-index-for-field-index against every BPRF (`bprf_1`–`bprf_5`) and HPRF (`hprf_1`–`hprf_36`) `.mat` vector and `packet_info.txt`.
   - Uses `sts_trailing_gap=False` specifically to match April 2020 Draft D08 vectors where the trailing 512-chip silent gap on SP2/SP3 was omitted; default `HRPConfig.sts_trailing_gap=True` conforms to published IEEE Std 802.15.4-2024 Figures 16-10 and 16-12.
2. **Legacy `802.15.4a` Round-Trip Demodulation (`tests/test_legacy_prf.py`):**
   - Independent coherent BPM-BPSK demodulator verifies all 12 Table 16-4 `(mean_prf, data_rate)` combinations, Viterbi rate 1/2 and rate 1, short and 64-symbol long SFD, and wideband channels.
3. **HPRF Round-Trip & Extended Lengths (`tests/test_hprf_roundtrip.py`):**
   - Verifies HPRF `K=3` and `K=7`, `124.8`/`249.6 MHz`, `SP0`–`SP3`, multi-segment STS, `extra_sts_gap`, and PSDU lengths up to `4095` bytes.
4. **Unit & Compliance Tests (`tests/test_components.py`, `tests/test_rs_waveform.py`):**
   - Covers GF(2^6) RS generator polynomial, SECDED parity, PRBS scrambler, Annex G DRBG vectors, §16.4.4 / §16.4.5 pulse compliance, custom `PulseConfig` subclasses, active-power invariance across `idle_time`, semantic markers, and `.wv` round-trip serialization.
