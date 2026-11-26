# PHY Modes & Configuration

[HRPConfig][uwb_hrp.HRPConfig] is an immutable, self-validating dataclass covering every Clause 16 HRP UWB operating mode defined in IEEE Std 802.15.4-2024. Any omitted (`None`) parameter is resolved to a mode- and channel-compliant default; invalid combinations raise `ValueError` immediately on construction.

## Operating Mode Overview

| Feature | `802.15.4a` (Legacy) | `BPRF` | `HPRF` |
| --- | --- | --- | --- |
| **Data Mean PRF (MHz)** | `3.9`, `15.6`, `62.4` | `62.4` | `124.8`, `249.6` |
| **Preamble Code Length** | `31` (`3.9`/`15.6`) or `127` (`62.4`) | `127` (`9..24`) | `91` (`25..32`) |
| **Preamble Mean PRF (MHz)** | `4.03` / `16.1` (`L=31`), `62.89` (`L=127`) | `62.89` | `111.09` |
| **Modulation** | BPM-BPSK | BPM-BPSK | HPRF BPSK |
| **FEC / Constraint Length** | `RS6(63,55)` + `K=3` (rate 1/2 or 1) | `RS6(63,55)` + `K=3` (rate 1/2) | `K=3` (with RS) or `K=7` (no RS) |
| **Data Rates (Mb/s)** | `0.11`, `0.85`, `1.70`, `6.81`, `27.24` | `6.81` | `6.81`, `7.8`, `27.24`, `31.2` |
| **PHR Data Rates (Mb/s)** | `0.11`, `0.85`, or `6.81` | `0.85` (DRBM_LP) or `6.81` (DRBM_HP) | `3.9`, `7.8`, `15.6`, `31.2` |
| **PSR (`preamble_duration`)** | `16`, `64`, `1024`, `4096` | `16`, `64`, `1024`, `4096` | `16`, `24`, `32`, `48`, `64`, `96`, `128`, `256` |
| **SFD Selectors** | `0` (short), `"long"` (64-sym) | `0`, `1`, `2`, `3`, `4` | `0`, `1`, `2`, `3`, `4` |
| **Max PSDU Length (bytes)** | `127` | `127` | `4095` (Table 16-16 extension) |
| **STS Support** | No (`SP0` only) | `SP0`–`SP3` (`1` seg × `64`) | `SP0`–`SP3` (`1..4` segs × `16..256`) |

## Legacy `802.15.4a` Rate Matrix (Table 16-4)

All 12 BPM-BPSK `(mean_prf, data_rate)` combinations are supported:

| `mean_prf` (MHz) | Valid `data_rate` (Mb/s) | Viterbi Rate | `N_burst` | `N_hop` | `N_cpb` | Chips / Symbol |
| --- | --- | --- | --- | --- | --- | --- |
| `3.9` | `0.11` / `0.85` / `1.70` / `6.81` | `0.5` / `0.5` / `0.5` / `1.0` | `128` | `32` | `32` / `4` / `2` / `1` | `4096` / `512` / `256` / `128` |
| `15.6` | `0.11` / `0.85` / `6.81` / `27.24` | `0.5` / `0.5` / `0.5` / `1.0` | `32` | `8` | `128` / `16` / `2` / `1` | `4096` / `512` / `64` / `32` |
| `62.4` | `0.11` / `0.85` / `6.81` / `27.24` | `0.5` / `0.5` / `0.5` / `0.5` | `8` | `2` | `512` / `64` / `8` / `2` | `4096` / `512` / `64` / `16` |

!!! note "Wideband channel restrictions"
    Per Tables 16-4 and 16-5, `mean_prf=3.9 MHz` and `preamble_mean_prf=4.03 MHz` are not permitted on wideband channels (`4`, `7`, `11`, `15`). Additionally, `preamble_duration=4096` is disallowed when `preamble_mean_prf=4.03 MHz` (Table 16-6 footnote a).

## HPRF Modulation Matrix (Tables 16-21 & 16-22)

In `HPRF` mode, `(mean_prf, constraint_length)` uniquely determines `data_rate` and `phr_data_rate`:

| `mean_prf` (MHz) | `constraint_length` | Outer RS | `data_rate` (Mb/s) | `phr_data_rate` (Mb/s) | Chips / Data Sym | Chips / PHR Sym |
| --- | --- | --- | --- | --- | --- | --- |
| `124.8` | `3` | `RS6(63,55)` | `6.81` | `3.9` | `64` | `128` |
| `124.8` | `7` | None | `7.8` | `7.8` | `64` | `64` |
| `249.6` | `3` | `RS6(63,55)` | `27.24` | `15.6` | `16` | `32` |
| `249.6` | `7` | None | `31.2` | `31.2` | `16` | `16` |

### Extended PSDU Lengths (Table 16-16)

In `HPRF`, `psdu_length` may be up to `4095` octets. If `psdu_size_extension` is left as `None`, `HRPConfig` selects the smallest valid extension automatically:

- `0`: `0..1023` bytes (PHR bits `A1`/`A0` available for `extra_sts_gap_index` in `SP2`)
- `1`: `0..2047` bytes (`A0` carries bit 10 of PSDU length)
- `2`: `0..4095` bytes (`A1`/`A0` carry bits 11 and 10 of PSDU length)

## Channels & Preamble Codes (Tables 16-7 to 16-9, 16-27)

All 16 IEEE 802.15.4-2024 UWB channels (`0..15`) are built in:

- **Narrowband (`499.2 MHz`, `Tp = 2.00 ns`, `Tw = 0.5 ns`):** Channels `0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14`
- **Wideband (`1081.6`–`1354.97 MHz`, `Tp = 0.74`–`0.92 ns`, `Tw = 0.2 ns`):** Channels `4, 7, 11, 15`

When `code_index=None`, `HRPConfig` picks the first non-DPS code of the required length allowed on `channel` (for example, code `9` for BPRF on channel `9`, and code `25` for HPRF on any channel).

```python
from uwb_hrp import HRPConfig

# Wideband legacy 802.15.4a frame on channel 4 (1331.2 MHz bandwidth)
cfg_wb = HRPConfig(mode="802.15.4a", channel=4, mean_prf=15.6, data_rate=6.81, psdu_length=32)
print(cfg_wb.code_index, cfg_wb.bandwidth / 1e6, cfg_wb.pulse_duration_tp * 1e9)
```
