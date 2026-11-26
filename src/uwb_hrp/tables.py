"""Normative tables and constants from IEEE Std 802.15.4-2024 Clause 16."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Peak pulse repetition frequency (chip rate) in Hz (Table 16-4)
PEAK_PRF_HZ: float = 499.2e6
CHIP_DURATION_S: float = 1.0 / PEAK_PRF_HZ

# Default Annex G STS DRBG seed values
DEFAULT_STS_KEY: bytes = bytes.fromhex("14148674d1d336aaf86050a814eb220f")
DEFAULT_STS_V: bytes = bytes.fromhex("362eeb34c44fa8fbd37ec3ca1f9a3de4")

# STS gap duration in chips (Section 16.2.9.1)
STS_GAP_CHIPS: int = 512
STS_SEGMENT_UNIT_CHIPS: int = 512
STS_EXTRA_GAP_UNIT_CHIPS: int = 4

WIDEBAND_CHANNELS: frozenset[int] = frozenset({4, 7, 11, 15})
NARROWBAND_CHANNELS: frozenset[int] = frozenset({0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14})
ALL_CHANNELS: frozenset[int] = WIDEBAND_CHANNELS | NARROWBAND_CHANNELS


@dataclass(frozen=True)
class ChannelSpec:
    """Channel frequency and pulse timing parameters (Tables 16-27 and 16-28)."""

    channel: int
    band_group: int
    center_freq_mhz: float
    bandwidth_mhz: float
    pulse_duration_tp_ns: float
    main_lobe_tw_ns: float


# Tables 16-27 and 16-28
CHANNEL_SPECS: dict[int, ChannelSpec] = {
    0: ChannelSpec(0, 0, 499.2, 499.2, 2.00, 0.5),
    1: ChannelSpec(1, 1, 3494.4, 499.2, 2.00, 0.5),
    2: ChannelSpec(2, 1, 3993.6, 499.2, 2.00, 0.5),
    3: ChannelSpec(3, 1, 4492.8, 499.2, 2.00, 0.5),
    4: ChannelSpec(4, 1, 3993.6, 1331.2, 0.75, 0.2),
    5: ChannelSpec(5, 2, 6489.6, 499.2, 2.00, 0.5),
    6: ChannelSpec(6, 2, 6988.8, 499.2, 2.00, 0.5),
    7: ChannelSpec(7, 2, 6489.6, 1081.6, 0.92, 0.2),
    8: ChannelSpec(8, 2, 7488.0, 499.2, 2.00, 0.5),
    9: ChannelSpec(9, 2, 7987.2, 499.2, 2.00, 0.5),
    10: ChannelSpec(10, 2, 8486.4, 499.2, 2.00, 0.5),
    11: ChannelSpec(11, 2, 7987.2, 1331.2, 0.75, 0.2),
    12: ChannelSpec(12, 2, 8985.6, 499.2, 2.00, 0.5),
    13: ChannelSpec(13, 2, 9484.8, 499.2, 2.00, 0.5),
    14: ChannelSpec(14, 2, 9984.0, 499.2, 2.00, 0.5),
    15: ChannelSpec(15, 2, 9484.8, 1354.97, 0.74, 0.2),
}


@dataclass(frozen=True)
class PreambleCodeSpec:
    """Ternary preamble code metadata (Tables 16-7, 16-8, and 16-9)."""

    index: int
    sequence_str: str
    channels: frozenset[int]
    dps_only: bool = False

    @property
    def length(self) -> int:
        return len(self.sequence_str)

    @property
    def chips(self) -> np.ndarray:
        mapping = {"+": 1, "-": -1, "0": 0}
        return np.array([mapping[ch] for ch in self.sequence_str], dtype=np.int8)


_CH_1_2 = frozenset({0, 1, 8, 12}) | WIDEBAND_CHANNELS
_CH_3_4 = frozenset({2, 5, 9, 13}) | WIDEBAND_CHANNELS
_CH_5_6 = frozenset({3, 6, 10, 14}) | WIDEBAND_CHANNELS
_CH_9_12 = NARROWBAND_CHANNELS | WIDEBAND_CHANNELS

# Tables 16-7 (1..8), 16-8 (9..24), and 16-9 (25..32)
PREAMBLE_CODES: dict[int, PreambleCodeSpec] = {
    # Length 31 (Table 16-7)
    1: PreambleCodeSpec(1, "-0000+0-0+++0+-000+-+++00-+0-00", _CH_1_2),
    2: PreambleCodeSpec(2, "0+0+-0+0+000-++0-+---00+00++000", _CH_1_2),
    3: PreambleCodeSpec(3, "-+0++000-+-++00++0+00-0000-0+0-", _CH_3_4),
    4: PreambleCodeSpec(4, "0000+-00-00-++++0+-+000+0-0++0-", _CH_3_4),
    5: PreambleCodeSpec(5, "-0+-00+++-+000-+0+++0-0+0000-00", _CH_5_6),
    6: PreambleCodeSpec(6, "++00+00---+-0++-000+0+0-+0+0000", _CH_5_6),
    7: PreambleCodeSpec(7, "+0000+-0+0+00+000+0++---0-+00-+", WIDEBAND_CHANNELS),
    8: PreambleCodeSpec(8, "0+00-0-0++0000--+00-+0++-++0+00", WIDEBAND_CHANNELS),
    # Length 127 (Table 16-8)
    9: PreambleCodeSpec(
        9,
        "+00+000-0--00--+0+0+00-+-++0+0000++-000+00-00--0-+0+0--0-+++0++000+-0+00-0++-0+++00-+00+0+0-0++-+--+000000+00000-+0000-0-000--+",
        _CH_9_12,
    ),
    10: PreambleCodeSpec(
        10,
        "++00+0-+00+00+000000-000-00--000-0+-+0-0+-0-+00000+-00++0-0+00--+00++-+0+-0+0000-0-0-0-++-+0+00+0+000-+0+++000----+++0000+++0--",
        _CH_9_12,
    ),
    11: PreambleCodeSpec(
        11,
        "-+-0000+00--00000-0+0+0+-0+00+00+0-00-+++00+000-+0+0-0000+++++-+0+--0+-0++--0-000+0-+00+0+----000-000000-+00+-0++000++-00++-0-0",
        _CH_9_12,
    ),
    12: PreambleCodeSpec(
        12,
        "-+0++000000-0+0-+0---+-++00-+0++0+0+0+000-00-00-+00+-++000-+-0-++0-0++++0-00-0++00+0+00++-00+000+-000-0--+0000-0000--0+00000+--",
        _CH_9_12,
    ),
    13: PreambleCodeSpec(
        13,
        "+000--0000--++0-++++0-0++0+0-00-+0++00++-0++0+-+0-00+00-0--000-+-00+0000-0++-00000+-0-000000-00-+-++-+000-0+0+0+++-00--00+0+000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    14: PreambleCodeSpec(
        14,
        "+000++0-0+0-00+-0-+0-00+0+0000+0+-0000++00+0+++++-+0-0+-0--+0++--000---0+000+0+0-+-000000+-+-0--00++000-00+00++-00--++-00-00000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    15: PreambleCodeSpec(
        15,
        "0+-00+0-000-++0000---++000+0+-0-+00-+000--0-00--0--+++-+0-++00+-++0+00000+0-0+++-00+00+000-0000+00--+0++0+0+0-00-0-+-0+0++00000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    16: PreambleCodeSpec(
        16,
        "++0000+000+00+--0+-++0-000--00+-0+00++000+++00+0+0-0-+-0-0+00+00+0++----+00++--+0+-0--+000000-0-0000-+0--00+00000+-++000-0-+0+0",
        ALL_CHANNELS,
        dps_only=True,
    ),
    17: PreambleCodeSpec(
        17,
        "+--000-0-0000+-00000+000000+--+-++0-0+0+00+-00+++0-++0-00+0-+000++0+++-0--0+0+-0--00-00+000-++0000+0++-+-00+0+0+--00--0-000+00+",
        WIDEBAND_CHANNELS,
    ),
    18: PreambleCodeSpec(
        18,
        "--0+++0000+++----000+++0+-000+0+00+0+-++-0-0-0-0000+0-+0+-++00+--00+0-0++00-+00000+-0-+0-0+-+0-000--00-000-000000+00+00+-0+00++",
        WIDEBAND_CHANNELS,
    ),
    19: PreambleCodeSpec(
        19,
        "-0-++00-++000++0-+00+-000000-000----+0+00+-0+000-0--++0-+0--+0+-+++++0000-0+0+-000+00+++-00-0+00+00+0-+0+0+0-00000--00+0000-+-0",
        WIDEBAND_CHANNELS,
    ),
    20: PreambleCodeSpec(
        20,
        "--+00000+0--0000-0000+--0-000-+000+00-++00+0+00++0-00-0++++0-0++-0-+-000++-+00+-00-00-000+0+0+0++0+-00++-+---0+-0+0-000000++0+-",
        WIDEBAND_CHANNELS,
    ),
    21: PreambleCodeSpec(
        21,
        "+0+00--00-+++0+0+0-000+-++-+-00-000000-0-+00000-++0-0000+00-+-000--0-00+00-0+-+0++0-++00++0+-00-0+0++0-0++++-0++--0000--000+000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    22: PreambleCodeSpec(
        22,
        "0-00-++--00-++00+00-000++00--0-+-+000000-+-0+0+000+0---000--++0+--0-+0-0+-+++++0+00++0000-+0+0000+0+00-0+-0-+00-0+0-0++000+0000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    23: PreambleCodeSpec(
        23,
        "000++0+0-+-0-00-0+0+0++0+--00+0000-000+00+00-+++0-0+00000+0++-+00++-0+-+++--0--00-0--000+-00+-0-+0+000++---0000++-000-0+00-+000",
        ALL_CHANNELS,
        dps_only=True,
    ),
    24: PreambleCodeSpec(
        24,
        "+0+-0-000++-+00000+00--0+-0000-0-000000+--0-+0+--++00+----++0+00+00+0-0-+-0-0+0+00+++000++00+0-+00--000-0++-+0--+00+000+0000++0",
        ALL_CHANNELS,
        dps_only=True,
    ),
    # Length 91 (Table 16-9, HPRF)
    25: PreambleCodeSpec(
        25,
        "-0++++--+--+-++++-+---++--++++++-++-+00+--+0--+0+++++--+++--0--0+-+----0-+-+-+0+--++-+-+++0",
        ALL_CHANNELS,
    ),
    26: PreambleCodeSpec(
        26,
        "++0+-+---+++++-+-++--+--++---+-0+++0-++++-+0+0--0+-++-++++++--+-++00+++--0+-----+-0+-+-+---",
        ALL_CHANNELS,
    ),
    27: PreambleCodeSpec(
        27,
        "+++--+++---+-+-0-+--0++-+-+0-++++++++++--+--++-++0++-+-+--+--+++---0-+++-0+00---++-+--0-++0",
        ALL_CHANNELS,
    ),
    28: PreambleCodeSpec(
        28,
        "+++++--+++-++---+-+--0++---+-+0+-----+0+++--+-+--+-++-++++0-0-++00+-+++-++-+0-+0--+---+++0+",
        ALL_CHANNELS,
    ),
    29: PreambleCodeSpec(
        29,
        "+-0--+-++--0++00++-++-----++++++-0+--+-++--+-+++--+++++++-+++0+-+-0-0-++---+0---++0-+-+-++-",
        ALL_CHANNELS,
    ),
    30: PreambleCodeSpec(
        30,
        "-++0--0++-00--++-++-+--+++++---+++-+-0+-+-+0+++++-+++-+++-+0------++-++-+0----+-+-0+0+-+++-",
        ALL_CHANNELS,
    ),
    31: PreambleCodeSpec(
        31,
        "-+-++0+++--++-0++00--++--+++-+------0+++-+++++--+-++---+-+---+-+0-+--0+0+-++-++0+-+--0+++++",
        ALL_CHANNELS,
    ),
    32: PreambleCodeSpec(
        32,
        "-+++++++++++---+-++--++00-+-+0-+-0-++--+++-+++0--0++-+-+-0---++-0----++++-+-0+0-+-++-++--+-",
        ALL_CHANNELS,
    ),
}


@dataclass(frozen=True)
class PreambleTimingSpec:
    """Preamble spreading parameters (Tables 16-5 and 16-10)."""

    code_length: int
    delta_l: int
    chips_per_symbol: int
    preamble_peak_prf_mhz: float
    preamble_mean_prf_mhz: float


# Keyed by (code_length, preamble_mean_prf_mhz)
PREAMBLE_TIMING: dict[tuple[int, float], PreambleTimingSpec] = {
    (31, 4.03): PreambleTimingSpec(31, 64, 1984, 7.80, 4.03),
    (31, 16.1): PreambleTimingSpec(31, 16, 496, 31.20, 16.10),
    (127, 62.89): PreambleTimingSpec(127, 4, 508, 124.80, 62.89),
    (91, 111.09): PreambleTimingSpec(91, 4, 364, 124.80, 111.09),
}


# SFD sequences (Section 16.2.6.3 and Table 16-11)
SFD_SEQUENCES: dict[int | str, np.ndarray] = {
    0: np.array([0, 1, 0, -1, 1, 0, 0, -1], dtype=np.int8),
    1: np.array([-1, -1, 1, -1], dtype=np.int8),
    2: np.array([-1, -1, -1, 1, -1, -1, 1, -1], dtype=np.int8),
    3: np.array(
        [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1],
        dtype=np.int8,
    ),
    4: np.array(
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
        ],
        dtype=np.int8,
    ),
    "long": np.array(
        [
            0,
            1,
            0,
            -1,
            1,
            0,
            0,
            -1,
            0,
            1,
            0,
            -1,
            1,
            0,
            0,
            -1,
            -1,
            0,
            0,
            1,
            0,
            -1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            -1,
            0,
            -1,
            0,
            -1,
            0,
            0,
            1,
            0,
            -1,
            -1,
            0,
            -1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
        ],
        dtype=np.int8,
    ),
}


@dataclass(frozen=True)
class BPMBPSKParams:
    """BPM-BPSK data symbol structure parameters (Tables 16-4 and 16-12)."""

    mean_prf_mhz: float
    data_rate_mbps: float
    viterbi_rate: float
    n_burst: int
    n_hop: int
    n_cpb: int
    chips_per_symbol: int
    phr_rate_bits: tuple[int, int]


# Table 16-4 & Table 16-12 keyed by (mean_prf_mhz, data_rate_mbps)
BPM_BPSK_PARAMS: dict[tuple[float, float], BPMBPSKParams] = {
    # 3.90 MHz mean PRF (length-31 preamble)
    (3.9, 0.11): BPMBPSKParams(3.9, 0.11, 0.5, 128, 32, 32, 4096, (0, 0)),
    (3.9, 0.85): BPMBPSKParams(3.9, 0.85, 0.5, 128, 32, 4, 512, (0, 1)),
    (3.9, 1.70): BPMBPSKParams(3.9, 1.70, 0.5, 128, 32, 2, 256, (1, 0)),
    (3.9, 6.81): BPMBPSKParams(3.9, 6.81, 1.0, 128, 32, 1, 128, (1, 1)),
    # 15.60 MHz mean PRF (length-31 preamble)
    (15.6, 0.11): BPMBPSKParams(15.6, 0.11, 0.5, 32, 8, 128, 4096, (0, 0)),
    (15.6, 0.85): BPMBPSKParams(15.6, 0.85, 0.5, 32, 8, 16, 512, (0, 1)),
    (15.6, 6.81): BPMBPSKParams(15.6, 6.81, 0.5, 32, 8, 2, 64, (1, 0)),
    (15.6, 27.24): BPMBPSKParams(15.6, 27.24, 1.0, 32, 8, 1, 32, (1, 1)),
    # 62.40 MHz mean PRF (length-127 preamble; legacy & BPRF)
    (62.4, 0.11): BPMBPSKParams(62.4, 0.11, 0.5, 8, 2, 512, 4096, (0, 0)),
    (62.4, 0.85): BPMBPSKParams(62.4, 0.85, 0.5, 8, 2, 64, 512, (0, 1)),
    (62.4, 6.81): BPMBPSKParams(62.4, 6.81, 0.5, 8, 2, 8, 64, (1, 0)),
    (62.4, 27.24): BPMBPSKParams(62.4, 27.24, 0.5, 8, 2, 2, 16, (1, 1)),
}


@dataclass(frozen=True)
class HPRFModParams:
    """HPRF modulation parameters (Tables 16-21 and 16-22)."""

    mean_prf_mhz: float
    constraint_length: int
    data_rate_mbps: float
    phr_rate_mbps: float
    pulses_per_data_symbol: int
    chips_per_data_symbol: int
    pulses_per_phr_symbol: int
    chips_per_phr_symbol: int
    pulses_per_burst: int
    pulse_spacing_chips: int
    guard_chips: int


# Tables 16-21 & 16-22 keyed by (mean_prf_mhz, constraint_length)
HPRF_MOD_PARAMS: dict[tuple[float, int], HPRFModParams] = {
    (124.8, 3): HPRFModParams(124.8, 3, 6.81, 3.9, 16, 64, 32, 128, 8, 2, 16),
    (124.8, 7): HPRFModParams(124.8, 7, 7.8, 7.8, 16, 64, 16, 64, 8, 2, 16),
    (249.6, 3): HPRFModParams(249.6, 3, 27.24, 15.6, 8, 16, 16, 32, 4, 1, 4),
    (249.6, 7): HPRFModParams(249.6, 7, 31.2, 31.2, 8, 16, 8, 16, 4, 1, 4),
}

# Table 16-13: Preamble duration PHR field values (b11, b12)
PREAMBLE_DURATION_BITS: dict[int, tuple[int, int]] = {
    16: (0, 0),
    64: (0, 1),
    1024: (1, 0),
    4096: (1, 1),
}

# Valid HPRF PSR values (Section 16.2.6.2)
HPRF_VALID_PSR: frozenset[int] = frozenset({16, 24, 32, 48, 64, 96, 128, 256})

# Valid STS segment lengths in units of 512 chips (Table 16-18)
VALID_STS_SEGMENT_LENGTHS: frozenset[int] = frozenset({16, 32, 64, 128, 256})

# HPRF burst bit patterns (first_burst_bit, second_burst_bit) indexed by (g0, g1)
# Tables 16-23 & 16-25 (K = 3)
HPRF_SYMBOL_MAP_K3: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (0, 0),
    (1, 0): (1, 0),
    (0, 1): (1, 1),
    (1, 1): (0, 1),
}

# Tables 16-24 & 16-26 (K = 7)
HPRF_SYMBOL_MAP_K7: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (0, 0),
    (1, 0): (1, 0),
    (0, 1): (0, 1),
    (1, 1): (1, 1),
}
