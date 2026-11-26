"""IEEE Std 802.15.4-2024 HRP UWB PHY configuration and field index structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from uwb_hrp.tables import (
    BPM_BPSK_PARAMS,
    CHANNEL_SPECS,
    CHIP_DURATION_S,
    DEFAULT_STS_KEY,
    DEFAULT_STS_V,
    HPRF_MOD_PARAMS,
    HPRF_VALID_PSR,
    PEAK_PRF_HZ,
    PREAMBLE_CODES,
    PREAMBLE_DURATION_BITS,
    PREAMBLE_TIMING,
    SFD_SEQUENCES,
    VALID_STS_SEGMENT_LENGTHS,
    WIDEBAND_CHANNELS,
)

HRPMode = Literal["802.15.4a", "BPRF", "HPRF"]
SFDSelector = int | Literal["long"]


@dataclass(frozen=True)
class HRPConfig:
    """IEEE Std 802.15.4-2024 Clause 16 HRP UWB PHY frame configuration.

    This object encapsulates purely PHY-layer frame parameters (mode, channel,
    preamble code, PRF, data rate, PHR, SFD, PSDU length, and STS). Baseband
    pulse shaping and oversampling are configured separately via
    :class:`uwb_hrp.PulseConfig` and :class:`uwb_hrp.PulseShaper`.
    """

    mode: HRPMode = "HPRF"
    channel: int = 9
    code_index: int | None = None
    mean_prf: float | None = None
    preamble_mean_prf: float | None = None
    data_rate: float | None = None
    phr_data_rate: float | None = None
    preamble_duration: int | None = None
    sfd_number: SFDSelector | None = None
    psdu_length: int = 20
    ranging: bool = False
    constraint_length: Literal[3, 7] = 3
    sts_packet_config: Literal[0, 1, 2, 3] = 0
    num_sts_segments: int = 1
    sts_segment_length: int = 64
    sts_key: bytes = DEFAULT_STS_KEY
    sts_v: bytes = DEFAULT_STS_V
    extra_sts_gap: int = 0
    extra_sts_gap_index: int | None = None
    psdu_size_extension: Literal[0, 1, 2] | None = None
    sts_trailing_gap: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("802.15.4a", "BPRF", "HPRF"):
            raise ValueError(
                f"Unsupported mode {self.mode!r}; expected '802.15.4a', 'BPRF', or 'HPRF'."
            )

        if self.channel not in CHANNEL_SPECS:
            raise ValueError(f"Invalid channel {self.channel}; expected integer in 0..15.")

        if self.sts_packet_config not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid sts_packet_config {self.sts_packet_config}; expected 0, 1, 2, or 3."
            )

        if self.constraint_length not in (3, 7):
            raise ValueError(
                f"Invalid constraint_length {self.constraint_length}; expected 3 or 7."
            )

        if self.mode == "802.15.4a":
            self._init_legacy()
        elif self.mode == "BPRF":
            self._init_bprf()
        else:
            self._init_hprf()

        self._validate_sts()

    def _set(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    def _default_code_for_channel(self, code_length: int) -> int:
        for idx, spec in PREAMBLE_CODES.items():
            if spec.length == code_length and not spec.dps_only and self.channel in spec.channels:
                return idx
        raise ValueError(
            f"No non-DPS preamble code of length {code_length} available for "
            f"channel {self.channel}."
        )

    def _init_legacy(self) -> None:
        if self.sts_packet_config != 0:
            raise ValueError("STS is only supported in 'BPRF' and 'HPRF' modes, not '802.15.4a'.")
        if self.constraint_length != 3:
            raise ValueError("constraint_length must be 3 in '802.15.4a' mode.")
        if self.extra_sts_gap != 0 or self.extra_sts_gap_index is not None:
            raise ValueError("STS gap parameters are not valid in '802.15.4a' mode.")
        if self.psdu_size_extension is not None and self.psdu_size_extension != 0:
            raise ValueError("psdu_size_extension is only supported in 'HPRF' mode.")
        self._set("psdu_size_extension", 0)

        # Infer mean_prf from code_index if omitted
        if self.mean_prf is None:
            if self.code_index is not None and self.code_index in PREAMBLE_CODES:
                code_len = PREAMBLE_CODES[self.code_index].length
                self._set("mean_prf", 62.4 if code_len == 127 else 15.6)
            else:
                self._set("mean_prf", 15.6)

        if self.mean_prf not in (3.9, 15.6, 62.4):
            raise ValueError(
                f"Invalid mean_prf {self.mean_prf} MHz for '802.15.4a'; "
                "expected 3.9, 15.6, or 62.4."
            )

        if self.mean_prf == 3.9 and self.channel in WIDEBAND_CHANNELS:
            raise ValueError(
                f"mean_prf=3.9 MHz is not allowed on wideband channel {self.channel} (Table 16-4)."
            )

        expected_code_len = 127 if self.mean_prf == 62.4 else 31
        if self.code_index is None:
            self._set("code_index", self._default_code_for_channel(expected_code_len))
        self._validate_code_index(expected_code_len)

        if expected_code_len == 31:
            if self.preamble_mean_prf is None:
                self._set("preamble_mean_prf", 16.1)
            elif self.preamble_mean_prf not in (4.03, 16.1):
                raise ValueError(
                    f"Invalid preamble_mean_prf {self.preamble_mean_prf} MHz for length-31 codes; "
                    "expected 16.1 or 4.03."
                )
            if self.preamble_mean_prf == 4.03 and self.channel in WIDEBAND_CHANNELS:
                raise ValueError(
                    f"preamble_mean_prf=4.03 MHz is not allowed on wideband channel {self.channel} "
                    "(Table 16-5)."
                )
        else:
            if self.preamble_mean_prf is None:
                self._set("preamble_mean_prf", 62.89)
            elif self.preamble_mean_prf != 62.89:
                raise ValueError(
                    f"Invalid preamble_mean_prf {self.preamble_mean_prf} MHz for length-127 codes; "
                    "expected 62.89."
                )

        if self.data_rate is None:
            self._set("data_rate", 0.85)
        assert self.data_rate is not None
        if (self.mean_prf, self.data_rate) not in BPM_BPSK_PARAMS:
            valid_rates = sorted(r for prf, r in BPM_BPSK_PARAMS if prf == self.mean_prf)
            raise ValueError(
                f"Invalid data_rate {self.data_rate} Mb/s for mean_prf={self.mean_prf} MHz; "
                f"valid rates: {valid_rates}."
            )

        expected_phr_rate = 0.11 if self.data_rate == 0.11 else 0.85
        allowed_phr_rates: tuple[float, ...] = (
            (0.85, 6.81)
            if (self.mean_prf == 62.4 and self.data_rate == 6.81)
            else (expected_phr_rate,)
        )
        if self.phr_data_rate is None:
            self._set("phr_data_rate", expected_phr_rate)
        elif self.phr_data_rate not in allowed_phr_rates:
            raise ValueError(
                f"In '802.15.4a' mode (mean_prf={self.mean_prf}, data_rate={self.data_rate}), "
                f"phr_data_rate must be one of {allowed_phr_rates} Mb/s."
            )

        if self.preamble_duration is None:
            self._set("preamble_duration", 64)
        if self.preamble_duration not in PREAMBLE_DURATION_BITS:
            raise ValueError(
                f"Invalid preamble_duration {self.preamble_duration} for '802.15.4a'; "
                "expected 16, 64, 1024, or 4096."
            )
        if self.preamble_mean_prf == 4.03 and self.preamble_duration == 4096:
            raise ValueError(
                "preamble_duration=4096 is not allowed at preamble_mean_prf=4.03 MHz "
                "(Table 16-6 footnote a)."
            )

        if self.sfd_number is None:
            self._set("sfd_number", "long" if self.data_rate == 0.11 else 0)
        if self.sfd_number not in (0, "long"):
            raise ValueError(
                f"Invalid sfd_number {self.sfd_number!r} for '802.15.4a'; "
                "expected 0 (short) or 'long'."
            )

        if not (0 <= self.psdu_length <= 127):
            raise ValueError(
                f"psdu_length must be in 0..127 for '802.15.4a', got {self.psdu_length}."
            )

    def _init_bprf(self) -> None:
        if self.constraint_length != 3:
            raise ValueError("constraint_length must be 3 in 'BPRF' mode.")
        if self.psdu_size_extension is not None and self.psdu_size_extension != 0:
            raise ValueError("psdu_size_extension is only supported in 'HPRF' mode.")
        self._set("psdu_size_extension", 0)
        if self.extra_sts_gap_index is not None:
            raise ValueError("extra_sts_gap_index (PHR A1/A0) is only defined in 'HPRF' mode.")

        if self.mean_prf is None:
            self._set("mean_prf", 62.4)
        elif self.mean_prf != 62.4:
            raise ValueError(f"BPRF mode requires mean_prf=62.4 MHz, got {self.mean_prf}.")

        if self.code_index is None:
            self._set("code_index", self._default_code_for_channel(127))
        self._validate_code_index(127)

        if self.preamble_mean_prf is None:
            self._set("preamble_mean_prf", 62.89)
        elif self.preamble_mean_prf != 62.89:
            raise ValueError(
                f"BPRF mode requires preamble_mean_prf=62.89 MHz, got {self.preamble_mean_prf}."
            )

        if self.data_rate is None:
            self._set("data_rate", 6.81)
        elif self.data_rate != 6.81:
            raise ValueError(f"BPRF mode requires data_rate=6.81 Mb/s, got {self.data_rate}.")

        if self.phr_data_rate is None:
            self._set("phr_data_rate", 0.85)
        elif self.phr_data_rate not in (0.85, 6.81):
            raise ValueError(
                "BPRF mode requires phr_data_rate of 0.85 (DRBM_LP) or 6.81 (DRBM_HP), "
                f"got {self.phr_data_rate}."
            )

        if self.preamble_duration is None:
            self._set("preamble_duration", 64)
        if self.preamble_duration not in PREAMBLE_DURATION_BITS:
            raise ValueError(
                f"Invalid preamble_duration {self.preamble_duration} for 'BPRF'; "
                "expected 16, 64, 1024, or 4096."
            )

        if self.sfd_number is None:
            self._set("sfd_number", 2)
        if self.sfd_number not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"Invalid sfd_number {self.sfd_number!r} for 'BPRF'; "
                "expected 0, 1, 2, 3, or 4 (Table 16-11)."
            )

        if self.sts_packet_config == 3:
            self._set("psdu_length", 0)
        elif not (0 <= self.psdu_length <= 127):
            raise ValueError(f"psdu_length must be in 0..127 for 'BPRF', got {self.psdu_length}.")

    def _init_hprf(self) -> None:
        if self.mean_prf is None:
            if self.data_rate in (27.24, 31.2):
                self._set("mean_prf", 249.6)
            else:
                self._set("mean_prf", 124.8)
        if self.mean_prf not in (124.8, 249.6):
            raise ValueError(
                f"HPRF mode requires mean_prf of 124.8 or 249.6 MHz, got {self.mean_prf}."
            )

        if self.code_index is None:
            self._set("code_index", 25)
        self._validate_code_index(91)

        if self.preamble_mean_prf is None:
            self._set("preamble_mean_prf", 111.09)
        elif self.preamble_mean_prf != 111.09:
            raise ValueError(
                f"HPRF mode requires preamble_mean_prf=111.09 MHz, got {self.preamble_mean_prf}."
            )

        mod = HPRF_MOD_PARAMS[(self.mean_prf, self.constraint_length)]
        if self.data_rate is None:
            self._set("data_rate", mod.data_rate_mbps)
        elif self.data_rate != mod.data_rate_mbps:
            raise ValueError(
                f"In HPRF mode with mean_prf={self.mean_prf} and "
                f"constraint_length={self.constraint_length}, "
                f"data_rate must be {mod.data_rate_mbps} Mb/s, got {self.data_rate}."
            )

        if self.phr_data_rate is None:
            self._set("phr_data_rate", mod.phr_rate_mbps)
        elif self.phr_data_rate != mod.phr_rate_mbps:
            raise ValueError(
                f"In HPRF mode with mean_prf={self.mean_prf} and "
                f"constraint_length={self.constraint_length}, "
                f"phr_data_rate must be {mod.phr_rate_mbps} Mb/s, got {self.phr_data_rate}."
            )

        if self.preamble_duration is None:
            self._set("preamble_duration", 64)
        if self.preamble_duration not in HPRF_VALID_PSR:
            raise ValueError(
                f"Invalid preamble_duration {self.preamble_duration} for 'HPRF'; "
                f"expected one of {sorted(HPRF_VALID_PSR)}."
            )

        if self.sfd_number is None:
            self._set("sfd_number", 2)
        if self.sfd_number not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"Invalid sfd_number {self.sfd_number!r} for 'HPRF'; expected 0, 1, 2, 3, or 4."
            )

        if self.sts_packet_config == 3:
            self._set("psdu_length", 0)
            self._set("psdu_size_extension", 0)
        else:
            if not (0 <= self.psdu_length <= 4095):
                raise ValueError(
                    f"psdu_length must be in 0..4095 for 'HPRF', got {self.psdu_length}."
                )
            if self.psdu_size_extension is None:
                if self.psdu_length <= 1023:
                    self._set("psdu_size_extension", 0)
                elif self.psdu_length <= 2047:
                    self._set("psdu_size_extension", 1)
                else:
                    self._set("psdu_size_extension", 2)
            elif self.psdu_size_extension not in (0, 1, 2):
                raise ValueError(
                    f"psdu_size_extension must be 0, 1, or 2, got {self.psdu_size_extension}."
                )
            assert self.psdu_size_extension is not None
            max_len = (1023, 2047, 4095)[self.psdu_size_extension]
            if self.psdu_length > max_len:
                raise ValueError(
                    f"psdu_length={self.psdu_length} exceeds max {max_len} for "
                    f"psdu_size_extension={self.psdu_size_extension}."
                )

        if self.extra_sts_gap_index is not None:
            if self.sts_packet_config != 2:
                raise ValueError("extra_sts_gap_index is only valid when sts_packet_config == 2.")
            if not (0 <= self.extra_sts_gap_index <= 3):
                raise ValueError(
                    f"extra_sts_gap_index must be in 0..3, got {self.extra_sts_gap_index}."
                )
            if self.psdu_size_extension != 0:
                raise ValueError(
                    "extra_sts_gap_index uses PHR bits A1/A0 and conflicts with "
                    f"psdu_size_extension == {self.psdu_size_extension} (Table 16-16)."
                )

    def _validate_code_index(self, expected_length: int) -> None:
        assert self.code_index is not None
        if self.code_index not in PREAMBLE_CODES:
            raise ValueError(f"Invalid code_index {self.code_index}; expected integer in 1..32.")
        spec = PREAMBLE_CODES[self.code_index]
        if spec.length != expected_length:
            raise ValueError(
                f"Code index {self.code_index} has length {spec.length}, "
                f"expected length {expected_length} for mode={self.mode!r}, "
                f"mean_prf={self.mean_prf} MHz."
            )
        if self.channel not in spec.channels:
            raise ValueError(
                f"Code index {self.code_index} is not permitted on channel {self.channel} "
                f"(allowed channels: {sorted(spec.channels)})."
            )

    def _validate_sts(self) -> None:
        if self.sts_packet_config == 0:
            if self.extra_sts_gap != 0:
                raise ValueError("extra_sts_gap is only valid when sts_packet_config == 2.")
            return
        if self.mode == "BPRF":
            if self.num_sts_segments != 1:
                raise ValueError(
                    f"BPRF mode only supports num_sts_segments=1 (Table 16-17), "
                    f"got {self.num_sts_segments}."
                )
            if self.sts_segment_length != 64:
                raise ValueError(
                    f"BPRF mode only supports sts_segment_length=64 (Table 16-17), "
                    f"got {self.sts_segment_length}."
                )
        if self.num_sts_segments not in (1, 2, 3, 4):
            raise ValueError(f"num_sts_segments must be in 1..4, got {self.num_sts_segments}.")
        if self.sts_segment_length not in VALID_STS_SEGMENT_LENGTHS:
            raise ValueError(
                f"sts_segment_length must be one of {sorted(VALID_STS_SEGMENT_LENGTHS)}, "
                f"got {self.sts_segment_length}."
            )
        if len(self.sts_key) != 16:
            raise ValueError(f"sts_key must be 16 bytes (128 bits), got {len(self.sts_key)} bytes.")
        if len(self.sts_v) != 16:
            raise ValueError(f"sts_v must be 16 bytes (128 bits), got {len(self.sts_v)} bytes.")
        if self.sts_packet_config != 2 and self.extra_sts_gap != 0:
            raise ValueError("extra_sts_gap is only valid when sts_packet_config == 2.")
        if not (0 <= self.extra_sts_gap <= 127):
            raise ValueError(
                f"extra_sts_gap must be in 0..127 (units of 4 chips), got {self.extra_sts_gap}."
            )

    # --- Derived PHY properties ---

    @property
    def peak_prf(self) -> float:
        """Peak pulse repetition frequency (chip rate) in Hz (499.2 MHz)."""
        return PEAK_PRF_HZ

    @property
    def chip_duration(self) -> float:
        """Chip duration Tc in seconds."""
        return CHIP_DURATION_S

    @property
    def center_frequency(self) -> float:
        """Channel center frequency in Hz (Table 16-27)."""
        return CHANNEL_SPECS[self.channel].center_freq_mhz * 1e6

    @property
    def bandwidth(self) -> float:
        """Channel 3 dB bandwidth in Hz (Table 16-27)."""
        return CHANNEL_SPECS[self.channel].bandwidth_mhz * 1e6

    @property
    def pulse_duration_tp(self) -> float:
        """Reference pulse duration Tp in seconds (Table 16-28)."""
        return CHANNEL_SPECS[self.channel].pulse_duration_tp_ns * 1e-9

    @property
    def main_lobe_tw(self) -> float:
        """Minimum compliant cross-correlation main-lobe width Tw in seconds (Table 16-28)."""
        return CHANNEL_SPECS[self.channel].main_lobe_tw_ns * 1e-9

    @property
    def preamble_code_length(self) -> int:
        assert self.code_index is not None
        return PREAMBLE_CODES[self.code_index].length

    @property
    def preamble_spreading_factor(self) -> int:
        assert self.preamble_mean_prf is not None
        return PREAMBLE_TIMING[(self.preamble_code_length, self.preamble_mean_prf)].delta_l

    @property
    def chips_per_preamble_symbol(self) -> int:
        assert self.preamble_mean_prf is not None
        return PREAMBLE_TIMING[(self.preamble_code_length, self.preamble_mean_prf)].chips_per_symbol

    @property
    def sfd_length(self) -> int:
        assert self.sfd_number is not None
        return len(SFD_SEQUENCES[self.sfd_number])

    @property
    def has_phr_and_psdu(self) -> bool:
        return self.sts_packet_config != 3

    @property
    def uses_reed_solomon(self) -> bool:
        return self.constraint_length == 3

    @property
    def viterbi_rate(self) -> float:
        if self.mode == "HPRF":
            return 0.5
        assert self.mean_prf is not None and self.data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.data_rate)].viterbi_rate

    @property
    def chips_per_burst(self) -> int:
        if self.mode == "HPRF":
            assert self.mean_prf is not None
            return HPRF_MOD_PARAMS[(self.mean_prf, self.constraint_length)].pulses_per_burst
        assert self.mean_prf is not None and self.data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.data_rate)].n_cpb

    @property
    def burst_positions_per_symbol(self) -> int | None:
        if self.mode == "HPRF":
            return None
        assert self.mean_prf is not None and self.data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.data_rate)].n_burst

    @property
    def hop_bursts(self) -> int | None:
        if self.mode == "HPRF":
            return None
        assert self.mean_prf is not None and self.data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.data_rate)].n_hop

    @property
    def chips_per_data_symbol(self) -> int:
        assert self.mean_prf is not None
        if self.mode == "HPRF":
            return HPRF_MOD_PARAMS[(self.mean_prf, self.constraint_length)].chips_per_data_symbol
        assert self.data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.data_rate)].chips_per_symbol

    @property
    def chips_per_phr_symbol(self) -> int:
        assert self.mean_prf is not None
        if self.mode == "HPRF":
            return HPRF_MOD_PARAMS[(self.mean_prf, self.constraint_length)].chips_per_phr_symbol
        assert self.phr_data_rate is not None
        return BPM_BPSK_PARAMS[(self.mean_prf, self.phr_data_rate)].chips_per_symbol

    @property
    def sts_spreading_factor(self) -> int | None:
        if self.sts_packet_config == 0:
            return None
        return 8 if self.mode == "BPRF" else 4


@dataclass(frozen=True)
class HRPFieldIndices:
    """Chip-level 0-based slices and ranging marker offsets for an HRP UWB PPDU."""

    sync: slice
    sfd: slice
    sts1: slice | None
    phr: slice | None
    psdu: slice | None
    sts2: slice | None
    total_chips: int
    rmarker_chip: int
    srmarker_chips: tuple[int, ...]

    def to_1based_ranges(self) -> dict[str, tuple[int, int] | None]:
        """Return 1-based inclusive ``(start, end)`` ranges matching IEEE ``packet_info.txt``."""

        def _conv(s: slice | None) -> tuple[int, int] | None:
            if s is None or s.start == s.stop:
                return None
            return (s.start + 1, s.stop)

        return {
            "SYNC": _conv(self.sync),
            "SFD": _conv(self.sfd),
            "STS1": _conv(self.sts1),
            "PHR": _conv(self.phr),
            "PSDU": _conv(self.psdu),
            "STS2": _conv(self.sts2),
        }
