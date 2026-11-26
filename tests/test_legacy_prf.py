"""Comprehensive round-trip tests for legacy IEEE 802.15.4a / PRF < 62.4 MHz modes."""

from __future__ import annotations

import numpy as np
import pytest

from uwb_hrp import (
    HRPConfig,
    bits_lsb_to_bytes,
    bytes_to_bits_lsb,
    field_indices,
    generate_chips,
)
from uwb_hrp.fec import rs_encode
from uwb_hrp.phr import compute_secded
from uwb_hrp.scrambler import generate_scrambler_bits
from uwb_hrp.tables import BPM_BPSK_PARAMS, PREAMBLE_DURATION_BITS


def _demodulate_bpm_bpsk(
    chips: np.ndarray,
    scrambler: np.ndarray,
    n_burst: int,
    n_hop: int,
    n_cpb: int,
    scrambler_offset: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Coherent reference BPM-BPSK demodulator verifying exact hopping and scrambling."""
    chips_per_sym = n_burst * n_cpb
    assert len(chips) % chips_per_sym == 0
    n_syms = len(chips) // chips_per_sym
    m = n_hop.bit_length() - 1
    half_sym = chips_per_sym // 2

    g0_out = np.empty(n_syms, dtype=np.uint8)
    g1_out = np.empty(n_syms, dtype=np.uint8)

    for k in range(n_syms):
        sym = chips[k * chips_per_sym : (k + 1) * chips_per_sym]
        nz = np.flatnonzero(sym)
        assert len(nz) == n_cpb, f"Symbol {k}: expected {n_cpb} active chips, got {len(nz)}"

        off = scrambler_offset + k * n_cpb
        expected_h = sum(int(scrambler[off + j]) << j for j in range(m))
        g0 = 1 if nz[0] >= half_sym else 0
        expected_start = g0 * half_sym + expected_h * n_cpb
        assert np.array_equal(nz, np.arange(expected_start, expected_start + n_cpb)), (
            f"Symbol {k}: burst at {nz[0]}, expected {expected_start}"
        )

        burst = sym[expected_start : expected_start + n_cpb]
        scram_chips = 1 - 2 * scrambler[off : off + n_cpb].astype(np.int8)
        if np.array_equal(burst, scram_chips):
            g1 = 0
        elif np.array_equal(burst, -scram_chips):
            g1 = 1
        else:
            raise AssertionError(f"Symbol {k}: burst does not match scrambler polarity")

        g0_out[k] = g0
        g1_out[k] = g1

    return g0_out, g1_out, scrambler_offset + n_syms * n_cpb


@pytest.mark.parametrize(
    ("channel", "mean_prf", "data_rate", "code_index"),
    [
        # 3.90 MHz mean PRF (all 4 data rates, including Viterbi rate 1 at 6.81 Mb/s & N_cpb < m)
        (0, 3.9, 0.11, 1),
        (1, 3.9, 0.85, 2),
        (2, 3.9, 1.70, 3),
        (9, 3.9, 6.81, 4),
        # 15.60 MHz mean PRF (all 4 data rates, including Viterbi rate 1 at 27.24 Mb/s & wideband)
        (3, 15.6, 0.11, 5),
        (6, 15.6, 0.85, 6),
        (4, 15.6, 6.81, 7),
        (15, 15.6, 27.24, 8),
        # Legacy 62.40 MHz mean PRF (all 4 data rates, including 0.11 and 27.24 Mb/s)
        (5, 62.4, 0.11, 9),
        (9, 62.4, 0.85, 10),
        (11, 62.4, 6.81, 17),
        (14, 62.4, 27.24, 12),
    ],
)
def test_legacy_roundtrip_all_table_16_4_modes(
    channel: int,
    mean_prf: float,
    data_rate: float,
    code_index: int,
) -> None:
    payload = bytes((i * 37 + 11) & 0xFF for i in range(25))
    cfg = HRPConfig(
        mode="802.15.4a",
        channel=channel,
        mean_prf=mean_prf,
        data_rate=data_rate,
        code_index=code_index,
        preamble_duration=16,
        psdu_length=len(payload),
        ranging=True,
    )
    chips = generate_chips(payload, cfg)
    idx = field_indices(cfg)
    assert len(chips) == idx.total_chips
    assert idx.phr is not None and idx.psdu is not None
    assert idx.sts1 is None and idx.sts2 is None

    # Verify SFD length (64 for 0.11 Mb/s, 8 otherwise)
    expected_sfd_syms = 64 if data_rate == 0.11 else 8
    assert (idx.sfd.stop - idx.sfd.start) == expected_sfd_syms * cfg.chips_per_preamble_symbol

    # Coherent demodulation of PHR & PSDU
    assert cfg.phr_data_rate is not None
    phr_p = BPM_BPSK_PARAMS[(mean_prf, cfg.phr_data_rate)]
    data_p = BPM_BPSK_PARAMS[(mean_prf, data_rate)]
    m_max = max(phr_p.n_hop, data_p.n_hop).bit_length() - 1
    scram = generate_scrambler_bits(code_index, 21 * phr_p.n_cpb + 2000 * data_p.n_cpb + m_max)

    g0_phr, g1_phr, off = _demodulate_bpm_bpsk(
        chips[idx.phr], scram, phr_p.n_burst, phr_p.n_hop, phr_p.n_cpb, 0
    )
    g0_psdu, g1_psdu, _ = _demodulate_bpm_bpsk(
        chips[idx.psdu], scram, data_p.n_burst, data_p.n_hop, data_p.n_cpb, off
    )

    # Recover PHR bits H0..H18 from g0_phr[1:20] (Table 16-2 & Table 16-3)
    assert g0_phr[0] == 0
    phr_bits = g0_phr[1:20]
    assert tuple(phr_bits[:2].tolist()) == data_p.phr_rate_bits
    decoded_len = int("".join(str(int(b)) for b in phr_bits[2:9]), 2)
    assert decoded_len == len(payload)
    assert phr_bits[9] == 1  # ranging=True
    assert phr_bits[10] == 0  # reserved
    assert tuple(phr_bits[11:13].tolist()) == PREAMBLE_DURATION_BITS[16]
    assert phr_bits[13:19].tolist() == compute_secded(phr_bits[:13].tolist())

    # Recover RS-encoded PSDU bits and compare against expected RS encoding
    expected_rs = rs_encode(bytes_to_bits_lsb(payload))
    if data_p.viterbi_rate == 0.5:
        # Table 16-2: D0 is g0_phr[20], D1..DN-1 are g0_psdu[0:-1], T0 is g0_psdu[-1] == 0
        recovered_rs = np.concatenate((g0_phr[20:21], g0_psdu[:-1]))
        assert g0_psdu[-1] == 0  # tail bit T0
        # Also verify K=3 parity g1_phr and g1_psdu match Table 16-2
        full_in = np.concatenate((phr_bits, expected_rs, np.zeros(2, dtype=np.uint8)))
        padded = np.concatenate((np.zeros(2, dtype=np.uint8), full_in))
        expected_g1 = (padded[2:] ^ padded[:-2]).astype(np.uint8)
        assert np.array_equal(np.concatenate((g1_phr, g1_psdu)), expected_g1)
    else:
        # Table 16-3: PHR tail T0 is g0_phr[20] == 0; even RS bits -> g0_psdu, odd -> g1_psdu
        assert g0_phr[20] == 0
        recovered_rs = np.empty(len(g0_psdu) * 2, dtype=np.uint8)
        recovered_rs[0::2] = g0_psdu
        recovered_rs[1::2] = g1_psdu

    assert np.array_equal(recovered_rs, expected_rs)
    assert bits_lsb_to_bytes(recovered_rs[: len(payload) * 8]) == payload


@pytest.mark.parametrize("preamble_duration", [16, 64, 1024, 4096])
def test_legacy_preamble_durations_15_6_mhz(preamble_duration: int) -> None:
    cfg = HRPConfig(
        mode="802.15.4a",
        channel=9,
        mean_prf=15.6,
        data_rate=0.85,
        preamble_duration=preamble_duration,
        psdu_length=0,
    )
    chips = generate_chips(b"", cfg)
    idx = field_indices(cfg)
    assert len(chips) == idx.total_chips
    assert (idx.sync.stop - idx.sync.start) == preamble_duration * 496
    assert idx.psdu is None


def test_legacy_zero_length_psdu_phr_tail() -> None:
    cfg = HRPConfig(mode="802.15.4a", channel=9, mean_prf=3.9, data_rate=0.85, psdu_length=0)
    chips = generate_chips(None, cfg)
    idx = field_indices(cfg)
    assert idx.psdu is None
    assert idx.phr is not None
    assert (idx.phr.stop - idx.phr.start) == 21 * 512
    assert len(chips) == idx.total_chips
