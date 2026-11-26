"""Coherent round-trip demodulation tests for HPRF K=3 and K=7 modes (Section 16.3.4)."""

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
from uwb_hrp.tables import HPRF_MOD_PARAMS, HPRF_SYMBOL_MAP_K3, HPRF_SYMBOL_MAP_K7


def _demodulate_hprf(
    chips: np.ndarray,
    scrambler: np.ndarray,
    *,
    mean_prf: float,
    constraint_length: int,
    is_phr: bool,
    scrambler_offset: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Coherent reference HPRF demodulator verifying burst structure, guards, and scrambling."""
    params = HPRF_MOD_PARAMS[(mean_prf, constraint_length)]
    sym_map = HPRF_SYMBOL_MAP_K3 if constraint_length == 3 else HPRF_SYMBOL_MAP_K7
    inv_map = {v: k for k, v in sym_map.items()}

    n_bursts = 4 if (is_phr and constraint_length == 3) else 2
    pulses_per_burst = params.pulses_per_burst
    spacing = params.pulse_spacing_chips
    guard = params.guard_chips
    burst_slot = pulses_per_burst * spacing + guard
    chips_per_sym = n_bursts * burst_slot
    pulses_per_sym = n_bursts * pulses_per_burst

    assert len(chips) % chips_per_sym == 0
    n_syms = len(chips) // chips_per_sym

    g0_out = np.empty(n_syms, dtype=np.uint8)
    g1_out = np.empty(n_syms, dtype=np.uint8)

    for k in range(n_syms):
        sym = chips[k * chips_per_sym : (k + 1) * chips_per_sym].reshape(n_bursts, burst_slot)
        # Guard intervals and inter-pulse zero positions must be silent
        assert np.all(sym[:, pulses_per_burst * spacing :] == 0)
        if spacing > 1:
            assert np.all(sym[:, 1 : pulses_per_burst * spacing : spacing] == 0)

        active = sym[:, : pulses_per_burst * spacing : spacing].reshape(-1)
        assert np.all((active == 1) | (active == -1))

        off = scrambler_offset + k * pulses_per_sym
        scram_bits = scrambler[off : off + pulses_per_sym]
        rx_bits = ((1 - active) // 2).astype(np.uint8) ^ scram_bits
        bursts = rx_bits.reshape(n_bursts, pulses_per_burst)

        # Every pulse within a burst must carry the same unscrambled bit
        assert np.all(bursts == bursts[:, :1])
        b_vals = bursts[:, 0].tolist()
        if n_bursts == 4:
            assert b_vals[:2] == b_vals[2:]

        g0_val, g1_val = inv_map[(int(b_vals[0]), int(b_vals[1]))]
        g0_out[k] = g0_val
        g1_out[k] = g1_val

    return g0_out, g1_out, scrambler_offset + n_syms * pulses_per_sym


def _decode_conv_k7(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Exact noiseless sequential decoder for K=7 (133_8, 171_8) verifying both polynomials."""
    assert len(g0) == len(g1) and len(g0) >= 6
    state = [0, 0, 0, 0, 0, 0]  # [D^1 .. D^6]
    decoded = np.empty(len(g0), dtype=np.uint8)

    for t in range(len(g0)):
        # g0 = u ^ D^2 ^ D^3 ^ D^5 ^ D^6 (133_8 = 1011011)
        u = int(g0[t]) ^ state[1] ^ state[2] ^ state[4] ^ state[5]
        # Verify g1 = u ^ D^1 ^ D^2 ^ D^3 ^ D^6 (171_8 = 1111001)
        expected_g1 = u ^ state[0] ^ state[1] ^ state[2] ^ state[5]
        assert int(g1[t]) == expected_g1, f"K=7 parity mismatch at symbol {t}"
        decoded[t] = u
        state = [u, *state[:-1]]

    # Last 6 bits must be zero tail bits
    assert np.all(decoded[-6:] == 0)
    return decoded[:-6]


def _strip_rs_parity(rs_bits: np.ndarray, num_info_bits: int) -> np.ndarray:
    """Extract the systematic information bits from multi-block RS6(63,55) output."""
    chunks: list[np.ndarray] = []
    pos = 0
    remaining = num_info_bits
    while remaining > 0:
        data_len = min(330, remaining)
        chunks.append(rs_bits[pos : pos + data_len])
        pos += data_len + 48
        remaining -= data_len
    assert pos == len(rs_bits)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)


@pytest.mark.parametrize(
    ("mean_prf", "constraint_length", "psdu_length", "ranging", "sts_cfg", "gap_idx"),
    [
        (124.8, 3, 20, True, 0, None),
        (124.8, 3, 1024, False, 2, 3),  # Table 16-16 ext=1 (A0=len bit10, A1=gap_idx MSB)
        (124.8, 7, 0, True, 0, None),  # K=7 zero-byte PSDU (6 tail symbols)
        (124.8, 7, 37, False, 1, None),
        (249.6, 3, 4095, True, 0, None),  # Max HPRF length (ext=2: A1/A0 = len bits 11/10)
        (249.6, 7, 2048, True, 0, None),  # K=7 extended length
        (249.6, 7, 25, False, 2, 2),  # K=7 SP2 with gap index
    ],
)
def test_hprf_roundtrip_demodulation_and_phr_decoding(
    mean_prf: float,
    constraint_length: int,
    psdu_length: int,
    ranging: bool,
    sts_cfg: int,
    gap_idx: int | None,
) -> None:
    payload = bytes((i * 53 + 7) & 0xFF for i in range(psdu_length))
    cfg = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=mean_prf,
        code_index=25,
        preamble_duration=32,
        sfd_number=2,
        psdu_length=psdu_length,
        ranging=ranging,
        constraint_length=constraint_length,  # type: ignore[arg-type]
        sts_packet_config=sts_cfg,  # type: ignore[arg-type]
        extra_sts_gap=8 if sts_cfg == 2 else 0,
        extra_sts_gap_index=gap_idx,
    )
    chips = generate_chips(payload, cfg)
    idx = field_indices(cfg)
    assert len(chips) == idx.total_chips
    assert idx.phr is not None

    params = HPRF_MOD_PARAMS[(mean_prf, constraint_length)]
    phr_syms = 25 if constraint_length == 7 else 21
    psdu_chips = chips[idx.psdu] if idx.psdu is not None else np.empty(0, dtype=np.int8)
    psdu_syms = len(psdu_chips) // params.chips_per_data_symbol
    total_scram = (
        phr_syms * params.pulses_per_phr_symbol + psdu_syms * params.pulses_per_data_symbol
    )
    scram = generate_scrambler_bits(25, total_scram)

    g0_phr, g1_phr, off = _demodulate_hprf(
        chips[idx.phr],
        scram,
        mean_prf=mean_prf,
        constraint_length=constraint_length,
        is_phr=True,
        scrambler_offset=0,
    )
    g0_psdu, g1_psdu, _ = _demodulate_hprf(
        psdu_chips,
        scram,
        mean_prf=mean_prf,
        constraint_length=constraint_length,
        is_phr=False,
        scrambler_offset=off,
    )

    if constraint_length == 3:
        assert g0_phr[0] == 0
        phr_bits = g0_phr[1:20]
        expected_rs = rs_encode(bytes_to_bits_lsb(payload))
        recovered_rs = np.concatenate((g0_phr[20:21], g0_psdu[:-1]))
        assert g0_psdu[-1] == 0  # tail bit T0
        assert np.array_equal(recovered_rs, expected_rs)
        recovered_payload = bits_lsb_to_bytes(_strip_rs_parity(recovered_rs, psdu_length * 8))
    else:
        phr_bits = _decode_conv_k7(g0_phr, g1_phr)
        decoded_psdu_bits = _decode_conv_k7(g0_psdu, g1_psdu)
        assert len(decoded_psdu_bits) == psdu_length * 8
        recovered_payload = bits_lsb_to_bytes(decoded_psdu_bits)

    assert recovered_payload == payload

    # Verify Figure 16-8 HPRF PHR fields: A1, A0, L9..L0, R, SECDED
    a1, a0 = int(phr_bits[0]), int(phr_bits[1])
    low_10 = int("".join(str(int(b)) for b in phr_bits[2:12]), 2)
    assert low_10 == (psdu_length & 0x3FF)
    if cfg.psdu_size_extension == 2:
        assert (a1, a0) == ((psdu_length >> 11) & 1, (psdu_length >> 10) & 1)
    elif cfg.psdu_size_extension == 1:
        assert a0 == ((psdu_length >> 10) & 1)
        expected_a1 = ((gap_idx >> 1) & 1) if gap_idx is not None else 0
        assert a1 == expected_a1
    else:
        expected_a1 = ((gap_idx >> 1) & 1) if gap_idx is not None else 0
        expected_a0 = (gap_idx & 1) if gap_idx is not None else 0
        assert (a1, a0) == (expected_a1, expected_a0)

    assert int(phr_bits[12]) == int(ranging)
    assert phr_bits[13:19].tolist() == compute_secded(phr_bits[:13].tolist())
