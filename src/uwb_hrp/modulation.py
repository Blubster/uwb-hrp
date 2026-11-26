"""BPM-BPSK (Section 16.3.1) and HPRF BPSK (Section 16.3.4) modulators."""

from __future__ import annotations

import numpy as np

from uwb_hrp.config import HRPConfig
from uwb_hrp.fec import conv_encode_k3, conv_encode_k7, rs_encode
from uwb_hrp.phr import build_phr_bits
from uwb_hrp.scrambler import generate_scrambler_bits
from uwb_hrp.tables import (
    BPM_BPSK_PARAMS,
    HPRF_MOD_PARAMS,
    HPRF_SYMBOL_MAP_K3,
    HPRF_SYMBOL_MAP_K7,
)


def modulate_bpm_bpsk(
    g0: np.ndarray,
    g1: np.ndarray,
    scrambler: np.ndarray,
    n_burst: int,
    n_hop: int,
    n_cpb: int,
    scrambler_offset: int,
) -> tuple[np.ndarray, int]:
    """Modulate ``(g0, g1)`` symbol pairs with BPM-BPSK time hopping and scrambling.

    Implements Section 16.3.1 and Section 16.3.2 (including Note 3 when ``n_cpb < m``).
    Returns ``(chips, next_scrambler_offset)``.
    """
    n_syms = len(g0)
    chips_per_sym = n_burst * n_cpb
    if n_syms == 0:
        return np.empty(0, dtype=np.int8), scrambler_offset

    m = n_hop.bit_length() - 1
    half_sym = chips_per_sym // 2

    # Base scrambler index for each symbol k: scrambler_offset + k * n_cpb
    sym_offsets = scrambler_offset + np.arange(n_syms, dtype=np.int64) * n_cpb

    # Hopping position h^(k) = sum_{j=0..m-1} 2^j * s_{sym_offsets[k] + j}
    hop_indices = sym_offsets[:, None] + np.arange(m, dtype=np.int64)[None, :]
    hop_weights = (1 << np.arange(m, dtype=np.int64))[None, :]
    h = np.sum(scrambler[hop_indices].astype(np.int64) * hop_weights, axis=1)

    # Active burst chips: (1 - 2*g1) * (1 - 2*s_{sym_offsets[k] + n})
    burst_scram_idx = sym_offsets[:, None] + np.arange(n_cpb, dtype=np.int64)[None, :]
    polarity = (1 - 2 * g1.astype(np.int8))[:, None]
    burst_chips = polarity * (1 - 2 * scrambler[burst_scram_idx].astype(np.int8))

    # Place each symbol's burst into its symbol buffer
    out = np.zeros((n_syms, chips_per_sym), dtype=np.int8)
    start_cols = g0.astype(np.int64) * half_sym + h * n_cpb
    row_idx = np.arange(n_syms, dtype=np.int64)[:, None]
    col_idx = start_cols[:, None] + np.arange(n_cpb, dtype=np.int64)[None, :]
    out[row_idx, col_idx] = burst_chips

    return out.reshape(-1), scrambler_offset + n_syms * n_cpb


def _modulate_hprf_symbols(
    g0: np.ndarray,
    g1: np.ndarray,
    scrambler: np.ndarray,
    *,
    mean_prf: float,
    constraint_length: int,
    is_phr: bool,
    scrambler_offset: int,
) -> tuple[np.ndarray, int]:
    """Modulate ``(g0, g1)`` pairs in HPRF mode (Sections 16.3.4.2 and 16.3.4.3)."""
    n_syms = len(g0)
    if n_syms == 0:
        return np.empty(0, dtype=np.int8), scrambler_offset

    params = HPRF_MOD_PARAMS[(mean_prf, constraint_length)]
    sym_map = HPRF_SYMBOL_MAP_K3 if constraint_length == 3 else HPRF_SYMBOL_MAP_K7
    n_bursts = 4 if (is_phr and constraint_length == 3) else 2
    pulses_per_burst = params.pulses_per_burst
    spacing = params.pulse_spacing_chips
    guard = params.guard_chips
    burst_slot_chips = pulses_per_burst * spacing + guard
    chips_per_sym = n_bursts * burst_slot_chips
    pulses_per_sym = n_bursts * pulses_per_burst

    # Build 2x2 lookup table of burst bits for (g0, g1)
    lut = np.zeros((2, 2, 2), dtype=np.uint8)
    for (k0, k1), (b0, b1) in sym_map.items():
        lut[k0, k1, 0] = b0
        lut[k0, k1, 1] = b1

    # burst_vals shape: (n_syms, 2) -> tile to (n_syms, n_bursts) -> repeat per pulse
    burst_pair = lut[g0, g1]
    burst_bits = np.tile(burst_pair, (1, n_bursts // 2))  # (n_syms, n_bursts)
    pulse_bits = np.repeat(burst_bits, pulses_per_burst, axis=1)  # (n_syms, pulses_per_sym)

    scram_slice = scrambler[scrambler_offset : scrambler_offset + n_syms * pulses_per_sym].reshape(
        n_syms, pulses_per_sym
    )
    scrambled = pulse_bits ^ scram_slice
    pulses = (1 - 2 * scrambled.astype(np.int8)).reshape(n_syms, n_bursts, pulses_per_burst)

    # Scatter pulses into active burst slots separated by guard intervals
    out = np.zeros((n_syms, n_bursts, burst_slot_chips), dtype=np.int8)
    out[:, :, : pulses_per_burst * spacing : spacing] = pulses
    return out.reshape(n_syms * chips_per_sym), scrambler_offset + n_syms * pulses_per_sym


def encode_and_modulate_phr_psdu(
    psdu_bits: np.ndarray,
    cfg: HRPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode and modulate PHR and PSDU into ``(phr_chips, psdu_chips)``.

    Implements:
    - Table 16-2 (Viterbi rate 0.5) & Table 16-3 (Viterbi rate 1) for 802.15.4a and BPRF
    - Section 16.3.4 (K=3 with RS and K=7 without RS) for HPRF
    """
    if not cfg.has_phr_and_psdu:
        empty = np.empty(0, dtype=np.int8)
        return empty, empty

    assert cfg.code_index is not None and cfg.mean_prf is not None
    phr_bits = build_phr_bits(cfg)  # 19 bits H0..H18

    if cfg.mode == "HPRF":
        return _encode_modulate_hprf(phr_bits, psdu_bits, cfg)
    return _encode_modulate_bpm_bpsk(phr_bits, psdu_bits, cfg)


def _encode_modulate_bpm_bpsk(
    phr_bits: np.ndarray,
    psdu_bits: np.ndarray,
    cfg: HRPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    assert cfg.code_index is not None and cfg.mean_prf is not None
    assert cfg.phr_data_rate is not None and cfg.data_rate is not None

    phr_p = BPM_BPSK_PARAMS[(cfg.mean_prf, cfg.phr_data_rate)]
    data_p = BPM_BPSK_PARAMS[(cfg.mean_prf, cfg.data_rate)]
    rs_bits = rs_encode(psdu_bits)
    n_rs = len(rs_bits)

    if data_p.viterbi_rate == 0.5:
        # Table 16-2: PHR (19 bits) + RS-coded PSDU (N bits) + 2 tail bits -> N + 21 symbols
        # First 21 symbols (0..20) sent at PHR rate; remaining N symbols sent at data rate.
        if n_rs == 0:
            # Zero-length PSDU: full 21 PHR symbols with D0=D1=0 (Section 16.2.7.3 / Table 16-2)
            g0_all, g1_all = conv_encode_k3(phr_bits, append_tail=True)
            g0_phr, g1_phr = g0_all, g1_all
            g0_psdu = np.empty(0, dtype=np.uint8)
            g1_psdu = np.empty(0, dtype=np.uint8)
        else:
            combined = np.concatenate((phr_bits, rs_bits))
            g0_all, g1_all = conv_encode_k3(combined, append_tail=True)
            g0_phr, g1_phr = g0_all[:21], g1_all[:21]
            g0_psdu, g1_psdu = g0_all[21:], g1_all[21:]
    else:
        # Table 16-3 (Viterbi rate 1): PHR encoded separately with K=3 + 2 tail bits (21 symbols);
        # PSDU bypasses conv encoder (even RS bits -> position g0, odd RS bits -> polarity g1).
        g0_phr, g1_phr = conv_encode_k3(phr_bits, append_tail=True)
        g0_psdu = rs_bits[0::2]
        g1_psdu = rs_bits[1::2]

    m_max = max(phr_p.n_hop, data_p.n_hop).bit_length() - 1
    total_scram = len(g0_phr) * phr_p.n_cpb + len(g0_psdu) * data_p.n_cpb + m_max
    scrambler = generate_scrambler_bits(cfg.code_index, total_scram)

    phr_chips, off = modulate_bpm_bpsk(
        g0_phr, g1_phr, scrambler, phr_p.n_burst, phr_p.n_hop, phr_p.n_cpb, 0
    )
    psdu_chips, _ = modulate_bpm_bpsk(
        g0_psdu, g1_psdu, scrambler, data_p.n_burst, data_p.n_hop, data_p.n_cpb, off
    )
    return phr_chips, psdu_chips


def _encode_modulate_hprf(
    phr_bits: np.ndarray,
    psdu_bits: np.ndarray,
    cfg: HRPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    assert cfg.code_index is not None and cfg.mean_prf is not None
    params = HPRF_MOD_PARAMS[(cfg.mean_prf, cfg.constraint_length)]

    if cfg.constraint_length == 3:
        rs_bits = rs_encode(psdu_bits)
        if len(rs_bits) == 0:
            g0_phr, g1_phr = conv_encode_k3(phr_bits, append_tail=True)
            g0_psdu = np.empty(0, dtype=np.uint8)
            g1_psdu = np.empty(0, dtype=np.uint8)
        else:
            combined = np.concatenate((phr_bits, rs_bits))
            g0_all, g1_all = conv_encode_k3(combined, append_tail=True)
            g0_phr, g1_phr = g0_all[:21], g1_all[:21]
            g0_psdu, g1_psdu = g0_all[21:], g1_all[21:]
    else:
        # K = 7: no RS; 6 zero tail bits appended separately to PHR and to PSDU (Section 16.3.3.3)
        g0_phr, g1_phr = conv_encode_k7(phr_bits, append_tail=True)
        g0_psdu, g1_psdu = conv_encode_k7(psdu_bits, append_tail=True)

    total_scram = (
        len(g0_phr) * params.pulses_per_phr_symbol + len(g0_psdu) * params.pulses_per_data_symbol
    )
    scrambler = generate_scrambler_bits(cfg.code_index, total_scram)

    phr_chips, off = _modulate_hprf_symbols(
        g0_phr,
        g1_phr,
        scrambler,
        mean_prf=cfg.mean_prf,
        constraint_length=cfg.constraint_length,
        is_phr=True,
        scrambler_offset=0,
    )
    psdu_chips, _ = _modulate_hprf_symbols(
        g0_psdu,
        g1_psdu,
        scrambler,
        mean_prf=cfg.mean_prf,
        constraint_length=cfg.constraint_length,
        is_phr=False,
        scrambler_offset=off,
    )
    return phr_chips, psdu_chips
