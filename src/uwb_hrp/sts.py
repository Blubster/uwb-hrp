"""Scrambled Timestamp Sequence (STS) DRBG and field generation (Section 16.2.9)."""

from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uwb_hrp.config import HRPConfig
from uwb_hrp.tables import STS_EXTRA_GAP_UNIT_CHIPS, STS_GAP_CHIPS, STS_SEGMENT_UNIT_CHIPS


def sts_drbg_blocks(key: bytes, v_init: bytes, num_blocks: int) -> bytes:
    """Run the AES-128 CTR-mode DRBG (Figure 16-11) for `num_blocks` iterations.

    `v_init` is `VUpper96 || VCounter` (16 bytes big-endian). The lower 32-bit
    counter increments modulo `2^32` after each 128-bit block.

    Args:
        key: 16-byte (128-bit) AES key.
        v_init: 16-byte initial `VUpper96 || VCounter` block.
        num_blocks: Number of 128-bit DRBG blocks to generate.

    Returns:
        Concatenated `16 * max(0, num_blocks)` pseudo-random bytes.

    Raises:
        ValueError: If `key` or `v_init` is not 16 bytes long.
    """
    if len(key) != 16:
        raise ValueError(f"STS key must be 16 bytes, got {len(key)}.")
    if len(v_init) != 16:
        raise ValueError(f"STS V must be 16 bytes, got {len(v_init)}.")
    if num_blocks <= 0:
        return b""

    v_upper = v_init[:12]
    counter = int.from_bytes(v_init[12:], "big")

    # Build all 16-byte counter blocks in one contiguous buffer and encrypt in a single AES-ECB call
    blocks = bytearray(num_blocks * 16)
    for i in range(num_blocks):
        offset = i * 16
        blocks[offset : offset + 12] = v_upper
        blocks[offset + 12 : offset + 16] = ((counter + i) & 0xFFFFFFFF).to_bytes(4, "big")

    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    return encryptor.update(bytes(blocks)) + encryptor.finalize()


def sts_drbg_bits(key: bytes, v_init: bytes, num_bits: int) -> np.ndarray:
    """Generate `num_bits` pseudo-random bits from the STS DRBG (MSB first).

    Args:
        key: 16-byte (128-bit) AES key.
        v_init: 16-byte initial `VUpper96 || VCounter` block.
        num_bits: Number of pseudo-random bits to generate.

    Returns:
        1D `uint8` array of length `max(0, num_bits)` containing `{0, 1}`.

    Raises:
        ValueError: If `key` or `v_init` is not 16 bytes long.
    """
    if num_bits <= 0:
        return np.empty(0, dtype=np.uint8)
    num_blocks = (num_bits + 127) // 128
    raw = sts_drbg_blocks(key, v_init, num_blocks)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    return bits[:num_bits]


def advance_sts_v(v_init: bytes, num_blocks: int) -> bytes:
    """Return `VUpper96 || ((VCounter + num_blocks) mod 2^32)` (Section 16.2.9.2).

    Args:
        v_init: 16-byte `VUpper96 || VCounter` state.
        num_blocks: Number of 128-bit DRBG iterations to advance `VCounter`.

    Returns:
        Updated 16-byte `V` value with unchanged `VUpper96`.

    Raises:
        ValueError: If `v_init` is not 16 bytes long.
    """
    if len(v_init) != 16:
        raise ValueError(f"STS V must be 16 bytes, got {len(v_init)}.")
    counter = (int.from_bytes(v_init[12:], "big") + num_blocks) & 0xFFFFFFFF
    return v_init[:12] + counter.to_bytes(4, "big")


def sts_drbg_blocks_per_packet(cfg: HRPConfig) -> int:
    """Return the number of 128-bit DRBG iterations consumed by one packet's STS.

    Args:
        cfg: Validated [HRPConfig][uwb_hrp.HRPConfig].

    Returns:
        Number of 128-bit AES blocks consumed per packet (`0` for SP0).
    """
    if cfg.sts_packet_config == 0:
        return 0
    delta_l = cfg.sts_spreading_factor
    assert delta_l is not None
    total_pulses = (
        cfg.sts_segment_length * STS_SEGMENT_UNIT_CHIPS // delta_l
    ) * cfg.num_sts_segments
    return (total_pulses + 127) // 128


def sts_field_length_chips(cfg: HRPConfig, *, trailing_gap: bool) -> int:
    """Compute total STS field length in chips including gaps and SP2 extra gap.

    Args:
        cfg: Validated [HRPConfig][uwb_hrp.HRPConfig].
        trailing_gap: Whether to include the trailing 512-chip silent gap after
            the final active STS segment.

    Returns:
        Total STS field length in chips (`0` for SP0).
    """
    if cfg.sts_packet_config == 0:
        return 0
    extra_chips = cfg.extra_sts_gap * STS_EXTRA_GAP_UNIT_CHIPS if cfg.sts_packet_config == 2 else 0
    active_per_seg = cfg.sts_segment_length * STS_SEGMENT_UNIT_CHIPS
    num_gaps = cfg.num_sts_segments + (1 if trailing_gap else 0)
    return extra_chips + cfg.num_sts_segments * active_per_seg + num_gaps * STS_GAP_CHIPS


def generate_sts(cfg: HRPConfig, *, trailing_gap: bool) -> np.ndarray:
    """Generate the complete STS field chips (Section 16.2.9.3).

    Args:
        cfg: HRP UWB configuration with `sts_packet_config` in `{1, 2, 3}`.
        trailing_gap: If `True`, append the trailing 512-chip silent gap after
            the final STS segment (IEEE Std 802.15.4-2024 Figures 16-10 & 16-12);
            set `False` only for legacy Draft D08 SP2/SP3 compatibility.

    Returns:
        1D `int8` ternary chip array for the STS field (empty for SP0).
    """
    if cfg.sts_packet_config == 0:
        return np.empty(0, dtype=np.int8)

    delta_l = cfg.sts_spreading_factor
    assert delta_l is not None

    active_chips_per_seg = cfg.sts_segment_length * STS_SEGMENT_UNIT_CHIPS
    pulses_per_seg = active_chips_per_seg // delta_l
    total_pulses = pulses_per_seg * cfg.num_sts_segments

    bits = sts_drbg_bits(cfg.sts_key, cfg.sts_v, total_pulses)
    # 0 -> +1, 1 -> -1
    polarities = (1 - 2 * bits.astype(np.int8)).reshape(cfg.num_sts_segments, pulses_per_seg)

    extra_chips = cfg.extra_sts_gap * STS_EXTRA_GAP_UNIT_CHIPS if cfg.sts_packet_config == 2 else 0
    total_chips = sts_field_length_chips(cfg, trailing_gap=trailing_gap)
    out = np.zeros(total_chips, dtype=np.int8)

    cursor = extra_chips + STS_GAP_CHIPS
    for seg_idx in range(cfg.num_sts_segments):
        out[cursor : cursor + active_chips_per_seg : delta_l] = polarities[seg_idx]
        cursor += active_chips_per_seg + STS_GAP_CHIPS

    return out
