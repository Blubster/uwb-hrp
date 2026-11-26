"""Forward Error Correction (Reed-Solomon, Convolutional) and FCS (CRC-16/32).

Implements IEEE Std 802.15.4-2024:
- Section 16.3.3.2: Systematic RS6(63, 55) over GF(2^6)
- Section 16.3.3.3: K=3 (g0=010, g1=101) and K=7 (133_8, 171_8) convolutional encoders
- Section 7.2.11: 2-octet ITU-T CRC-16 and 4-octet ANSI X3.66 CRC-32 FCS
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# GF(2^6) primitive polynomial: 1 + x + x^6 = 0b1000011 = 0x43, alpha = 2
_GF_PRIM_POLY = 0x43
_GF_SIZE = 64
_GF_EXP = np.zeros(2 * (_GF_SIZE - 1), dtype=np.int32)
_GF_LOG = np.zeros(_GF_SIZE, dtype=np.int32)

_val = 1
for _i in range(_GF_SIZE - 1):
    _GF_EXP[_i] = _val
    _GF_EXP[_i + _GF_SIZE - 1] = _val
    _GF_LOG[_val] = _i
    _val <<= 1
    if _val & _GF_SIZE:
        _val ^= _GF_PRIM_POLY

# Multiplication lookup table for fast RS encoding
_GF_MUL_TABLE = np.zeros((_GF_SIZE, _GF_SIZE), dtype=np.int32)
for _a in range(1, _GF_SIZE):
    for _b in range(1, _GF_SIZE):
        _GF_MUL_TABLE[_a, _b] = _GF_EXP[_GF_LOG[_a] + _GF_LOG[_b]]


def _build_rs_generator() -> np.ndarray:
    """Compute the RS6(63, 55) generator polynomial in GF(2^6).

    Evaluates `g(x) = prod_{k=1..8} (x + alpha^k)` with highest-degree
    coefficient first.

    Returns:
        1D `int32` array of 9 generator coefficients in GF(2^6).
    """
    poly = [1]
    for k in range(1, 9):
        root = int(_GF_EXP[k])
        new_poly = [0] * (len(poly) + 1)
        for idx, coeff in enumerate(poly):
            new_poly[idx] ^= coeff
            new_poly[idx + 1] ^= int(_GF_MUL_TABLE[coeff, root])
        poly = new_poly
    return np.array(poly, dtype=np.int32)


# Normative RS6(63,55) generator: [1, 55, 61, 37, 48, 47, 20, 6, 22]
RS_GENERATOR_POLY: np.ndarray = _build_rs_generator()
_RS_GEN_COEFFS: tuple[int, ...] = tuple(int(c) for c in RS_GENERATOR_POLY[1:])
RS_MAX_INFO_BITS: int = 330
RS_PARITY_BITS: int = 48


def rs_encode_block(bits: np.ndarray) -> np.ndarray:
    """Encode a single block of `1 <= I <= 330` bits using systematic RS6(63, 55).

    Follows Section 16.3.3.2 steps (a) through (e):

    - Left-pads with `330 - I` zeros.
    - Packs 6-bit LSB-first symbols `D_0..D_54`.
    - Computes 8 parity symbols `P(x) = x^8 * D(x) mod g(x)`.
    - Unpacks LSB-first and removes the `330 - I` dummy bits.

    Args:
        bits: 1D binary array of at most 330 information bits.

    Returns:
        1D `uint8` systematic codeword of length `len(bits) + 48` (or empty if
        `len(bits) == 0`).

    Raises:
        ValueError: If `len(bits) > 330`.
    """
    n_bits = len(bits)
    if n_bits == 0:
        return np.empty(0, dtype=np.uint8)
    if n_bits > RS_MAX_INFO_BITS:
        raise ValueError(f"Single RS block supports at most {RS_MAX_INFO_BITS} bits, got {n_bits}.")

    pad = RS_MAX_INFO_BITS - n_bits
    padded = np.zeros(RS_MAX_INFO_BITS, dtype=np.uint8)
    padded[pad:] = bits

    # Bit-to-symbol conversion: D_k = sum_{j=0..5} d_{6k+j} * 2^j (LSB first)
    grouped = padded.reshape(55, 6)
    weights = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    symbols = grouped @ weights

    # Systematic LFSR division for P(x) = x^8 * D(x) mod g(x)
    rem = [0, 0, 0, 0, 0, 0, 0, 0]
    g0, g1, g2, g3, g4, g5, g6, g7 = _RS_GEN_COEFFS
    mul = _GF_MUL_TABLE
    for sym in symbols:
        fb = int(sym) ^ rem[0]
        if fb:
            row = mul[fb]
            rem[0] = rem[1] ^ int(row[g0])
            rem[1] = rem[2] ^ int(row[g1])
            rem[2] = rem[3] ^ int(row[g2])
            rem[3] = rem[4] ^ int(row[g3])
            rem[4] = rem[5] ^ int(row[g4])
            rem[5] = rem[6] ^ int(row[g5])
            rem[6] = rem[7] ^ int(row[g6])
            rem[7] = int(row[g7])
        else:
            rem[0] = rem[1]
            rem[1] = rem[2]
            rem[2] = rem[3]
            rem[3] = rem[4]
            rem[4] = rem[5]
            rem[5] = rem[6]
            rem[6] = rem[7]
            rem[7] = 0

    # Unpack 8 parity symbols LSB-first into 48 parity bits
    parity_syms = np.array(rem, dtype=np.int32)[:, None]
    shifts = np.arange(6, dtype=np.int32)[None, :]
    parity_bits = ((parity_syms >> shifts) & 1).reshape(RS_PARITY_BITS).astype(np.uint8)

    return np.concatenate((bits.astype(np.uint8, copy=False), parity_bits))


def rs_encode(bits: np.ndarray) -> np.ndarray:
    """Encode an arbitrary-length PSDU bit array with RS6(63, 55) in 330-bit blocks.

    Args:
        bits: 1D binary PSDU bit array in LSB-first transmission order.

    Returns:
        1D `uint8` concatenated RS-encoded bit stream (48 parity bits per
        330-bit block).
    """
    n_bits = len(bits)
    if n_bits == 0:
        return np.empty(0, dtype=np.uint8)
    blocks = [
        rs_encode_block(bits[start : start + RS_MAX_INFO_BITS])
        for start in range(0, n_bits, RS_MAX_INFO_BITS)
    ]
    return np.concatenate(blocks)


def rs_encoded_length_bits(psdu_length_octets: int) -> int:
    """Return total RS-encoded bit length for a PSDU of `psdu_length_octets` bytes.

    Args:
        psdu_length_octets: PSDU length in octets (`>= 0`).

    Returns:
        Total bit count after block-wise RS6(63, 55) parity insertion.
    """
    n_bits = psdu_length_octets * 8
    if n_bits == 0:
        return 0
    num_blocks = (n_bits + RS_MAX_INFO_BITS - 1) // RS_MAX_INFO_BITS
    return n_bits + num_blocks * RS_PARITY_BITS


def conv_encode_k3(bits: np.ndarray, *, append_tail: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Encode bits with the K=3 systematic rate-1/2 convolutional code (Section 16.3.3.3).

    Generator polynomials: `g0 = [010]_2` (`x[n-1]`), `g1 = [101]_2`
    (`x[n] ^ x[n-2]`). Initial state is all-zero; when `append_tail` is `True`,
    two zero tail bits are appended so the encoder returns to the all-zero state
    (Table 16-2).

    Args:
        bits: 1D binary input bit array.
        append_tail: Whether to append 2 zero flush bits (default `True`).

    Returns:
        Tuple `(g0, g1)` of 1D `uint8` coded output bit streams.
    """
    src = np.asarray(bits, dtype=np.uint8)
    if append_tail:
        src = np.concatenate((src, np.zeros(2, dtype=np.uint8)))
    padded = np.concatenate((np.zeros(2, dtype=np.uint8), src))
    g0 = padded[1:-1].copy()
    g1 = (padded[2:] ^ padded[:-2]).astype(np.uint8)
    return g0, g1


def conv_encode_k7(bits: np.ndarray, *, append_tail: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Encode bits with the optional HPRF K=7 rate-1/2 convolutional code (Section 16.3.3.3).

    Generator polynomials `(133_8, 171_8)` with MSB = current input (`D^0`):

    - `g0(n) = x[n] ^ x[n-2] ^ x[n-3] ^ x[n-5] ^ x[n-6]`
    - `g1(n) = x[n] ^ x[n-1] ^ x[n-2] ^ x[n-3] ^ x[n-6]`

    Args:
        bits: 1D binary input bit array.
        append_tail: Whether to append 6 zero flush bits (default `True`).

    Returns:
        Tuple `(g0, g1)` of 1D `uint8` coded output bit streams.
    """
    src = np.asarray(bits, dtype=np.uint8)
    if append_tail:
        src = np.concatenate((src, np.zeros(6, dtype=np.uint8)))
    p = np.concatenate((np.zeros(6, dtype=np.uint8), src))
    g0 = (p[6:] ^ p[4:-2] ^ p[3:-3] ^ p[1:-5] ^ p[:-6]).astype(np.uint8)
    g1 = (p[6:] ^ p[5:-1] ^ p[4:-2] ^ p[3:-3] ^ p[:-6]).astype(np.uint8)
    return g0, g1


def bytes_to_bits_lsb(data: bytes | bytearray) -> np.ndarray:
    """Serialize octets into IEEE 802.15.4 transmission order (LSB of each octet first).

    Args:
        data: Input byte sequence.

    Returns:
        1D `uint8` array of length `8 * len(data)` containing bits `{0, 1}`.
    """
    if len(data) == 0:
        return np.empty(0, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="little")


def bits_lsb_to_bytes(bits: Sequence[int] | np.ndarray) -> bytes:
    """Pack IEEE 802.15.4 LSB-first transmission-order bits back into octets.

    Args:
        bits: Binary bit sequence whose length is a multiple of 8.

    Returns:
        Packed `bytes` object of length `len(bits) // 8`.

    Raises:
        ValueError: If `len(bits)` is not divisible by 8.
    """
    arr = np.asarray(bits, dtype=np.uint8)
    if len(arr) % 8 != 0:
        raise ValueError(f"Bit length must be a multiple of 8, got {len(arr)}.")
    if len(arr) == 0:
        return b""
    return np.packbits(arr, bitorder="little").tobytes()


def compute_fcs16_bits(bits: Sequence[int] | np.ndarray) -> np.ndarray:
    """Compute IEEE 802.15.4 Section 7.2.11 2-octet ITU-T CRC-16 over transmission bits.

    Generator polynomial: `G16(x) = x^16 + x^12 + x^5 + 1`.

    Args:
        bits: Sequence of transmission-order frame bits `{0, 1}`.

    Returns:
        1D `uint8` array of 16 FCS bits `[r0, ..., r15]` in transmission order
        (`r0` first).
    """
    reg = 0
    for bit in bits:
        fb = ((reg >> 15) & 1) ^ (int(bit) & 1)
        reg = (reg << 1) & 0xFFFF
        if fb:
            reg ^= 0x1021
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def compute_fcs16(data: bytes | bytearray) -> bytes:
    """Compute IEEE 802.15.4 Section 7.2.11 2-octet FCS for `data`.

    Args:
        data: MAC frame bytes (MHR + MAC payload) excluding the FCS field.

    Returns:
        2-byte CRC-16 FCS ready to append to `data`.
    """
    fcs_bits = compute_fcs16_bits(bytes_to_bits_lsb(data))
    return bits_lsb_to_bytes(fcs_bits)


def compute_fcs32_bits(bits: Sequence[int] | np.ndarray) -> np.ndarray:
    """Compute IEEE 802.15.4 Section 7.2.11 4-octet ANSI X3.66 CRC-32 over bits.

    Generator polynomial: `G32(x) = x^32 + x^26 + x^23 + x^22 + x^16 + x^12
    + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1`.
    Pads calculation fields shorter than 32 bits by appending trailing zero bits
    per Section 7.2.11.

    Args:
        bits: Sequence of transmission-order frame bits `{0, 1}`.

    Returns:
        1D `uint8` array of 32 FCS bits in transmission order (highest-degree
        remainder coefficient first).
    """
    bit_list = [int(b) & 1 for b in bits]
    if len(bit_list) < 32:
        bit_list = bit_list + [0] * (32 - len(bit_list))
    reg = 0xFFFFFFFF
    for bit in bit_list:
        fb = ((reg >> 31) & 1) ^ bit
        reg = (reg << 1) & 0xFFFFFFFF
        if fb:
            reg ^= 0x04C11DB7
    fcs = (~reg) & 0xFFFFFFFF
    return np.array([(fcs >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)


def compute_fcs32(data: bytes | bytearray) -> bytes:
    """Compute IEEE 802.15.4 Section 7.2.11 4-octet FCS for `data`.

    Args:
        data: MAC frame bytes (MHR + MAC payload) excluding the FCS field.

    Returns:
        4-byte CRC-32 FCS ready to append to `data`.
    """
    fcs_bits = compute_fcs32_bits(bytes_to_bits_lsb(data))
    return bits_lsb_to_bytes(fcs_bits)
