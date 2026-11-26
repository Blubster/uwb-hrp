"""PRBS scrambler LFSR for HRP UWB PHY (IEEE Std 802.15.4-2024 Section 16.3.2)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from uwb_hrp.tables import PREAMBLE_CODES

_LFSR_PERIOD: int = (1 << 15) - 1  # 32767 (maximal-length polynomial 1 + D^14 + D^15)


def lfsr_initial_state(code_index: int) -> np.ndarray:
    """Compute the 15-bit LFSR initial state `(s_{-15}, ..., s_{-1})`.

    Per Section 16.3.2, removes all zeros from ternary preamble code `C_i`,
    replaces `-1` with `0`, and takes the first 15 bits.

    Args:
        code_index: Ternary preamble code index in `1..32`.

    Returns:
        1D `uint8` array of 15 initial LFSR state bits.
    """
    spec = PREAMBLE_CODES[code_index]
    bits = [1 if ch == "+" else 0 for ch in spec.sequence_str if ch in "+-"]
    return np.array(bits[:15], dtype=np.uint8)


@lru_cache(maxsize=32)
def _full_period_sequence(code_index: int) -> np.ndarray:
    """Precompute one complete 32767-bit period of the LFSR for `code_index`."""
    init = lfsr_initial_state(code_index)
    # s_n = s_{n-14} ^ s_{n-15}
    buf = np.empty(15 + _LFSR_PERIOD, dtype=np.uint8)
    buf[:15] = init
    # Vectorized recurrence in chunks of 14 (since s_n depends on n-14 and n-15)
    pos = 15
    end = 15 + _LFSR_PERIOD
    while pos < end:
        step = min(14, end - pos)
        buf[pos : pos + step] = buf[pos - 14 : pos - 14 + step] ^ buf[pos - 15 : pos - 15 + step]
        pos += step
    out = buf[15:].copy()
    out.setflags(write=False)
    return out


def generate_scrambler_bits(code_index: int, num_bits: int) -> np.ndarray:
    """Generate `num_bits` outputs `s_0, ..., s_{num_bits-1}` of the PRBS scrambler.

    Args:
        code_index: Ternary preamble code index in `1..32` used to seed the LFSR.
        num_bits: Number of scrambler bits to generate.

    Returns:
        1D `uint8` array of length `max(0, num_bits)` containing `{0, 1}`.
    """
    if num_bits <= 0:
        return np.empty(0, dtype=np.uint8)
    period = _full_period_sequence(code_index)
    if num_bits <= _LFSR_PERIOD:
        return period[:num_bits].copy()
    reps = (num_bits + _LFSR_PERIOD - 1) // _LFSR_PERIOD
    return np.tile(period, reps)[:num_bits]
