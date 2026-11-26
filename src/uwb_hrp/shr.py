"""Synchronization Header (SHR: SYNC and SFD) generation (Section 16.2.6)."""

from __future__ import annotations

import numpy as np

from uwb_hrp.config import HRPConfig
from uwb_hrp.tables import PREAMBLE_CODES, SFD_SEQUENCES


def build_preamble_symbol(code_index: int, delta_l: int) -> np.ndarray:
    """Construct preamble symbol `S_i = C_i (x) delta_L` (Section 16.2.6.2, Figure 16-6).

    Args:
        code_index: Ternary preamble code index in `1..32`.
        delta_l: Spreading factor inserting `delta_l - 1` zero chips after each
            code chip.

    Returns:
        1D `int8` spread preamble symbol of length `len(C_i) * delta_l`.
    """
    code = PREAMBLE_CODES[code_index].chips
    symbol = np.zeros(len(code) * delta_l, dtype=np.int8)
    symbol[::delta_l] = code
    return symbol


def generate_sync(cfg: HRPConfig) -> np.ndarray:
    """Generate the SYNC field: `N_sync` repetitions of `S_i`.

    Args:
        cfg: Validated [HRPConfig][uwb_hrp.HRPConfig].

    Returns:
        1D `int8` ternary SYNC chip array.
    """
    assert cfg.code_index is not None and cfg.preamble_duration is not None
    sym = build_preamble_symbol(cfg.code_index, cfg.preamble_spreading_factor)
    return np.tile(sym, cfg.preamble_duration)


def generate_sfd(cfg: HRPConfig) -> np.ndarray:
    """Generate the SFD field: selected SFD sequence spread by `S_i` (Section 16.2.6.3).

    Args:
        cfg: Validated [HRPConfig][uwb_hrp.HRPConfig].

    Returns:
        1D `int8` ternary SFD chip array.
    """
    assert cfg.code_index is not None and cfg.sfd_number is not None
    sym = build_preamble_symbol(cfg.code_index, cfg.preamble_spreading_factor)
    sfd_seq = SFD_SEQUENCES[cfg.sfd_number]
    return np.kron(sfd_seq, sym).astype(np.int8, copy=False)
