"""PHY Header (PHR) formatting and SECDED parity encoding (Section 16.2.7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uwb_hrp.config import HRPConfig
from uwb_hrp.tables import BPM_BPSK_PARAMS, PREAMBLE_DURATION_BITS


def compute_secded(b: Sequence[int]) -> list[int]:
    """Compute the 6-bit SECDED parity field `[b13..b18]` (Section 16.2.7.1).

    Args:
        b: Exactly 13 PHR information bits `[b0..b12]`.

    Returns:
        List of 6 parity bits `[b13, b14, b15, b16, b17, b18]`.

    Raises:
        ValueError: If `len(b) != 13`.
    """
    if len(b) != 13:
        raise ValueError(f"SECDED requires exactly 13 information bits (b0..b12), got {len(b)}.")
    b18 = b[1] ^ b[0] ^ b[8] ^ b[6] ^ b[4] ^ b[3] ^ b[10] ^ b[11]
    b17 = b[0] ^ b[6] ^ b[5] ^ b[3] ^ b[2] ^ b[9] ^ b[10] ^ b[12]
    b16 = b[1] ^ b[8] ^ b[7] ^ b[3] ^ b[2] ^ b[9] ^ b[10]
    b15 = b[8] ^ b[7] ^ b[6] ^ b[5] ^ b[4] ^ b[9] ^ b[10]
    b14 = b[12] ^ b[11]
    b13 = (
        b[0]
        ^ b[1]
        ^ b[2]
        ^ b[3]
        ^ b[4]
        ^ b[5]
        ^ b[6]
        ^ b[7]
        ^ b[8]
        ^ b[9]
        ^ b[10]
        ^ b[11]
        ^ b[12]
        ^ b14
        ^ b15
        ^ b16
        ^ b17
        ^ b18
    )
    return [b13, b14, b15, b16, b17, b18]


def build_phr_bits(cfg: HRPConfig) -> np.ndarray:
    """Build the 19-bit PHR field `H0..H18` for `cfg` (Sections 16.2.7.1-16.2.7.3).

    Args:
        cfg: Validated [HRPConfig][uwb_hrp.HRPConfig] with `sts_packet_config != 3`.

    Returns:
        1D `uint8` array of 19 PHR bits (`13` info bits + `6` SECDED parity bits).

    Raises:
        ValueError: If `cfg.sts_packet_config == 3` (SP3 carries no PHR).
    """
    if not cfg.has_phr_and_psdu:
        raise ValueError("STS packet configuration 3 has no PHR field.")

    if cfg.mode == "HPRF":
        return _build_hprf_phr(cfg)
    return _build_legacy_or_bprf_phr(cfg)


def _build_legacy_or_bprf_phr(cfg: HRPConfig) -> np.ndarray:
    """Build Figure 16-7 PHR (DataRate, FrameLength, Ranging, Reserved, PreDur, SECDED)."""
    assert (
        cfg.mean_prf is not None and cfg.data_rate is not None and cfg.preamble_duration is not None
    )
    dr0, dr1 = BPM_BPSK_PARAMS[(cfg.mean_prf, cfg.data_rate)].phr_rate_bits
    len_bits = [(cfg.psdu_length >> (6 - i)) & 1 for i in range(7)]
    ranging = 1 if cfg.ranging else 0
    reserved = 0
    pd0, pd1 = PREAMBLE_DURATION_BITS[cfg.preamble_duration]
    info = [dr0, dr1, *len_bits, ranging, reserved, pd0, pd1]
    return np.array([*info, *compute_secded(info)], dtype=np.uint8)


def _build_hprf_phr(cfg: HRPConfig) -> np.ndarray:
    """Build Figure 16-8 HPRF PHR (A1, A0, 10-bit PayloadLength, Ranging, SECDED)."""
    assert cfg.psdu_size_extension is not None
    length = cfg.psdu_length
    len_low10 = [(length >> (9 - i)) & 1 for i in range(10)]

    if cfg.psdu_size_extension == 2:
        a1 = (length >> 11) & 1
        a0 = (length >> 10) & 1
    elif cfg.psdu_size_extension == 1:
        # Table 16-16: A0 is bit 10 of PSDU length; A1 is unused (0)
        a1 = 0
        a0 = (length >> 10) & 1
    elif cfg.extra_sts_gap_index is not None:
        a1 = (cfg.extra_sts_gap_index >> 1) & 1
        a0 = cfg.extra_sts_gap_index & 1
    else:
        a1 = 0
        a0 = 0

    ranging = 1 if cfg.ranging else 0
    info = [a1, a0, *len_low10, ranging]
    return np.array([*info, *compute_secded(info)], dtype=np.uint8)
