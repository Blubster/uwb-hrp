"""Top-level HRP UWB frame and baseband waveform generation API."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from uwb_hrp.config import HRPConfig, HRPFieldIndices
from uwb_hrp.fec import bytes_to_bits_lsb, rs_encoded_length_bits
from uwb_hrp.modulation import encode_and_modulate_phr_psdu
from uwb_hrp.pulse import PulseConfig, PulseShaper
from uwb_hrp.shr import generate_sfd, generate_sync
from uwb_hrp.sts import (
    advance_sts_v,
    generate_sts,
    sts_drbg_blocks_per_packet,
    sts_field_length_chips,
)
from uwb_hrp.tables import STS_EXTRA_GAP_UNIT_CHIPS, STS_GAP_CHIPS, STS_SEGMENT_UNIT_CHIPS

SinglePSDUInput = bytes | bytearray | memoryview | Sequence[int] | np.ndarray | None
PSDUInput = SinglePSDUInput | Sequence[bytes | bytearray | memoryview]


@dataclass(frozen=True)
class HRPWaveformResult:
    """Result returned by :func:`generate_waveform`.

    Attributes:
        waveform: 1D float64 pulse-shaped baseband waveform at ``sample_rate`` Hz
            (including any trailing per-packet ``idle_time`` zeros).
        chips: Ternary chip sequence ``{-1, 0, +1}`` at 499.2 MHz. Has shape
            ``(chips_per_packet,)`` when ``num_packets == 1`` and
            ``(chips_per_packet, num_packets)`` when ``num_packets > 1``.
        indices: Chip-level field slices and ranging marker offsets for a single packet.
        sample_rate: Output waveform sample rate in Hz.
        pulse_taps: Sampled baseband pulse shaping impulse response taps.
        num_packets: Number of packets generated.
        idle_samples: Number of silent zero samples appended after each packet.
        next_sts_v: Next 16-byte STS DRBG ``V`` value after the generated packet(s)
            when STS is enabled (``sts_packet_config > 0``), or ``None`` when STS is
            disabled. Pass this to ``dataclasses.replace(cfg, sts_v=res.next_sts_v)``
            to stream consecutive frames with continuously advancing ``VCounter``.
    """

    waveform: np.ndarray
    chips: np.ndarray
    indices: HRPFieldIndices
    sample_rate: float
    pulse_taps: np.ndarray
    num_packets: int = 1
    idle_samples: int = 0
    next_sts_v: bytes | None = None


def _normalize_single_psdu_bits(
    psdu: SinglePSDUInput,
    cfg: HRPConfig,
    *,
    pad_or_truncate: bool = False,
) -> np.ndarray:
    """Convert bytes or a 1D bit sequence into a validated uint8 bit array for one packet."""
    if not cfg.has_phr_and_psdu:
        if psdu is not None and len(psdu) != 0 and not pad_or_truncate:
            raise ValueError(
                "STS packet configuration 3 carries no PSDU; pass None or empty bytes."
            )
        return np.empty(0, dtype=np.uint8)

    expected_bits = cfg.psdu_length * 8
    if psdu is None:
        if expected_bits == 0:
            return np.empty(0, dtype=np.uint8)
        raise ValueError(
            f"Expected PSDU of {cfg.psdu_length} bytes ({expected_bits} bits), got None."
        )

    if isinstance(psdu, (bytes, bytearray, memoryview)):
        raw = bytes(psdu)
        bits = bytes_to_bits_lsb(raw)
    else:
        bits = np.asarray(psdu, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"PSDU bit input must be 1D, got shape {bits.shape}.")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("PSDU bit input must contain only 0 and 1.")

    if expected_bits == 0:
        if len(bits) != 0 and not pad_or_truncate:
            raise ValueError(f"Expected PSDU of 0 bits (0 bytes), got {len(bits)} bits.")
        return np.empty(0, dtype=np.uint8)

    if len(bits) != expected_bits:
        if not pad_or_truncate:
            raise ValueError(
                f"Expected PSDU of {expected_bits} bits ({cfg.psdu_length} bytes), "
                f"got {len(bits)} bits."
            )
        if len(bits) == 0:
            raise ValueError("Cannot repeat an empty PSDU to fill a non-zero psdu_length.")
        return np.resize(bits, expected_bits) if len(bits) < expected_bits else bits[:expected_bits]

    return bits


def _normalize_multi_psdu_bits(
    psdu: PSDUInput,
    cfg: HRPConfig,
    num_packets: int,
    *,
    pad_or_truncate: bool = False,
) -> list[np.ndarray]:
    """Normalize single- or multi-packet PSDU input into ``num_packets`` bit arrays."""
    if num_packets < 1:
        raise ValueError(f"num_packets must be >= 1, got {num_packets}.")

    # Explicit sequence of per-packet byte buffers (e.g., [b"pkt1", b"pkt2"])
    if (
        isinstance(psdu, Sequence)
        and not isinstance(psdu, (bytes, bytearray, memoryview, str))
        and len(psdu) > 0
        and isinstance(psdu[0], (bytes, bytearray, memoryview))
    ):
        if len(psdu) != num_packets:
            raise ValueError(f"Expected {num_packets} per-packet payloads, got {len(psdu)}.")
        pkt_buffers: list[bytes | bytearray | memoryview] = []
        for pkt in psdu:
            if not isinstance(pkt, (bytes, bytearray, memoryview)):
                raise TypeError(
                    f"Expected bytes-like per-packet payload, got {type(pkt).__name__}."
                )
            pkt_buffers.append(pkt)
        return [
            _normalize_single_psdu_bits(pkt, cfg, pad_or_truncate=pad_or_truncate)
            for pkt in pkt_buffers
        ]

    if num_packets == 1:
        single_psdu: SinglePSDUInput = psdu  # type: ignore[assignment]
        return [_normalize_single_psdu_bits(single_psdu, cfg, pad_or_truncate=pad_or_truncate)]

    if not cfg.has_phr_and_psdu:
        if psdu is not None and len(psdu) != 0 and not pad_or_truncate:
            raise ValueError(
                "STS packet configuration 3 carries no PSDU; pass None or empty bytes."
            )
        return [np.empty(0, dtype=np.uint8) for _ in range(num_packets)]

    bits_per_pkt = cfg.psdu_length * 8
    if bits_per_pkt == 0:
        if psdu is not None and len(psdu) != 0 and not pad_or_truncate:
            raise ValueError(f"Expected PSDU of 0 bits (0 bytes), got {len(psdu)} elements.")
        return [np.empty(0, dtype=np.uint8) for _ in range(num_packets)]

    if psdu is None:
        raise ValueError(
            f"Expected PSDU for {num_packets} packets ({cfg.psdu_length * num_packets} bytes), "
            "got None."
        )

    if isinstance(psdu, (bytes, bytearray, memoryview)):
        flat_bits = bytes_to_bits_lsb(bytes(psdu))
    else:
        flat_bits = np.asarray(psdu, dtype=np.uint8)
        if flat_bits.ndim != 1:
            raise ValueError(f"PSDU bit input must be 1D, got shape {flat_bits.shape}.")
        if np.any((flat_bits != 0) & (flat_bits != 1)):
            raise ValueError("PSDU bit input must contain only 0 and 1.")

    total_needed = bits_per_pkt * num_packets
    if len(flat_bits) != total_needed:
        if not pad_or_truncate:
            raise ValueError(
                f"Expected {total_needed} bits ({cfg.psdu_length * num_packets} bytes) for "
                f"{num_packets} packets, got {len(flat_bits)} bits."
            )
        if len(flat_bits) == 0:
            raise ValueError("Cannot repeat an empty PSDU to fill a non-zero psdu_length.")
        stream = (
            np.resize(flat_bits, total_needed)
            if len(flat_bits) < total_needed
            else flat_bits[:total_needed]
        )
    else:
        stream = flat_bits

    return [stream[i * bits_per_pkt : (i + 1) * bits_per_pkt] for i in range(num_packets)]


def field_indices(cfg: HRPConfig) -> HRPFieldIndices:
    """Compute exact chip-level field slices and ranging marker positions for ``cfg``."""
    assert cfg.preamble_duration is not None

    sync_len = cfg.preamble_duration * cfg.chips_per_preamble_symbol
    sfd_len = cfg.sfd_length * cfg.chips_per_preamble_symbol

    sync_slice = slice(0, sync_len)
    sfd_slice = slice(sync_len, sync_len + sfd_len)
    cursor = sfd_slice.stop
    rmarker_chip = cursor  # First chip after SFD in all STS configs (Figure 16-3)

    sts1_slice: slice | None = None
    phr_slice: slice | None = None
    psdu_slice: slice | None = None
    sts2_slice: slice | None = None
    srmarkers: list[int] = []

    seg_step = STS_GAP_CHIPS + cfg.sts_segment_length * STS_SEGMENT_UNIT_CHIPS

    if cfg.sts_packet_config in (1, 3):
        trailing = True if cfg.sts_packet_config == 1 else cfg.sts_trailing_gap
        sts1_len = sts_field_length_chips(cfg, trailing_gap=trailing)
        sts1_slice = slice(cursor, cursor + sts1_len)
        srmarkers = [cursor + i * seg_step for i in range(cfg.num_sts_segments + 1)]
        cursor += sts1_len

    if cfg.has_phr_and_psdu:
        if cfg.mode == "HPRF" and cfg.constraint_length == 7:
            phr_syms = 25
            psdu_syms = cfg.psdu_length * 8 + 6
        else:
            phr_syms = 21
            n_rs = rs_encoded_length_bits(cfg.psdu_length)
            psdu_syms = n_rs if cfg.viterbi_rate == 0.5 else n_rs // 2

        phr_len = phr_syms * cfg.chips_per_phr_symbol
        psdu_len = psdu_syms * cfg.chips_per_data_symbol
        phr_slice = slice(cursor, cursor + phr_len)
        cursor += phr_len
        psdu_slice = slice(cursor, cursor + psdu_len) if psdu_len > 0 else None
        cursor += psdu_len

    if cfg.sts_packet_config == 2:
        sts2_len = sts_field_length_chips(cfg, trailing_gap=cfg.sts_trailing_gap)
        sts2_slice = slice(cursor, cursor + sts2_len)
        srmarker0 = cursor + cfg.extra_sts_gap * STS_EXTRA_GAP_UNIT_CHIPS
        srmarkers = [srmarker0 + i * seg_step for i in range(cfg.num_sts_segments + 1)]
        cursor += sts2_len

    return HRPFieldIndices(
        sync=sync_slice,
        sfd=sfd_slice,
        sts1=sts1_slice,
        phr=phr_slice,
        psdu=psdu_slice,
        sts2=sts2_slice,
        total_chips=cursor,
        rmarker_chip=rmarker_chip,
        srmarker_chips=tuple(srmarkers),
    )


def _generate_single_packet_chips(psdu_bits: np.ndarray, cfg: HRPConfig) -> np.ndarray:
    """Generate the 1D ternary chip array for one already-validated packet."""
    parts: list[np.ndarray] = [generate_sync(cfg), generate_sfd(cfg)]

    if cfg.sts_packet_config in (1, 3):
        trailing = True if cfg.sts_packet_config == 1 else cfg.sts_trailing_gap
        parts.append(generate_sts(cfg, trailing_gap=trailing))

    if cfg.has_phr_and_psdu:
        phr_chips, psdu_chips = encode_and_modulate_phr_psdu(psdu_bits, cfg)
        parts.append(phr_chips)
        if len(psdu_chips) > 0:
            parts.append(psdu_chips)

    if cfg.sts_packet_config == 2:
        parts.append(generate_sts(cfg, trailing_gap=cfg.sts_trailing_gap))

    return np.concatenate(parts)


def generate_chips(
    psdu: PSDUInput,
    cfg: HRPConfig,
    *,
    num_packets: int = 1,
    pad_or_truncate: bool = False,
    advance_sts_counter: bool = True,
) -> np.ndarray:
    """Generate the 499.2 MHz ternary chip sequence ``{-1, 0, +1}`` for one or more HRP PPDUs.

    Args:
        psdu: Payload bytes, LSB-first bit sequence, sequence of per-packet byte
            buffers, or ``None`` (for zero-length PSDU or STS packet configuration 3).
        cfg: PHY frame configuration (:class:`HRPConfig`).
        num_packets: Number of consecutive packets to generate (default ``1``).
        pad_or_truncate: If ``True``, cyclically repeat short bit/byte streams or
            truncate long streams to ``num_packets * cfg.psdu_length * 8`` bits.
            If ``False`` (default), strictly validate exact input length.
        advance_sts_counter: If ``True`` (default), advance the 32-bit STS DRBG counter
            ``VCounter`` across consecutive packets per IEEE 802.15.4-2024 Section
            16.2.9.2 / Annex G. Set ``False`` to reuse ``cfg.sts_v`` for each packet.

    Returns:
        1D ``int8`` array of shape ``(chips_per_packet,)`` when ``num_packets == 1``,
        or 2D ``int8`` array of shape ``(chips_per_packet, num_packets)`` when
        ``num_packets > 1``.
    """
    pkt_bits_list = _normalize_multi_psdu_bits(
        psdu, cfg, num_packets, pad_or_truncate=pad_or_truncate
    )
    if num_packets == 1:
        return _generate_single_packet_chips(pkt_bits_list[0], cfg)

    blocks_per_pkt = sts_drbg_blocks_per_packet(cfg) if advance_sts_counter else 0
    columns: list[np.ndarray] = []
    for pkt_idx, pkt_bits in enumerate(pkt_bits_list):
        pkt_cfg = (
            replace(cfg, sts_v=advance_sts_v(cfg.sts_v, pkt_idx * blocks_per_pkt))
            if blocks_per_pkt > 0 and pkt_idx > 0
            else cfg
        )
        columns.append(_generate_single_packet_chips(pkt_bits, pkt_cfg))
    return np.column_stack(columns)


def generate_waveform(
    psdu: PSDUInput,
    cfg: HRPConfig,
    pulse: PulseConfig | PulseShaper | None = None,
    *,
    num_packets: int = 1,
    idle_time: float = 0.0,
    pad_or_truncate: bool = False,
    advance_sts_counter: bool = True,
) -> HRPWaveformResult:
    """Generate one or more IEEE 802.15.4-2024 HRP UWB frames and baseband waveform.

    Args:
        psdu: Payload bytes, LSB-first bit sequence, sequence of per-packet byte
            buffers, or ``None`` (for zero-length PSDU or STS packet configuration 3).
        cfg: PHY frame configuration (:class:`HRPConfig`).
        pulse: Optional :class:`PulseConfig` subclass (e.g., :class:`ButterworthPulse`
            or :class:`RRCPulse`) or pre-built :class:`PulseShaper`. Defaults to
            :class:`ButterworthPulse` (4th-order baseband Butterworth, equivalent to
            an 8th-order RF bandpass filter per Figure 16-23) at 4 samples/pulse
            matched to ``cfg.channel``.
        num_packets: Number of consecutive packets to generate (default ``1``).
        idle_time: Silent idle duration in seconds appended after each packet
            (default ``0.0``).
        pad_or_truncate: If ``True``, cyclically repeat/truncate ``psdu`` to the
            required total bit count.
        advance_sts_counter: If ``True`` (default), advance the 32-bit STS DRBG counter
            across consecutive packets per IEEE 802.15.4-2024 Section 16.2.9.2.

    Returns:
        :class:`HRPWaveformResult` containing the 1D baseband waveform, ternary chips,
        single-packet field indices, sample rate, impulse response taps, packet count,
        per-packet idle sample count, and ``next_sts_v``.
    """
    if idle_time < 0.0:
        raise ValueError(f"idle_time must be >= 0, got {idle_time}.")

    chips = generate_chips(
        psdu,
        cfg,
        num_packets=num_packets,
        pad_or_truncate=pad_or_truncate,
        advance_sts_counter=advance_sts_counter,
    )
    indices = field_indices(cfg)

    shaper = pulse if isinstance(pulse, PulseShaper) else PulseShaper.for_hrp_config(cfg, pulse)
    idle_samples = math.floor(idle_time * shaper.sample_rate + 0.5) if idle_time > 0.0 else 0

    if num_packets == 1:
        pkt_wave = shaper.shape(chips)
        waveform = (
            np.concatenate((pkt_wave, np.zeros(idle_samples, dtype=np.float64)))
            if idle_samples > 0
            else pkt_wave
        )
    else:
        segments: list[np.ndarray] = []
        idle_pad = np.zeros(idle_samples, dtype=np.float64) if idle_samples > 0 else None
        for pkt_idx in range(num_packets):
            segments.append(shaper.shape(chips[:, pkt_idx]))
            if idle_pad is not None:
                segments.append(idle_pad)
        waveform = np.concatenate(segments)

    next_sts_v: bytes | None
    if cfg.sts_packet_config == 0:
        next_sts_v = None
    elif advance_sts_counter:
        next_sts_v = advance_sts_v(cfg.sts_v, num_packets * sts_drbg_blocks_per_packet(cfg))
    else:
        next_sts_v = cfg.sts_v

    return HRPWaveformResult(
        waveform=waveform,
        chips=chips,
        indices=indices,
        sample_rate=shaper.sample_rate,
        pulse_taps=shaper.taps,
        num_packets=num_packets,
        idle_samples=idle_samples,
        next_sts_v=next_sts_v,
    )
