"""Verification against all 93 official IEEE P802.15.4z HRP UWB .mat test vectors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from uwb_hrp import HRPConfig, compute_fcs16, compute_fcs32, field_indices, generate_chips

VECTORS_DIR = Path(__file__).resolve().parent / "ieee_test_vectors"
PACKET_INFO_PATH = VECTORS_DIR / "packet_info.txt"


@dataclass(frozen=True)
class IEEEVectorSpec:
    name: str
    mat_filename: str
    mode: str
    code_index: int
    psr: int
    sfd_number: int
    phr_rate: float | None
    psdu_length: int
    psdu_pattern: str | None
    mean_prf: float | None
    constraint_length: int
    sts_packet_config: int
    num_sts_segments: int
    sts_segment_length: int
    extra_sts_gap: int
    extra_sts_gap_index: int | None
    sts_key: bytes | None
    sts_v: bytes | None
    expected_ranges: dict[str, tuple[int, int] | None]


def _parse_packet_info(path: Path) -> list[IEEEVectorSpec]:
    text = path.read_text(encoding="utf-8")
    header_re = re.compile(r"^((?:BPRF|HPRF) Set #[0-9a-z]+) \(([^)]+\.mat)\)\s*$", re.MULTILINE)
    matches = list(header_re.finditer(text))
    specs: list[IEEEVectorSpec] = []

    for idx, match in enumerate(matches):
        name = match.group(1)
        mat_filename = match.group(2)
        block_start = match.end()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        mode = "BPRF" if name.startswith("BPRF") else "HPRF"

        shr_m = re.search(r"SYNC code 802_15_4z?_code(\d+), PSR (\d+) symbols", block)
        assert shr_m is not None, f"Failed to parse SHR for {name}"
        code_index = int(shr_m.group(1))
        psr = int(shr_m.group(2))

        sfd_m = re.search(r"phyHrpUwbSfdSelector (\d+)", block)
        assert sfd_m is not None, f"Failed to parse SFD for {name}"
        sfd_number = int(sfd_m.group(1))

        phr_m = re.search(r"PHR\s*:\s*(.+)", block)
        assert phr_m is not None, f"Failed to parse PHR for {name}"
        phr_str = phr_m.group(1).strip()
        phr_rate: float | None
        if phr_str == "No PHR":
            phr_rate = None
        elif "975 kbit/s" in phr_str:
            phr_rate = 0.85
        elif "7.80 Mbit/s" in phr_str:
            phr_rate = 6.81 if mode == "BPRF" else 7.8
        elif "3.90 Mbit/s" in phr_str:
            phr_rate = 3.9
        elif "15.60 Mbit/s" in phr_str:
            phr_rate = 15.6
        elif "31.20 Mbit/s" in phr_str:
            phr_rate = 31.2
        else:
            raise ValueError(f"Unexpected PHR description {phr_str!r} in {name}")

        len_m = re.search(r"PSDU Length\s*:\s*(\d+) bytes", block)
        psdu_length = int(len_m.group(1)) if len_m else 0

        data_m = re.search(r"PSDU Data\s*:\s*(\S+)", block)
        psdu_pattern = data_m.group(1).strip() if data_m else None

        prf_m = re.search(r"mean PRF ([0-9.]+) MHz", block)
        mean_prf = float(prf_m.group(1)) if prf_m else (62.4 if mode == "BPRF" else 124.8)

        cl_m = re.search(r"Conv encoding K=(\d+)", block)
        constraint_length = int(cl_m.group(1)) if cl_m else 3

        sts1_m = re.search(r"STS1 \(pre-PSDU\)\s*:\s*(.+)", block)
        sts2_m = re.search(r"STS2 \(post-PSDU\)\s*:\s*(.+)", block)
        assert sts1_m is not None and sts2_m is not None
        sts1_str = sts1_m.group(1).strip()
        sts2_str = sts2_m.group(1).strip()

        num_sts_segments = 1
        sts_segment_length = 64
        if sts1_str.startswith("yes"):
            seg_m = re.search(r"(\d+) segments?, segment length (\d+)", sts1_str)
            assert seg_m is not None
            num_sts_segments = int(seg_m.group(1))
            sts_segment_length = int(seg_m.group(2))
            sts_packet_config = 3 if phr_rate is None else 1
        elif sts2_str.startswith("yes"):
            seg_m = re.search(r"(\d+) segments?, segment length (\d+)", sts2_str)
            assert seg_m is not None
            num_sts_segments = int(seg_m.group(1))
            sts_segment_length = int(seg_m.group(2))
            sts_packet_config = 2
        else:
            sts_packet_config = 0

        extra_sts_gap = 0
        extra_sts_gap_index: int | None = None
        gap_m = re.search(r"STS2 Gap\s*:\s*gap index (\d+), gap length (\d+)", block)
        if gap_m:
            extra_sts_gap_index = int(gap_m.group(1))
            extra_sts_gap = int(gap_m.group(2))

        key_m = re.search(r"DRBG seed key value\s*:\s*([0-9a-fA-F]{32})", block)
        val_m = re.search(r"DRBG seed data value\s*:\s*([0-9a-fA-F]{32})", block)
        sts_key = bytes.fromhex(key_m.group(1)) if key_m else None
        sts_v = bytes.fromhex(val_m.group(1)) if val_m else None

        expected_ranges: dict[str, tuple[int, int] | None] = {}
        for field_name in ("SYNC", "SFD", "STS1", "PHR", "PSDU", "STS2"):
            fm = re.search(rf"{field_name}\s*:\s*(?:\[(\d+)\.\.(\d+)\]|(N/A))", block)
            assert fm is not None, f"Missing {field_name} range in {name}"
            if fm.group(3) == "N/A":
                expected_ranges[field_name] = None
            else:
                expected_ranges[field_name] = (int(fm.group(1)), int(fm.group(2)))

        specs.append(
            IEEEVectorSpec(
                name=name,
                mat_filename=mat_filename,
                mode=mode,
                code_index=code_index,
                psr=psr,
                sfd_number=sfd_number,
                phr_rate=phr_rate,
                psdu_length=psdu_length,
                psdu_pattern=psdu_pattern,
                mean_prf=mean_prf,
                constraint_length=constraint_length,
                sts_packet_config=sts_packet_config,
                num_sts_segments=num_sts_segments,
                sts_segment_length=sts_segment_length,
                extra_sts_gap=extra_sts_gap,
                extra_sts_gap_index=extra_sts_gap_index,
                sts_key=sts_key,
                sts_v=sts_v,
                expected_ranges=expected_ranges,
            )
        )

    return specs


def _build_psdu(spec: IEEEVectorSpec) -> bytes | None:
    if spec.sts_packet_config == 3 or spec.psdu_length == 0:
        return None
    assert spec.psdu_pattern is not None
    if spec.psdu_pattern == "0xBABE..BABE":
        return (b"\xba\xbe" * (spec.psdu_length // 2 + 1))[: spec.psdu_length]
    if spec.psdu_pattern == "0xABAB..AB":
        return b"\xab" * spec.psdu_length
    if spec.psdu_pattern == "0xBABE..BABE+CRC16":
        body = (b"\xba\xbe" * (spec.psdu_length // 2))[: spec.psdu_length - 2]
        return body + compute_fcs16(body)
    if spec.psdu_pattern == "0xBABE..BABE+CRC32":
        body = (b"\xba\xbe" * (spec.psdu_length // 2))[: spec.psdu_length - 4]
        return body + compute_fcs32(body)
    raise ValueError(f"Unknown PSDU pattern {spec.psdu_pattern!r}")


ALL_VECTOR_SPECS = _parse_packet_info(PACKET_INFO_PATH)


def test_all_93_ieee_vectors_discovered() -> None:
    mat_files = sorted(p.name for p in VECTORS_DIR.glob("*.mat"))
    parsed_files = sorted(s.mat_filename for s in ALL_VECTOR_SPECS)
    assert len(ALL_VECTOR_SPECS) == 93
    assert parsed_files == mat_files


@pytest.mark.parametrize("spec", ALL_VECTOR_SPECS, ids=lambda s: s.mat_filename)
def test_ieee_test_vector(spec: IEEEVectorSpec) -> None:
    kwargs: dict[str, object] = {
        "mode": spec.mode,
        "channel": 9,
        "code_index": spec.code_index,
        "mean_prf": spec.mean_prf,
        "preamble_duration": spec.psr,
        "sfd_number": spec.sfd_number,
        "psdu_length": spec.psdu_length,
        "ranging": False,
        "constraint_length": spec.constraint_length,
        "sts_packet_config": spec.sts_packet_config,
        "num_sts_segments": spec.num_sts_segments,
        "sts_segment_length": spec.sts_segment_length,
        "extra_sts_gap": spec.extra_sts_gap,
        "extra_sts_gap_index": spec.extra_sts_gap_index,
        # IEEE P802.15-20-0002-01-004z (Draft D08) omits the trailing 512 silent zero
        # chips at end-of-frame in SP2/SP3 (SP1 always retains the gap before PHR).
        "sts_trailing_gap": False,
    }
    if spec.phr_rate is not None:
        kwargs["phr_data_rate"] = spec.phr_rate
    if spec.sts_key is not None:
        kwargs["sts_key"] = spec.sts_key
    if spec.sts_v is not None:
        kwargs["sts_v"] = spec.sts_v

    cfg = HRPConfig(**kwargs)  # type: ignore[arg-type]
    psdu = _build_psdu(spec)

    # 1. Exact field index ranges
    indices = field_indices(cfg)
    assert indices.to_1based_ranges() == spec.expected_ranges

    # 2. Exact chip-for-chip comparison against IEEE .mat vector
    chips = generate_chips(psdu, cfg)
    mat = loadmat(VECTORS_DIR / spec.mat_filename)
    expected = mat["pulses_all"].ravel().astype(np.int8)

    assert len(chips) == len(expected), f"Length mismatch: {len(chips)} != {len(expected)}"
    assert np.array_equal(chips, expected), (
        f"Chip mismatch in {spec.mat_filename} at first index "
        f"{int(np.flatnonzero(chips != expected)[0])}"
    )
