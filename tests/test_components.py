"""Unit tests for normative IEEE 802.15.4-2024 Clause 16 building blocks and pulse shaping."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace

import numpy as np
import pytest

import uwb_hrp
from uwb_hrp import (
    ButterworthPulse,
    HRPConfig,
    PulseConfig,
    PulseShaper,
    RRCPulse,
    compute_fcs16_bits,
    compute_fcs32_bits,
    generate_waveform,
)
from uwb_hrp.fec import _GF_EXP, _GF_MUL_TABLE, RS_GENERATOR_POLY, rs_encode, rs_encode_block
from uwb_hrp.phr import compute_secded
from uwb_hrp.scrambler import generate_scrambler_bits, lfsr_initial_state
from uwb_hrp.sts import sts_drbg_bits, sts_drbg_blocks
from uwb_hrp.tables import DEFAULT_STS_KEY, DEFAULT_STS_V, PREAMBLE_CODES, ChannelSpec


def test_preamble_code_table_integrity() -> None:
    assert len(PREAMBLE_CODES) == 32
    for idx in range(1, 9):
        assert PREAMBLE_CODES[idx].length == 31
    for idx in range(9, 25):
        assert PREAMBLE_CODES[idx].length == 127
    for idx in range(25, 33):
        assert PREAMBLE_CODES[idx].length == 91
    for spec in PREAMBLE_CODES.values():
        assert set(spec.sequence_str) <= {"+", "-", "0"}


def test_table_16_20_lfsr_code6() -> None:
    init = lfsr_initial_state(6)
    assert "".join(map(str, init.tolist())) == "111000101101101"
    out = generate_scrambler_bits(6, 16)
    assert "".join(map(str, out.tolist())) == "0010011101101110"


def test_annex_g_sts_drbg_test_vector() -> None:
    blocks = sts_drbg_blocks(DEFAULT_STS_KEY, DEFAULT_STS_V, 2)
    b0 = blocks[:16].hex().upper()
    b1 = blocks[16:32].hex().upper()
    assert b0 == "7AA6F63EF917AE47115EB6FE3B5A5791"
    assert b1 == "41DA0C7503566357EBF38B2C12BB3E92"

    c_bits = sts_drbg_bits(DEFAULT_STS_KEY, DEFAULT_STS_V, 256)
    expected_c = (
        "0111101010100110111101100011111011111001000101111010111001000111"
        "0001000101011110101101101111111000111011010110100101011110010001"
        "0100000111011010000011000111010100000011010101100110001101010111"
        "1110101111110011100010110010110000010010101110110011111010010010"
    )
    assert "".join(map(str, c_bits.tolist())) == expected_c


def test_section_7_2_11_fcs16_ack_example() -> None:
    ack_bits = [int(c) for c in "010000000000000001010110"]
    fcs = compute_fcs16_bits(ack_bits)
    assert "".join(map(str, fcs.tolist())) == "0010011110011110"


def test_section_7_2_11_fcs32_residue_and_short_padding() -> None:
    # Verify normative 3-octet Ack example from IEEE 802.15.4-2024 Section 7.2.11 (pp. 84-85)
    ack_bits = [int(c) for c in "010000000000000001010110"]
    ack_fcs = compute_fcs32_bits(ack_bits)
    assert "".join(map(str, ack_fcs.tolist())) == "01011101001010011111101000101000"

    # Verify receiver unique remainder polynomial: 0xC704DD7B (Section 7.2.11)
    msg_bits = [1, 0, 1, 1, 0, 0, 1, 0] * 6
    fcs_bits = compute_fcs32_bits(msg_bits)
    reg = 0xFFFFFFFF
    for bit in [*msg_bits, *fcs_bits.tolist()]:
        fb = ((reg >> 31) & 1) ^ bit
        reg = (reg << 1) & 0xFFFFFFFF
        if fb:
            reg ^= 0x04C11DB7
    assert reg == 0xC704DD7B

    # Verify short (< 4 octet) input appends trailing zero bits to 32 bits
    short_bits = [1, 0, 1, 0, 1, 1, 0, 0]
    padded_bits = short_bits + [0] * 24
    assert np.array_equal(compute_fcs32_bits(short_bits), compute_fcs32_bits(padded_bits))


def test_rs_generator_polynomial_and_syndrome_zero() -> None:
    # Section 16.3.3.2: g(x) = x^8 + 55x^7 + 61x^6 + 37x^5 + 48x^4 + 47x^3 + 20x^2 + 6x + 22
    assert RS_GENERATOR_POLY.tolist() == [1, 55, 61, 37, 48, 47, 20, 6, 22]

    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=160, dtype=np.uint8)
    cw_bits = rs_encode_block(bits)
    assert len(cw_bits) == 208

    # Reconstruct full 378-bit codeword (with leading zeros) and verify U(alpha^k) == 0 for k=1..8
    full_bits = np.concatenate((np.zeros(330 - 160, dtype=np.uint8), cw_bits))
    weights = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    syms = full_bits.reshape(63, 6) @ weights
    for k in range(1, 9):
        root = int(_GF_EXP[k])
        val = 0
        for sym in syms:
            val = int(_GF_MUL_TABLE[val, root]) ^ int(sym)
        assert val == 0

    # Multi-block check (> 330 bits)
    long_bits = rng.integers(0, 2, size=1016, dtype=np.uint8)
    encoded_long = rs_encode(long_bits)
    assert len(encoded_long) == 1016 + 4 * 48


def test_secded_single_error_detect_and_overall_parity() -> None:
    info = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    parity = compute_secded(info)
    codeword = [*info, *parity]
    # Overall parity (XOR of all 19 bits) must be 0
    assert sum(codeword) % 2 == 0


@pytest.mark.parametrize(
    ("channel", "spp"),
    [
        (0, 4),
        (3, 4),
        (9, 4),
        (4, 8),
        (7, 8),
        (11, 8),
        (15, 8),
    ],
)
@pytest.mark.parametrize("pulse_kind", ["butterworth", "rrc_0.5", "rrc_0.45"])
def test_pulse_compliance_section_16_4_4(channel: int, spp: int, pulse_kind: str) -> None:
    pulse_cfg: PulseConfig
    if pulse_kind == "butterworth":
        pulse_cfg = ButterworthPulse(samples_per_pulse=spp)
    elif pulse_kind == "rrc_0.5":
        pulse_cfg = RRCPulse(samples_per_pulse=spp, rolloff=0.5)
    else:
        pulse_cfg = RRCPulse(samples_per_pulse=spp, rolloff=0.45)

    shaper = PulseShaper(pulse_cfg, channel=channel)
    report = shaper.verify_compliance()
    assert report.compliant, f"Channel {channel}, {type(pulse_cfg).__name__}: {report}"
    assert report.xcorr_compliant
    assert report.psd_compliant
    assert report.main_lobe_width_ns >= report.required_tw_ns
    assert report.max_sidelobe <= 0.3
    assert report.psd_transition_max_dbr < -10.0
    assert report.psd_stopband_max_dbr < -18.0


def test_pulse_compliance_psd_mask_undersampling_and_min_precursor() -> None:
    # Undersampled RRC on wideband channel 4 is caught instead of silently re-synthesized
    rep_wb_spp1 = PulseShaper(
        RRCPulse(samples_per_pulse=1, rolloff=0.5), channel=4
    ).verify_compliance()
    assert not rep_wb_spp1.compliant
    assert not rep_wb_spp1.xcorr_compliant
    assert not rep_wb_spp1.psd_compliant

    rep_wb_spp2 = PulseShaper(
        RRCPulse(samples_per_pulse=2, rolloff=0.5), channel=4
    ).verify_compliance()
    assert not rep_wb_spp2.compliant
    assert not rep_wb_spp2.psd_compliant

    # Butterworth on wideband channel 4 at spp=4 has bilinear warping sidelobe > 0.3
    rep_wb_bw4 = PulseShaper(ButterworthPulse(samples_per_pulse=4), channel=4).verify_compliance()
    assert not rep_wb_bw4.xcorr_compliant
    assert not rep_wb_bw4.compliant

    # Butterworth order=4 at high oversampling (spp=8) exceeds Section 16.4.5 PSD mask slightly,
    # whereas order=5 satisfies both Section 16.4.4 and Section 16.4.5
    rep_bw4_hi = PulseShaper(
        ButterworthPulse(samples_per_pulse=8, order=4), channel=9
    ).verify_compliance()
    assert rep_bw4_hi.xcorr_compliant
    assert not rep_bw4_hi.psd_compliant
    assert not rep_bw4_hi.compliant

    rep_bw5_hi = PulseShaper(
        ButterworthPulse(samples_per_pulse=8, order=5), channel=9
    ).verify_compliance()
    assert rep_bw5_hi.compliant
    assert rep_bw5_hi.xcorr_compliant
    assert rep_bw5_hi.psd_compliant

    # Figure 16-24 minimum-precursor mask: causal Butterworth passes, symmetric RRC has precursors
    rep_bw4 = PulseShaper(
        ButterworthPulse(samples_per_pulse=4, order=4), channel=9
    ).verify_compliance()
    assert rep_bw4.min_precursor_mask_compliant
    rep_rrc = PulseShaper(RRCPulse(samples_per_pulse=4, rolloff=0.5), channel=9).verify_compliance()
    assert not rep_rrc.min_precursor_mask_compliant


def test_generate_waveform_api_and_reusable_shaper() -> None:
    cfg = HRPConfig(mode="HPRF", channel=9, mean_prf=124.8, psdu_length=20)
    payload = b"\xba\xbe" * 10

    # Default pulse shaping (ButterworthPulse: 4th-order baseband Butterworth, 4 samples/pulse)
    res_default = generate_waveform(payload, cfg)
    assert res_default.sample_rate == pytest.approx(499.2e6 * 4)
    assert len(res_default.waveform) == len(res_default.chips) * 4

    # Single packet with trailing idle time
    res_idle = generate_waveform(payload, cfg, idle_time=0.5e-6)
    expected_idle = round(0.5e-6 * res_idle.sample_rate)
    assert res_idle.num_packets == 1
    assert res_idle.idle_samples == expected_idle
    assert len(res_idle.waveform) == len(res_idle.chips) * 4 + expected_idle
    assert np.array_equal(res_idle.waveform[:-expected_idle], res_default.waveform)
    assert np.all(res_idle.waveform[-expected_idle:] == 0.0)

    # Custom RRCPulse & reusable PulseShaper produce identical output
    rrc_cfg = RRCPulse(samples_per_pulse=8, rolloff=0.45)
    shaper = PulseShaper.for_hrp_config(cfg, rrc_cfg)
    res_cfg = generate_waveform(payload, cfg, pulse=rrc_cfg)
    res_shaper = generate_waveform(payload, cfg, pulse=shaper)
    assert res_cfg.sample_rate == pytest.approx(499.2e6 * 8)
    assert len(res_cfg.waveform) == len(res_cfg.chips) * 8
    assert np.array_equal(res_cfg.waveform, res_shaper.waveform)
    assert np.array_equal(res_cfg.chips, res_default.chips)


def test_custom_pulse_config_subclass() -> None:
    @dataclass(frozen=True)
    class DeltaPulse(PulseConfig):
        amplitude: float = 2.0

        def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
            del channel_spec
            return np.array([self.amplitude], dtype=np.float64)

    cfg = HRPConfig(mode="BPRF", channel=9, preamble_duration=16, psdu_length=10)
    res = generate_waveform(b"\xaa" * 10, cfg, pulse=DeltaPulse(samples_per_pulse=2, amplitude=2.0))
    assert np.array_equal(res.waveform[::2], 2.0 * res.chips)
    assert np.all(res.waveform[1::2] == 0.0)


def test_version_and_config_immutability() -> None:
    assert isinstance(uwb_hrp.__version__, str) and len(uwb_hrp.__version__) > 0

    cfg = HRPConfig(mode="BPRF", channel=9, psdu_length=20)
    with pytest.raises(FrozenInstanceError):
        cfg.psdu_length = 999  # type: ignore[misc]

    bw = ButterworthPulse(samples_per_pulse=4)
    with pytest.raises(FrozenInstanceError):
        bw.order = 8  # type: ignore[misc]

    rrc = RRCPulse(samples_per_pulse=4)
    with pytest.raises(FrozenInstanceError):
        rrc.rolloff = 0.25  # type: ignore[misc]


def test_multipacket_idle_time_and_sts_counter_advance() -> None:
    cfg = HRPConfig(
        mode="BPRF",
        channel=9,
        code_index=9,
        preamble_duration=16,
        sfd_number=2,
        psdu_length=20,
        sts_packet_config=1,
        sts_key=DEFAULT_STS_KEY,
        sts_v=DEFAULT_STS_V,
    )
    pkt1 = b"\xba\xbe" * 10
    pkt2 = b"\xca\xfe" * 10

    # advance_sts_counter defaults to True per Section 16.2.9.2
    pulse_cfg = ButterworthPulse(samples_per_pulse=2)
    res_adv = generate_waveform(
        [pkt1, pkt2],
        cfg,
        pulse=pulse_cfg,
        num_packets=2,
        idle_time=1e-6,
    )
    assert res_adv.num_packets == 2
    assert res_adv.idle_samples == 998  # round(1e-6 * 998.4e6)
    assert res_adv.chips.shape == (res_adv.indices.total_chips, 2)
    assert len(res_adv.waveform) == 2 * (res_adv.indices.total_chips * 2 + 998)
    assert np.all(res_adv.waveform[-998:] == 0.0)
    assert res_adv.next_sts_v == bytes.fromhex("362eeb34c44fa8fbd37ec3ca1f9a3e24")

    # Verify single-packet streaming with res.next_sts_v matches multi-packet burst
    res_pkt1 = generate_waveform(pkt1, cfg, pulse=pulse_cfg)
    assert res_pkt1.next_sts_v == bytes.fromhex("362eeb34c44fa8fbd37ec3ca1f9a3e04")
    res_pkt2 = generate_waveform(pkt2, replace(cfg, sts_v=res_pkt1.next_sts_v), pulse=pulse_cfg)
    assert np.array_equal(res_adv.chips[:, 0], res_pkt1.chips)
    assert np.array_equal(res_adv.chips[:, 1], res_pkt2.chips)
    assert res_pkt2.next_sts_v == res_adv.next_sts_v

    sts_slice = res_adv.indices.sts1
    assert sts_slice is not None
    assert not np.array_equal(res_adv.chips[sts_slice, 0], res_adv.chips[sts_slice, 1])

    # With advance_sts_counter=False, STS repeats across packets and next_sts_v is unchanged
    res_same = generate_waveform(
        [pkt1, pkt2],
        cfg,
        pulse=pulse_cfg,
        num_packets=2,
        advance_sts_counter=False,
    )
    assert np.array_equal(res_same.chips[sts_slice, 0], res_same.chips[sts_slice, 1])
    assert res_same.next_sts_v == cfg.sts_v

    # When STS is disabled, next_sts_v is None
    res_no_sts = generate_waveform(pkt1, replace(cfg, sts_packet_config=0), pulse=pulse_cfg)
    assert res_no_sts.next_sts_v is None


def test_hprf_edge_cases_k7_zero_psdu_and_4095_bytes() -> None:
    # HPRF K=7 with 0-byte PSDU still emits 6 convolutional tail symbols (Section 16.3.3.3)
    cfg_k7_zero = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=124.8,
        constraint_length=7,
        psdu_length=0,
    )
    res = generate_waveform(b"", cfg_k7_zero)
    assert res.indices.phr is not None
    assert res.indices.phr.stop - res.indices.phr.start == 25 * 64
    assert res.indices.psdu is not None
    assert res.indices.psdu.stop - res.indices.psdu.start == 6 * 64
    assert len(res.chips) == res.indices.total_chips

    # HPRF maximum 4095-byte PSDU (Table 16-16 psdu_size_extension == 2)
    cfg_4095 = HRPConfig(
        mode="HPRF",
        channel=9,
        mean_prf=249.6,
        constraint_length=3,
        psdu_length=4095,
    )
    assert cfg_4095.psdu_size_extension == 2
    res_max = generate_waveform(b"\xab" * 4095, cfg_4095)
    assert len(res_max.chips) == res_max.indices.total_chips


def test_config_validation_errors() -> None:
    with pytest.raises(ValueError, match="wideband"):
        HRPConfig(mode="802.15.4a", channel=4, mean_prf=3.9)
    with pytest.raises(ValueError, match="4096"):
        HRPConfig(
            mode="802.15.4a",
            channel=9,
            mean_prf=3.9,
            preamble_mean_prf=4.03,
            preamble_duration=4096,
        )
    with pytest.raises(ValueError, match="STS"):
        HRPConfig(mode="802.15.4a", sts_packet_config=1)
    with pytest.raises(ValueError, match="psdu_length"):
        HRPConfig(mode="BPRF", psdu_length=128)
    # Table 16-11 / Section 16.7: BPRF supports all HRP-ERDEV SFD selectors 0..4 (0 & 2 mandatory)
    for sfd in (0, 1, 2, 3, 4):
        cfg_bprf_sfd = HRPConfig(mode="BPRF", sfd_number=sfd)
        assert cfg_bprf_sfd.sfd_number == sfd
    with pytest.raises(ValueError, match="sfd_number"):
        HRPConfig(mode="BPRF", sfd_number="long")
    with pytest.raises(ValueError, match="sfd_number"):
        HRPConfig(mode="BPRF", sfd_number=5)
    with pytest.raises(ValueError, match="constraint_length"):
        HRPConfig(mode="BPRF", constraint_length=7)
    with pytest.raises(ValueError, match="conflicts"):
        HRPConfig(
            mode="HPRF",
            sts_packet_config=2,
            psdu_length=4095,
            extra_sts_gap_index=1,
        )
    # Table 16-16: psdu_size_extension == 1 uses A0 for bit 10, conflicting with 2-bit gap index
    with pytest.raises(ValueError, match="conflicts"):
        HRPConfig(
            mode="HPRF",
            sts_packet_config=2,
            psdu_length=1024,
            extra_sts_gap_index=3,
        )
    # Extended PSDU in HPRF SP2 without extra_sts_gap_index is valid (e.g. hprf_34e.mat)
    cfg_ext1 = HRPConfig(
        mode="HPRF",
        sts_packet_config=2,
        psdu_length=1024,
    )
    assert cfg_ext1.psdu_size_extension == 1
    # Table 16-17: BPRF STS only supports 1 segment of length 64
    with pytest.raises(ValueError, match="num_sts_segments=1"):
        HRPConfig(mode="BPRF", sts_packet_config=1, num_sts_segments=2)
    with pytest.raises(ValueError, match="sts_segment_length=64"):
        HRPConfig(mode="BPRF", sts_packet_config=1, sts_segment_length=32)
    with pytest.raises(ValueError, match=r"code_index|length"):
        HRPConfig(mode="HPRF", code_index=9)
    # Wideband Butterworth filter rejects undersampled samples_per_pulse where BW >= fs
    with pytest.raises(ValueError, match="samples_per_pulse"):
        PulseShaper(ButterworthPulse(samples_per_pulse=2), channel=4)
