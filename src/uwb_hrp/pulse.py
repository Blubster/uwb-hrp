"""Baseband pulse shaping and Section 16.4.4 compliance verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal

from uwb_hrp.config import HRPConfig
from uwb_hrp.tables import CHANNEL_SPECS, PEAK_PRF_HZ, ChannelSpec


def rrc_pulse(
    samples_per_pulse: int,
    *,
    tp_s: float = 1.0 / PEAK_PRF_HZ,
    rolloff: float = 0.5,
    span_chips: int = 8,
) -> np.ndarray:
    """Generate the Section 16.4.4 Root Raised Cosine (RRC) pulse, unit-energy normalized.

    Uses the exact closed-form expression for r(t) with analytical limits at
    ``t = 0`` and ``t = +/- Tp / (4 * beta)``.
    """
    fs = PEAK_PRF_HZ * samples_per_pulse
    half_samples = (span_chips * samples_per_pulse) // 2
    t = np.arange(-half_samples, half_samples + 1, dtype=np.float64) / fs
    u = t / tp_s
    beta = float(rolloff)

    out = np.empty_like(u)
    zero_mask = np.isclose(u, 0.0, atol=1e-12)
    sing_mask = np.isclose(np.abs(u), 1.0 / (4.0 * beta), atol=1e-12)
    reg_mask = ~(zero_mask | sing_mask)

    if np.any(zero_mask):
        out[zero_mask] = (1.0 - beta + 4.0 * beta / np.pi) / np.sqrt(tp_s)

    if np.any(sing_mask):
        arg = np.pi / (4.0 * beta)
        out[sing_mask] = (beta / np.sqrt(2.0 * tp_s)) * (
            (1.0 + 2.0 / np.pi) * np.sin(arg) + (1.0 - 2.0 / np.pi) * np.cos(arg)
        )

    ur = u[reg_mask]
    num = np.cos((1.0 + beta) * np.pi * ur) + np.sin((1.0 - beta) * np.pi * ur) / (4.0 * beta * ur)
    den = 1.0 - (4.0 * beta * ur) ** 2
    out[reg_mask] = (4.0 * beta / (np.pi * np.sqrt(tp_s))) * (num / den)

    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def _butterworth_iir(
    samples_per_pulse: int,
    *,
    bandwidth_hz: float = PEAK_PRF_HZ,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Design the causal baseband digital Butterworth IIR filter ``(b, a)``."""
    fs = PEAK_PRF_HZ * samples_per_pulse
    wn = bandwidth_hz / fs  # (bandwidth_hz / 2) / (fs / 2)
    if not (0.0 < wn < 1.0):
        min_spp = int(np.floor(bandwidth_hz / PEAK_PRF_HZ)) + 1
        raise ValueError(
            f"samples_per_pulse={samples_per_pulse} is too low for bandwidth "
            f"{bandwidth_hz / 1e6:.2f} MHz (normalized cutoff Wn={wn:.3f} >= 1.0); "
            f"use samples_per_pulse >= {min_spp}."
        )
    b, a = signal.butter(order, wn, btype="low")
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def butterworth_pulse(
    samples_per_pulse: int,
    *,
    bandwidth_hz: float = PEAK_PRF_HZ,
    order: int = 4,
    span_chips: int = 12,
) -> np.ndarray:
    """Generate the causal baseband Butterworth impulse response (Figure 16-23).

    ``bandwidth_hz`` is the double-sided 3 dB RF bandwidth (e.g., 499.2 MHz for
    narrowband channels, corresponding to a baseband lowpass cutoff of
    ``bandwidth_hz / 2``). ``order=4`` is the baseband lowpass equivalent of an
    8th-order RF bandpass Butterworth filter.
    """
    b, a = _butterworth_iir(samples_per_pulse, bandwidth_hz=bandwidth_hz, order=order)
    n_samples = span_chips * samples_per_pulse + 1
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    return np.asarray(signal.lfilter(b, a, impulse), dtype=np.float64)


@dataclass(frozen=True)
class PulseConfig(ABC):
    """Abstract base class for immutable baseband pulse-shaping configurations.

    Subclasses (such as :class:`ButterworthPulse` and :class:`RRCPulse`) define
    shape-specific parameters and implement :meth:`build_taps` (and optionally
    :meth:`build_filter` for IIR or custom shaping). New pulse shapes can be
    added by subclassing ``PulseConfig`` as a ``@dataclass(frozen=True)`` without
    modifying :class:`PulseShaper` or :func:`uwb_hrp.generate_waveform`.

    Attributes:
        samples_per_pulse: Oversampling factor relative to the 499.2 MHz peak PRF.
        bandwidth_hz: Optional explicit 3 dB double-sided RF bandwidth in Hz. When
            ``None``, the channel bandwidth from Table 16-27 is used.
    """

    samples_per_pulse: int = 4
    bandwidth_hz: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.samples_per_pulse < 1:
            raise ValueError(f"samples_per_pulse must be >= 1, got {self.samples_per_pulse}.")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive, got {self.bandwidth_hz}.")

    @property
    def sample_rate(self) -> float:
        """Output baseband waveform sample rate in Hz (``499.2e6 * samples_per_pulse``)."""
        return PEAK_PRF_HZ * self.samples_per_pulse

    @abstractmethod
    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        """Return the sampled baseband impulse response for ``channel_spec``."""

    def build_filter(
        self,
        channel_spec: ChannelSpec,
        taps: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a callable that filters a zero-stuffed upsampled chip stream.

        The default implementation applies peak-aligned FIR convolution using
        ``taps`` so sample ``k * samples_per_pulse`` aligns with chip ``k``'s peak.
        IIR pulses may override this method to apply direct recursive filtering.
        """
        peak_idx = int(np.argmax(np.abs(taps)))

        def _fir_filter(upsampled: np.ndarray) -> np.ndarray:
            full = signal.fftconvolve(upsampled, taps, mode="full")
            return np.asarray(full[peak_idx : peak_idx + len(upsampled)], dtype=np.float64)

        return _fir_filter


@dataclass(frozen=True)
class ButterworthPulse(PulseConfig):
    """Causal baseband Butterworth pulse configuration (Figure 16-23).

    Attributes:
        samples_per_pulse: Oversampling factor relative to 499.2 MHz (default ``4``).
        order: Baseband lowpass Butterworth order (default ``4``, equivalent to an
            8th-order RF bandpass filter per Figure 16-23).
        span_chips: Truncated impulse response length in chip periods for ``taps``
            inspection and compliance verification (default ``12``).
        bandwidth_hz: Optional explicit double-sided 3 dB RF bandwidth in Hz.
    """

    order: int = 4
    span_chips: int = field(default=12, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}.")
        if self.span_chips < 2:
            raise ValueError(f"span_chips must be >= 2, got {self.span_chips}.")

    def _effective_bandwidth(self, channel_spec: ChannelSpec) -> float:
        return (
            self.bandwidth_hz if self.bandwidth_hz is not None else channel_spec.bandwidth_mhz * 1e6
        )

    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        return butterworth_pulse(
            self.samples_per_pulse,
            bandwidth_hz=self._effective_bandwidth(channel_spec),
            order=self.order,
            span_chips=self.span_chips,
        )

    def build_filter(
        self,
        channel_spec: ChannelSpec,
        taps: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        del taps  # Direct causal IIR filtering avoids truncation error
        b, a = _butterworth_iir(
            self.samples_per_pulse,
            bandwidth_hz=self._effective_bandwidth(channel_spec),
            order=self.order,
        )

        def _iir_filter(upsampled: np.ndarray) -> np.ndarray:
            return np.asarray(signal.lfilter(b, a, upsampled), dtype=np.float64)

        return _iir_filter


@dataclass(frozen=True)
class RRCPulse(PulseConfig):
    """Root Raised Cosine (RRC) baseband pulse configuration (Section 16.4.4).

    Attributes:
        samples_per_pulse: Oversampling factor relative to 499.2 MHz (default ``4``).
        rolloff: Roll-off factor beta in ``(0, 1]`` (``0.5`` reference or ``0.45``).
        span_chips: Impulse response span in chip periods (default ``8``, covering
            at least +/-3 chip periods per Section 16.4.4).
        bandwidth_hz: Optional explicit bandwidth in Hz (sets ``Tp = 1 / bandwidth_hz``).
    """

    rolloff: float = 0.5
    span_chips: int = field(default=8, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.rolloff <= 1.0):
            raise ValueError(f"rolloff must be in (0, 1], got {self.rolloff}.")
        if self.span_chips < 2:
            raise ValueError(f"span_chips must be >= 2, got {self.span_chips}.")

    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        tp_s = (
            1.0 / self.bandwidth_hz
            if self.bandwidth_hz is not None
            else channel_spec.pulse_duration_tp_ns * 1e-9
        )
        return rrc_pulse(
            self.samples_per_pulse,
            tp_s=tp_s,
            rolloff=self.rolloff,
            span_chips=self.span_chips,
        )


@dataclass(frozen=True)
class PulseComplianceReport:
    """IEEE Std 802.15.4-2024 Section 16.4.4 cross-correlation compliance report."""

    compliant: bool
    peak_correlation: float
    main_lobe_width_ns: float
    required_tw_ns: float
    max_sidelobe: float


class PulseShaper:
    """Reusable baseband pulse shaper for IEEE 802.15.4-2024 HRP UWB chip streams."""

    def __init__(self, pulse_cfg: PulseConfig | None = None, *, channel: int = 9) -> None:
        if channel not in CHANNEL_SPECS:
            raise ValueError(f"Invalid channel {channel}; expected integer in 0..15.")
        self.config: PulseConfig = pulse_cfg if pulse_cfg is not None else ButterworthPulse()
        self.channel: int = channel
        self._ch_spec = CHANNEL_SPECS[channel]
        self._taps: np.ndarray = self.config.build_taps(self._ch_spec)
        self._filter_fn: Callable[[np.ndarray], np.ndarray] = self.config.build_filter(
            self._ch_spec, self._taps
        )

    @classmethod
    def for_hrp_config(
        cls,
        hrp_cfg: HRPConfig,
        pulse_cfg: PulseConfig | None = None,
    ) -> PulseShaper:
        """Create a ``PulseShaper`` matched to the channel bandwidth of ``hrp_cfg``."""
        return cls(pulse_cfg=pulse_cfg, channel=hrp_cfg.channel)

    @property
    def taps(self) -> np.ndarray:
        """Baseband impulse response taps."""
        return self._taps.copy()

    @property
    def sample_rate(self) -> float:
        """Output baseband waveform sample rate in Hz."""
        return self.config.sample_rate

    @property
    def samples_per_pulse(self) -> int:
        return self.config.samples_per_pulse

    def shape(self, chips: np.ndarray) -> np.ndarray:
        """Upsample ternary chips ``{-1, 0, +1}`` and apply baseband pulse shaping."""
        chips_arr = np.asarray(chips, dtype=np.float64)
        spp = self.config.samples_per_pulse
        if len(chips_arr) == 0:
            return np.empty(0, dtype=np.float64)

        upsampled = np.zeros(len(chips_arr) * spp, dtype=np.float64)
        upsampled[::spp] = chips_arr
        return self._filter_fn(upsampled)

    def verify_compliance(self) -> PulseComplianceReport:
        """Verify pulse compliance against Section 16.4.4 cross-correlation limits.

        Checks against the reference RRC pulse (beta = 0.5, Tp from Table 16-28):
        - Continuous main-lobe width at ``|phi(tau)| >= 0.8`` is at least ``Tw``;
        - Maximum sidelobe peak is at most ``0.3``.
        """
        # Evaluate on a fine time grid (at least 32 samples/chip) for accurate Tw and peak detection
        fine_spp = max(self.config.samples_per_pulse, 32)
        if fine_spp == self.config.samples_per_pulse:
            p = self._taps
        else:
            fine_cfg = replace(self.config, samples_per_pulse=fine_spp)
            p = PulseShaper(fine_cfg, channel=self.channel).taps

        span = getattr(self.config, "span_chips", 8)
        ref = rrc_pulse(
            fine_spp,
            tp_s=self._ch_spec.pulse_duration_tp_ns * 1e-9,
            rolloff=0.5,
            span_chips=max(span, 12),
        )

        # Normalized cross-correlation |phi(tau)|
        xcorr = np.abs(signal.correlate(p, ref, mode="full")) / (
            float(np.linalg.norm(p)) * float(np.linalg.norm(ref))
        )
        peak_idx = int(np.argmax(xcorr))
        peak_val = float(xcorr[peak_idx])

        # Main-lobe width at |phi(tau)| >= 0.8 around peak_idx
        dt_ns = 1e9 / (PEAK_PRF_HZ * fine_spp)
        above = xcorr >= 0.8
        left = peak_idx
        while left > 0 and above[left - 1]:
            left -= 1
        right = peak_idx
        while right < len(xcorr) - 1 and above[right + 1]:
            right += 1
        main_lobe_width_ns = (right - left) * dt_ns if above[peak_idx] else 0.0

        # Identify sidelobe peaks: local maxima outside the main lobe (past first local minima)
        ml_left = peak_idx
        while ml_left > 0 and xcorr[ml_left - 1] <= xcorr[ml_left]:
            ml_left -= 1
        ml_right = peak_idx
        while ml_right < len(xcorr) - 1 and xcorr[ml_right + 1] <= xcorr[ml_right]:
            ml_right += 1

        local_maxima, _ = signal.find_peaks(xcorr)
        sidelobe_peaks = [float(xcorr[i]) for i in local_maxima if i < ml_left or i > ml_right]
        max_sidelobe = max(sidelobe_peaks, default=0.0)

        tw_ns = self._ch_spec.main_lobe_tw_ns
        compliant = (main_lobe_width_ns >= tw_ns) and (max_sidelobe <= 0.3)
        return PulseComplianceReport(
            compliant=compliant,
            peak_correlation=peak_val,
            main_lobe_width_ns=main_lobe_width_ns,
            required_tw_ns=tw_ns,
            max_sidelobe=max_sidelobe,
        )
