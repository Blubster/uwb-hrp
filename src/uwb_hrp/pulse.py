"""Baseband pulse shaping and Section 16.4.4 compliance verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

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
    """Generate the Section 16.4.4 Root Raised Cosine (RRC) pulse.

    Uses the exact closed-form expression for `r(t)` with analytical limits at
    `t = 0` and `t = +/- Tp / (4 * beta)`, normalized to unit L2 energy.

    Args:
        samples_per_pulse: Oversampling factor relative to the 499.2 MHz peak
            PRF.
        tp_s: Reference pulse duration `Tp` in seconds (Table 16-28).
        rolloff: Roll-off factor `beta` in `(0, 1]` (`0.5` reference or `0.45`).
        span_chips: Total impulse response length in chip periods.

    Returns:
        1D `float64` unit-energy RRC impulse response taps.
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
    """Design the causal baseband digital Butterworth IIR filter `(b, a)`.

    Args:
        samples_per_pulse: Oversampling factor relative to 499.2 MHz.
        bandwidth_hz: Double-sided 3 dB RF bandwidth in Hz.
        order: Baseband lowpass filter order.

    Returns:
        Tuple `(b, a)` of `float64` numerator and denominator IIR coefficients.

    Raises:
        ValueError: If `samples_per_pulse` is too low for the requested
            `bandwidth_hz` (normalized cutoff `Wn >= 1.0`).
    """
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

    `bandwidth_hz` is the double-sided 3 dB RF bandwidth (e.g., 499.2 MHz for
    narrowband channels, corresponding to a baseband lowpass cutoff of
    `bandwidth_hz / 2`). `order=4` is the baseband lowpass equivalent of an
    8th-order RF bandpass Butterworth filter.

    Args:
        samples_per_pulse: Oversampling factor relative to 499.2 MHz.
        bandwidth_hz: Double-sided 3 dB RF bandwidth in Hz.
        order: Baseband lowpass Butterworth order (default `4`).
        span_chips: Truncated impulse response length in chip periods.

    Returns:
        1D `float64` truncated causal impulse response of length
        `span_chips * samples_per_pulse + 1`.

    Raises:
        ValueError: If `samples_per_pulse` is too low for `bandwidth_hz`.
    """
    b, a = _butterworth_iir(samples_per_pulse, bandwidth_hz=bandwidth_hz, order=order)
    n_samples = span_chips * samples_per_pulse + 1
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    return np.asarray(signal.lfilter(b, a, impulse), dtype=np.float64)


@dataclass(frozen=True)
class PulseConfig(ABC):
    """Abstract base class for immutable baseband pulse-shaping configurations.

    Subclasses (such as [ButterworthPulse][uwb_hrp.ButterworthPulse] and
    [RRCPulse][uwb_hrp.RRCPulse]) define shape-specific parameters and implement
    [build_taps][uwb_hrp.PulseConfig.build_taps] (and optionally
    [build_filter][uwb_hrp.PulseConfig.build_filter] for IIR or custom shaping).
    New pulse shapes can be added by subclassing `PulseConfig` as a
    `@dataclass(frozen=True)` without modifying [PulseShaper][uwb_hrp.PulseShaper]
    or [generate_waveform][uwb_hrp.generate_waveform].

    Attributes:
        samples_per_pulse: Oversampling factor relative to the 499.2 MHz peak PRF.
        bandwidth_hz: Optional explicit 3 dB double-sided RF bandwidth in Hz. When
            `None`, the channel bandwidth from Table 16-27 is used.
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
        """Output baseband waveform sample rate in Hz (`499.2e6 * samples_per_pulse`)."""
        return PEAK_PRF_HZ * self.samples_per_pulse

    @abstractmethod
    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        """Build the sampled baseband impulse response for `channel_spec`.

        Args:
            channel_spec: Target channel specification (Tables 16-27 and 16-28).

        Returns:
            1D `float64` sampled baseband impulse response taps.
        """

    def build_filter(
        self,
        channel_spec: ChannelSpec,
        taps: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build a callable that filters a zero-stuffed upsampled chip stream.

        The default implementation applies peak-aligned FIR convolution using
        `taps` so sample `k * samples_per_pulse` aligns with chip `k`'s peak.
        IIR pulses may override this method to apply direct recursive filtering.

        Args:
            channel_spec: Target channel specification (Tables 16-27 and 16-28).
            taps: Impulse response taps returned by `build_taps`.

        Returns:
            Callable accepting a 1D upsampled chip stream and returning the
            shaped `float64` baseband waveform of the same length.
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
        samples_per_pulse: Oversampling factor relative to 499.2 MHz (default `4`).
        order: Baseband lowpass Butterworth order (default `4`, equivalent to an
            8th-order RF bandpass filter per Figure 16-23).
        span_chips: Truncated impulse response length in chip periods for `taps`
            inspection and compliance verification (default `12`).
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
        """Return explicit `bandwidth_hz` or the default channel bandwidth in Hz."""
        return (
            self.bandwidth_hz if self.bandwidth_hz is not None else channel_spec.bandwidth_mhz * 1e6
        )

    def build_taps(self, channel_spec: ChannelSpec) -> np.ndarray:
        """Build truncated causal Butterworth impulse response taps.

        Args:
            channel_spec: Target channel specification.

        Returns:
            1D `float64` causal Butterworth impulse response taps.
        """
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
        """Build a direct causal IIR Butterworth filter to avoid truncation error.

        Args:
            channel_spec: Target channel specification.
            taps: Unused truncated impulse response (direct IIR filtering is used).

        Returns:
            Callable applying `scipy.signal.lfilter` with the Butterworth IIR
            coefficients.
        """
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
        samples_per_pulse: Oversampling factor relative to 499.2 MHz (default `4`).
        rolloff: Roll-off factor `beta` in `(0, 1]` (`0.5` reference or `0.45`).
        span_chips: Impulse response span in chip periods (default `8`, covering
            at least +/-3 chip periods per Section 16.4.4).
        bandwidth_hz: Optional explicit bandwidth in Hz (sets `Tp = 1 / bandwidth_hz`).
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
        """Build unit-energy closed-form RRC impulse response taps.

        Args:
            channel_spec: Target channel specification (provides default `Tp`).

        Returns:
            1D `float64` unit-energy RRC impulse response taps.
        """
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
    """IEEE Std 802.15.4-2024 Sections 16.4.4 & 16.4.5 pulse compliance report.

    Attributes:
        compliant: `True` if both mandatory Section 16.4.4 cross-correlation
            and Section 16.4.5 transmit PSD mask requirements are met.
        peak_correlation: Peak normalized cross-correlation `max |phi(tau)|`
            against the Section 16.4.4 reference RRC pulse (`beta = 0.5`).
        main_lobe_width_ns: Continuous main-lobe width at `|phi(tau)| >= 0.8` (ns).
        required_tw_ns: Minimum required main-lobe width `Tw` from Table 16-28 (ns).
        max_sidelobe: Maximum sidelobe peak magnitude outside the main lobe (limit `0.3`).
        xcorr_compliant: Section 16.4.4 cross-correlation compliance flag.
        psd_compliant: Section 16.4.5 transmit PSD mask compliance flag.
        psd_transition_max_dbr: Peak relative PSD in `0.65/Tp < |f| < 0.8/Tp`
            (limit `< -10 dBr`; `+inf` if sampling rate cannot resolve `0.65/Tp`).
        psd_stopband_max_dbr: Peak relative PSD in `|f| > 0.8/Tp` up to Nyquist
            (limit `< -18 dBr`; `-inf` if `0.8/Tp` exceeds Nyquist).
        min_precursor_mask_compliant: Informational check against the recommended
            Figure 16-24 minimum-precursor time-domain mask.
    """

    compliant: bool
    peak_correlation: float
    main_lobe_width_ns: float
    required_tw_ns: float
    max_sidelobe: float
    xcorr_compliant: bool = True
    psd_compliant: bool = True
    psd_transition_max_dbr: float = float("-inf")
    psd_stopband_max_dbr: float = float("-inf")
    min_precursor_mask_compliant: bool = False


class PulseShaper:
    """Reusable baseband pulse shaper for IEEE 802.15.4-2024 HRP UWB chip streams.

    Attributes:
        config: Active [PulseConfig][uwb_hrp.PulseConfig] instance.
        channel: Target IEEE 802.15.4-2024 UWB channel number (`0..15`).
    """

    def __init__(self, pulse_cfg: PulseConfig | None = None, *, channel: int = 9) -> None:
        """Initialize the pulse shaper for a pulse configuration and UWB channel.

        Args:
            pulse_cfg: Pulse configuration subclass (defaults to
                [ButterworthPulse][uwb_hrp.ButterworthPulse]).
            channel: Target UWB channel number in `0..15` (default `9`).

        Raises:
            ValueError: If `channel` is not in `0..15` or the pulse filter cannot
                be constructed at the configured oversampling rate.
        """
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
        """Create a `PulseShaper` matched to the channel bandwidth of `hrp_cfg`.

        Args:
            hrp_cfg: PHY configuration whose `channel` selects the bandwidth and
                reference `Tp`/`Tw` parameters.
            pulse_cfg: Optional [PulseConfig][uwb_hrp.PulseConfig] subclass
                (defaults to [ButterworthPulse][uwb_hrp.ButterworthPulse]).

        Returns:
            Configured [PulseShaper][uwb_hrp.PulseShaper] instance.
        """
        return cls(pulse_cfg=pulse_cfg, channel=hrp_cfg.channel)

    @property
    def taps(self) -> np.ndarray:
        """Copy of the sampled baseband impulse response taps."""
        return self._taps.copy()

    @property
    def sample_rate(self) -> float:
        """Output baseband waveform sample rate in Hz."""
        return self.config.sample_rate

    @property
    def samples_per_pulse(self) -> int:
        """Oversampling factor relative to the 499.2 MHz peak PRF."""
        return self.config.samples_per_pulse

    def shape(self, chips: np.ndarray) -> np.ndarray:
        """Upsample ternary chips `{-1, 0, +1}` and apply baseband pulse shaping.

        Args:
            chips: 1D ternary chip sequence at 499.2 MHz.

        Returns:
            1D `float64` baseband waveform of length `len(chips) * samples_per_pulse`.
        """
        chips_arr = np.asarray(chips, dtype=np.float64)
        spp = self.config.samples_per_pulse
        if len(chips_arr) == 0:
            return np.empty(0, dtype=np.float64)

        upsampled = np.zeros(len(chips_arr) * spp, dtype=np.float64)
        upsampled[::spp] = chips_arr
        return self._filter_fn(upsampled)

    def verify_compliance(self) -> PulseComplianceReport:
        """Verify pulse compliance against Sections 16.4.4 and 16.4.5.

        Evaluates the actual configured pulse taps (at `self.samples_per_pulse`):

        - **Section 16.4.4 (mandatory):** Normalized cross-correlation against the
          reference RRC pulse (`beta = 0.5`, `Tp` from Table 16-28) has
          continuous main-lobe width at `|phi(tau)| >= 0.8` of at least `Tw`
          and maximum sidelobe peak `<= 0.3`.
        - **Section 16.4.5 (mandatory):** Transmit PSD is `< -10 dBr` for
          `0.65/Tp < |f - fc| < 0.8/Tp` and `< -18 dBr` for `|f - fc| > 0.8/Tp`.
        - **Figure 16-24 (recommended, informational):** Minimum-precursor
          time-domain amplitude mask relative to the first peak.

        Returns:
            [PulseComplianceReport][uwb_hrp.PulseComplianceReport] summarizing
            cross-correlation, PSD mask, and minimum-precursor results.
        """
        spp = self.config.samples_per_pulse
        tp_s = self._ch_spec.pulse_duration_tp_ns * 1e-9
        fs = PEAK_PRF_HZ * spp

        # 1. Section 16.4.5 Transmit PSD mask on the actual configured discrete-time pulse
        nfft = max(16384, 1 << (len(self._taps) * 4).bit_length())
        spectrum = np.fft.rfft(self._taps, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        mag = np.abs(spectrum)
        peak_mag = float(np.max(mag))
        if peak_mag > 0.0:
            psd_dbr = 20.0 * np.log10(np.maximum(mag / peak_mag, 1e-15))
        else:
            psd_dbr = np.full_like(mag, -300.0)

        trans_mask = (freqs > 0.65 / tp_s) & (freqs < 0.8 / tp_s)
        stop_mask = freqs > 0.8 / tp_s
        if np.any(trans_mask):
            psd_transition_max_dbr = float(np.max(psd_dbr[trans_mask]))
            psd_stopband_max_dbr = (
                float(np.max(psd_dbr[stop_mask])) if np.any(stop_mask) else float("-inf")
            )
            psd_compliant = (psd_transition_max_dbr < -10.0) and (psd_stopband_max_dbr < -18.0)
        else:
            # Sampling rate too low (fs/2 <= 0.65/Tp) to resolve the transmit PSD mask
            psd_transition_max_dbr = float("inf")
            psd_stopband_max_dbr = float("inf")
            psd_compliant = False

        # 2. Section 16.4.4 cross-correlation via bandlimited interpolation of actual taps
        factor = max(1, (32 + spp - 1) // spp)
        fine_spp = spp * factor
        p = (
            np.asarray(signal.resample_poly(self._taps, factor, 1), dtype=np.float64)
            if factor > 1
            else self._taps
        )

        span = getattr(self.config, "span_chips", 8)
        ref = rrc_pulse(
            fine_spp,
            tp_s=tp_s,
            rolloff=0.5,
            span_chips=max(span, 12),
        )

        # Normalized cross-correlation |phi(tau)|
        p_norm = float(np.linalg.norm(p))
        ref_norm = float(np.linalg.norm(ref))
        if p_norm == 0.0 or ref_norm == 0.0:
            xcorr = np.zeros(len(p) + len(ref) - 1, dtype=np.float64)
        else:
            xcorr = np.abs(signal.correlate(p, ref, mode="full")) / (p_norm * ref_norm)
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
        xcorr_compliant = (main_lobe_width_ns >= tw_ns) and (max_sidelobe <= 0.3)

        # 3. Figure 16-24 recommended minimum-precursor time-domain mask (informational)
        min_precursor_ok = self._check_min_precursor_mask(p, fine_spp, tp_s)

        compliant = xcorr_compliant and psd_compliant
        return PulseComplianceReport(
            compliant=compliant,
            peak_correlation=peak_val,
            main_lobe_width_ns=main_lobe_width_ns,
            required_tw_ns=tw_ns,
            max_sidelobe=max_sidelobe,
            xcorr_compliant=xcorr_compliant,
            psd_compliant=psd_compliant,
            psd_transition_max_dbr=psd_transition_max_dbr,
            psd_stopband_max_dbr=psd_stopband_max_dbr,
            min_precursor_mask_compliant=min_precursor_ok,
        )

    @staticmethod
    def _check_min_precursor_mask(p_fine: np.ndarray, fine_spp: int, tp_s: float) -> bool:
        """Check interpolated pulse against Figure 16-24 minimum-precursor mask.

        Args:
            p_fine: Bandlimited-interpolated pulse samples at `fine_spp` samples/chip.
            fine_spp: Effective oversampling factor of `p_fine`.
            tp_s: Reference pulse duration `Tp` in seconds.

        Returns:
            `True` if all samples lie within Figure 16-24 upper and lower bounds.
        """
        if len(p_fine) == 0:
            return False
        abs_peak_idx = int(np.argmax(np.abs(p_fine)))
        peak_mag = float(np.abs(p_fine[abs_peak_idx]))
        if peak_mag == 0.0:
            return False

        # Orient positive and scale peak magnitude to 1.0
        p = (p_fine if p_fine[abs_peak_idx] >= 0.0 else -p_fine) / peak_mag

        # Locate first peak amplitude: max before first >1.25% drop on the main pulse
        run_max = 0.0
        first_peak_idx = abs_peak_idx
        for i, val in enumerate(p):
            v = float(val)
            if v > run_max:
                run_max = v
                first_peak_idx = i
            elif run_max > 0.5 and v < run_max - 0.0125:
                break

        dt_tp = 1.0 / (PEAK_PRF_HZ * fine_spp * tp_s)
        t_tp = (np.arange(len(p), dtype=np.float64) - first_peak_idx) * dt_tp

        # Figure 16-24 upper and lower bounds in units of Tp
        upper = np.where(t_tp < -1.25, 0.015, np.where(t_tp <= 1.0, 1.0, 0.3))
        lower = np.where(t_tp < 0.0, -0.015, np.where(t_tp <= 2.0, -0.5, -0.3))
        tol = 1e-9
        return bool(np.all(p <= upper + tol) and np.all(p >= lower - tol))
