from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from ..core.arrays import ArrayLike1D, _as_1d_array


@dataclass(frozen=True)
class RippleCorrectionTarget:
    """Frequency-domain target used to remove smooth-band ripple."""

    measured_freq_hz: np.ndarray
    measured_s21: np.ndarray
    smooth_magnitude_db: np.ndarray
    linear_phase_rad: np.ndarray
    ripple_magnitude_db: np.ndarray
    ripple_phase_rad: np.ndarray
    correction_freq_hz: np.ndarray
    correction_response: np.ndarray
    phase_delay_s: float


@dataclass(frozen=True)
class FIRCorrectionDesign:
    """Low-rate prototype and refined output-rate FIR correction filters."""

    prototype_sample_rate_hz: float
    output_sample_rate_hz: float
    correction_band_hz: float
    prototype_taps: np.ndarray
    interpolated_taps: np.ndarray
    refined_taps: np.ndarray
    prototype_time_s: np.ndarray
    output_time_s: np.ndarray
    target: RippleCorrectionTarget
    notes: tuple[str, ...] = ()


def _validate_frequency_response(freq_hz: ArrayLike1D, s21: ArrayLike1D) -> tuple[np.ndarray, np.ndarray]:
    freq = _as_1d_array(freq_hz, name="freq_hz", dtype=float)
    response = _as_1d_array(s21, name="s21", dtype=complex)
    if freq.shape != response.shape:
        raise ValueError("freq_hz and s21 must have the same shape.")
    if freq.size < 3:
        raise ValueError("At least three S21 samples are required.")
    if np.any(~np.isfinite(freq)) or np.any(~np.isfinite(response)):
        raise ValueError("freq_hz and s21 must contain finite values.")
    if np.any(np.diff(freq) <= 0.0):
        raise ValueError("freq_hz must be strictly increasing.")
    if not np.isclose(freq[0], 0.0):
        raise ValueError("freq_hz must start at 0 Hz for FIR target construction.")
    spacing = np.diff(freq)
    if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=max(1e-6, abs(spacing[0]) * 1e-9)):
        raise ValueError("freq_hz must be uniformly spaced.")
    return freq, response


def _odd_window(requested: int, size: int) -> int:
    window = min(max(3, int(requested)), size if size % 2 else size - 1)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("The measured frequency band is too short for smoothing.")
    return window


def build_ripple_correction_target(
    freq_hz: ArrayLike1D,
    s21: ArrayLike1D,
    *,
    correction_band_hz: float,
    correction_freq_hz: ArrayLike1D,
    smoothing_points: int = 31,
    max_correction_db: float = 3.0,
    edge_taper_fraction: float = 0.18,
    phase_fit_band_hz: float | None = None,
    phase_delay_s: float = 0.0,
) -> RippleCorrectionTarget:
    """Build a bounded correction target from a trusted S21 band.

    The smooth magnitude baseline is removed in dB and the dominant linear
    phase is removed before constructing the correction.  Above the trusted
    band the correction is unity, so unknown high-frequency S21 is not
    invented by the equalizer.
    """

    freq, response = _validate_frequency_response(freq_hz, s21)
    correction_freq = _as_1d_array(correction_freq_hz, name="correction_freq_hz", dtype=float)
    if np.any(np.diff(correction_freq) <= 0.0) or correction_freq[0] < 0.0:
        raise ValueError("correction_freq_hz must be non-negative and increasing.")
    if correction_band_hz <= 0.0 or correction_band_hz > freq[-1]:
        raise ValueError("correction_band_hz must lie inside the measured frequency range.")
    if correction_freq[-1] < correction_band_hz:
        raise ValueError("correction_freq_hz must reach the correction band edge.")
    if not 0.0 <= edge_taper_fraction < 1.0:
        raise ValueError("edge_taper_fraction must be in [0, 1).")
    if max_correction_db < 0.0:
        raise ValueError("max_correction_db must be non-negative.")

    band_mask = freq <= correction_band_hz
    band_freq = freq[band_mask]
    band_s21 = response[band_mask]
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(band_s21), 1e-300))
    phase_rad = np.unwrap(np.angle(band_s21))
    smooth_window = _odd_window(smoothing_points, band_freq.size)
    smooth_magnitude_db = savgol_filter(magnitude_db, smooth_window, 3, mode="interp")

    fit_band = correction_band_hz if phase_fit_band_hz is None else float(phase_fit_band_hz)
    fit_mask = band_freq <= fit_band
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("phase_fit_band_hz leaves too few samples for a linear fit.")
    phase_slope, phase_intercept = np.polyfit(band_freq[fit_mask], phase_rad[fit_mask], 1)
    linear_phase_rad = phase_slope * band_freq + phase_intercept
    ripple_magnitude_db = magnitude_db - smooth_magnitude_db
    ripple_phase_rad = phase_rad - linear_phase_rad

    eval_freq = np.minimum(correction_freq, correction_band_hz)
    ripple_db_eval = np.interp(eval_freq, band_freq, ripple_magnitude_db)
    ripple_phase_eval = np.interp(eval_freq, band_freq, ripple_phase_rad)
    taper_start = correction_band_hz * (1.0 - edge_taper_fraction)
    taper = np.ones_like(correction_freq)
    if edge_taper_fraction > 0.0:
        fade = (correction_freq > taper_start) & (correction_freq < correction_band_hz)
        taper[fade] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (correction_freq[fade] - taper_start)
                / (correction_band_hz - taper_start)
            )
        )
    taper[correction_freq >= correction_band_hz] = 0.0

    correction_db = np.clip(-ripple_db_eval, -max_correction_db, max_correction_db) * taper
    correction_phase = -ripple_phase_eval * taper
    correction = 10.0 ** (correction_db / 20.0) * np.exp(1j * correction_phase)
    if phase_delay_s:
        correction *= np.exp(-1j * 2.0 * np.pi * correction_freq * phase_delay_s)

    return RippleCorrectionTarget(
        measured_freq_hz=band_freq.copy(),
        measured_s21=band_s21.copy(),
        smooth_magnitude_db=smooth_magnitude_db.copy(),
        linear_phase_rad=linear_phase_rad.copy(),
        ripple_magnitude_db=ripple_magnitude_db.copy(),
        ripple_phase_rad=ripple_phase_rad.copy(),
        correction_freq_hz=correction_freq.copy(),
        correction_response=correction,
        phase_delay_s=float(phase_delay_s),
    )


def _second_difference_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((max(0, size - 2), size), dtype=float)
    if size >= 3:
        indices = np.arange(size - 2)
        matrix[indices, indices] = 1.0
        matrix[indices, indices + 1] = -2.0
        matrix[indices, indices + 2] = 1.0
    return matrix


def design_real_fir_from_response(
    freq_hz: ArrayLike1D,
    target_response: ArrayLike1D,
    *,
    sample_rate_hz: float,
    tap_count: int,
    delay_s: float | None = None,
    weights: ArrayLike1D | None = None,
    regularization: float = 1e-4,
    prior_taps: ArrayLike1D | None = None,
    prior_strength: float = 0.0,
    smoothness_strength: float = 0.0,
) -> np.ndarray:
    """Fit a real causal FIR to a positive-frequency complex response."""

    freq = _as_1d_array(freq_hz, name="freq_hz", dtype=float)
    target = _as_1d_array(target_response, name="target_response", dtype=complex)
    if freq.shape != target.shape:
        raise ValueError("freq_hz and target_response must have the same shape.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if tap_count < 3:
        raise ValueError("tap_count must be at least 3.")
    if np.any(freq < 0.0) or np.any(freq > sample_rate_hz / 2.0):
        raise ValueError("freq_hz must lie between DC and Nyquist.")
    if regularization < 0.0 or prior_strength < 0.0 or smoothness_strength < 0.0:
        raise ValueError("regularization strengths must be non-negative.")

    if delay_s is None:
        delay_s = 0.5 * (tap_count - 1) / sample_rate_hz
    if delay_s < 0.0:
        raise ValueError("delay_s must be non-negative.")

    if weights is None:
        weight = np.ones(freq.size, dtype=float)
    else:
        weight = _as_1d_array(weights, name="weights", dtype=float)
        if weight.shape != freq.shape:
            raise ValueError("weights must have the same shape as freq_hz.")
        if np.any(weight < 0.0):
            raise ValueError("weights must be non-negative.")

    n = np.arange(tap_count, dtype=float)
    basis = np.exp(-1j * 2.0 * np.pi * freq[:, None] * n[None, :] / sample_rate_hz)
    delayed_target = target * np.exp(-1j * 2.0 * np.pi * freq * delay_s)
    weighted_real = basis.real * weight[:, None]
    weighted_imag = basis.imag * weight[:, None]
    rhs = np.concatenate([delayed_target.real * weight, delayed_target.imag * weight])
    matrix = np.vstack([weighted_real, weighted_imag])

    if regularization > 0.0:
        matrix = np.vstack([matrix, np.sqrt(regularization) * np.eye(tap_count)])
        rhs = np.concatenate([rhs, np.zeros(tap_count)])
    if prior_taps is not None:
        prior = _as_1d_array(prior_taps, name="prior_taps", dtype=float)
        if prior.shape != (tap_count,):
            raise ValueError("prior_taps must have tap_count entries.")
        if prior_strength > 0.0:
            matrix = np.vstack([matrix, np.sqrt(prior_strength) * np.eye(tap_count)])
            rhs = np.concatenate([rhs, np.sqrt(prior_strength) * prior])
    if smoothness_strength > 0.0:
        second_difference = _second_difference_matrix(tap_count)
        matrix = np.vstack([matrix, np.sqrt(smoothness_strength) * second_difference])
        rhs = np.concatenate([rhs, np.zeros(second_difference.shape[0])])

    taps, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    dc_gain = float(np.sum(taps))
    if np.isfinite(dc_gain) and not np.isclose(dc_gain, 0.0):
        taps = taps / dc_gain
    return np.asarray(taps, dtype=float)


def interpolate_fir_time_domain(
    taps: ArrayLike1D,
    *,
    input_sample_rate_hz: float,
    output_sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a causal prototype impulse response onto a finer time grid."""

    prototype = _as_1d_array(taps, name="taps", dtype=float)
    if input_sample_rate_hz <= 0.0 or output_sample_rate_hz <= 0.0:
        raise ValueError("sample rates must be positive.")
    if output_sample_rate_hz < input_sample_rate_hz:
        raise ValueError("output_sample_rate_hz must not be lower than input_sample_rate_hz.")

    prototype_time = np.arange(prototype.size, dtype=float) / input_sample_rate_hz
    output_size = int(round((prototype.size - 1) * output_sample_rate_hz / input_sample_rate_hz)) + 1
    output_time = np.arange(output_size, dtype=float) / output_sample_rate_hz
    interpolated = np.interp(output_time, prototype_time, prototype)
    dc_gain = float(np.sum(interpolated))
    if np.isfinite(dc_gain) and not np.isclose(dc_gain, 0.0):
        interpolated = interpolated / dc_gain
    return output_time, np.asarray(interpolated, dtype=float)


def frequency_response(
    taps: ArrayLike1D,
    freq_hz: ArrayLike1D,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """Evaluate a real FIR frequency response at arbitrary non-negative frequencies."""

    coefficients = _as_1d_array(taps, name="taps", dtype=float)
    freq = _as_1d_array(freq_hz, name="freq_hz", dtype=float)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if np.any(freq < 0.0) or np.any(freq > sample_rate_hz / 2.0):
        raise ValueError("freq_hz must lie between DC and Nyquist.")
    n = np.arange(coefficients.size, dtype=float)
    return np.exp(-1j * 2.0 * np.pi * freq[:, None] * n[None, :] / sample_rate_hz) @ coefficients


def design_interpolated_ripple_correction_fir(
    freq_hz: ArrayLike1D,
    s21: ArrayLike1D,
    *,
    correction_band_hz: float,
    prototype_sample_rate_hz: float,
    output_sample_rate_hz: float = 2.0e9,
    prototype_tap_count: int = 121,
    smoothing_points: int = 31,
    max_correction_db: float = 3.0,
    edge_taper_fraction: float = 0.18,
    prototype_outside_band_weight: float = 0.08,
    output_outside_band_weight: float = 0.03,
    regularization: float = 1e-4,
    prior_strength: float = 1e-2,
    smoothness_strength: float = 1e-5,
) -> FIRCorrectionDesign:
    """Design a low-rate ripple FIR, interpolate it, and refine it at output rate.

    The prototype Nyquist limit is enforced, so the prototype sample rate must
    be at least twice the trusted correction-band edge.
    """

    freq, response = _validate_frequency_response(freq_hz, s21)
    if prototype_sample_rate_hz <= 0.0 or output_sample_rate_hz <= 0.0:
        raise ValueError("sample rates must be positive.")
    if output_sample_rate_hz < prototype_sample_rate_hz:
        raise ValueError("output_sample_rate_hz must be at least the prototype rate.")
    if correction_band_hz > prototype_sample_rate_hz / 2.0:
        raise ValueError(
            "correction_band_hz exceeds prototype Nyquist; use a higher prototype "
            "sample rate or reduce the correction band."
        )
    if correction_band_hz > freq[-1]:
        raise ValueError("correction_band_hz exceeds the measured frequency range.")
    if prototype_outside_band_weight < 0.0 or output_outside_band_weight < 0.0:
        raise ValueError("outside-band weights must be non-negative.")

    prototype_delay_s = 0.5 * (prototype_tap_count - 1) / prototype_sample_rate_hz
    prototype_freq = np.linspace(0.0, prototype_sample_rate_hz / 2.0, 2401)
    prototype_target = build_ripple_correction_target(
        freq,
        response,
        correction_band_hz=correction_band_hz,
        correction_freq_hz=prototype_freq,
        smoothing_points=smoothing_points,
        max_correction_db=max_correction_db,
        edge_taper_fraction=edge_taper_fraction,
        phase_delay_s=0.0,
    )
    prototype_weights = np.where(prototype_freq <= correction_band_hz, 1.0, prototype_outside_band_weight)
    prototype_taps = design_real_fir_from_response(
        prototype_freq,
        prototype_target.correction_response,
        sample_rate_hz=prototype_sample_rate_hz,
        tap_count=prototype_tap_count,
        delay_s=prototype_delay_s,
        weights=prototype_weights,
        regularization=regularization,
        smoothness_strength=smoothness_strength,
    )

    _, interpolated_taps = interpolate_fir_time_domain(
        prototype_taps,
        input_sample_rate_hz=prototype_sample_rate_hz,
        output_sample_rate_hz=output_sample_rate_hz,
    )
    output_delay_s = prototype_delay_s
    output_freq = np.linspace(0.0, output_sample_rate_hz / 2.0, 4001)
    output_target = build_ripple_correction_target(
        freq,
        response,
        correction_band_hz=correction_band_hz,
        correction_freq_hz=output_freq,
        smoothing_points=smoothing_points,
        max_correction_db=max_correction_db,
        edge_taper_fraction=edge_taper_fraction,
        phase_delay_s=0.0,
    )
    output_weights = np.where(output_freq <= correction_band_hz, 1.0, output_outside_band_weight)
    refined_taps = design_real_fir_from_response(
        output_freq,
        output_target.correction_response,
        sample_rate_hz=output_sample_rate_hz,
        tap_count=interpolated_taps.size,
        delay_s=output_delay_s,
        weights=output_weights,
        regularization=regularization,
        prior_taps=interpolated_taps,
        prior_strength=prior_strength,
        smoothness_strength=smoothness_strength,
    )

    notes = (
        "The correction target removes a smoothed dB baseline and a fitted linear phase trend.",
        "The correction is tapered to unity at the trusted-band edge.",
        "The output-rate FIR is refined around the time-interpolated prototype to prevent new interpolation ripple.",
    )
    return FIRCorrectionDesign(
        prototype_sample_rate_hz=float(prototype_sample_rate_hz),
        output_sample_rate_hz=float(output_sample_rate_hz),
        correction_band_hz=float(correction_band_hz),
        prototype_taps=prototype_taps,
        interpolated_taps=interpolated_taps,
        refined_taps=refined_taps,
        prototype_time_s=np.arange(prototype_taps.size, dtype=float) / prototype_sample_rate_hz,
        output_time_s=np.arange(refined_taps.size, dtype=float) / output_sample_rate_hz,
        target=output_target,
        notes=notes,
    )


def design_fir_sequence_from_s21(
    freq_hz: ArrayLike1D,
    s21: ArrayLike1D,
    *,
    correction_band_hz: float,
    output_sample_rate_hz: float,
    prototype_sample_rate_hz: float | None = None,
    prototype_tap_count: int = 121,
    smoothing_points: int = 31,
    max_correction_db: float = 3.0,
    edge_taper_fraction: float = 0.18,
    prototype_outside_band_weight: float = 0.08,
    output_outside_band_weight: float = 0.03,
    regularization: float = 1e-4,
    prior_strength: float = 1e-2,
    smoothness_strength: float = 1e-5,
) -> np.ndarray:
    """Return the final real FIR sequence designed from prepared complex S21.

    ``freq_hz`` and ``s21`` must already be on the uniformly spaced grid
    required by :func:`design_interpolated_ripple_correction_fir`.  The
    function designs the Nyquist-safe prototype, interpolates it in the time
    domain, and refines it at ``output_sample_rate_hz``.  By default the
    prototype rate is chosen as ``2 * correction_band_hz`` so that the full
    trusted band is represented without folding.

    Only the final FIR coefficients are returned.  Use
    :func:`design_interpolated_ripple_correction_fir` when prototype,
    interpolated, target, or diagnostic metadata are also needed.
    """

    correction_band = float(correction_band_hz)
    prototype_rate = (
        2.0 * correction_band
        if prototype_sample_rate_hz is None
        else float(prototype_sample_rate_hz)
    )
    design = design_interpolated_ripple_correction_fir(
        freq_hz,
        s21,
        correction_band_hz=correction_band,
        prototype_sample_rate_hz=prototype_rate,
        output_sample_rate_hz=float(output_sample_rate_hz),
        prototype_tap_count=prototype_tap_count,
        smoothing_points=smoothing_points,
        max_correction_db=max_correction_db,
        edge_taper_fraction=edge_taper_fraction,
        prototype_outside_band_weight=prototype_outside_band_weight,
        output_outside_band_weight=output_outside_band_weight,
        regularization=regularization,
        prior_strength=prior_strength,
        smoothness_strength=smoothness_strength,
    )
    return np.asarray(design.refined_taps, dtype=float).copy()
