# S21-to-Response Workflow

This workflow turns measured S21 into impulse response, step response, rise-edge metrics, phase residuals, and group-delay statistics.

## 1. Read and Identify the Data

Use `read_s21()` when the file format may vary. It currently handles:

- Touchstone `.s2p`,
- MATLAB `.mat` files containing frequency and S21-like arrays,
- already constructed `RawS21Data` objects.

The minimum normalized representation is:

- `freq_hz`: frequency samples in Hz,
- `s21`: complex S21 samples.

## 2. Check the Frequency Grid

IFFT-friendly data should satisfy:

```python
freqs = df * np.arange(n)
```

That means the grid starts at `0 Hz` and has uniform spacing. If not, preprocess the data before time-domain conversion.

## 3. Preprocess dB and Phase Separately

Use `prepare_s21()` with `PreprocessConfig`.

The recommended order is:

1. Extract raw complex S21.
2. Convert to `S21dB` and unwrap phase immediately.
3. Add a `0 Hz` anchor when needed, with `phase0 = 0`.
4. Interpolate `S21dB` and unwrapped phase separately to the target grid.
5. Reconstruct complex S21.

Do not interpolate real and imaginary parts directly for repo-style `.s2p` workflows.

## 4. Apply Single-Pass Equivalent When Needed

If metadata says the measurement is a round trip, construct the single-pass equivalent:

```python
s21_single = 10 ** ((s21_db / 2) / 20) * np.exp(1j * phase / 2)
```

The phase should already be unwrapped before halving.

## 5. Choose IFFT Parameters from the Data

Use `choose_ifft_parameters()` or pass an `IfftConfig` into `analyze_rise_from_s21()`.

Do not hard-code one old parameter set. The library considers:

- frequency spacing and time-window length,
- maximum usable frequency and native time resolution,
- band-edge magnitude,
- the required observation window after alignment.

## 6. Measure the Step Response

`analyze_rise_from_s21()` returns a `TimeDomainResponse` containing:

- impulse response,
- raw and normalized step response,
- delay estimate from low-frequency phase,
- aligned time axis,
- `1%`, `10%`, `50%`, `90%`, `95%` crossing times,
- overshoot and plateau stability metrics.

## 7. Analyze Phase and Group Delay

Use `analyze_phase_group_delay()` to get:

- unwrapped single-pass phase,
- low-frequency linear phase fit,
- residual phase after bulk delay compensation,
- group-delay curve,
- trusted low-frequency GD mean/std/peak-to-peak statistics.

Always state the trusted frequency interval when reporting `GD std` or `GD pp`.