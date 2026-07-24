# API Reference Map

This is a curated map of the public API exported by `transmission_line_response`. The top-level package re-exports common APIs for convenience, while the implementation lives in functional subpackages.

## Core Data Containers

Implementation path: `transmission_line_response.core`

- `RawS21Data`: raw frequency/S21 samples after reading a measurement source.
- `PreparedS21`: S21 on a zero-start uniform grid with preprocessing notes and single-pass-equivalent metadata.

## Reading Data

Implementation path: `transmission_line_response.io`

- `read_touchstone_2port(path)` / `read_s2p(path)`: read Touchstone 2-port files.
- `read_mat_s21(path)`: read S21-like arrays from MATLAB `.mat` files.
- `read_s21(path)`: dispatch to a supported S21 reader.
- `read_scope_csv(path)`: read oscilloscope CSV traces.
- `subtract_scope_background(trace, background)`: subtract a measured background trace.
- `Touchstone2Port`, `ScopeTrace`: reader result containers.

## S21 Preprocessing and Transforms

Implementation path: `transmission_line_response.sparams`

- `PreprocessConfig`: frequency-grid and single-pass-equivalent settings.
- `repo_roundtrip_s2p_config(...)`: preset for the repository's common round-trip `.s2p` workflow.
- `check_fft_ready_grid(freq_hz)`: verify `df * arange(n)` compatibility.
- `add_dc_anchor(...)`: add a `0 Hz` dB/phase anchor.
- `prepare_s21(source, config)`: read and preprocess S21 onto the working grid.
- `s21_to_impulse_response(...)`: low-level S21 IFFT recovery.
- `impulse_to_step_response(...)`: integrate impulse response.
- `group_delay_from_s21(s21, freqs)`: compute group delay from the unwrapped phase derivative.
- `filter_waveform_by_s21(...)`: apply S21 to a waveform.
- `select_usable_s21_band(...)`: truncate/taper a noisy high-frequency tail.

## Analytical Models and Fitting

Implementation paths: `sparams.multiexp`, `sparams.standing_wave`, `sparams.fitting`

- `simple_multiexp_s21(...)`, `simple_multiexp_step_response(...)`: smooth multiexponential models.
- `oscillatory_multiexp_s21(...)`: oscillatory multiexponential model.
- `standing_wave_s21(...)`: single-reflector standing-wave correction model.
- `fit_simple_multiexp(...)`: fit a smooth multiexponential model to complex S21.
- `calibrate_s21_by_simple_multiexp(...)`: notebook-compatible fitting/calibration helper.
- `SimpleMultiexpFitResult`, `S21BandwidthWindow`: model result containers.

## Time-Domain and Phase Analysis

Implementation path: `transmission_line_response.analysis`

- `IfftConfig`: response-recovery and normalization settings.
- `choose_ifft_parameters(prepared, config)`: data-aware IFFT parameter selection.
- `recover_time_domain_response(prepared, config)`: recover impulse/step response from prepared S21.
- `analyze_rise_from_s21(source, ...)`: high-level one-call rise analysis.
- `PhaseAnalysisConfig`: phase-fit and trusted-band settings.
- `fit_linear_phase(freq_ghz, phase_rad)`: fit bulk delay from phase slope.
- `compensate_linear_phase(...)`: subtract fitted linear phase.
- `analyze_phase_group_delay(source, ...)`: high-level phase/GD analysis.
- `RiseMetrics`, `TimeDomainResponse`, `PhaseGroupDelayResult`: analysis result containers.

## Plotting and Reports

Implementation path: `transmission_line_response.output`

- `plot_rise_analysis(path, response)`: single trace rise-edge figure.
- `plot_rise_comparison(path, responses)`: compare aligned step responses.
- `plot_phase_group_delay(path, result)`: phase, residual phase, and group-delay figure.
- `write_time_domain_csv(path, response)`: export time-domain arrays.
- `write_rise_report(path, response)`: Markdown rise report.
- `write_phase_group_delay_report(path, result)`: Markdown phase/GD report.

Plotting imports are lazy at the top-level package boundary, so numerical code can import the package without importing `matplotlib`.

## Waveform Helpers

Implementation path: `transmission_line_response.signals`

- `normalize_step_trace(...)`, `measure_rise_time(...)`: measured waveform normalization and rise-time extraction.
- `ACZWave`, `FlattopWave`, `CosEnv`, `Differential`, `gen_alpha_drag`: waveform generation utilities kept for compatibility with existing lab scripts.

## Compatibility Facades

These old module paths remain intentionally supported:

- `transmission_line_response.s21`
- `transmission_line_response.preprocess`
- `transmission_line_response.time_domain`
- `transmission_line_response.phase`
- `transmission_line_response.plots`
- `transmission_line_response.reports`
- `transmission_line_response.types`
- `transmission_line_response.waveforms`
- `transmission_line_response.generate_waveform`
