# API Reference Map

This is a curated map of the public API exported by `transmission_line_response`.

## Reading Data

- `read_touchstone_2port(path)` / `read_s2p(path)`: read Touchstone 2-port files.
- `read_s21(path)`: format-dispatching S21 reader.
- `read_scope_csv(path)`: read oscilloscope CSV traces.
- `subtract_scope_background(trace, background)`: subtract a measured background trace.
- `RawS21Data`, `PreparedS21`, `Touchstone2Port`, `ScopeTrace`: core data containers.

## S21 Preprocessing

- `PreprocessConfig`: frequency-grid and single-pass-equivalent settings.
- `repo_roundtrip_s2p_config(...)`: preset for the repository's common round-trip `.s2p` workflow.
- `check_fft_ready_grid(freq_hz)`: verify `df * arange(n)` compatibility.
- `add_dc_anchor(...)`: add `0 Hz` dB/phase anchor.
- `prepare_s21(source, config)`: read and preprocess S21 onto the working grid.

## Time-Domain Recovery

- `IfftConfig`: response-recovery and normalization settings.
- `choose_ifft_parameters(prepared, config)`: data-aware IFFT parameter selection.
- `recover_time_domain_response(prepared, config)`: recover impulse/step response from prepared S21.
- `analyze_rise_from_s21(source, ...)`: high-level one-call rise analysis.
- `RiseMetrics`, `TimeDomainResponse`: result containers.

## Phase and Group Delay

- `PhaseAnalysisConfig`: phase-fit and trusted-band settings.
- `fit_linear_phase(freq_ghz, phase_rad)`: fit bulk delay from phase slope.
- `compensate_linear_phase(...)`: subtract fitted linear phase.
- `group_delay_from_s21(s21, freqs)`: compute group delay from unwrapped phase derivative.
- `analyze_phase_group_delay(source, ...)`: high-level phase/GD analysis.

## Plotting and Reports

- `plot_rise_analysis(path, response)`: single trace rise-edge figure.
- `plot_rise_comparison(path, responses)`: compare aligned step responses.
- `plot_phase_group_delay(path, result)`: phase, residual phase, and group-delay figure.
- `write_time_domain_csv(path, response)`: export time-domain arrays.
- `write_rise_report(path, response)`: Markdown rise report.
- `write_phase_group_delay_report(path, result)`: Markdown phase/GD report.

## Lower-Level Models and Filters

- `s21_to_impulse_response(...)`: low-level S21 IFFT recovery.
- `impulse_to_step_response(...)`: integrate impulse response.
- `filter_waveform_by_s21(...)`: apply S21 to a waveform.
- `simple_multiexp_s21(...)`, `fit_simple_multiexp(...)`: smooth multiexponential modeling.
- `standing_wave_s21(...)`: simple standing-wave correction model.

## Waveform Helpers

- `normalize_step_trace(...)`, `measure_rise_time(...)`: measured waveform normalization and rise-time extraction.
- `ACZWave`, `FlattopWave`, `CosEnv`, `Differential`, `gen_alpha_drag`: waveform generation utilities kept for compatibility with existing lab scripts.