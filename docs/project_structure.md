# Project Structure

The repository is organized as a small installable analysis package plus examples, docs, and tests.

```text
transmission-line-response/
├─ src/transmission_line_response/   # package source code
├─ examples/                         # runnable user examples
├─ docs/                             # workflow and API documentation
├─ tests/                            # synthetic-data unit tests
├─ pyproject.toml                    # build metadata and console scripts
└─ README.md                         # public entry point
```

## Source Modules

- `io.py` owns file readers and raw data containers for Touchstone, MATLAB, and scope CSV files.
- `preprocess.py` owns S21 normalization onto a usable frequency grid.
- `time_domain.py` owns high-level impulse/step recovery and rise-edge metrics.
- `phase.py` owns phase compensation and group-delay analysis.
- `s21.py` keeps lower-level numerical primitives and model fitting utilities.
- `plots.py` and `reports.py` own generated artifacts.
- `waveforms.py` and `generate_waveform.py` contain time-domain waveform helpers.
- `cli.py` is the command-line interface for one-file S2P analysis.

## File Management Rules

- Keep measured raw datasets outside this package repository unless a tiny public sample is deliberately added.
- Put generated figures, CSV files, and Markdown reports under `output/` or `analysis_output/`.
- Keep reusable code in `src/transmission_line_response/`; keep one-off study scripts outside the package or under `examples/` only when they are general enough.
- Add tests with synthetic data when changing numerical behavior.