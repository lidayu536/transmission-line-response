# Project Structure

The repository is organized as an installable analysis package plus examples, docs, and tests.

```text
transmission-line-response/
|-- src/transmission_line_response/
|   |-- core/              # shared array helpers and data containers
|   |-- io/                # Touchstone, MATLAB, scope CSV, and dispatch readers
|   |-- sparams/           # S21 preprocessing, transforms, filtering, models, fitting
|   |-- analysis/          # rise-edge/time-domain and phase/group-delay workflows
|   |-- output/            # plotting and report writers
|   |-- signals/           # waveform generation and waveform metrics
|   |-- cli.py             # command-line S2P analysis
|   |-- s21.py             # compatibility facade
|   |-- phase.py           # compatibility facade
|   `-- time_domain.py     # compatibility facade
|-- examples/              # runnable user examples
|-- docs/                  # workflow and API documentation
|-- tests/                 # synthetic-data unit tests
|-- pyproject.toml         # build metadata and console scripts
`-- README.md              # public entry point
```

## Source Responsibilities

### `core/`

- `arrays.py`: shared 1D array validation and typing.
- `types.py`: `RawS21Data` and `PreparedS21` containers.

### `io/`

- `touchstone.py`: Touchstone 2-port reader and `Touchstone2Port`.
- `matlab.py`: MATLAB S21 reader.
- `scope.py`: oscilloscope CSV reader and background subtraction.
- `readers.py`: format-dispatching `read_s21()` entry point.

### `sparams/`

- `preprocess.py`: DC anchor, dB/phase interpolation, and single-pass equivalent construction.
- `transforms.py`: group delay, S21-to-impulse response, and impulse-to-step utilities.
- `filtering.py`: waveform filtering by S21.
- `band.py`: usable-band selection and tapering.
- `multiexp.py`: analytical multiexponential responses.
- `standing_wave.py`: standing-wave correction model.
- `fitting.py`: multiexponential fitting and calibration helpers.
- `models.py`: compatibility facade for old model imports.

### `analysis/`

- `rise.py`: IFFT parameter selection, time-domain response recovery, step normalization, and rise metrics.
- `phase.py`: linear phase compensation and trusted low-frequency group-delay statistics.

### `output/`

- `plots.py`: PNG plotting routines. Imported lazily from the top-level package.
- `reports.py`: Markdown and CSV writers.

### `signals/`

- `generation.py`: waveform generation helpers kept for existing lab workflows.
- `waveform_metrics.py`: measured waveform normalization and rise-time extraction.

## Compatibility Rules

The preferred implementation imports are subpackage-based, for example:

```python
from transmission_line_response.sparams import prepare_s21, s21_to_impulse_response
from transmission_line_response.analysis import analyze_rise_from_s21
```

Existing imports remain supported through thin facades:

```python
from transmission_line_response.s21 import s21_to_impulse_response
from transmission_line_response.time_domain import analyze_rise_from_s21
```

## File Management Rules

- Keep measured raw datasets outside this package repository unless a tiny public sample is deliberately added.
- Put generated figures, CSV files, and Markdown reports under `output/` or `analysis_output/`.
- Put reusable implementation code in the functional subpackages under `src/transmission_line_response/`.
- Keep one-off study scripts outside the package or under `examples/` only when they are general enough.
- Add synthetic-data tests when changing numerical behavior or public API paths.
