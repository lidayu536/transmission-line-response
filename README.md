# transmission-line-response

`transmission-line-response` analyzes analog transmission-line and measurement-chain transfer characteristics from S-parameter data. The Python import name is `transmission_line_response`.

The current focus is S21 preprocessing, single-pass equivalent construction, impulse/step response recovery, rise-edge metrics, phase compensation, group-delay analysis, and report-ready figures/tables.

The package also provides bounded ripple-correction FIR design. A low-rate prototype can be designed from a trusted S21 band, interpolated in time to a higher output rate, and refined against the measured band without inventing unknown high-frequency S21.

## Installation

For local development from this repository:

```powershell
python -m pip install -e .
```

For the existing local `_ComputingPackages` workflow, adding the parent directory to `PYTHONPATH` is still supported through a compatibility shim.

```python
import sys
sys.path.insert(0, r"D:\PostGraduate\SuperconductingQuantumComputing\_ComputingPackages")

from transmission_line_response import analyze_rise_from_s21
```

For a trusted low-frequency S21 band, use `design_fir_sequence_from_s21` when only the final FIR sequence is needed. It automatically chooses a prototype sample rate of at least twice the trusted-band edge, performs time-domain interpolation, and returns the FIR at the requested output sample rate. Use `design_interpolated_ripple_correction_fir` instead when prototype, interpolation, target, or diagnostic metadata are also needed.

## Quick Start

```python
from pathlib import Path

from transmission_line_response import (
    IfftConfig,
    analyze_rise_from_s21,
    plot_rise_analysis,
    repo_roundtrip_s2p_config,
    write_rise_report,
)

s2p_path = Path(r"D:\path\to\line.s2p")
response = analyze_rise_from_s21(
    s2p_path,
    preprocess_config=repo_roundtrip_s2p_config(skip_initial_points=1),
    ifft_config=IfftConfig(align_offset_ns=-2.0),
    label="line under test",
)

figure = plot_rise_analysis("output/rise.png", response)
write_rise_report("output/rise_report.md", response, figure_path=figure)
```

Command-line entry points:

```powershell
tlr-analyze-s2p "D:\path\to\line.s2p" --roundtrip --output-dir "output\line_analysis"
```

or, directly from a source checkout:

```powershell
python examples/analyze_single_s2p.py "D:\path\to\line.s2p" --roundtrip
```

## Package Layout

The implementation is grouped by responsibility under `src/`. `pyproject.toml` maps the Python package name `transmission_line_response` directly to `src/`, so there is no extra package-name directory under `src`.

- `core/`: shared array typing and data containers.
- `io/`: file readers split by source type, including Touchstone, MATLAB, and oscilloscope CSV.
- `sparams/`: S-parameter preprocessing, frequency-to-time transforms, filtering, bandwidth selection, and analytical S21 models.
- `analysis/`: high-level rise-edge, time-domain response, phase, and group-delay analysis.
- `output/`: plotting and Markdown/CSV report writers.
- `signals/`: waveform generation and measured waveform normalization utilities.
- Top-level files inside `src/`, such as `s21.py`, `phase.py`, and `time_domain.py`, are compatibility facades that re-export the new subpackage APIs.

Plotting functions are imported lazily at the package top level, so pure numerical workflows do not import `matplotlib` until a plot is actually requested.

## Documentation

- [Project structure](docs/project_structure.md)
- [S21-to-response workflow](docs/s21_workflow.md)
- [API reference map](docs/api_reference.md)
- [Examples](examples/README.md)

## Output Convention

Generated analysis artifacts should live outside the package source, usually under an `output/` or `analysis_output/` directory. The repository `.gitignore` excludes these generated folders by default.

A complete S21 analysis usually produces:

- `*_rise_analysis.png`
- `*_phase_group_delay.png`
- `*_time_domain.csv`
- `*_rise_report.md`
- `*_phase_group_delay_report.md`

## Development Checks

```powershell
python -m unittest discover -s tests
python examples/basic_s21_to_response.py
```

The tests use synthetic or temporary data only; measured datasets are intentionally kept out of the public package.
