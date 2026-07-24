# transmission-line-response

`transmission-line-response` analyzes analog transmission-line and measurement-chain transfer characteristics from S-parameter data.  The Python import name is `transmission_line_response`.

The current focus is S21 preprocessing, single-pass equivalent construction, impulse/step response recovery, rise-edge metrics, phase compensation, group-delay analysis, and report-ready figures/tables.

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

## Feature Areas

- `io.py`: read Touchstone `.s2p`, MATLAB `.mat`, and oscilloscope CSV data.
- `preprocess.py`: put S21 on a zero-start uniform grid, add a DC anchor, interpolate dB and unwrapped phase, and construct single-pass equivalents.
- `time_domain.py`: choose IFFT parameters, recover impulse/step responses, normalize steps, and measure rise-edge metrics.
- `phase.py`: fit linear phase, compensate bulk delay, select a trusted low-frequency group-delay band, and compute GD statistics.
- `s21.py`: lower-level S21 math, IFFT conversion, filtering, multiexponential models, and fitting helpers.
- `plots.py` and `reports.py`: generate report-ready PNG, CSV, and Markdown outputs.
- `generate_waveform.py` and `waveforms.py`: waveform generation and measured waveform rise-edge utilities.

## Documentation

- [Project structure](docs/project_structure.md)
- [S21-to-response workflow](docs/s21_workflow.md)
- [API reference map](docs/api_reference.md)
- [Examples](examples/README.md)

## Output Convention

Generated analysis artifacts should live outside the package source, usually under an `output/` or `analysis_output/` directory.  The repository `.gitignore` excludes these generated folders by default.

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