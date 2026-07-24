from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from signal_process import (
    IfftConfig,
    analyze_phase_group_delay,
    analyze_rise_from_s21,
    plot_phase_group_delay,
    plot_rise_analysis,
    repo_roundtrip_s2p_config,
    write_phase_group_delay_report,
    write_rise_report,
    write_time_domain_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one S2P file with the reusable S21 response toolkit.")
    parser.add_argument("s2p_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "s21_response_toolkit_example")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--skip-initial-points", type=int, default=1)
    parser.add_argument("--roundtrip", action="store_true", help="Build the single-pass equivalent by dB/2 and phase/2.")
    parser.add_argument("--grid-stop-ghz", type=float, default=5.0)
    parser.add_argument("--grid-step-mhz", type=float, default=5.0)
    parser.add_argument("--align-offset-ns", type=float, default=-2.0)
    parser.add_argument("--linear-extend", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()

    preprocess = repo_roundtrip_s2p_config(
        grid_stop_hz=args.grid_stop_ghz * 1.0e9,
        grid_step_hz=args.grid_step_mhz * 1.0e6,
        skip_initial_points=args.skip_initial_points,
    )
    if not args.roundtrip:
        preprocess = replace(preprocess, roundtrip_to_single_pass=False)

    linear_extend = None
    if args.linear_extend == "true":
        linear_extend = True
    elif args.linear_extend == "false":
        linear_extend = False

    ifft = IfftConfig(align_offset_ns=args.align_offset_ns, linear_extend=linear_extend)
    response = analyze_rise_from_s21(args.s2p_path, preprocess_config=preprocess, ifft_config=ifft, label=args.label)
    phase = analyze_phase_group_delay(response.prepared)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = response.prepared.label.replace("/", "_").replace("\\", "_").replace(" ", "_")
    rise_png = plot_rise_analysis(output_dir / f"{stem}_rise_analysis.png", response, title=response.prepared.label)
    phase_png = plot_phase_group_delay(output_dir / f"{stem}_phase_group_delay.png", phase, title=response.prepared.label)
    write_time_domain_csv(output_dir / f"{stem}_time_domain.csv", response)
    write_rise_report(output_dir / f"{stem}_rise_report.md", response, figure_path=rise_png)
    write_phase_group_delay_report(output_dir / f"{stem}_phase_group_delay_report.md", phase, figure_path=phase_png)

    metrics = response.metrics
    print(f"label={response.prepared.label}")
    print(f"t10_ns={metrics.t10_ns}")
    print(f"t90_ns={metrics.t90_ns}")
    print(f"t95_ns={metrics.t95_ns}")
    print(f"rise_10_90_ns={metrics.rise_10_90_ns}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

