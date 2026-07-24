from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .phase import PhaseGroupDelayResult
from .time_domain import TimeDomainResponse


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "未定义"
    return f"{value:.{digits}f}"


def write_time_domain_csv(path: str | Path, response: TimeDomainResponse) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ns", "aligned_time_ns", "impulse_real", "impulse_imag", "step_raw", "step_normalized"])
        writer.writerows(
            zip(
                response.time_ns,
                response.aligned_time_ns,
                np.real(response.impulse),
                np.imag(response.impulse),
                response.step_raw,
                response.step_normalized,
            )
        )
    return out_path


def write_rise_report(
    path: str | Path,
    response: TimeDomainResponse,
    *,
    figure_path: str | Path | None = None,
) -> Path:
    out_path = Path(path)
    prepared = response.prepared
    fig_name = Path(figure_path).name if figure_path is not None else None
    lines: list[str] = []
    lines.append(f"# {prepared.label} 上升沿分析报告")
    lines.append("")
    lines.append(f"- 生成时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    if prepared.source_path is not None:
        lines.append(f"- 源文件：`{prepared.source_path}`")
    lines.append(f"- 数据类型：`{prepared.source_format}`")
    lines.append(f"- 单程等效：`{prepared.single_pass_assumed}`")
    lines.append(f"- 工作频率网格：`{prepared.freq_hz[0] / 1e9:.3f} ~ {prepared.freq_hz[-1] / 1e9:.3f} GHz`，步长 `{prepared.grid_step_hz / 1e6:.3f} MHz`")
    lines.append(f"- IFFT 参数：`interp_multiple={response.ifft_choice.interp_multiple}`，`extend_multiple={response.ifft_choice.extend_multiple}`，`linear_extend={response.ifft_choice.linear_extend}`")
    lines.append("")
    lines.append("## 预处理记录")
    lines.append("")
    for note in prepared.notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 上升沿指标")
    lines.append("")
    lines.append(f"- 低频线性相位对应的总体时延估计：`{response.phase_delay_ns:.3f} ns`")
    lines.append(f"- 对齐参考时刻：`{response.align_reference_ns:.3f} ns`")
    lines.append(f"- `1%` 位置：`{_fmt(response.metrics.t01_ns)} ns`")
    lines.append(f"- `10%` 位置：`{_fmt(response.metrics.t10_ns)} ns`")
    lines.append(f"- `50%` 位置：`{_fmt(response.metrics.t50_ns)} ns`")
    lines.append(f"- `90%` 位置：`{_fmt(response.metrics.t90_ns)} ns`")
    lines.append(f"- `95%` 位置：`{_fmt(response.metrics.t95_ns)} ns`")
    lines.append(f"- `10%-90%` 上升沿长度：`{_fmt(response.metrics.rise_10_90_ns)} ns`")
    lines.append(f"- 平台期标准差：`{response.metrics.plateau_std:.6g}`")
    lines.append(f"- 平台期峰峰值：`{response.metrics.plateau_pp:.6g}`")
    lines.append(f"- 过冲量：`{response.metrics.overshoot:.6g}`")
    lines.append("")
    lines.append("## IFFT 参数选择说明")
    lines.append("")
    lines.append(f"- {response.ifft_choice.rationale}")
    if fig_name is not None:
        lines.append("")
        lines.append("## 图")
        lines.append("")
        lines.append(f"![{prepared.label} 上升沿分析图]({fig_name})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_phase_group_delay_report(
    path: str | Path,
    result: PhaseGroupDelayResult,
    *,
    figure_path: str | Path | None = None,
) -> Path:
    out_path = Path(path)
    prepared = result.prepared
    fig_name = Path(figure_path).name if figure_path is not None else None
    lines: list[str] = []
    lines.append(f"# {prepared.label} 相位与群时延分析报告")
    lines.append("")
    lines.append(f"- 生成时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    if prepared.source_path is not None:
        lines.append(f"- 源文件：`{prepared.source_path}`")
    lines.append(f"- 单程等效：`{prepared.single_pass_assumed}`")
    lines.append(f"- 线性相位拟合点数：`{result.config.fit_points}`")
    lines.append(f"- 低频线性相位对应的总体时延估计：`{result.delay_ns:.3f} ns`")
    lines.append(f"- 可信低频区间：`{result.trusted_start_ghz:.3f} ~ {result.trusted_stop_ghz:.3f} GHz`")
    lines.append("")
    lines.append("## 群时延统计")
    lines.append("")
    lines.append(f"- `GD mean`：`{result.trusted_gd_mean_ns:.6g} ns`")
    lines.append(f"- `GD std`：`{result.trusted_gd_std_ns:.6g} ns`")
    lines.append(f"- `GD pp`：`{result.trusted_gd_pp_ns:.6g} ns`")
    lines.append("")
    lines.append("这里的 `GD std` 和 `GD pp` 都是在该样本自己的可信低频区间内，由群时延曲线直接统计得到的量；它们不是拟合参数。")
    if fig_name is not None:
        lines.append("")
        lines.append("## 图")
        lines.append("")
        lines.append(f"![{prepared.label} 相位与群时延图]({fig_name})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
