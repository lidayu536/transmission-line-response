# signal_process

`signal_process` 是一个面向超导量子计算实验数据的信号处理工具包。目前重点包括微波 S 参数到时域响应的恢复、上升沿指标分析、相位补偿与群时延分析，同时保留已有的常用波形生成函数。

## 功能

- 读取 Touchstone `.s2p`、MATLAB `.mat` 和示波器 CSV 数据。
- 将 S21 数据预处理到 FFT 友好的频率网格：可选跳过异常首点、补 `0 Hz`、对 `S21dB` 和 `unwrap` 后相位分别插值。
- 根据测量说明构造一来一回数据的单程等效：`dB / 2`，`phase / 2`。
- 根据频率间隔、频带边缘幅度和需要观察的时间窗选择 IFFT 插值/延拓参数。
- 从 S21 恢复冲激响应和阶跃响应，计算 `1%/10%/50%/90%/95%` 上升沿位置、过冲和平台期稳定性。
- 分析解包相位、线性相位补偿后的残余相位，以及可信低频区间内的群时延统计。
- 生成常用 PNG 图、CSV 表格和 Markdown 报告。
- 提供 `ACZWave`、`FlattopWave`、`CosEnv`、`Differential`、`gen_alpha_drag` 等波形工具。

## 快速使用

如果 `_ComputingPackages` 已经在 `PYTHONPATH` 中，可以直接导入：

```python
from pathlib import Path

from signal_process import (
    IfftConfig,
    analyze_rise_from_s21,
    plot_rise_analysis,
    repo_roundtrip_s2p_config,
    write_rise_report,
)

s2p_path = Path(r"D:\AITools\Codex\workspace_0\MicrowaveProcess\S21_to_response\20260630_Z4.1\06.10.s2p")

response = analyze_rise_from_s21(
    s2p_path,
    preprocess_config=repo_roundtrip_s2p_config(skip_initial_points=1),
    ifft_config=IfftConfig(align_offset_ns=-2.0),
    label="Z4.1 / 06.10",
)

fig = plot_rise_analysis("output/example/rise.png", response)
write_rise_report("output/example/rise_report.md", response, figure_path=fig)
```

命令行示例：

```powershell
python -m signal_process.examples.analyze_single_s2p `
  "D:\AITools\Codex\workspace_0\MicrowaveProcess\S21_to_response\20260630_Z4.1\06.10.s2p" `
  --roundtrip `
  --output-dir "output\signal_process_example"
```

## S21 标准处理口径

1. 读入数据后立刻提取 `freqs` 和复数 `S21`。
2. 对相位先 `unwrap`，再进行补 0 频和插值。
3. 若频率轴不满足 `freqs = df * arange(n)`，先补 `0 Hz` 锚点，默认 `phase0 = 0`。
4. 对 `S21dB` 和解包相位分别插值到统一工作网格，默认 `0 ~ 5 GHz`、`5 MHz`。
5. 若测量是一来一回，用 `dB / 2` 和 `phase / 2` 构造单程等效。
6. IFFT 参数需要根据数据特征选择，不把某一组旧参数当成固定模板。
7. 报告里记录是否补 0、0 频幅度估计方式、是否折半、频率网格、相位拟合窗口、群时延可信区间和延拓参数。

## 目录说明

- `generate_waveform.py`：已有波形生成和简单频谱工具。
- `s21.py`：S21 到冲激响应、群时延、多指数模型、滤波等底层函数。
- `io.py`：Touchstone、MATLAB 和示波器 CSV 读取。
- `preprocess.py`：S21 补点、插值、单程等效。
- `time_domain.py`：时域响应恢复和上升沿指标。
- `phase.py`：相位拟合、相位补偿和群时延统计。
- `plots.py`：常用图形输出。
- `reports.py`：Markdown/CSV 报告输出。
- `examples/`：可直接运行的示例脚本。
