# Examples

These scripts are intentionally small and can be run either after installing the package or directly from a source checkout.

## Analyze one measured S2P file

```powershell
python examples/analyze_single_s2p.py `
  "D:\path\to\line.s2p" `
  --roundtrip `
  --skip-initial-points 1 `
  --output-dir "output\line_analysis"
```

The script generates:

- a rise-edge PNG,
- a phase/group-delay PNG,
- a time-domain CSV,
- Markdown reports for both analyses.

## Synthetic smoke test

```powershell
python examples/basic_s21_to_response.py
```

This creates an analytic one-pole S21 in memory and prints the 10-90% rise time. It is useful for checking that imports and the core response pipeline work before using measured data.