# Performance and Resource Envelope

Milestone 6.1 introduces transparent limits so the free public demo fails clearly instead of exhausting a shared session.

## Default input limits

| Boundary | Default | Reason |
|---|---:|---|
| Upload size | 25 MB | Keeps parsing and session memory predictable on free hosting |
| Rows | 250,000 | Bounds whole-dataset profiling and deterministic cleaning rules |
| Columns | 500 | Prevents extremely wide files from overwhelming the interface |
| Expanded XLSX content | 250 MB | Reduces archive-expansion risk before workbook parsing |
| XLSX archive members | 10,000 | Rejects structurally abnormal workbooks |

The limits are implemented by `DatasetLimits` in `ods/loader.py`. Local developers can pass a reviewed custom instance to `load_dataset`; increasing a limit does not guarantee the machine has enough memory.

## Reproducible check

```bash
python -m scripts.benchmark --rows 100000 --max-seconds 30
```

The benchmark constructs deterministic mixed-type customer data, profiles it, generates cleaning recommendations, creates the four-card starter dashboard, and computes every card. CI runs the same check with 25,000 rows and a deliberately generous 20-second ceiling to catch severe regressions without relying on fragile micro-benchmarks.

Runtime varies by processor, available memory, Python version, and dependency build. Treat the result as a regression signal, not a universal speed guarantee.

### Release-candidate observation

On 2026-07-14, the clean Python 3.12 lockfile environment used for local release verification completed the 100,000-row benchmark in **1.776 seconds**. This is a traceable observation from one environment, not a promised runtime for other machines.
