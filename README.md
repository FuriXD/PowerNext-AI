operating states.

### 4. Preserve duplicate evidence

Historical exact duplicates are all Invalid and can carry inconsistent
reference values.  Duplicate test records are therefore marked as corrupted
and assigned `Invalid`, while retaining their model reference estimate for a
complete audit trail.

## Decision hierarchy

```text
Missing or duplicate data
        ↓
Engineering-rule or sensor-twin violation
        ↓
Predicted invalid test condition
        ↓
Rare but coherent operating regime
        ↓
Normal record
```

This hierarchy records the most actionable root cause when several indicators
are present.

## Validation results

All metrics below are calculated with five-fold out-of-fold validation on the
supplied historical data.

| Measure | Result |
| --- | ---: |
| Reference prediction MAE | 0.9370 |
| Reference prediction RMSE | 2.3521 |
| Reference prediction R² | 0.9516 |
| Invalid-class F1 | 0.9249 |
| Invalid-class balanced accuracy | 0.9354 |

The program verifies the generated submission before writing it: schema,
column order, row count, test-ID order, missing predictions, numeric reference
values, and permitted validity labels are all checked.

## Current supplied-test run

The included `summary.json` records the latest run on the supplied 350-row test
set:

| Finding | Records |
| --- | ---: |
| Normal | 286 |
| Sensor error | 34 |
| Corrupted record | 23 |
| Invalid condition | 2 |
| Genuine new regime | 5 |
| Exact duplicate records | 8 |

The generated submission contains 303 `Valid` and 47 `Invalid` predictions.

## Output files

| File | Description |
| --- | --- |
| `output/submission.csv` | Upload-ready prediction file matching `Sample.csv`. |
| `output/scored_test.csv` | Full diagnostic output for every test record. |
| `output/summary.json` | Run totals, models, features, threshold, and diagnoses. |
| `output/metrics_baseline.json` | Cross-validated model-performance metrics. |
| `output/rule_config.json` | Learned engineering ranges and sensor tolerances. |

## Optional arguments

```bash
python3 industrial_pipeline.py --help
```

Useful options include:

- `--reference-column` and `--validity-column` for custom target names;
- `--time-column` and `--asset-column` to enable time-aware rate/frozen-sensor
  checks;
- `--rules rules.json` to provide approved laboratory limits and tolerances;
- `--id-column` when the submission identifier uses a nonstandard name.

## Reproducibility

The random state is fixed, all generated artifacts are written to the selected
output directory, and the complete methodology is documented in
[`methodology.md`](methodology.md).  Re-running the command above recreates the
submission and its associated audit trail.

## Project files

```text
industrial_pipeline.py   Main reproducible pipeline
requirements.txt         Python dependencies
methodology.md           Detailed method and validation rationale
training_data.csv        Historical labelled tests
test_data.csv            New tests to score
Sample.csv               Submission template
prediction.csv           Included submission prediction file
summary.json             Included run summary
output/                  Reproduced diagnostic artifacts
```
