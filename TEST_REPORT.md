# PowerNext AI test report

**Status:** PASS  
**Tested:** 9 September 2026  
**Scope:** `industrial_pipeline.py` against the repository's supplied hackathon files.

## Test inputs

| Input | Purpose |
| --- | --- |
| `training_data.csv` | Model training and learned quality-rule bounds |
| `test_data.csv` | Records scored by the pipeline |
| `Sample.csv` | Required submission schema |

## Validation performed

The pipeline was compiled, then run with:

```bash
python3 industrial_pipeline.py \
  --train training_data.csv \
  --test test_data.csv \
  --out <temporary-output-directory> \
  --submission-template Sample.csv
```

The resulting artifacts were checked for:

- successful Python compilation;
- one scored row for every test record;
- a `submission.csv` whose columns and order exactly match `Sample.csv`;
- preservation of every `Test_ID` in the original order;
- non-empty reference-parameter and validity predictions;
- validity confidence values between 0 and 1;
- only supported diagnosis values; and
- diagnosis totals that reconcile to the total number of test records.

The validity-model upgrade was also measured with stratified five-fold, out-of-fold validation on `training_data.csv`. This keeps every evaluated prediction separate from the rows used to fit that fold.

## Results

| Check | Result |
| --- | --- |
| Test records scored | 350 / 350 |
| Submission schema | Pass — `Test_ID`, `Predicted_Reference_Parameter`, `Validity_Label` |
| IDs preserved | Pass |
| Missing submission predictions | 0 |
| Validity predictions | 311 `Valid`, 39 `Invalid` |
| Validity-confidence range | 0.4229–0.9957 |
| Diagnosis-total reconciliation | Pass — 350 / 350 |

### Validity-model validation

| Metric | Original classifier | Improved classifier |
| --- | ---: | ---: |
| Invalid F1 | 0.4598 | 0.9249 |
| Balanced accuracy | 0.6493 | 0.9354 |

The improved classifier uses sensor-consistency features (summary, spread, missing count, and pairwise differences) and an out-of-fold-calibrated Invalid threshold of **0.375**. The threshold and validation metrics are saved in `summary.json` for each run.

### Diagnosis distribution

| Diagnosis | Records |
| --- | ---: |
| Normal | 292 |
| Sensor error | 34 |
| Corrupted record | 17 |
| Invalid condition | 2 |
| Genuine new regime | 5 |

## Notes

scikit-learn emitted a local environment warning while attempting to detect physical CPU cores. It falls back to logical-core detection and does not affect the pipeline's outputs or the test result.

The test data does not contain ground-truth target labels, so this run verifies execution, output integrity, and submission compatibility—not predictive accuracy. Predictive performance should be measured separately with a held-out labeled validation set or cross-validation on `training_data.csv`.
