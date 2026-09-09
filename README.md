# PowerNext AI

**Industrial test-data quality, validity classification, and calibrated reference-parameter prediction.**

PowerNext AI processes electrical-system laboratory test data where automated
measurements may be noisy, incomplete, duplicated, or unreliable.  Given
historical tests with calibrated reference values and Valid/Invalid labels, it
learns the normal operating behaviour and produces an upload-ready prediction
file for new tests.

The project is designed around a practical laboratory principle: **an unusual
test is not necessarily a bad test**.  It separates coherent changes in
operating behaviour from corrupted records and isolated sensor faults.

## What it produces

For each new test record, the pipeline produces:

- `Predicted_Reference_Parameter` — estimated calibrated reference value.
- `Validity_Label` — `Valid` or `Invalid` classification.
- A detailed audit record with anomaly scores, engineering-rule flags,
  duplicate evidence, and identified sensor-fault channels.

The submission file exactly follows the schema and ID ordering in `Sample.csv`.

## Quick start

### Requirements

- Python 3.10 or later
- `pandas`, `numpy`, and `scikit-learn`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the pipeline on the supplied data:

```bash
python3 industrial_pipeline.py \
  --train training_data.csv \
  --test test_data.csv \
  --out output \
  --submission-template Sample.csv
```

The final upload file is written to:

```text
output/submission.csv
```

## Input data

| File | Purpose |
| --- | --- |
| `training_data.csv` | Historical measurements with `Reference_Parameter` and `Validity_Label`. |
| `test_data.csv` | New test records to score. |
| `Sample.csv` | Required submission schema and test-ID order. |

The supplied dataset contains eight numerical measurement columns: applied
voltage, load current, ambient temperature, test duration, and four sensor
channels.  The program detects the target columns by name; command-line
overrides are available for different schemas.

## How it works

### 1. Learn normal equipment behaviour

An ExtraTrees regressor learns the relationship between operating conditions,
sensor readings, and the calibrated `Reference_Parameter` from historical
tests.

### 2. Identify unreliable data

The pipeline combines several independent checks:

- robust operating-range and cross-sensor consistency rules;
- missing-value, rate-of-change, and frozen-sensor checks;
- **sensor twins**, which estimate each reliable sensor from operating inputs
  and flag isolated departures;
- exact duplicate-record detection, independent of `Test_ID`;
- multivariate novelty checks using Isolation Forest, PCA reconstruction error,
  and robust Mahalanobis distance.

### 3. Predict validity without mistaking novelty for failure

An ExtraTrees classifier uses raw measurements and sensor-consistency features
to predict `Valid`/`Invalid`.  Its Invalid threshold is selected from
stratified out-of-fold predictions to focus on the minority Invalid class.

A rare record is retained as `genuine_new_regime` only when it is internally
coherent: it must pass engineering checks, show no localized sensor fault, and
be predicted Valid.  This prevents automatic rejection of meaningful new
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
