# PowerNext AI — industrial data-quality and prediction pipeline

This pipeline treats the supplied task as three linked but separate problems:

1. predict a continuous/categorical **reference parameter**;
2. predict the **Valid/Invalid** label; and
3. diagnose unusual records as a physical/sensor fault, corrupted record, or a coherent new operating regime.

It applies explainable engineering checks before statistical models. A record that is rare but coherent across its sensors is retained and marked `genuine_new_regime`; a record that violates range, cross-sensor, frozen-sensor, or rate-of-change constraints is marked erroneous.

## Quick start

Install the small runtime dependency set, then run against the included hackathon data:

```bash
python3 -m pip install -r requirements.txt
python3 industrial_pipeline.py \
  --train training_data.csv --test test_data.csv --out pipeline_output \
  --submission-template Sample.csv
```

This produces `pipeline_output/submission.csv` with exactly the same columns and order as `Sample.csv`, ready to upload, alongside the full diagnostic output.

## General usage

```bash
python3 industrial_pipeline.py \
  --train data/train.csv --test data/test.csv --out results
```

The script detects target columns named `Reference Parameter`/`reference_parameter` and `Valid/Invalid`/`valid_invalid`, with a case-insensitive match. Override unusual names explicitly:

```bash
python3 industrial_pipeline.py --train train.csv --test test.csv --out results \
  --reference-column RefParam --validity-column QA_Status --time-column cycle
```

Output:

- `results/scored_test.csv` — predictions, anomaly scores, and one auditable primary reason per row.
- `results/summary.json` — counts by reason, model metadata, and aggregate quality statistics.
- `results/rule_config.json` — learned bounds/tolerances used for the run.
- `results/submission.csv` — optional upload-ready file, created with `--submission-template`.

`invalid_probability` estimates the chance that a record is Invalid. The pipeline selects an Invalid cutoff using stratified out-of-fold predictions on the training data, maximizing Invalid F1 and using balanced accuracy as a tie-breaker. This avoids the misleadingly high accuracy that can arise when Invalid records are rare. `validity_confidence` is the probability associated with the final predicted label. Neither value is an engineering safety score; use the rule flags and diagnosis for that purpose.

## Rule configuration

By default, numerical hardware envelopes use robust training-data limits (0.1–99.9 percentiles with a 10% margin), which makes the starter pipeline usable without equipment specifications. For production, pass a JSON file of approved physical limits and tolerances:

```json
{
  "ranges": {"Temperature": [-40, 180], "Current": [0, 500]},
  "sensor_groups": [["S1", "S2", "S3"]],
  "max_rate": {"Temperature": 5.0},
  "frozen_window": 5
}
```

Use `--rules rules.json`. Rate checks are only applied when `--time-column` is provided; input must be ordered within each asset (or use `--asset-column`).

The sensor-twin layer trains one surrogate per informative sensor on valid records only, using the operating inputs. A localized, high residual is logged as `sensor_error` and identifies the affected channel. It deliberately ignores sensors whose valid-record relationship is too noisy to be reliable. The validity classifier additionally learns sensor mean, spread, missing-count, and pairwise-difference features, which make cross-sensor inconsistency easier to detect. PCA reconstruction, Isolation Forest, and robust Mahalanobis provide complementary multivariate anomaly signals. A rare record is classified as `genuine_new_regime` only when it has no rule violation, no localized sensor residual, and a valid predicted condition.
