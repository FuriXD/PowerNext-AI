#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 41
REFERENCE_ALIASES = ("reference parameter", "reference_parameter", "reference", "target")
VALIDITY_ALIASES = ("valid/invalid", "valid_invalid", "validity", "validity_label", "is_valid", "valid")


def normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def resolve_column(frame: pd.DataFrame, requested: str | None, aliases: tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in frame:
            raise ValueError(f"{kind} column '{requested}' was not found")
        return requested
    mapping = {normalise(c): c for c in frame.columns}
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    raise ValueError(f"Cannot infer {kind} column. Pass --{kind.replace(' ', '-')}-column.")


def load_rules(path: str | None, train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    user = json.loads(Path(path).read_text()) if path else {}
    ranges = dict(user.get("ranges", {}))
    for col in features:
        if col not in ranges:
            lo, hi = train[col].quantile([0.001, 0.999])
            span = max(float(hi - lo), 1e-9)
            ranges[col] = [float(lo - 0.1 * span), float(hi + 0.1 * span)]
    groups = user.get("sensor_groups")
    if not groups:
        sensors = [c for c in features if normalise(c).startswith(("s", "sensor"))]
        groups = [sensors] if len(sensors) >= 2 else []
    tolerances = user.get("cross_sensor_tolerance", {})
    for group in groups:
        name = "|".join(group)
        if name not in tolerances:
            spread = train[group].max(axis=1) - train[group].min(axis=1)
            tolerances[name] = float(spread.quantile(0.999) * 1.15)
    return {"ranges": ranges, "sensor_groups": groups, "cross_sensor_tolerance": tolerances,
            "max_rate": user.get("max_rate", {}), "frozen_window": int(user.get("frozen_window", 5))}


def engineering_rules(df: pd.DataFrame, rules: dict[str, Any], time_col: str | None, asset_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    range_bad = pd.Series(False, index=df.index)
    missing = df.isna().any(axis=1)
    for col, (lo, hi) in rules["ranges"].items():
        range_bad |= df[col].lt(lo) | df[col].gt(hi)
    cross_bad = pd.Series(False, index=df.index)
    for group in rules["sensor_groups"]:
        if all(c in df for c in group):
            tolerance = rules["cross_sensor_tolerance"]["|".join(group)]
            cross_bad |= (df[group].max(axis=1) - df[group].min(axis=1)).gt(tolerance)
    rate_bad = pd.Series(False, index=df.index)
    frozen = pd.Series(False, index=df.index)
    if time_col:
        ordered = df.sort_values(([asset_col] if asset_col else []) + [time_col])
        grouping = ordered.groupby(asset_col, sort=False) if asset_col else [(None, ordered)]
        for _, part in grouping:
            raw_time = part[time_col]
            numeric_time = pd.to_numeric(raw_time, errors="coerce")
            if numeric_time.isna().any():
                numeric_time = pd.to_datetime(raw_time, errors="raise").astype("int64") / 1e9
            dt = numeric_time.diff().replace(0, np.nan)
            for col, max_rate in rules["max_rate"].items():
                rate_bad.loc[part.index] |= part[col].diff().abs().div(dt).gt(max_rate).fillna(False)
            for col in rules["ranges"]:
                unchanged = part[col].diff().eq(0)
                frozen.loc[part.index] |= unchanged.rolling(rules["frozen_window"], min_periods=rules["frozen_window"]).sum().eq(rules["frozen_window"])
    out["rule_range"] = range_bad
    out["rule_cross_sensor"] = cross_bad
    out["rule_rate"] = rate_bad
    out["rule_frozen"] = frozen
    out["rule_missing"] = missing
    return out


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame, excluded: list[str]) -> tuple[list[str], Pipeline, np.ndarray, np.ndarray]:
    features = [c for c in train.columns if c not in excluded and pd.api.types.is_numeric_dtype(train[c]) and c in test]
    if len(features) < 2:
        raise ValueError("Need at least two shared numeric feature columns.")
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return features, prep, prep.fit_transform(train[features]), prep.transform(test[features])


def valid_mask(y: pd.Series) -> pd.Series:
    return y.astype(str).str.strip().str.lower().isin(("valid", "1", "true", "yes", "y"))


def sensor_twin_checks(train: pd.DataFrame, test: pd.DataFrame, is_valid: pd.Series, features: list[str]) -> pd.DataFrame:
    """Flag isolated departures from a normal, operating-condition sensor twin.

    Only channels whose valid-record out-of-fold R² is at least 0.50 are used:
    an unpredictable/noisy channel cannot provide reliable fault evidence.
    """
    sensor_cols = [c for c in features if normalise(c).startswith(("sensor", "s_"))]
    operating_cols = [c for c in features if c not in sensor_cols]
    result = pd.DataFrame(index=test.index)
    result["sensor_fault_channels"] = ""
    result["rule_sensor_residual"] = False
    if not sensor_cols or not operating_cols:
        return result
    channel_flags: dict[str, pd.Series] = {}
    for sensor in sensor_cols:
        fit_rows = is_valid & train[sensor].notna()
        if fit_rows.sum() < 25:
            continue
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
        folds = KFold(n_splits=min(5, int(fit_rows.sum())), shuffle=True, random_state=RANDOM_STATE)
        x_fit, y_fit = train.loc[fit_rows, operating_cols], train.loc[fit_rows, sensor]
        oof = cross_val_predict(model, x_fit, y_fit, cv=folds, n_jobs=-1)
        residual = y_fit.to_numpy() - oof
        baseline = np.sum((y_fit.to_numpy() - y_fit.mean()) ** 2)
        r2 = 1 - np.sum(residual ** 2) / baseline if baseline else 0.0
        if r2 < 0.50:
            continue
        scale = max(np.median(np.abs(residual - np.median(residual))) * 1.4826, 1e-6)
        threshold = float(np.quantile(np.abs(residual - np.median(residual)) / scale, .99))
        model.fit(x_fit, y_fit)
        z = np.abs(test[sensor] - model.predict(test[operating_cols])) / scale
        result[f"twin_residual_{sensor}"] = z
        channel_flags[sensor] = z.gt(threshold).fillna(False)
    if channel_flags:
        flags = pd.DataFrame(channel_flags)
        result["rule_sensor_residual"] = flags.any(axis=1)
        result["sensor_fault_channels"] = flags.apply(lambda row: ",".join(row.index[row].tolist()), axis=1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reference-column"); ap.add_argument("--validity-column"); ap.add_argument("--time-column"); ap.add_argument("--asset-column"); ap.add_argument("--rules")
    args = ap.parse_args()
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    ref = resolve_column(train, args.reference_column, REFERENCE_ALIASES, "reference")
    validity = resolve_column(train, args.validity_column, VALIDITY_ALIASES, "validity")
    excluded = [ref, validity] + [x for x in (args.time_column, args.asset_column) if x]
    features, prep, x_train, x_test = build_feature_matrix(train, test, excluded)
    rules = load_rules(args.rules, train, features)
    test_rules = engineering_rules(test[features + [x for x in (args.time_column, args.asset_column) if x]], rules, args.time_column, args.asset_column)
    normal_train = valid_mask(train[validity])
    twin_checks = sensor_twin_checks(train, test, normal_train, features)
    x_normal = x_train[normal_train.to_numpy()] if normal_train.sum() >= 10 else x_train
    iso = IsolationForest(contamination="auto", random_state=RANDOM_STATE).fit(x_normal)
    iso_score = -iso.score_samples(x_test)
    pca = PCA(n_components=min(max(1, x_normal.shape[1] - 1), x_normal.shape[0] - 1), random_state=RANDOM_STATE).fit(x_normal)
    reconstruction = np.mean((x_test - pca.inverse_transform(pca.transform(x_test))) ** 2, axis=1)
    train_reconstruction = np.mean((x_normal - pca.inverse_transform(pca.transform(x_normal))) ** 2, axis=1)
    # Robust, diagonal Mahalanobis avoids brittle covariance inversion in small datasets.
    median, mad = np.median(x_normal, axis=0), np.median(np.abs(x_normal - np.median(x_normal, axis=0)), axis=0) + 1e-6
    mahal = np.sum(((x_test - median) / (1.4826 * mad)) ** 2, axis=1)
    q_iso = np.quantile(-iso.score_samples(x_normal), .99); q_rec = np.quantile(train_reconstruction, .99); q_mah = np.quantile(np.sum(((x_normal-median)/(1.4826*mad))**2, axis=1), .99)
    statistical = (iso_score > q_iso) | (reconstruction > q_rec) | (mahal > q_mah)
    rule_bad = test_rules.any(axis=1).to_numpy()
    # A regime is rare but coherent: multivariate anomaly without a physical inconsistency.
    gmm = GaussianMixture(n_components=min(3, len(x_normal)), random_state=RANDOM_STATE).fit(x_normal)
    regime = gmm.predict(x_test)
    # Predict label separately from anomaly diagnosis.
    validity_model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    validity_model.fit(x_train, train[validity].astype(str))
    valid_prediction = validity_model.predict(x_test)
    # Severity ordering preserves the actionable root cause. A model-only invalid
    # record is an invalid operating condition, not evidence of a sensor failure.
    reasons = np.where(statistical, "genuine_new_regime", "normal").astype(object)
    reasons[~valid_mask(pd.Series(valid_prediction)).to_numpy()] = "invalid_condition"
    # A local residual in an otherwise physically plausible record is a measurement
    # fault. It takes precedence over the learned invalid label because it supplies
    # the root cause, not merely its downstream classification.
    reasons[twin_checks["rule_sensor_residual"].to_numpy()] = "sensor_error"
    reasons[rule_bad] = "sensor_error"
    reasons[test_rules["rule_missing"].to_numpy()] = "corrupted_record"
    target = train[ref]
    if pd.api.types.is_numeric_dtype(target):
        ref_model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1).fit(x_train, target)
    else:
        ref_model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1).fit(x_train, target.astype(str))
    out = test.copy()
    out["predicted_reference_parameter"] = ref_model.predict(x_test)
    out["predicted_validity"] = valid_prediction
    out["diagnosis"] = reasons
    out["operating_regime"] = regime
    out["isolation_score"] = iso_score; out["twin_reconstruction_error"] = reconstruction; out["robust_mahalanobis"] = mahal
    for c in test_rules: out[c] = test_rules[c].to_numpy()
    for c in twin_checks: out[c] = twin_checks[c].to_numpy()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "scored_test.csv", index=False)
    (out_dir / "rule_config.json").write_text(json.dumps(rules, indent=2))
    diagnosis_counts = {key: int((out["diagnosis"] == key).sum()) for key in ("normal", "sensor_error", "corrupted_record", "invalid_condition", "genuine_new_regime")}
    summary = {"rows": int(len(out)), "diagnosis_counts": diagnosis_counts,
               "predicted_validity_counts": {str(k): int(v) for k, v in out["predicted_validity"].value_counts().items()},
               "features": features, "reference_column": ref, "validity_column": validity,
               "models": {"reference": type(ref_model).__name__, "validity": type(validity_model).__name__, "anomaly": ["IsolationForest", "PCA surrogate reconstruction", "robust Mahalanobis", "GaussianMixture"]}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
