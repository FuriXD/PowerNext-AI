from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any
import math


def identify_abnormal_records(
    records: list[dict[str, Any]],
    *,
    min_history: int = 5,
    outlier_zscore: float = 4.0,
    shift_zscore: float = 3.0,
    shift_window: int = 4,
) -> list[dict[str, Any]]:
    """Classify records as normal, genuine change, or likely invalid measurements.

    Returns one result per input record with:
    - status: "normal" or "abnormal"
    - reason: one of "sensor_error", "corrupted_data", "invalid_test_condition",
      "genuine_change", or None.
    """

    if shift_window < 2:
        raise ValueError("shift_window must be >= 2")

    results: list[dict[str, Any]] = []
    clean_history: list[float] = []
    last_timestamp: datetime | None = None
    shift_indices: set[int] = set()

    for index, record in enumerate(records):
        abnormal = _record_base_error(record, last_timestamp)
        if abnormal is not None:
            results.append({"index": index, "status": "abnormal", "reason": abnormal})
            continue

        timestamp = _parse_timestamp(record["timestamp"])
        last_timestamp = timestamp
        value = float(record["value"])

        if index in shift_indices:
            clean_history.append(value)
            results.append({"index": index, "status": "normal", "reason": "genuine_change"})
            continue

        min_bound = record.get("expected_min")
        max_bound = record.get("expected_max")
        if (min_bound is not None and value < float(min_bound)) or (
            max_bound is not None and value > float(max_bound)
        ):
            results.append({"index": index, "status": "abnormal", "reason": "sensor_error"})
            continue

        if len(clean_history) < min_history:
            clean_history.append(value)
            results.append({"index": index, "status": "normal", "reason": None})
            continue

        med = median(clean_history)
        spread = _robust_spread(clean_history)
        z_score = abs(value - med) / spread

        if z_score >= outlier_zscore:
            if _is_genuine_shift(
                records,
                start_index=index,
                baseline=med,
                spread=spread,
                shift_zscore=shift_zscore,
                shift_window=shift_window,
            ):
                shift_indices.update(range(index, index + shift_window))
                clean_history.append(value)
                results.append({"index": index, "status": "normal", "reason": "genuine_change"})
            else:
                results.append({"index": index, "status": "abnormal", "reason": "sensor_error"})
            continue

        clean_history.append(value)
        results.append({"index": index, "status": "normal", "reason": None})

    return results


def _record_base_error(record: Any, last_timestamp: datetime | None) -> str | None:
    if not isinstance(record, dict):
        return "corrupted_data"

    if record.get("test_condition_valid") is False:
        return "invalid_test_condition"

    if "timestamp" not in record or "value" not in record:
        return "corrupted_data"

    timestamp = _parse_timestamp(record["timestamp"])
    if timestamp is None:
        return "corrupted_data"

    if last_timestamp is not None and timestamp <= last_timestamp:
        return "corrupted_data"

    try:
        value = float(record["value"])
    except (TypeError, ValueError):
        return "corrupted_data"

    if not math.isfinite(value):
        return "sensor_error"

    return None


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None

    timestamp = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return None


def _robust_spread(values: list[float]) -> float:
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations)
    spread = 1.4826 * mad
    return spread if spread > 1e-9 else 1.0


def _is_genuine_shift(
    records: list[dict[str, Any]],
    *,
    start_index: int,
    baseline: float,
    spread: float,
    shift_zscore: float,
    shift_window: int,
) -> bool:
    end_index = min(len(records), start_index + shift_window)
    if end_index - start_index < shift_window:
        return False

    direction: int | None = None
    values: list[float] = []

    for record in records[start_index:end_index]:
        if _record_base_error(record, None) is not None:
            return False

        value = float(record["value"])
        diff = value - baseline
        z_score = abs(diff) / spread
        if z_score < shift_zscore:
            return False

        sign = 1 if diff > 0 else -1
        if direction is None:
            direction = sign
        elif sign != direction:
            return False

        values.append(value)

    mean_value = sum(values) / len(values)
    window_scatter = max(abs(v - mean_value) for v in values)
    return window_scatter <= spread * shift_zscore
