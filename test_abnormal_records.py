import unittest

from abnormal_records import identify_abnormal_records


class IdentifyAbnormalRecordsTests(unittest.TestCase):
    def test_flags_corrupted_and_invalid_condition_records(self):
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "bad-ts", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:02Z", "value": "oops"},
            {"timestamp": "2026-01-01T00:00:03Z", "value": 11.0, "test_condition_valid": False},
        ]

        result = identify_abnormal_records(records, min_history=1)
        self.assertEqual(result[1]["reason"], "corrupted_data")
        self.assertEqual(result[2]["reason"], "corrupted_data")
        self.assertEqual(result[3]["reason"], "invalid_test_condition")

    def test_flags_isolated_outlier_as_sensor_error(self):
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:01Z", "value": 10.1},
            {"timestamp": "2026-01-01T00:00:02Z", "value": 9.9},
            {"timestamp": "2026-01-01T00:00:03Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:04Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:05Z", "value": 80.0},
            {"timestamp": "2026-01-01T00:00:06Z", "value": 10.1},
            {"timestamp": "2026-01-01T00:00:07Z", "value": 10.0},
        ]

        result = identify_abnormal_records(records)
        self.assertEqual(result[5]["status"], "abnormal")
        self.assertEqual(result[5]["reason"], "sensor_error")

    def test_identifies_persistent_shift_as_genuine_change(self):
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:01Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:02Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:03Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:04Z", "value": 10.0},
            {"timestamp": "2026-01-01T00:00:05Z", "value": 20.0},
            {"timestamp": "2026-01-01T00:00:06Z", "value": 20.2},
            {"timestamp": "2026-01-01T00:00:07Z", "value": 19.8},
            {"timestamp": "2026-01-01T00:00:08Z", "value": 20.1},
        ]

        result = identify_abnormal_records(records)
        self.assertEqual(result[5]["status"], "normal")
        self.assertEqual(result[5]["reason"], "genuine_change")

    def test_detects_bounds_violations_as_sensor_error(self):
        records = [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0, "expected_min": 5, "expected_max": 15},
            {"timestamp": "2026-01-01T00:00:01Z", "value": 16.0, "expected_min": 5, "expected_max": 15},
        ]

        result = identify_abnormal_records(records, min_history=1)
        self.assertEqual(result[1]["reason"], "sensor_error")


if __name__ == "__main__":
    unittest.main()
