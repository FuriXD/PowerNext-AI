# PowerNext-AI
Harness the power of AI to solve real-world challenges in India’s power sector.

## Task 01 - Identify Abnormal Records

This repository now includes `identify_abnormal_records` in
`/home/runner/work/PowerNext-AI/PowerNext-AI/abnormal_records.py`.

The classifier marks records as:
- `sensor_error` (impossible values, non-finite values, isolated spikes)
- `corrupted_data` (malformed records, invalid/missing fields, bad timestamp order)
- `invalid_test_condition` (explicitly invalid test context)

It also distinguishes a **genuine operating behavior shift** from an isolated
measurement error by checking whether large deviations persist across a
consecutive window.
