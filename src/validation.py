"""Validate labeled CSV inputs before inference. No uploaded models are accepted."""
import csv
import io

import numpy as np
import pandas as pd

from src.config import CLASSES, FEATURES, TARGET

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 20000

def validate_frame(frame):
    if frame.empty:
        raise ValueError('The CSV must contain at least one data row.')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'Upload at most {MAX_ROWS:,} test rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    expected = set(FEATURES + [TARGET])
    missing, extra = expected - set(frame.columns), set(frame.columns) - expected
    if missing or extra:
        raise ValueError(f'Column mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}.')
    labels = frame[TARGET]
    if labels.isna().any() or not labels.isin(CLASSES).all():
        raise ValueError(f'Class must contain only these labels: {", ".join(CLASSES)}.')
    try:
        features = frame[FEATURES].apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('All feature values must be numeric or blank.') from exc
    if np.isinf(features.to_numpy()).any():
        raise ValueError('Infinite feature values are not allowed.')
    if features.isna().all().any():
        raise ValueError('A feature column cannot be entirely blank.')
    return features, labels.copy()

def read_test_csv(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('Provide a nonempty CSV no larger than 10 MiB.')
    try:
        text = payload.decode('utf-8-sig')
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows)
        if len(header) != len(set(header)):
            raise ValueError('Duplicate column names are not allowed.')
        count = 0
        for row in rows:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError('Every data row must have the same number of fields as the header.')
            count += 1
            if count > MAX_ROWS:
                raise ValueError(f'Upload at most {MAX_ROWS:,} test rows.')
        frame = pd.read_csv(io.StringIO(text), float_precision='round_trip')
    except (UnicodeError, csv.Error, StopIteration, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ValueError('Upload a valid UTF-8 CSV with a header row.') from exc
    validate_frame(frame)
    return frame
