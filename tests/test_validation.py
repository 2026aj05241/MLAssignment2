import numpy as np
import pandas as pd
import pytest
from src.config import CLASSES, FEATURES, TARGET
from src.validation import read_test_csv, validate_frame

def frame():
    data = pd.DataFrame(np.ones((2, len(FEATURES))), columns=FEATURES)
    data[TARGET] = CLASSES[:2]
    return data

def test_reordered_features_and_blank_cells():
    data = frame()
    data.loc[0, FEATURES[0]] = np.nan
    features, target = validate_frame(data[data.columns[::-1]])
    assert list(features.columns) == FEATURES
    assert target.tolist() == CLASSES[:2]

@pytest.mark.parametrize('kind', ['missing', 'extra', 'unknown', 'infinite', 'text', 'empty', 'allblank'])
def test_invalid_inputs(kind):
    data = frame()
    if kind == 'missing': data = data.drop(columns=FEATURES[0])
    elif kind == 'extra': data['ID'] = 1
    elif kind == 'unknown': data.loc[0, TARGET] = 'UNKNOWN'
    elif kind == 'infinite': data.loc[0, FEATURES[0]] = np.inf
    elif kind == 'text': data[FEATURES[0]] = ['bad', 'bad']
    elif kind == 'empty': data = data.iloc[:0]
    elif kind == 'allblank': data[FEATURES[0]] = np.nan
    with pytest.raises(ValueError): validate_frame(data)

def test_duplicate_headers_rejected_before_pandas_renames_them():
    with pytest.raises(ValueError, match='Duplicate'):
        read_test_csv(b'Area,Area,Class\n1,2,SEKER\n')

def test_malformed_row_rejected():
    payload = frame().to_csv(index=False).encode() + b'1,2\n'
    with pytest.raises(ValueError, match='same number'):
        read_test_csv(payload)
