import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.config import FEATURES, ROOT
from src.evaluation import evaluate
from src.train import check_partitions, digest

def test_split_is_disjoint_and_reconstructs_unique_source():
    train, test = check_partitions()
    raw = pd.read_csv(ROOT / 'data/raw/dry_bean.csv', float_precision='round_trip')
    assert len(train) == 10834 and len(test) == 2709
    assert len(pd.concat([train, test]).drop_duplicates()) == len(raw.drop_duplicates()) == 13543

@pytest.mark.parametrize('artifact', sorted((ROOT / 'model').glob('*.joblib')), ids=lambda p:p.stem)
def test_saved_artifact_reproduces_metrics_and_training_statistics(artifact):
    train, test = check_partitions()
    info = json.loads((ROOT / 'model/metadata.json').read_text())
    assert digest(artifact) == info['models'][artifact.stem]['sha256']
    model = joblib.load(artifact)
    expected = json.loads((ROOT / f'results/{artifact.stem}.json').read_text())['metrics']
    actual = evaluate(model, test)['metrics']
    assert actual == pytest.approx(expected)
    if 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        assert scaler.n_samples_seen_ == len(train)
        np.testing.assert_allclose(scaler.mean_, train[FEATURES].mean(), rtol=1e-12)
