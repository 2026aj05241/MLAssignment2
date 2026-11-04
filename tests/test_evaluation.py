import numpy as np
import pandas as pd
import pytest
from src.config import CLASSES, FEATURES, TARGET
from src.evaluation import evaluate

class PerfectModel:
    classes_ = np.array(CLASSES)
    def predict(self, features): return np.array(CLASSES[:len(features)])
    def predict_proba(self, features): return np.eye(7)[:len(features)]

def test_perfect_model_scores_and_auc_edge_case():
    data = pd.DataFrame(np.ones((7, 16)), columns=FEATURES)
    data[TARGET] = CLASSES
    result = evaluate(PerfectModel(), data)
    assert all(v == pytest.approx(1.0) for v in result['metrics'].values())
    result = evaluate(PerfectModel(), data.iloc[:1])
    assert result['metrics']['AUC'] is None
    assert result['metrics']['F1'] == pytest.approx(1/7)
    assert result['auc_note']
