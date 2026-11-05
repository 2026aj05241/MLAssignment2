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

def test_metrics_match_independent_confusion_and_pairwise_calculations():
    class ImperfectModel:
        classes_ = np.array(CLASSES)
        indices = np.array([0, 2, 2, 3, 6, 5, 1])
        def predict(self, features): return self.classes_[self.indices]
        def predict_proba(self, features):
            scores = np.full((7, 7), 0.05)
            scores[np.arange(7), self.indices] = 0.70
            return scores
    data = pd.DataFrame(np.ones((7, 16)), columns=FEATURES)
    data[TARGET] = CLASSES
    model = ImperfectModel()
    result = evaluate(model, data)
    matrix = np.zeros((7, 7))
    for actual, predicted in enumerate(model.indices): matrix[actual, predicted] += 1
    tp = matrix.diagonal()
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(7), where=matrix.sum(axis=0)!=0)
    recall = tp / matrix.sum(axis=1)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(7), where=(precision+recall)!=0)
    total = matrix.sum()
    rows, cols = matrix.sum(axis=1), matrix.sum(axis=0)
    mcc = (matrix.trace()*total - rows@cols) / np.sqrt((total**2-cols@cols)*(total**2-rows@rows))
    probabilities = model.predict_proba(None)
    aucs = []
    for label in range(7):
        positive = probabilities[label, label]
        negatives = np.delete(probabilities[:, label], label)
        aucs.append(np.mean((positive > negatives) + 0.5*(positive == negatives)))
    assert result['metrics'] == pytest.approx({'Accuracy': matrix.trace()/total,
        'Precision': precision.mean(), 'Recall': recall.mean(), 'F1': f1.mean(),
        'MCC': mcc, 'AUC': np.mean(aucs)})
