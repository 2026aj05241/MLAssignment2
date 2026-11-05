"""Consistent seven-class evaluation shared by offline training and the app."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from src.config import CLASSES
from src.validation import validate_frame

def evaluate(model, frame):
    features, target = validate_frame(frame)
    if list(model.classes_) != CLASSES:
        raise ValueError('Saved model class order does not match the project schema.')
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    present = set(target.unique())
    auc = None
    note = None
    if present == set(CLASSES):
        auc = float(roc_auc_score(target, probabilities, labels=CLASSES,
                                  multi_class='ovr', average='macro'))
    else:
        note = ('AUC is unavailable: the CSV must include all seven classes for '
                'the fixed seven-class macro one-vs-rest AUC.')
    common = dict(labels=CLASSES, average='macro', zero_division=0)
    metrics = {
        'Accuracy': float(accuracy_score(target, predicted)), 'AUC': auc,
        'Precision': float(precision_score(target, predicted, **common)),
        'Recall': float(recall_score(target, predicted, **common)),
        'F1': float(f1_score(target, predicted, **common)),
        # sklearn defines the zero-denominator case as 0; avoid its single-label warning.
        'MCC': (0.0 if len(set(target) | set(predicted)) == 1
                else float(matthews_corrcoef(target, predicted))),
    }
    return {
        'metrics': metrics, 'auc_note': note,
        'confusion_matrix': confusion_matrix(target, predicted, labels=CLASSES).tolist(),
        'classification_report': classification_report(target, predicted, labels=CLASSES,
                                  output_dict=True, zero_division=0),
        'predicted': predicted.tolist(), 'probabilities': np.asarray(probabilities).tolist(),
    }
