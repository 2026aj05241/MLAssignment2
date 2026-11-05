"""Read only trusted repository artifacts. This module never trains a model."""
import hashlib
import json
from importlib.metadata import version

import joblib

from src.config import CLASSES, FEATURES, ROOT, TARGET

ORDER = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'svm']

def read_metadata():
    info = json.loads((ROOT / 'model/metadata.json').read_text())
    if (info.get('schema_version') != 1 or info.get('features') != FEATURES
            or info.get('classes') != CLASSES or info.get('target') != TARGET
            or set(info.get('models', {})) != set(ORDER)):
        raise ValueError('Saved model metadata does not match the required five-model schema.')
    if info['versions']['scikit-learn'] != version('scikit-learn'):
        raise ValueError('Install the pinned scikit-learn version from requirements.txt before loading models.')
    return info

def load_model(slug, expected_hash):
    if slug not in ORDER:
        raise ValueError('Unknown model selection.')
    path = ROOT / f'model/{slug}.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'The saved {slug} artifact does not match its recorded checksum.')
    model = joblib.load(path)
    if list(model.classes_) != CLASSES or list(model.feature_names_in_) != FEATURES:
        raise ValueError('Saved pipeline features or class order are incompatible.')
    return model
