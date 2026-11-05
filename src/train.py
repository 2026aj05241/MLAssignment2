"""Offline training only: python -m src.train --model logistic_regression."""
import argparse
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd

from src.config import CLASSES, FEATURES, METRICS, ROOT, SEED, TARGET
from src.evaluation import evaluate
from src.validation import validate_frame

def available_models():
    return sorted(p.stem for p in (ROOT / 'src/models').glob('*.py') if p.stem != '__init__')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_partitions():
    summary = json.loads((ROOT / 'data/split_summary.json').read_text())
    for name, expected in summary['sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'{name} differs from the recorded split. Rebuild data first.')
    train = pd.read_csv(ROOT / 'data/train_data.csv', float_precision='round_trip')
    test = pd.read_csv(ROOT / 'test_data.csv', float_precision='round_trip')
    train_hash = set(pd.util.hash_pandas_object(train[FEATURES], index=False))
    test_hash = set(pd.util.hash_pandas_object(test[FEATURES], index=False))
    if train_hash & test_hash:
        raise ValueError('Identical feature rows occur in both partitions.')
    return train, test

def train_one(slug, pipeline=None, selection=None):
    module = import_module(f'src.models.{slug}')
    train, test = check_partitions()
    features, target = validate_frame(train)
    pipeline = module.build() if pipeline is None else pipeline
    start = time.perf_counter()
    pipeline.fit(features, target)
    fit_seconds = time.perf_counter() - start
    report = evaluate(pipeline, test)
    path = ROOT / f'model/{slug}.joblib'
    joblib.dump(pipeline, path, compress=3)
    loaded = joblib.load(path)
    if not np.array_equal(pipeline.predict(test[FEATURES]), loaded.predict(test[FEATURES])):
        raise AssertionError('Reloaded predictions do not match.')
    if not np.allclose(pipeline.predict_proba(test[FEATURES]), loaded.predict_proba(test[FEATURES])):
        raise AssertionError('Reloaded probabilities do not match.')
    summary = {k: v for k, v in report.items() if k not in {'predicted', 'probabilities'}}
    summary.update({'model': module.NAME, 'slug': slug, 'fit_seconds': fit_seconds,
                    'selection': selection or 'Fixed baseline configuration; no test-set tuning.'})
    (ROOT / f'results/{slug}.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if not isinstance(selection, dict):
        # A manual baseline retrain invalidates any previous five-model CV selection.
        (ROOT / 'results/cross_validation.json').unlink(missing_ok=True)
    metadata_path = ROOT / 'model/metadata.json'
    info = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    info.update({'schema_version': 1, 'features': FEATURES, 'target': TARGET, 'classes': CLASSES,
                 'seed': SEED, 'python': platform.python_version(),
                 'versions': {name: metadata.version(name) for name in
                              ['scikit-learn', 'numpy', 'pandas', 'scipy', 'joblib']},
                 'primary_metric': 'macro F1',
                 'metric_conventions': 'Precision/Recall/F1: macro over all 7 classes, zero_division=0; AUC: macro OVR from predict_proba; MCC: multiclass.',
                 'train_sha256': digest(ROOT / 'data/train_data.csv'),
                 'test_sha256': digest(ROOT / 'test_data.csv')})
    info.setdefault('models', {})[slug] = {
        'name': module.NAME, 'file': path.name, 'sha256': digest(path),
        'parameters': {k: v for k, v in pipeline.get_params().items()
                       if k.startswith('classifier__') and isinstance(v, (str, int, float, bool, type(None)))},
    }
    metadata_path.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    rows = []
    for name in available_models():
        report_path = ROOT / f'results/{name}.json'
        if report_path.exists():
            item = json.loads(report_path.read_text())
            rows.append({'ML Model Name': item['model'], **item['metrics']})
    pd.DataFrame(rows, columns=['ML Model Name'] + METRICS).to_csv(ROOT / 'results/model_comparison.csv', index=False)
    print(module.NAME, {k: round(v, 6) if v is not None else None for k, v in report['metrics'].items()}, flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=available_models(), required=True)
    args = parser.parse_args()
    train_one(args.model)

if __name__ == '__main__':
    main()
