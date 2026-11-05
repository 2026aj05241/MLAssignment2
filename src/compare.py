"""Reproduce all models with training-only selection, then held-out evaluation."""
import argparse
import json
from importlib import import_module

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from src.config import ROOT, SEED
from src.train import available_models, check_partitions, train_one
from src.validation import validate_frame

# Small, explicit search spaces fixed before the comparative search.
GRIDS = {
    'logistic_regression': {'classifier__C': [0.1, 1.0, 10.0]},
    'decision_tree': {'classifier__max_depth': [8, 12, None],
                      'classifier__min_samples_leaf': [3]},
    'knn': {'classifier__n_neighbors': [5, 9, 15]},
    'naive_bayes': {'classifier__var_smoothing': [1e-9, 1e-7, 1e-5]},
    'svm': {'classifier__C': [1.0, 3.0]},
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true', help='Use fixed baseline configurations without CV.')
    args = parser.parse_args()
    train, _ = check_partitions()
    features, target = validate_frame(train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    selection_rows = []
    with threadpool_limits(limits=2):
        for slug in available_models():
            module = import_module(f'src.models.{slug}')
            pipeline = module.build()
            selection = 'Fixed baseline configuration; no test-set tuning.'
            if not args.baseline:
                # SVC probability calibration is unnecessary for macro-F1 selection.
                # The final SVC restores probability=True and fits calibration only on training data.
                if slug == 'svm':
                    pipeline.set_params(classifier__probability=False)
                search = GridSearchCV(pipeline, GRIDS[slug], scoring='f1_macro', cv=cv,
                                      n_jobs=1, error_score='raise', refit=False)
                search.fit(features, target)
                pipeline = module.build().set_params(**search.best_params_)
                selection = {'method': '3-fold stratified training-only CV', 'primary_metric': 'macro F1',
                             'best_params': search.best_params_, 'best_cv_f1': float(search.best_score_)}
                selection_rows.append({'slug': slug, 'model': module.NAME, **selection,
                    'candidates': [{'params': params, 'mean_f1': float(mean), 'std_f1': float(std)}
                                   for params, mean, std in zip(search.cv_results_['params'],
                                       search.cv_results_['mean_test_score'], search.cv_results_['std_test_score'])]})
                print(f'{module.NAME}: CV F1={search.best_score_:.6f}, {search.best_params_}', flush=True)
            train_one(slug, pipeline=pipeline, selection=selection)
    if selection_rows:
        winner = max(selection_rows, key=lambda row: row['best_cv_f1'])
        record = {'method': '3-fold stratified CV on training data only', 'primary_metric': 'macro F1',
                  'selected_model': winner['slug'], 'models': selection_rows,
                  'note': 'Test scores are descriptive final evaluation, not inputs to parameter or winner selection.'}
        (ROOT / 'results/cross_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    else:
        (ROOT / 'results/cross_validation.json').unlink(missing_ok=True)
    print(pd.read_csv(ROOT / 'results/model_comparison.csv').to_string(index=False))

if __name__ == '__main__':
    main()
