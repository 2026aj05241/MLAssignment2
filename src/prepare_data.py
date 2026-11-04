"""Rebuild a deduplicated, stratified split from the attributed source CSV."""
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURES, ROOT, SEED, TARGET

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    raw_path = ROOT / 'data/raw/dry_bean.csv'
    raw = pd.read_csv(raw_path, float_precision='round_trip')
    assert list(raw.columns) == FEATURES + [TARGET]
    assert raw.shape == (13611, 17)
    assert not raw.isna().any().any()
    if raw.groupby(FEATURES, dropna=False)[TARGET].nunique().max() > 1:
        raise ValueError('Identical feature rows have conflicting targets.')
    clean = raw.drop_duplicates(subset=FEATURES).copy()
    train, test = train_test_split(clean, test_size=0.2, stratify=clean[TARGET], random_state=SEED)
    train.to_csv(ROOT / 'data/train_data.csv', index=False)
    test.to_csv(ROOT / 'test_data.csv', index=False)
    assert set(train.index).isdisjoint(test.index)
    summary = {
        'seed': SEED, 'test_fraction': 0.2, 'raw_rows': len(raw),
        'duplicates_removed': len(raw) - len(clean), 'unique_rows': len(clean),
        'train_rows': len(train), 'test_rows': len(test), 'feature_count': len(FEATURES),
        'missing_values': int(raw.isna().sum().sum()),
        'class_counts_raw': {str(k): int(v) for k, v in raw[TARGET].value_counts().items()},
        'class_counts_train': {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        'class_counts_test': {str(k): int(v) for k, v in test[TARGET].value_counts().items()},
        'train_source_indices': train.index.tolist(), 'test_source_indices': test.index.tolist(),
        'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                   [raw_path, ROOT / 'data/train_data.csv', ROOT / 'test_data.csv']},
    }
    (ROOT / 'data/split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(clean)} unique rows: {len(train)} training / {len(test)} test')

if __name__ == '__main__':
    main()
