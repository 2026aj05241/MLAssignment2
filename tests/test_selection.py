import json
from src.config import ROOT

def test_selected_settings_match_artifacts_and_training_only_cv():
    record = json.loads((ROOT / 'results/cross_validation.json').read_text())
    metadata = json.loads((ROOT / 'model/metadata.json').read_text())
    assert len(record['models']) == 5
    winner = max(record['models'], key=lambda row: row['best_cv_f1'])
    assert record['selected_model'] == winner['slug']
    for row in record['models']:
        for key, value in row['best_params'].items():
            assert metadata['models'][row['slug']]['parameters'][key] == value
