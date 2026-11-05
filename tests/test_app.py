from pathlib import Path
from io import BytesIO
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest
from src.config import ROOT

APP = str(Path(__file__).resolve().parents[1] / 'app.py')

def test_included_data_displays_metrics_without_training():
    with patch.object(Pipeline, 'fit', side_effect=AssertionError('App must never train')):
        app = AppTest.from_file(APP, default_timeout=30).run()
        assert not app.exception
        assert not app.error
        assert len(app.metric) == 6
        assert app.metric[0].label == 'Accuracy'
        assert len(app.selectbox[0].options) == 5

def test_upload_mode_waits_for_file():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio[0].set_value('Upload a labeled test CSV').run()
    assert not app.exception
    assert any('upload your own' in item.value for item in app.info)

def test_all_models_match_recorded_metrics_and_display_comparison():
    expected = pd.read_csv(ROOT / 'results/model_comparison.csv').set_index('ML Model Name')
    app = AppTest.from_file(APP, default_timeout=30).run()
    for slug, label in [('logistic_regression', 'Logistic Regression'), ('decision_tree', 'Decision Tree'),
                        ('knn', 'KNN'), ('naive_bayes', 'Naive Bayes'), ('svm', 'Support Vector Machine')]:
        app.selectbox[0].set_value(slug).run()
        assert not app.exception and not app.error
        assert float(app.metric[0].value) == pytest.approx(expected.loc[label, 'Accuracy'], abs=0.00005)
        assert len(app.dataframe[0].value) == 5
    app.checkbox[0].set_value(True).run()
    assert not app.exception

def test_uploaded_data_uses_same_results_without_training():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio[0].set_value('Upload a labeled test CSV').run()
    with patch('streamlit.file_uploader', return_value=BytesIO((ROOT / 'test_data.csv').read_bytes())), \
         patch.object(Pipeline, 'fit', side_effect=AssertionError('App must never train')):
        app.run()
    assert not app.exception and not app.error
    assert len(app.metric) == 6
    assert len(app.dataframe[0].value) == 5

def test_uploaded_subset_shows_auc_warning_and_invalid_csv_shows_error():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio[0].set_value('Upload a labeled test CSV').run()
    data = pd.read_csv(ROOT / 'test_data.csv').iloc[:1]
    with patch('streamlit.file_uploader', return_value=BytesIO(data.to_csv(index=False).encode())):
        app.run()
    assert not app.exception and not app.error
    assert app.metric[1].value == 'N/A'
    assert app.warning
    with patch('streamlit.file_uploader', return_value=BytesIO(b'wrong,Class\n1,SEKER\n')):
        app.run()
    assert not app.exception and app.error
