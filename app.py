"""Compare trusted saved models on labeled test data, without training."""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CLASSES, FEATURES, METRICS, ROOT, TARGET
from src.evaluation import evaluate
from src.inference import ORDER, load_model, read_metadata
from src.validation import read_test_csv

@st.cache_resource(show_spinner=False)
def saved_model(slug, checksum):
    return load_model(slug, checksum)

@st.cache_data(show_spinner=False, ttl=300, max_entries=4)
def evaluate_all(payload, signatures):
    frame = read_test_csv(payload)
    return {slug: evaluate(saved_model(slug, checksum), frame) for slug, checksum in signatures}

def show_metrics(metrics):
    for column, name in zip(st.columns(6), METRICS):
        value = metrics[name]
        column.metric(name, 'N/A' if value is None else f'{value:.4f}')

def main():
    st.set_page_config(page_title='Bean Atlas | Model Comparison', page_icon='🌱', layout='wide')
    st.title('Bean Atlas')
    st.write('A closer look at seven bean varieties, through five classification models.')
    st.caption('SHAPE TO VARIETY · 16 measurements · 5 saved models · 1 shared test set')
    try:
        info = read_metadata()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Model setup is incomplete: {exc}')
        st.stop()

    with st.sidebar:
        st.header('Evaluation desk')
        st.download_button('Download test_data.csv', (ROOT / 'test_data.csv').read_bytes(),
                           file_name='test_data.csv', mime='text/csv')
        source = st.radio('Test data source', ['Included held-out test data', 'Upload a labeled test CSV'])
        slug = st.selectbox('Model', ORDER, format_func=lambda key: info['models'][key]['name'])
        st.caption('Upload test data only: all 16 feature columns plus Class. Maximum: 10 MiB / 20,000 rows.')
        with st.expander('CSV format and labels'):
            st.write('Features: ' + ', '.join(FEATURES))
            st.write('Class labels: ' + ', '.join(CLASSES))
            st.write('Column order may vary. Blank feature cells use training medians. Extra columns are rejected.')

    if source == 'Upload a labeled test CSV':
        upload = st.file_uploader('Upload test CSV', type=['csv'])
        if upload is None:
            st.info('Download the sample test data or upload your own labeled CSV to begin.')
            st.stop()
        payload = upload.getvalue()
    else:
        payload = (ROOT / 'test_data.csv').read_bytes()
    try:
        frame = read_test_csv(payload)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    signatures = tuple((key, info['models'][key]['sha256']) for key in ORDER)
    try:
        with st.spinner('Comparing five saved models…'):
            results = evaluate_all(payload, signatures)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Unable to evaluate saved models: {exc}')
        st.stop()
    result = results[slug]
    st.subheader(info['models'][slug]['name'])
    st.caption(f'{len(frame):,} test rows · {frame[TARGET].nunique()} observed classes · results from the selected test data')
    show_metrics(result['metrics'])
    if result['auc_note']:
        st.warning(result['auc_note'])
    st.caption('Precision, Recall, and F1: macro average over all seven classes. AUC: macro one-vs-rest. MCC: multiclass; 0 when its denominator is zero.')
    if frame[FEATURES].isna().any().any():
        st.info('Blank feature cells were replaced using medians learned from training data.')

    overview_tab, detail_tab, data_tab = st.tabs(['All-model comparison', 'Selected model details', 'Test data'])
    with overview_tab:
        comparison = pd.DataFrame([{'Model': info['models'][key]['name'], **results[key]['metrics']} for key in ORDER])
        st.subheader('Five models, the same test rows')
        st.dataframe(comparison.set_index('Model').style.format('{:.4f}', na_rep='N/A'), use_container_width=True)
        st.bar_chart(comparison.set_index('Model')[['F1', 'Accuracy']], horizontal=True)
        st.download_button('Download comparison CSV', comparison.to_csv(index=False),
                           file_name='uploaded_test_comparison.csv', mime='text/csv')
        cv_path = ROOT / 'results/cross_validation.json'
        if cv_path.exists():
            import json
            selected = json.loads(cv_path.read_text())['selected_model']
            st.info('Training-validation selection: ' + info['models'][selected]['name'] +
                    '. The table above describes this test data and does not change the selected model.')
        st.caption('Uploaded samples may have a different distribution from the original test set; scores can change.')
    with detail_tab:
        st.subheader('Confusion matrix')
        st.caption('Rows: actual class. Columns: predicted class. Empty rows correspond to absent actual classes.')
        matrix = pd.DataFrame(result['confusion_matrix'], index=CLASSES, columns=CLASSES)
        normalized = st.checkbox('Normalize by actual-class count')
        display = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) if normalized else matrix
        st.dataframe(display.style.format('{:.2f}' if normalized else '{:.0f}').background_gradient(cmap='Greens'),
                     use_container_width=True)
        st.subheader('Classification report')
        st.dataframe(pd.DataFrame(result['classification_report']).T, use_container_width=True)
        predictions = frame.copy()
        predictions['Predicted Class'] = result['predicted']
        probabilities = np.array(result['probabilities'])
        label_indices = [CLASSES.index(label) for label in result['predicted']]
        predictions['Predicted class probability'] = probabilities[np.arange(len(frame)), label_indices]
        st.download_button('Download predictions CSV', predictions.to_csv(index=False),
                           file_name=f'{slug}_predictions.csv', mime='text/csv')
    with data_tab:
        st.subheader('Class distribution')
        counts = frame[TARGET].value_counts().reindex(CLASSES, fill_value=0).rename('Rows')
        st.bar_chart(counts)
        st.caption('Preview: first 100 test rows.')
        st.dataframe(frame.head(100), use_container_width=True)
    st.caption('Saved-model evaluation only. Uploaded test data is processed in memory and is never used for training.')

if __name__ == '__main__':
    main()
