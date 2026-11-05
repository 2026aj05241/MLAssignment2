# Classification Model Comparison

## Project status

Snapshot 11: Complete app with five-model comparison, downloads, normalized confusion matrices, customized theme, and robustness fixes. All 25 tests pass.

## Problem statement

Compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes,
and Support Vector Machine on one public classification dataset. Evaluate all
five models on the same held-out test partition using Accuracy, AUC, Precision,
Recall, F1, and Matthews Correlation Coefficient (MCC).

The completed Streamlit app will load saved preprocessing/model pipelines,
evaluate an uploaded CSV containing input features and the target column, and
display per-model and comparative results without retraining.

## Dataset description

The [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry),
DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), contains 13,611
records, 16 numeric image-derived geometry features, and seven bean varieties.
It is attributed to Murat Koklu (2020), under CC BY 4.0. See `data/README.md`
for attribution, feature descriptions, and preparation details.

After removing 68 duplicate feature rows, 13,543 unique records remain.
The fixed seed 42, stratified 80/20 split produces 10,834 training records
and 2,709 test records. No missing values were found. The target is `Class`.
Original column spellings are preserved. No target or ID is used as a predictor.

Run `python -m src.prepare_data` to rebuild the committed split. Source row
indices, per-class counts, and SHA-256 hashes are in `data/split_summary.json`.
Preprocessing and any parameter selection use training data only.

## GitHub repository link

Pending manual repository creation and publication by the project owner.

## Models used

Model implementations and saved pipelines are added incrementally. Gaussian
Naive Bayes is used because the predictors are continuous measurements.
Precision, Recall, and F1 are unweighted macro averages over the seven fixed
classes (`zero_division=0`). AUC is macro one-vs-rest from class probabilities.
MCC uses the multiclass definition. The primary comparison metric is macro F1,
chosen before model comparison to give equal weight to minority classes.

The dashes below indicate results that have not yet been computed.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.921373 | 0.994860 | 0.932894 | 0.930681 | 0.931617 | 0.904968 |
| Decision Tree | 0.895533 | 0.982145 | 0.908113 | 0.908042 | 0.907863 | 0.873724 |
| KNN | 0.915467 | 0.989935 | 0.930199 | 0.926540 | 0.927922 | 0.897874 |
| Naive Bayes | 0.897010 | 0.991143 | 0.908075 | 0.908248 | 0.907220 | 0.876171 |
| Support Vector Machine | 0.923957 | 0.994430 | 0.935927 | 0.934556 | 0.935202 | 0.908019 |

## Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Pending training and evaluation. |
| Decision Tree | Pending training and evaluation. |
| KNN | Pending training and evaluation. |
| Naive Bayes | Pending training and evaluation. |
| Support Vector Machine | Pending training and evaluation. |
| Overall winner for this dataset | Pending evaluation; the selection criterion will be defined before comparing results. |

## Local setup

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app accepts the included held-out data or a labeled CSV upload, lets you
select one of five saved models, and displays six metrics, a confusion matrix,
and a classification report. A separate comparison tab evaluates all five
models on the same selected data, with downloadable comparison and prediction
CSVs, row-normalized confusion matrices, and class distributions. It never
retrains on uploaded data. Dependencies
are pinned; use Python 3.11 for training and deployment.

CSV schema: all 16 original feature names plus `Class`, with no extra columns.
Column order may differ. Targets must match the seven uppercase class names.
Blank feature cells use training medians; entirely blank feature columns,
non-numeric values, infinite values, duplicate headers, and malformed rows are
rejected. Limits: 10 MiB and 20,000 rows. AUC displays N/A when any of the seven
classes is absent; macro Precision/Recall/F1 always use all seven classes,
including zero contribution for absent classes. Uploaded data is processed in
memory and is not written to project files.

## Training and validation

Train any currently implemented model offline, for example:

```bash
python -m src.train --model logistic_regression
```

Reproduce the complete comparison with training-only parameter selection:

```bash
python -m src.compare
```

This evaluates small predefined grids using three stratified training folds,
then fits each selected configuration on all training rows. Fold-local pipelines
prevent preprocessing leakage. `results/cross_validation.json` records every
candidate mean/std macro F1, selected settings, and the CV-selected overall model.
SVM probability calibration is enabled only for the final training fit; CV uses
class predictions and does not need calibration. `python -m src.compare --baseline`
rebuilds the original fixed configurations instead and removes the CV record.

Artifacts include preprocessing fitted only on training data. Training checks
partition hashes and cross-partition duplicate rows, then verifies saved-model
predictions and probabilities after reloading. Model metadata records versions,
class order, parameters, and artifact hashes.

Install development dependencies to run the focused checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/          # Shared data preparation, evaluation, and training code
├── data/         # Dataset provenance and reproducible source data access
├── model/        # Five fitted pipelines and metadata, added in later stages
├── results/      # Generated comparison results, added in later stages
└── reports/      # Submission PDF, added in later stages
```

Empty directories contain `.gitkeep` files so they can be tracked by Git.
Saved model files and the eventual `test_data.csv` must remain tracked.

## Deployment and submission

Deployment is pending. The project owner will create commits, publish the
GitHub repository, and deploy to Streamlit Community Cloud manually after the
complete app passes verification. The final PDF will include the required
README content and verified project links.
