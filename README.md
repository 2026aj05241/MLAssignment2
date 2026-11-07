# Bean Atlas: Classification Model Comparison

## Problem statement

Classify dry bean varieties using image-derived geometric measurements.
Compare the following five classifiers on one shared dataset:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

Evaluate each classifier using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1
- Matthews Correlation Coefficient (MCC)

Expose saved-model evaluation through a Streamlit application.

## Dataset description

Dataset: [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry).
Attribution: Murat Koklu (2020), UCI Machine Learning Repository.
DOI: [10.24432/C50S4B](https://doi.org/10.24432/C50S4B).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The dataset is credited to its creators; project code and presentation were
written for this assignment.

The source has 13,611 instances, 16 numeric input features, and seven classes.
The target is `Class`. Neither target values nor identifiers are predictors.
The dataset exceeds the minimum 500 instances and 12 input features. It has
no missing values. The source XLSX is converted to the included CSV; original
feature names and labels are retained, including `AspectRation` and `roundness`.

| Input feature | Meaning |
| --- | --- |
| Area | Pixel area of the bean region |
| Perimeter | Boundary length |
| MajorAxisLength | Length of the major axis |
| MinorAxisLength | Length of the minor axis |
| AspectRation | Major-to-minor axis ratio; original source spelling |
| Eccentricity | Eccentricity of the fitted ellipse |
| ConvexArea | Area of the convex hull |
| EquivDiameter | Diameter of a circle with equivalent area |
| Extent | Region area relative to its bounding rectangle |
| Solidity | Region area relative to convex area |
| roundness | Roundness descriptor; original source capitalization |
| Compactness | Compactness descriptor |
| ShapeFactor1 | Source-defined shape descriptor 1 |
| ShapeFactor2 | Source-defined shape descriptor 2 |
| ShapeFactor3 | Source-defined shape descriptor 3 |
| ShapeFactor4 | Source-defined shape descriptor 4 |

Sixty-eight exact duplicate feature/target records were removed before splitting.
There are no conflicting labels among identical feature rows. The 13,543 unique
records are split 80/20 with stratification and seed 42: 10,834 training rows and
2,709 held-out test rows. The source has no batch/plant group IDs, so independence
at that level cannot be verified.

| Class | Source rows | Training rows | Test rows |
| --- | --- | --- | --- |
| BARBUNYA | 1322 | 1057 | 265 |
| BOMBAY | 522 | 418 | 104 |
| CALI | 1630 | 1304 | 326 |
| DERMASON | 3546 | 2837 | 709 |
| HOROZ | 1928 | 1488 | 372 |
| SEKER | 2027 | 1621 | 406 |
| SIRA | 2636 | 2109 | 527 |

Source rows include duplicates; training and test counts follow deduplication.
`data/source.json` records the source archive checksum. `data/split_summary.json`
records source indices, split counts, and hashes. `data/README.md` contains
retrieval and attribution details. The committed CSVs allow offline training.

## GitHub repository link

[https://github.com/2026aj05241/MLAssignment2](https://github.com/2026aj05241/MLAssignment2)


## Models used

All models use the same training and test rows. Every saved artifact is a fitted
scikit-learn pipeline containing a training-median imputer and classifier.
Logistic Regression, KNN, Gaussian Naive Bayes, and SVM also use StandardScaler;
Decision Tree uses the original feature scale. Each CV fold fits its own
preprocessing. Although the source has no missing values, the fitted imputer
supports occasional blank numeric cells at inference.

Gaussian Naive Bayes is appropriate for continuous predictors. Multinomial
Naive Bayes is not selected because these measurements are not count features.
SVM uses an RBF kernel and `probability=True` for the final fit; calibration is
performed internally using training data only. During macro-F1 grid search,
probability estimation is disabled because selection uses predicted labels.

### Evaluation conventions

Accuracy is the fraction of correct labels. Precision, Recall, and F1 are
unweighted macro averages across the fixed seven classes, with undefined
per-class divisions assigned zero. AUC is macro one-vs-rest ROC AUC calculated
from class probabilities in the recorded class order. MCC is the multiclass
coefficient, with zero returned when its denominator is zero.

For uploaded subsets missing any class, AUC is displayed as N/A because the fixed
seven-class OVR definition is unavailable. Macro Precision/Recall/F1 retain all
seven classes, including zero contribution for absent classes; this is explained
in the app. SVC label predictions follow its decision rule, which can differ
from the highest calibrated probability.

### Comparison table: held-out test results

All values below were calculated from the same 2,709 held-out rows using saved
pipelines. Values are proportions, not percentages, rounded to six decimals.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.921373 | 0.994860 | 0.932894 | 0.930681 | 0.931617 | 0.904968 |
| Decision Tree | 0.895533 | 0.982145 | 0.908113 | 0.908042 | 0.907863 | 0.873724 |
| KNN | 0.915467 | 0.989935 | 0.930199 | 0.926540 | 0.927922 | 0.897874 |
| Naive Bayes | 0.897010 | 0.991143 | 0.908075 | 0.908248 | 0.907220 | 0.876171 |
| Support Vector Machine | 0.923957 | 0.994430 | 0.935927 | 0.934556 | 0.935202 | 0.908019 |

### Hyperparameter tuning

Model settings were chosen using three-fold stratified cross-validation on the
training data, with seed 42 and macro F1 as the selection metric. Preprocessing
was fitted within each training fold. The selected models were then fitted on
the full training set and evaluated on the held-out test set.

| Model | Selected configuration | CV macro F1 | Fold std. |
| --- | --- | --- | --- |
| Logistic Regression | C=10, scaled | 0.936915 | 0.000540 |
| Decision Tree | max_depth=8, min_samples_leaf=3 | 0.917646 | 0.003150 |
| KNN | k=15, distance weights, scaled | 0.936527 | 0.001887 |
| Naive Bayes | Gaussian, var_smoothing=1e-9, scaled | 0.906827 | 0.006418 |
| Support Vector Machine | RBF, C=3, gamma=scale, scaled | 0.943299 | 0.002624 |

Search settings are defined in `src/compare.py`, and candidate scores are
recorded in `results/cross_validation.json`. SVM was selected using the highest
average validation macro F1. Test results were reserved for final evaluation.
Fold standard deviations describe variation across folds, not confidence intervals.

## Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline: test F1 0.931617 and AUC 0.994860, the highest AUC here. It is close to SVM while using a simpler decision boundary. SIRA is its weakest class, with F1 0.875585. |
| Decision Tree | Test F1 0.907863 and AUC 0.982145. Depth-8 regularization was selected by training CV. Its single-tree probability estimates and class separation are weaker here than the leading models. SIRA F1 is 0.830116. |
| KNN | Scaling and distance weighting yield test F1 0.927922. Training CV selected 15 neighbors. It is competitive but requires retaining training vectors and doing distance calculations at inference. SIRA F1 is 0.864564. |
| Naive Bayes | Test F1 0.907220 with AUC 0.991143. Its simple Gaussian conditional-independence model is a useful baseline, but overlapping and related geometry features plausibly limit it. This is an interpretation, not a causal experiment. BARBUNYA is weakest, F1 0.832347. |
| Support Vector Machine | Highest training CV macro F1, 0.943299. Test accuracy 0.923957, F1 0.935202, and MCC 0.908019 are highest in this comparison. SIRA remains difficult, F1 0.872280. Logistic Regression has slightly higher AUC. |
| Overall winner for this dataset | SVM, selected by training CV macro F1 and supported by the strongest test F1. The test F1 advantage over Logistic Regression is only 0.003585; one split does not establish a universal or statistically significant advantage. |


## Local setup

Use Python 3.11. Commands below run from the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. Runtime dependencies are
pinned in `requirements.txt`; development/report tools are separate in
`requirements-dev.txt`. Training metadata records the actual library versions.

## Training and reproducibility

The included artifacts are ready for inference. To reproduce the full pipeline:

```bash
python -m src.prepare_data
python -m src.compare
```

To reproduce fixed initial baselines instead of CV-selected settings:

```bash
python -m src.compare --baseline
```

To train one initial model configuration:

```bash
python -m src.train --model logistic_regression
```

Other model names are `decision_tree`, `knn`, `naive_bayes`, and `svm`.
A manual baseline retrain removes the old overall CV record to prevent stale
selection claims. Re-run the full comparison to restore final selected models,
and regenerate documentation before submitting changed results.

Training checks file hashes and rejects identical feature rows crossing the
split. Saved pipelines are reloaded and checked against pre-save predictions
and probabilities. `model/metadata.json` records feature order, class order,
versions, parameters, and checksums. Only trusted repository joblib files are
loaded. The app does not accept uploaded model files.

## Streamlit features and CSV contract

The app provides CSV upload, a five-model dropdown, all six metrics, a confusion
matrix with optional row normalization, a classification report, and a visible
all-model comparison for the selected test data. It includes downloadable sample
test data, comparison results, predictions, and class-distribution charts.

Upload a UTF-8 CSV containing exactly the 16 documented features plus `Class`.
Column order may vary. Class labels must match the seven uppercase names shown
in the class-count table. Extra/index columns, missing headers, duplicate
headers, unknown or blank targets, malformed rows, nonnumeric feature values,
infinite values, and entirely blank feature columns are rejected. Individual
blank numeric cells use saved training medians. Limits are 10 MiB / 20,000 rows.

Uploaded data is processed in memory; it is not saved to project files or used
for training. Bounded in-memory result caching expires after five minutes.
Downloaded prediction probabilities refer to the predicted label. App inference
loads saved pipelines and never calls training code.

## Validation and checks

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

The current suite contains 27 passing tests: schema and malformed-input checks,
independent confusion-matrix and pairwise-AUC calculations, partition isolation,
artifact checksums, reloaded metrics, training-only scaling statistics, selected
parameter consistency, all five app selections, CSV-upload evaluation,
missing-class handling, app execution with training explicitly forbidden, and publication-link/PDF
finalization in an isolated temporary project.
Local Streamlit server startup and its health/HTML endpoints also passed.
Streamlit AppTest verifies app behavior; it does not substitute for a hosted
Streamlit smoke test. Hosted deployment remains the owner's manual step.

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
├── .python-version
├── DEPLOYMENT.md
├── deployment_links.json
├── .streamlit/config.toml
├── test_data.csv
├── data/
│   ├── README.md
│   ├── raw/dry_bean.csv
│   ├── source.json
│   ├── train_data.csv
│   └── split_summary.json
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── train.py
│   ├── compare.py
│   └── models/
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json
├── results/
├── tests/
├── scripts/
│   ├── download_data.py
│   ├── build_report.py
│   └── finalize_links.py
└── reports/submission.pdf
```
