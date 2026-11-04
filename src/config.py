"""Dataset and reproducibility constants shared by training and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = 'Class'
FEATURES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3',
    'ShapeFactor4',
]
CLASSES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
SOURCE_URL = 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
