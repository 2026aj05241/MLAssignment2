# Dataset provenance

Dry Bean dataset, donated by Murat Koklu, UCI Machine Learning Repository (2020).
Source: https://archive.ics.uci.edu/dataset/602/dry
DOI: https://doi.org/10.24432/C50S4B
License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/ . The data is attributed to its
creators; this project does not claim ownership of the dataset.

The source contains 13,611 rows, 16 numeric predictors, and seven bean-variety
labels. Predictors are image-derived geometry measurements: area, perimeter,
major/minor axis lengths, aspect ratio, eccentricity, convex area, equivalent
diameter, extent, solidity, roundness, compactness, and four shape factors.
Original names, including `AspectRation` and lowercase `roundness`, are retained.
`Class` is the target; it is never an input feature. There are no identifiers.

`raw/dry_bean.csv` is a CSV conversion of the original XLSX sheet. The conversion
changes the storage format but not the feature set or labels. `source.json`
records the original download hash and attribution. No missing values were
found. Sixty-eight identical feature/target rows are removed before a seeded
80/20 stratified split. This prevents identical records crossing partitions;
the source does not provide plant/batch IDs, so independence at that level
cannot be verified.

Rebuild the split from the project root:

```bash
python -m src.prepare_data
```

`split_summary.json` records class counts, source row indices, and file hashes.
`train_data.csv` is for training and internal cross-validation. The root-level
`test_data.csv` is for final evaluation and app upload. Do not tune models using
the test partition. Optional original-source retrieval requires openpyxl:

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python -m src.prepare_data
```
