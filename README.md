# Student Success Prediction — Machine Learning

Final project for **Machine Learning** (B.Sc. Mathematical Engineering & AI — iMAT, ICAI, Universidad Pontificia Comillas, 2025/26). Individual project.

Predicting dropout, enrollment and graduation in higher education with three tasks:

1. **Multiclass classification** — final student status (*Dropout / Enrolled / Graduate*).
2. **Regression** — 2nd-semester grade average, with explicit leakage control.
3. **Unsupervised learning** — student profiles via PCA, K-Means and hierarchical clustering.

## Results (held-out test set)

| Task | Best model | Metric | Test |
|------|------------|--------|------|
| Classification | Gradient Boosting | Macro F1 | 0.657 |
| Regression | Random Forest | RMSE | 2.728 |

## Methodology highlights

- **No leakage**: every 2nd-semester variable (contemporaneous with the target) and the classification target are excluded from the regression features.
- **Macro F1** as primary classification metric — classes are imbalanced, accuracy alone is misleading.
- All preprocessing (scaling, one-hot) inside scikit-learn `Pipeline`s, fitted on train only.
- Model selection by cross-validation on train; the test set is touched **once**, at the end.
- Baselines (majority class / mean) for every task; interpretability via feature importances and coefficients.
- Clustering stability checked with repeated sub-sampling.
- Fixed seed (`random_state=42`).

## Structure

```
├── data/student_success.csv     # Dataset used in the project
├── notebooks/
│   ├── 01_eda.ipynb             # EDA and variable sheet
│   ├── 02_clasificacion.ipynb   # Task 1: classification
│   ├── 03_regresion.ipynb       # Task 2: regression
│   └── 04_no_supervisado.ipynb  # Task 3: unsupervised
├── src/                         # Reusable modules (loading, preprocessing, pipelines, utils)
├── results/ficha_variables.csv  # Variable description sheet
└── report/memoria.pdf           # Full report (NeurIPS format, Spanish)
```

## Data

`data/student_success.csv` — Spanish version (renamed columns, translated categories) of the public UCI dataset **[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)** (Realinho et al., CC BY 4.0). 4,424 students, `;`-separated.

## Reproduce

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook   # run notebooks 01 → 04 in order
```

Python 3.12. Code, comments and report are in Spanish.

## Author

Ignacio Mascarell Agúndez
