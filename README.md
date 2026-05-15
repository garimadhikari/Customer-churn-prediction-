# Customer Churn Prediction Using Machine Learning

---

## Project Overview

This project implements and compares four supervised machine learning classifiers: Logistic Regression, Random Forest, Gradient Boosting, and ANN for customer churn prediction across two datasets of contrasting scale (10k and 505k records). Key contributions include a data-leakage investigation, ANN architecture search, and cross-dataset scale analysis.

---

## Repository Structure

```
├── Bank-churn.ipynb                          # Main Jupyter Notebook (all experiments)
├── CustomerChurnRecords.csv                  # Primary dataset (10,000 records)
├── customer_churn_datasettrainingmaster.csv  # Large dataset — training split (440,833 records)
├── customer_churn_datasettestingmaster.csv   # Large dataset — testing split (64,374 records)
├── outputs/                                  # Auto-created by notebook (saved plots & models)
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python version:** 3.10 recommended.
> Tested on: Windows 11, macOS 13, Ubuntu 22.04.

### 2. Launch Jupyter Notebook

```bash
jupyter notebook Bank-churn.ipynb
```

Or open in JupyterLab:

```bash
jupyter lab Bank-churn.ipynb
```

### 3. Run all cells

In Jupyter: **Kernel → Restart & Run All**

The notebook runs all experiments in order:
1. EDA on the primary dataset
2. Preprocessing (label encoding, feature engineering, SMOTE, scaling)
3. Model training and evaluation — WITH Complain (10k dataset)  
4. Data leakage investigation (Complain feature correlation revealed)
5. Model training and evaluation — WITHOUT Complain (clean 10k experiment)
6. ANN architecture search (on the clean 10k dataset)
7. Large dataset (505k) — merge, re-split, train, evaluate all 4 models
8. Final summary plots
Outputs (plots, saved models) are written to the `outputs/` folder automatically.

---

## Reproducibility

All random states are fixed at `random_state=42`. Results should be identical across runs on the same hardware. Minor numerical differences may appear across operating systems due to floating-point implementation differences in TensorFlow.

---

## Datasets

| File | Source | Records | Target |
|---|---|---|---|
| `CustomerChurnRecords.csv` | Kaggle — Bank Customer Churn | 10,000 | `Exited` |
| `customer_churn_datasettrainingmaster.csv` | Kaggle — Customer Churn Dataset | 440,833 | `Churn` |
| `customer_churn_datasettestingmaster.csv` | Kaggle — Customer Churn Dataset | 64,374 | `Churn` |

> **Note:** The large dataset training/testing files are merged and re-split inside the notebook using a stratified 80/20 split to correct a churn distribution mismatch in the original pre-split files.

---

## Key Results

| Dataset | Best F1 Model | F1 | Best ROC-AUC Model | ROC-AUC |
|---|---|---|---|---|
| 10k — no Complain | Gradient Boosting | 0.611 | Gradient Boosting | 0.857 |
| 505k — large dataset | Random Forest | 0.946 | ANN | 0.954 |

> **Key finding:** ANN showed the greatest improvement of any model when scaled from 10k → 505k records (+0.42 F1), eliminating overfitting entirely (gap: 0.20 → 0.00). On small data, Gradient Boosting is the most reliable choice.

> Initial scores on the 10k dataset were inflated (F1 > 0.99) due to the `Complain` feature being a near-perfect leaky predictor (r ≈ 1.00 with target). All final results exclude this feature.

---

## Dependencies

See `requirements.txt`. Core packages:

- `scikit-learn` 1.3.0
- `tensorflow` 2.13.0
- `imbalanced-learn` 0.11.0
- `pandas` 2.0.3, `numpy` 1.24.3
- `matplotlib` 3.7.2, `seaborn` 0.12.2
