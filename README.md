# DS-3002 Data Mining — Assignment #4
## Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks

**Spring 2026 | BSDS | FAST-NUCES**  
**Student ID:** i232564

---

## Project Overview

This assignment builds a complete ML pipeline for **CardioAI Labs**, a fictional health-tech startup building decision-support tools for cardiologists. The pipeline covers:

- **Unsupervised Learning** — K-Means & Hierarchical Clustering to discover hidden patient subgroups
- **Ensemble Methods** — Random Forest (Bagging) & XGBoost (Boosting) for heart disease prediction
- **Neural Networks** — Single-Layer Perceptron (SLP) & Multi-Layer Perceptron (MLP) on tabular data
- **CNN** — Lightweight convolutional network for handwritten digit recognition (MNIST)
- **Streamlit Dashboard** — Interactive front-end for real-time heart disease screening

---

## Datasets

### Dataset 1 — UCI Heart Disease (Cleveland)
Used for Parts: Pre, A, B, C, E

| Field | Detail |
|-------|--------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data |
| Size | 303 rows × 14 columns |
| Task | Binary classification: heart disease present (1) vs absent (0) |

### Dataset 2 — MNIST Handwritten Digits
Used for Part D (CNN only)

| Field | Detail |
|-------|--------|
| Load method | `from tensorflow.keras.datasets import mnist` (built into Keras) |
| Subset used | First 12,000 training + 2,000 test images |
| Task | 10-class digit classification (0–9) |

---

## Repo Structure

```
DS3002-Assignment4/
├── notebooks/
│   └── i232564_DS_A_Assignment4.ipynb   # Main notebook (all parts)
├── app/
│   ├── app.py                           # Streamlit CardioAI dashboard
│   ├── model.pkl                        # Trained Random Forest model
│   ├── scaler.pkl                       # Fitted StandardScaler
│   ├── feature_names.pkl                # Feature column names
│   ├── sample_patient.json              # Sample test patient data
│   └── requirements.txt                 # App dependencies
├── report/
│   └── DS3002_Assignment4_Report.pdf    # Full written report
└── README.md
```

---

## How to Run the Notebook

### Option 1 — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/i232564_DS_A_Assignment4.ipynb`
3. Click **Runtime → Run all**
4. The notebook downloads the dataset automatically from UCI URL — no manual download needed

### Option 2 — Local Jupyter
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap tensorflow joblib

# Launch Jupyter
jupyter notebook notebooks/i232564_DS_A_Assignment4.ipynb
```

> **Note:** All random seeds are fixed (`SEED = 42`) for full reproducibility.

---

## How to Run the Streamlit App

The app is a **CardioAI Heart Disease Screener** built with Streamlit, powered by the trained Random Forest model.

### Step 1 — Install dependencies
```bash
cd app
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
```
http://localhost:8501
```

The app will open with a patient input form pre-populated with a real test patient. Edit any field and click **🔍 Predict** to get a prediction with confidence score and feature importance chart.

---

## Requirements

### Notebook (`notebooks/`)
```
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
imbalanced-learn>=0.12
xgboost>=2.0
shap>=0.44
tensorflow>=2.16
scipy>=1.12
joblib>=1.3
```

### App (`app/requirements.txt`)
```
streamlit>=1.30
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
joblib>=1.3
shap>=0.44
```

---

## Results Summary

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Random Forest | ~85% | ~0.85 | ~0.92 |
| XGBoost | ~84% | ~0.84 | ~0.91 |
| MLP (Neural Net) | ~83% | ~0.83 | ~0.90 |
| CNN (MNIST) | ~98% | ~0.98 | — |

---

## GitHub Repository
[https://github.com/emanali-dev/DS3002-Assignment4](https://github.com/emanali-dev/DS3002-Assignment4)
