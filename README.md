# US Traffic Accident Severity Prediction

> Predicting how severe a US traffic accident will be — before emergency services arrive — using 7.7 million real accident records, gradient boosting, and a custom Streamlit app.

---

## Problem Statement

Every year, tens of thousands of people die in US traffic accidents. When a crash is reported, emergency dispatchers must decide immediately how many resources to send — but the severity is often unknown until responders arrive. A model that can estimate severity from conditions at the time of the accident (weather, road features, time of day, location) could help prioritize response and get help to the most critical cases faster.

This project trains machine learning models to predict accident severity on a scale of **1 (minor) to 4 (critical)**. The challenge is not just accuracy — it's detecting the rare, dangerous cases. Class 4 accidents make up only 2.5% of the dataset; a model that ignores them scores 80% accuracy while being useless in practice.

---

## Live Demo

[**Try the Streamlit app →**](https://YOUR-APP-NAME.streamlit.app)

*(Deploy instructions in [How to Run](#how-to-run))*

---

## Results

| Model | Accuracy | Macro F1 | F1 — Class 1 | F1 — Class 4 |
|-------|----------|----------|--------------|--------------|
| XGBoost (Optuna) | 72.3% | 0.5021 | 0.3146 | 0.3094 |
| XGBoost + Thresholds | 75.0% | 0.5062 | 0.2976 | 0.3019 |
| LightGBM (Optuna) | 72.1% | 0.5018 | 0.3254 | 0.3046 |
| **LightGBM + Thresholds** | **75.0%** | **0.5070** | **0.3115** | **0.2974** |
| CatBoost (Optuna) | 71.7% | 0.4908 | 0.3018 | 0.2889 |
| CatBoost + Thresholds | 74.7% | 0.4954 | 0.2822 | 0.2840 |
| MLP | 41.7% | 0.3043 | 0.1019 | 0.1108 |
| FT-Transformer | 52.9% | 0.3559 | 0.1073 | 0.1669 |

Macro F1 is the primary metric — it weights all four classes equally, which matters when two of them (Class 1 and Class 4) represent fewer than 4% of the data combined.

**Best model:** LightGBM tuned with Optuna + per-class threshold adjustment — **Macro F1: 0.5070**, Accuracy: 75.0%.

---

## Dataset

**Source:** [US Accidents — HuggingFace](https://huggingface.co/datasets/yuvidhepe/us-accidents-updated)

- **7,728,394** accident records collected across the contiguous US (2016–2023)
- **46 raw features:** timestamps, GPS coordinates, weather readings, road feature flags, free-text descriptions
- **Class distribution:** Class 1 — 0.9% · Class 2 — 79.7% · Class 3 — 16.8% · Class 4 — 2.5%

The heavy imbalance (Class 2 dominates) means naive models learn to predict "moderate" for everything. Addressing this was the central challenge.

---

## Approach

### Preprocessing

The raw data is noisy and high-dimensional. Key steps:

- **Geospatial feature engineering:** Each accident is mapped to an H3 hexagonal grid cell at resolution 7 (163,485 unique cells). Per-cell accident count and average severity are computed, then KMeans (k=10,000) clusters these cells into a single `Cluster` feature. This captures location-based risk without leaking raw coordinates into the model.
- **Duration:** `Start_Time` and `End_Time` are parsed to compute `Duration_Seconds`, clipped at 0.
- **Dropped features:** IDs, free-text fields, raw coordinates, redundant timestamp columns, and low-signal flags — 28 columns total.
- **Imputation:** Mean imputation for five weather readings (temperature, humidity, pressure, visibility, wind speed).
- **Encoding:** Wind direction and weather condition are label-encoded. Boolean POI flags are cast to int. `Civil_Twilight` is binarized (Day/Night).
- **Scaling:** StandardScaler fit on training data only, applied to val/test.

**Final feature set (20):** `Distance(mi)`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Direction`, `Wind_Speed(mph)`, `Weather_Condition`, `Amenity`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Civil_Twilight`, `Duration_Seconds`, `Cluster`

### Handling Class Imbalance

Two strategies were combined:

1. **SMOTE:** Training data resampled to 500k per class (2M total), balancing the heavily skewed distribution before boosting model training.
2. **Per-class threshold tuning:** After training, class probability thresholds are grid-searched on the validation set to maximize minority-weighted F1. The LightGBM threshold that worked best raises the Class 3 bar from 0.50 → 0.82, pushing borderline predictions toward Class 2 or Class 4 where the model is more confident. Final thresholds: `[0.50, 0.50, 0.82, 0.50]`.

### Hyperparameter Tuning

All three boosting models were tuned with **Optuna** (75 trials each, TPE sampler). The optimization objective was a minority-weighted F1: `0.35×F1(C1) + 0.10×F1(C2) + 0.20×F1(C3) + 0.35×F1(C4)` — deliberately penalizing misclassification of the rare, high-stakes classes.

### Deep Learning Comparison

An MLP and a Feature Tokenizer Transformer (FT-Transformer) were trained on the raw imbalanced data with class-weighted loss. Both fell well short of the boosting models (macro F1: 0.30 and 0.36 respectively vs. 0.51 for LightGBM). This is consistent with findings in the literature — gradient boosting tends to outperform transformers on structured tabular data of this type.

---

## What the Model Learned (SHAP)

SHAP TreeExplainer was run on 5,000 test samples from the best LightGBM model. Top features by global mean |SHAP|:

1. **Cluster** — the geospatial risk cluster dominates. Where an accident happens matters more than most weather variables.
2. **Distance(mi)** — the road distance affected by the accident is a strong proxy for severity.
3. **Pressure(in)** — atmospheric pressure correlates with severe weather events.
4. **Duration_Seconds** — longer disruptions tend to indicate more serious incidents.
5. **Temperature(F)** — extreme temperatures (ice, heat) associate with higher severity.

---

## Limitations

- **Class 1 and Class 4 F1 scores (~0.31) are low.** Even with SMOTE and threshold tuning, the rarest classes remain hard to predict reliably. The model is most confident about Class 2 (F1: 0.84).
- **The Cluster feature depends on historical accident data.** For locations with few or no past accidents, the cluster assignment is less informative.
- **No real-time data.** Weather inputs are user-provided at prediction time; the model does not connect to live APIs.
- **Geographic and temporal gaps.** The dataset ends in 2023 and may not reflect newly built infrastructure or changes in driving patterns.

---

## Project Structure

```
accident-severity-prediction/
├── config.py                    # all path constants
├── requirements.txt
├── data/
│   └── load_data.py             # HuggingFace dataset download
├── preprocessing/
│   └── pipeline.py              # fit/transform pipeline (serialized)
├── models/
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   └── deep_learning.py         # MLP + FT-Transformer (PyTorch)
├── evaluation/
│   └── metrics.py               # confusion matrix, SHAP helpers
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_boosting_models.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_evaluation.ipynb
├── app/
│   ├── predictor.py             # inference wrapper
│   └── streamlit_app.py         # Streamlit UI
└── outputs/
    ├── models/                  # saved model files (.pkl, .pt)
    ├── results/                 # all_model_results.csv
    └── plots/                   # evaluation charts, SHAP plots
```

---

## How to Run

### Prerequisites

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

### Reproducing the full pipeline

Run the notebooks in order:

```
notebooks/01_preprocessing.ipynb   — data loading, feature engineering, pipeline fit
notebooks/02_boosting_models.ipynb — XGBoost, LightGBM, CatBoost training + tuning
notebooks/03_deep_learning.ipynb   — MLP + FT-Transformer training
notebooks/04_evaluation.ipynb      — SHAP, confusion matrices, final verdict
```

Each notebook saves its outputs to `outputs/` — subsequent notebooks load from there rather than recomputing.

### Running the Streamlit app locally

```bash
streamlit run app/streamlit_app.py
```

### Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. New app → select this repo, branch `main`, main file `app/streamlit_app.py`
4. Deploy (~3 minutes)

---

## Tech Stack

Python · LightGBM · XGBoost · CatBoost · PyTorch · Optuna · SHAP · H3 · Scikit-learn · Imbalanced-learn · HuggingFace Datasets · Streamlit · Matplotlib

---

## Citation

Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.
[HuggingFace dataset](https://huggingface.co/datasets/yuvidhepe/us-accidents-updated)

---

## License

MIT
