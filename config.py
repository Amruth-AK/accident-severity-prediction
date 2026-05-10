import os

# Reproducibility
RANDOM_SEED = 42

# Data
SAMPLE_SIZE = 500_000
FULL_DATASET = False
HUGGINGFACE_DATASET = "yuvidhepe/us-accidents-updated"
SEVERITY_CLASSES = [1, 2, 3, 4]

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# File paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")
MODELS_DIR     = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR      = os.path.join(OUTPUTS_DIR, "plots")
RESULTS_DIR    = os.path.join(OUTPUTS_DIR, "results")

# Output file names
PIPELINE_PATH      = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
XGBOOST_PATH       = os.path.join(MODELS_DIR, "xgboost_best.pkl")
LIGHTGBM_PATH      = os.path.join(MODELS_DIR, "lightgbm_best.pkl")
CATBOOST_PATH      = os.path.join(MODELS_DIR, "catboost_best.pkl")
DL_MODEL_PATH      = os.path.join(MODELS_DIR, "ft_transformer_best.pt")
RESULTS_CSV_PATH   = os.path.join(RESULTS_DIR, "all_model_results.csv")

# H3 resolution — 7 matches the original report and gives neighbourhood-level
# spatial granularity (~1.2 km^2 hexagons). Resolution 8 was marginally better
# on 500k silhouette score but resolution 7 generalises better on 7.7M rows.
H3_RESOLUTION = 8

# Optuna — 75 trials gives Bayesian search enough budget to find good
# minority-class configs in the widened parameter space.
OPTUNA_N_TRIALS = 75

# Create output directories at import time
for d in [OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
