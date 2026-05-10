"""
preprocessing/pipeline.py

Preprocessing pipeline matching the original Preprocessing.ipynb exactly.

Steps (verbatim from original):
  1. Parse timestamps -> Duration_Seconds
  2. Clean Wind_Direction abbreviations
  3. H3 resolution 7 -> lat_lng cell IDs
  4. KMeans(n_clusters=10000) on [accident_count, avg_severity] -> Cluster
  5. Drop original column list (including Precipitation(in))
  6. Mean-impute Temperature, Humidity, Pressure, Visibility, Wind_Speed
  7. Drop remaining NaN rows + duplicates
  8. Bool-encode POI columns, map Civil_Twilight (Day=1, Night=0),
     raw label-encode Wind_Direction + Weather_Condition
  9. StandardScaler fit on training data

No cyclic time encoding, no interaction features, no grouped weather,
no winsorization, no variance filter.

Entry point:
    pipeline = PreprocessingPipeline()
    X_train, y_train = pipeline.fit_transform(df_train)
    X_val,   y_val   = pipeline.transform(df_val)
    pipeline.save()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import h3
import joblib

from config import RANDOM_SEED, PIPELINE_PATH


# ---------------------------------------------------------------------------
# Column lists (verbatim from Preprocessing.ipynb)
# ---------------------------------------------------------------------------

_COLS_TO_DROP = [
    'ID', 'Start_Lat', 'Street', 'Zipcode', 'End_Lng', 'Description',
    'City', 'County', 'State', 'Country', 'Timezone', 'Airport_Code',
    'Weather_Timestamp', 'Wind_Chill(F)', 'Precipitation(in)', 'Bump',
    'Roundabout', 'Station', 'Turning_Loop', 'Sunrise_Sunset',
    'Nautical_Twilight', 'Astronomical_Twilight', 'Source',
    'Start_Time', 'End_Time', 'lat_lng', 'Start_Lng', 'End_Lat',
]

_MEAN_IMPUTE_COLS = [
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)',
]

_BOOL_COLS = [
    'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
]


# ---------------------------------------------------------------------------
# Step 1+2: Timestamps + Wind Direction
# ---------------------------------------------------------------------------

def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for suffix in ['.000000000', '.000000']:
        df['Start_Time'] = df['Start_Time'].str.replace(suffix, '', regex=False)
        df['End_Time']   = df['End_Time'].str.replace(suffix, '', regex=False)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['End_Time']   = pd.to_datetime(df['End_Time'],   format='mixed')
    df['Duration_Seconds'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()
    return df


def _clean_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df['Wind_Direction'] == 'Calm',     'Wind_Direction'] = 'CALM'
    df.loc[df['Wind_Direction'] == 'Variable', 'Wind_Direction'] = 'VAR'
    df.loc[df['Wind_Direction'] == 'North',    'Wind_Direction'] = 'N'
    df.loc[df['Wind_Direction'] == 'East',     'Wind_Direction'] = 'E'
    df.loc[df['Wind_Direction'] == 'South',    'Wind_Direction'] = 'S'
    df.loc[df['Wind_Direction'] == 'West',     'Wind_Direction'] = 'W'
    return df


# ---------------------------------------------------------------------------
# Steps 3+4: H3 + KMeans
# ---------------------------------------------------------------------------

def _h3_kmeans(
    df: pd.DataFrame,
    resolution: int = 7,
    n_clusters: int = 10_000,
    is_train: bool = True,
    kmeans_model=None,
) -> tuple:
    """
    Converts coordinates to H3 cells, clusters on per-cell
    [accident_count, avg_severity] aggregates (verbatim from original).
    Returns (df_with_cluster, kmeans_model).
    """
    df = df.copy()
    df = df.dropna(subset=['Start_Lat', 'Start_Lng'])

    df['lat_lng'] = df.apply(
        lambda r: h3.geo_to_h3(r['Start_Lat'], r['Start_Lng'], resolution),
        axis=1,
    )

    if is_train:
        h3_agg = df.groupby('lat_lng').agg(
            accident_count=('Severity', 'size'),
            avg_severity=('Severity', 'mean'),
        ).reset_index()

        # Clamp n_clusters to number of unique cells (can be < 10k on sample)
        actual_k = min(n_clusters, len(h3_agg))
        print(f"      KMeans: {actual_k} clusters on {len(h3_agg)} unique H3 cells")
        km = KMeans(n_clusters=actual_k, random_state=RANDOM_SEED, n_init=10)
        h3_agg['Cluster'] = km.fit_predict(h3_agg[['accident_count', 'avg_severity']])

        # Store lookup for transform()
        km._cell_cluster = h3_agg.set_index('lat_lng')['Cluster'].to_dict()
        km._fallback_cluster = 0
        df = df.merge(h3_agg[['lat_lng', 'Cluster']], on='lat_lng', how='left')
    else:
        cell_map = kmeans_model._cell_cluster
        fallback  = kmeans_model._fallback_cluster
        df['Cluster'] = df['lat_lng'].map(cell_map).fillna(fallback).astype(int)
        km = kmeans_model

    return df, km


# ---------------------------------------------------------------------------
# Step 9: silhouette evaluation helper (display-only, not used in pipeline)
# ---------------------------------------------------------------------------

def evaluate_h3_resolutions(
    df: pd.DataFrame, resolutions: list = [5, 6, 7, 8]
) -> dict:
    """
    Compares H3 resolutions using silhouette score on a 10k subsample.
    Score is computed in the cluster-feature space [accident_count, avg_severity]
    — the same space KMeans was fitted on.
    """
    sample = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Severity']).sample(
        min(10_000, len(df)), random_state=RANDOM_SEED
    )
    results = {}
    for res in resolutions:
        df_tmp, _ = _h3_kmeans(sample, resolution=res, n_clusters=50, is_train=True)
        labels = df_tmp['Cluster'].values
        cluster_features = df_tmp.groupby('lat_lng').agg(
            accident_count=('Severity', 'size'),
            avg_severity=('Severity', 'mean'),
        ).reindex(df_tmp['lat_lng'])[['accident_count', 'avg_severity']].values
        cluster_features = cluster_features[:len(labels)]
        score = silhouette_score(
            cluster_features, labels,
            sample_size=min(5000, len(labels)),
            random_state=RANDOM_SEED,
        )
        results[res] = round(score, 4)
        print(f"  Resolution {res}: silhouette = {score:.4f}")
    return results


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Encapsulates all stateful preprocessing objects.
    Mirrors the original Preprocessing.ipynb steps exactly.
    """

    def __init__(self):
        self.scaler          = None
        self.kmeans_model    = None
        self.weather_encoder = LabelEncoder()
        self.wind_encoder    = LabelEncoder()
        self._impute_means   = {}
        self.feature_cols    = None
        self.mi_scores       = None

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """Fits all transformations on training data. Returns (X, y)."""
        print("=== PreprocessingPipeline.fit_transform ===")
        df = df.copy()

        print("[1/8] Parsing timestamps -> Duration_Seconds...")
        df = _parse_timestamps(df)

        print("[2/8] Cleaning Wind_Direction...")
        df = _clean_wind_direction(df)

        print("[3/8] H3 resolution 7 + KMeans(10000 clusters)...")
        df, self.kmeans_model = _h3_kmeans(
            df, resolution=7, n_clusters=10_000, is_train=True
        )

        print("[4/8] Dropping columns...")
        drop_present = [c for c in _COLS_TO_DROP if c in df.columns]
        df = df.drop(columns=drop_present)

        print("[5/8] Mean-imputing numerical features...")
        present = [c for c in _MEAN_IMPUTE_COLS if c in df.columns]
        self._impute_means = df[present].mean().to_dict()
        df[present] = df[present].fillna(pd.Series(self._impute_means))

        print("[6/8] Dropping NaN rows + duplicates...")
        before = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        print(f"      Dropped {before - len(df)} rows")

        print("[7/8] Encoding features...")
        df = self._encode(df, is_train=True)

        # Separate target before scaling
        y = df['Severity'].values
        df = df.drop(columns=['Severity'])

        print("[8/8] StandardScaler...")
        self.feature_cols = df.columns.tolist()
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df)

        print(f"fit_transform complete. Features: {len(self.feature_cols)}")
        return X, y

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> tuple:
        """Applies fitted transformations to val/test data. Returns (X, y)."""
        df = df.copy()

        df = _parse_timestamps(df)
        df = _clean_wind_direction(df)
        df, _ = _h3_kmeans(
            df, resolution=7, n_clusters=10_000,
            is_train=False, kmeans_model=self.kmeans_model,
        )

        drop_present = [c for c in _COLS_TO_DROP if c in df.columns]
        df = df.drop(columns=drop_present)

        present = [c for c in _MEAN_IMPUTE_COLS if c in df.columns]
        df[present] = df[present].fillna(
            pd.Series({k: v for k, v in self._impute_means.items() if k in present})
        )

        df = df.dropna()
        df = df.drop_duplicates()

        df = self._encode(df, is_train=False)

        y = df['Severity'].values if 'Severity' in df.columns else None
        if 'Severity' in df.columns:
            df = df.drop(columns=['Severity'])

        # Align to training feature set
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols]

        X = self.scaler.transform(df)
        return X, y

    # ------------------------------------------------------------------
    def _encode(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Encoding steps from original (verbatim order)."""
        df = df.copy()

        # Bool POI columns
        bool_present = [c for c in _BOOL_COLS if c in df.columns]
        df[bool_present] = df[bool_present].astype(int)

        # Civil_Twilight: Day=1, Night=0
        if 'Civil_Twilight' in df.columns:
            df['Civil_Twilight'] = df['Civil_Twilight'].map({'Day': 1, 'Night': 0})

        # Label-encode Wind_Direction
        if 'Wind_Direction' in df.columns:
            df['Wind_Direction'] = df['Wind_Direction'].fillna('CALM')
            if is_train:
                df['Wind_Direction'] = self.wind_encoder.fit_transform(
                    df['Wind_Direction'].astype(str)
                )
            else:
                known = set(self.wind_encoder.classes_)
                df['Wind_Direction'] = df['Wind_Direction'].apply(
                    lambda x: x if str(x) in known else 'CALM'
                )
                df['Wind_Direction'] = self.wind_encoder.transform(
                    df['Wind_Direction'].astype(str)
                )

        # Label-encode Weather_Condition (raw, no grouping)
        if 'Weather_Condition' in df.columns:
            df['Weather_Condition'] = df['Weather_Condition'].fillna('Clear')
            if is_train:
                df['Weather_Condition'] = self.weather_encoder.fit_transform(
                    df['Weather_Condition'].astype(str)
                )
            else:
                known = set(self.weather_encoder.classes_)
                df['Weather_Condition'] = df['Weather_Condition'].apply(
                    lambda x: x if str(x) in known else 'Clear'
                )
                df['Weather_Condition'] = self.weather_encoder.transform(
                    df['Weather_Condition'].astype(str)
                )

        return df

    # ------------------------------------------------------------------
    def compute_mutual_information(self, X: np.ndarray, y: np.ndarray) -> pd.Series:
        """Compute MI scores for display in notebook."""
        mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
        self.mi_scores = pd.Series(mi, index=self.feature_cols).sort_values(ascending=False)
        return self.mi_scores

    # ------------------------------------------------------------------
    def save(self, path: str = PIPELINE_PATH):
        joblib.dump(self, path)
        print(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str = PIPELINE_PATH):
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    is_train: bool = True,
    pipeline: PreprocessingPipeline = None,
):
    if is_train:
        pipeline = PreprocessingPipeline()
        X, y = pipeline.fit_transform(df)
    else:
        assert pipeline is not None, "Pass a fitted pipeline when is_train=False"
        X, y = pipeline.transform(df)
    return X, y, pipeline
