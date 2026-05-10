import joblib
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PIPELINE_PATH, LIGHTGBM_PATH
from preprocessing.pipeline import PreprocessingPipeline

LGB_THRESHOLDS = [0.50, 0.50, 0.82, 0.50]   # C1, C2, C3, C4 — from notebook 02


class Predictor:

    SEVERITY_DESCRIPTIONS = {
        1: "Severity 1 — Minor incident. Short-term traffic impact, no emergency response expected.",
        2: "Severity 2 — Moderate incident. Noticeable traffic slowdown. Possible emergency response.",
        3: "Severity 3 — Significant disruption. Major traffic impact. Emergency services likely required.",
        4: "Severity 4 — Critical incident. Severe long-term traffic impact. Immediate emergency response.",
    }

    SEVERITY_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}

    def __init__(self):
        self.pipeline = PreprocessingPipeline.load(PIPELINE_PATH)
        self.model    = joblib.load(LIGHTGBM_PATH)

    def predict(self, input_dict: dict) -> dict:
        """
        Takes a raw input dict with accident conditions and returns a prediction.

        Required keys: Start_Time (str), End_Time (str), Start_Lat, Start_Lng,
        Temperature(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph),
        Weather_Condition, Distance(mi), Civil_Twilight, Wind_Direction,
        Amenity, Crossing, Give_Way, Junction, No_Exit, Railway,
        Stop, Traffic_Calming, Traffic_Signal.

        Returns:
            {
                "severity":      int (1-4),
                "probabilities": {1: float, 2: float, 3: float, 4: float},
                "description":   str,
                "color":         str (hex),
            }
        """
        df    = pd.DataFrame([input_dict])
        X, _  = self.pipeline.transform(df)
        proba  = self.model.predict_proba(X)[0]
        scaled = proba / LGB_THRESHOLDS
        severity = int(np.argmax(scaled) + 1)   # 0-indexed → 1-indexed
        return {
            "severity":      severity,
            "probabilities": {i + 1: float(proba[i]) for i in range(4)},
            "description":   self.SEVERITY_DESCRIPTIONS[severity],
            "color":         self.SEVERITY_COLORS[severity],
        }
