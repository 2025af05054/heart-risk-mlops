"""Offline inference with the packaged model (no API needed).

    python scripts/predict_cli.py scripts/sample_request.json
"""
import json
import sys

import joblib
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]

model = joblib.load("models/model.joblib")
records = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scripts/sample_request.json"))
records = records if isinstance(records, list) else [records]
X = pd.DataFrame(records, columns=FEATURES).astype(float)
for rec, p in zip(records, model.predict_proba(X)[:, 1]):
    label = "disease" if p >= 0.5 else "no_disease"
    print(json.dumps({"label": label, "probability_disease": round(float(p), 4),
                      "confidence": round(float(max(p, 1 - p)), 4)}))
