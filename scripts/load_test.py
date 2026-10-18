"""Send realistic traffic to the API so the Grafana dashboard has data.

    python scripts/load_test.py --url http://localhost:8000 --n 300 --rps 5
Samples rows from the cleaned dataset; ~5% of requests are deliberately invalid
so error-rate panels light up.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rps", type=float, default=5)
    ap.add_argument("--data", default="data/processed/heart_clean.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.data)[FEATURES]
    codes = {}
    for i in range(a.n):
        row = df.sample(1).iloc[0].to_dict()
        rec = {k: (None if pd.isna(v) else (float(v) if k == "oldpeak" else int(v)))
               for k, v in row.items()}
        if random.random() < 0.05:
            rec["age"] = -1
        code = post(f"{a.url}/predict", rec)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(1 / a.rps)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{a.n} sent, status counts: {codes}")
    print("done:", codes)


if __name__ == "__main__":
    main()
