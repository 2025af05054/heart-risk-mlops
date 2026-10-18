"""FastAPI service exposing the heart-disease classifier.

Endpoints
  GET  /health      liveness probe
  GET  /ready       readiness probe (model loaded)
  GET  /model-info  model metadata
  POST /predict     JSON patient record(s) -> prediction + confidence
  GET  /metrics     Prometheus metrics
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)

from api.schemas import PatientFeatures, Prediction, PredictRequest, PredictResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
META_PATH = Path(os.getenv("MODEL_META_PATH", "models/model_meta.json"))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
FEATURE_ORDER = list(PatientFeatures.model_fields)


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart_api")
log.handlers = [handler]
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
log.propagate = False

# ---------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of served predictions",
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
PROBA = Histogram("model_probability_disease", "Predicted P(disease)",
                  buckets=[i / 10 for i in range(1, 11)])
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded", ["model_name", "version"])

state: dict = {}


def load_model() -> None:
    state["model"] = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    state["meta"] = meta
    state["name"] = meta.get("model_name", "unknown")
    state["version"] = (meta.get("mlflow_run_id") or "local")[:8]
    MODEL_LOADED.labels(state["name"], state["version"]).set(1)
    log.info("model loaded", extra={"extra_fields": {"model": state["name"],
                                                     "version": state["version"]}})


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": rid, "method": request.method, "path": endpoint,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model": state["name"]}


@app.get("/model-info")
def model_info():
    return {"model_name": state["name"], "model_version": state["version"],
            "threshold": THRESHOLD, "features": FEATURE_ORDER, "metadata": state["meta"]}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest | PatientFeatures, request: Request):
    records = payload.instances if isinstance(payload, PredictRequest) else [payload]
    frame = pd.DataFrame([r.model_dump() for r in records], columns=FEATURE_ORDER).astype(float)
    proba = state["model"].predict_proba(frame)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= THRESHOLD)
        label = "disease" if pred else "no_disease"
        conf = float(p if pred else 1 - p)
        PREDICTIONS.labels(label).inc()
        CONFIDENCE.observe(conf)
        PROBA.observe(float(p))
        out.append(Prediction(prediction=pred, label=label, confidence=round(conf, 4),
                              probability_disease=round(float(p), 4)))
    log.info("prediction", extra={"extra_fields": {
        "request_id": getattr(request.state, "request_id", None), "n": len(out),
        "labels": [o.label for o in out], "proba": [o.probability_disease for o in out]}})
    return PredictResponse(model_name=state["name"], model_version=state["version"],
                           predictions=out)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.error("unhandled error", extra={"extra_fields": {"error": repr(exc)}})
    return JSONResponse(status_code=500, content={"detail": "internal error"})
