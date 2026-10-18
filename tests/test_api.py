import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_PATIENT


@pytest.fixture(scope="module")
def client(tiny_model_files):
    import api.app as app_module

    app_module.MODEL_PATH, app_module.META_PATH = tiny_model_files
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model"] == "test_lr"


def test_predict_single(client):
    r = client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    p = body["predictions"][0]
    assert p["prediction"] in (0, 1)
    assert 0.5 <= p["confidence"] <= 1.0
    assert p["label"] == ("disease" if p["prediction"] else "no_disease")
    assert r.headers["x-request-id"]


def test_predict_batch(client):
    r = client.post("/predict", json={"instances": [SAMPLE_PATIENT, SAMPLE_PATIENT]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_optional_fields_may_be_missing(client):
    patient = {k: v for k, v in SAMPLE_PATIENT.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=patient).status_code == 200


@pytest.mark.parametrize("bad", [
    {**SAMPLE_PATIENT, "age": -5},
    {**SAMPLE_PATIENT, "cp": 9},
    {**SAMPLE_PATIENT, "unknown_field": 1},
    {"instances": []},
])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json=SAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text
