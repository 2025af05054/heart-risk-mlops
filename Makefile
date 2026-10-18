.PHONY: setup data eda train test lint serve mlflow docker-build docker-run compose deploy smoke load
PY ?= python

setup:        ## install everything
	pip install -r requirements.txt && pip install -e .
data:         ## download + clean dataset
	$(PY) -m heart.data
eda:          ## regenerate EDA figures
	$(PY) -m heart.eda
train:        ## train/tune all models with MLflow tracking
	$(PY) -m heart.train
test:
	pytest -v
lint:
	flake8 src api tests
serve:        ## run API locally
	uvicorn api.app:app --reload --port 8000
mlflow:       ## open MLflow UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
docker-build:
	docker build -t heart-risk-api:1.0.0 .
docker-run:
	docker run --rm -p 8000:8000 heart-risk-api:1.0.0
compose:      ## API + Prometheus + Grafana
	docker compose up --build
deploy:       ## Minikube deployment
	./scripts/deploy_minikube.sh
smoke:
	./scripts/smoke_test.sh
load:
	$(PY) scripts/load_test.py
