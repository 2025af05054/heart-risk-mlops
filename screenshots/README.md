# Screenshots checklist

Capture these on your machine and save them here; the report has a marked placeholder for each.

| File | What to capture |
|---|---|
| 01_mlflow_runs.png | MLflow UI run list comparing the 5 model-family runs |
| 02_mlflow_run_detail.png | One parent run: params, metrics, `evaluation/` artifacts |
| 03_actions_success.png | GitHub Actions run: lint → test → train → docker all green |
| 04_actions_artifacts.png | Artifacts section of that run |
| 05_docker_build_ps.png | `docker build` finishing + `docker ps` showing `(healthy)` |
| 06_docker_smoke.png | `./scripts/smoke_test.sh` against the container |
| 07_k8s_resources.png | `kubectl -n heart-mlops get pods,svc,ingress,hpa` |
| 08_k8s_predict.png | `curl /predict` via LoadBalancer IP and via `heart-api.local` |
| 09_grafana.png | Grafana dashboard after `python scripts/load_test.py` |
| 10_prometheus_targets.png | Prometheus Status → Targets, both API pods UP |
