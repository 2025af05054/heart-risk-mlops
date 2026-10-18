#!/usr/bin/env bash
# Build the image inside Minikube and deploy API + Prometheus + Grafana.
set -euo pipefail
cd "$(dirname "$0")/.."

minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=6g
minikube addons enable ingress
minikube addons enable metrics-server

# Build straight into Minikube's Docker daemon (no registry needed)
eval "$(minikube docker-env)"
docker build -t heart-risk-api:1.0.0 .

kubectl apply -k k8s/
kubectl -n heart-mlops rollout status deploy/heart-api --timeout=180s
kubectl -n heart-mlops get pods,svc,ingress,hpa -o wide

cat <<MSG

Deployed. Next steps:
  1. In a separate terminal:  minikube tunnel        (gives the LoadBalancer an EXTERNAL-IP)
  2. kubectl -n heart-mlops get svc heart-api         (copy EXTERNAL-IP)
  3. ./scripts/smoke_test.sh http://<EXTERNAL-IP>
  Ingress: add "\$(minikube ip) heart-api.local" to /etc/hosts, then ./scripts/smoke_test.sh http://heart-api.local
  Grafana: minikube service -n heart-mlops grafana     Prometheus: minikube service -n heart-mlops prometheus
MSG
