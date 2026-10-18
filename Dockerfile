# ---- build stage: install wheels into a virtualenv ----
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---- runtime stage ----
FROM python:3.13-slim
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI)"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1001 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_META_PATH=/app/models/model_meta.json
WORKDIR /app
COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/model_meta.json models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1
# single worker per container: Prometheus metrics live in-process; scale with replicas
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
