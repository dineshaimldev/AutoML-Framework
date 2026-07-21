FROM python:3.11-slim

WORKDIR /app

# System deps needed by scikit-learn / matplotlib / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY models_saved ./models_saved

EXPOSE 8000

# Default: serve the API. Override with `docker run ... automl run --config ...`
CMD ["uvicorn", "automl.api:app", "--host", "0.0.0.0", "--port", "8000"]
