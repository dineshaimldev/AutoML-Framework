# AutoML Mini-Framework

A lightweight, config-driven AutoML framework: automatically searches models and
hyperparameters with Bayesian optimization (Optuna), evaluates them thoroughly,
and generates a full HTML report — with MLflow experiment tracking and a
FastAPI serving layer.

> Status: 🚧 scaffold stage — architecture and CI are in place; pipeline logic
> is being built out module by module. See "Roadmap" below.

## Why this over AutoGluon / TPOT?

Mature AutoML libraries exist (AutoGluon, TPOT, H2O). This project isn't trying
to replace them — it's a smaller, fully-readable, from-scratch implementation
built to demonstrate the full lifecycle: preprocessing → Bayesian search →
evaluation → reporting → tracking → serving. Every stage is transparent and
swappable, which is the point.

## Architecture

```
raw CSV → preprocessing → Optuna search over model zoo → best model
              |                       |
              v                       v
        MLflow logging        evaluation (ROC, confusion matrix,
        (every trial)          feature importance, SHAP)
                                        |
                                        v
                              HTML report + FastAPI /predict
```

## Tech stack (100% free / open-source)

| Purpose               | Tool                          |
|------------------------|-------------------------------|
| Data handling          | Pandas, NumPy                 |
| Models                 | scikit-learn                  |
| Hyperparameter search  | Optuna (TPE sampler + pruning)|
| Explainability         | SHAP                          |
| Plots                  | Matplotlib, Seaborn           |
| Config & validation    | PyYAML, Pydantic              |
| Experiment tracking    | MLflow (local, no server)     |
| Serving                | FastAPI + Uvicorn              |
| Testing                | Pytest                        |
| Lint/format            | Ruff, Black, mypy              |
| CI                     | GitHub Actions                |
| Containerization       | Docker                        |

## Getting started

```bash
# clone and install in editable mode with dev tools
git clone <your-repo-url>
cd automl-framework
pip install -e ".[dev]"

# run tests
pytest

# run the pipeline against the example config
automl run --config configs/example_config.yaml
```

### Docker

```bash
docker build -t automl-framework .
docker run -p 8000:8000 automl-framework
```

## Configuration

Everything is driven by a single YAML file — see `configs/example_config.yaml`.
No code changes needed to point at a new dataset, add/remove models, or
change the optimization budget.

## Roadmap

- [x] Repo structure, packaging, config schema
- [x] CI (lint + test + docker build) on every push
- [ ] Data loading + preprocessing (auto column-type detection, imputation, scaling)
- [ ] Model zoo + Optuna search space per model
- [ ] Nested cross-validation
- [ ] Evaluation module (ROC, confusion matrix, feature importance, SHAP)
- [ ] Jinja2 HTML report generation
- [ ] MLflow logging integration
- [ ] FastAPI `/predict` wired to the best saved model
- [ ] Streamlit demo UI

## License

MIT
