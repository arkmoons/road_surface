# Road-Surface Condition Classifier — MLOps Final Project

An end-to-end machine-learning application that classifies a road image into one
of **6 surface conditions**: `dry`, `wet`, `water`, `fresh_snow`, `melted_snow`,
`ice`. Built with DevOps + MLOps practices: Git branching, DVC data versioning on
Azure, MLflow experiment tracking, Docker, and GitHub Actions CI/CD.

## What is in the repository

```
app/                      FastAPI service (app/main.py) + model loading
ml/
  train.py                transfer-learning training script (logs to MLflow)
  data_pipeline.py        builds the balanced v2 dataset from v1
  configs/                training configs
  registry/current_model.md   which run is "production"
  README_data.md          data provenance, v1 vs v2, commit↔version map
  README_experiments.md   MLflow experiment log and model selection
  PROVENANCE.md           model ↔ code ↔ data provenance
data/
  manifest.csv            filename → labels (Git-tracked)
  raw.dvc / processed.dvc DVC pointers (images live on Azure)
tests/                    pytest suite (API contract + model unit tests)
Dockerfile                slim production image
.dockerignore
pyproject.toml            ruff config (lint + format)
.github/workflows/ci.yml  GitHub Actions CI/CD
DEPLOYMENT.md             how staging / production are built
```

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# (Windows) or: source .venv/bin/activate && pip install -r requirements-dev.txt

# run the API (needs a trained model in models/current.pt)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# run the tests
.venv\Scripts\python.exe -m pytest -q

# lint / format
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
```

API:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | simple upload page |
| POST | `/predict` | multipart image → `{condition, confidence, probabilities}` |
| GET | `/health` | liveness |

## Data versioning (DVC + Azure)

The raw dataset (19,860 road images) is **not** stored in Git. DVC tracks it and
the actual bytes live in Azure Blob Storage. Git only stores `.dvc` pointers and
`data/manifest.csv`.

| version | dir | DVC md5 | files |
|---|---|---|---|
| v1 raw | `data/raw` | `226e861c295a314f530950af6879f8f3` | 19,860 |
| v2 balanced | `data/processed` | `e22aa7c0702400f7303f28d2c9ff17ca` | 4,920 |

```bash
dvc pull        # fetch both versions from Azure
dvc push        # upload a new version
```

See [`ml/README_data.md`](ml/README_data.md) for the full data story.

## Machine learning (MLflow)

Training fine-tunes a pretrained CNN (transfer learning) and logs every run to
MLflow (local SQLite backend `mlflow.db`): git commit, DVC data hash, hyper
parameters, accuracy / macro-F1, per-class F1, the model weights and a confusion
matrix.

```bash
# baseline: ResNet18 on v1, 5 epochs
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone resnet18 \
    --epochs 5 --run-name baseline-v1-resnet18

# improved: EfficientNet-B0 on v1, 10 epochs, lr 5e-4
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 \
    --epochs 10 --lr 5e-4 --run-name improved-v1-efficientnet_b0

# view the experiment history
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

See [`ml/README_experiments.md`](ml/README_experiments.md) for results and the
production-model decision.

## Testing

Three test groups (see `tests/`):
- **API contract** (`test_api.py`): `/health`, valid image → 200 + valid class +
  probabilities sum to 1, non-image → 400, corrupt image → 400, missing file → 422.
  The model is monkeypatched with a deterministic dummy, so the suite needs no
  weights or data and runs on CI.
- **Model unit** (`test_model.py`): architecture output shape, class list,
  preprocessing tensor shape, unknown-backbone rejection.

## Docker

```bash
docker build -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```

The image is slim (`python:3.12-slim`), installs only inference dependencies
(`requirements-api.txt`) with the CPU build of torch, and bakes in
`models/current.pt` + `ml/registry/current_model.md`.

## CI/CD (GitHub Actions)

Pipelines are defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Branch strategy

```
feature/*  ->  dev  ->  staging  ->  main
```

### What runs where

| Event | Pipeline |
|---|---|
| Pull request → `dev` / `staging` / `main` | **CI**: install, `ruff check`, `ruff format --check`, `pytest`, `docker build`. A failed CI blocks the merge. |
| Push / merge to `staging` | CI above, then **staging deploy**: create staging `.env` from staging GitHub secrets → build & push `road-surface:staging`. |
| Push / merge to `main` | CI above, then **production deploy**: create production `.env` from production GitHub secrets → build & push `road-surface:latest` + `road-surface:sha-<commit>`. |

### Commit → build → test → staging → prod

```
git commit
   │  push feature branch, open PR to dev/staging/main
   ▼
CI job: install deps → ruff lint → ruff format --check → pytest → docker build
   │  PR green + reviewed  →  merge
   ▼
push to staging ──► CI passes ──► create staging .env (secrets) ──► build/push :staging
   │  merge staging → main
   ▼
push to main ──► CI passes ──► create prod .env (secrets) ──► build/push :latest + :sha-<commit>
```

### Required repository secrets / variables

| Name | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_USERNAME` | deploy jobs | Docker Hub account |
| `DOCKERHUB_TOKEN` | deploy jobs | Docker Hub access token |
| `STAGING_MODEL_PATH` | staging deploy | staging `.env` value |
| `PROD_MODEL_PATH` | production deploy | production `.env` value |

## Related docs

- [`ml/README_data.md`](ml/README_data.md) — dataset, v1→v2, DVC versions
- [`ml/README_experiments.md`](ml/README_experiments.md) — MLflow runs, model choice
- [`ml/PROVENANCE.md`](ml/PROVENANCE.md) — model ↔ code ↔ data provenance
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — staging/production build details
