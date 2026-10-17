# Road Surface Condition Classifier

My MLOps course project. The app looks at a photo of a road and guesses the
surface: `dry`, `wet`, `water`, `fresh_snow`, `melted_snow` or `ice`.

I trained the model myself (fine-tuned a pretrained EfficientNet-B0), then added
the things the course asked for around it: git branches, DVC for the data,
MLflow for the experiments, Docker, and GitHub Actions for CI/CD.

## Files

- `app/main.py` – the FastAPI app. It loads the trained model and has
  `/predict` (send an image, get the class back).
- `ml/train.py` – the training script (logs every run to MLflow).
- `ml/data_pipeline.py` – makes the balanced v2 data from the raw v1 images.
- `ml/registry/current_model.md` – which run is the current model.
- `data/` – `manifest.csv` (file name -> labels) and the DVC pointer files.
- `tests/` – pytest tests.
- `Dockerfile`, `.dockerignore` – build the image.
- `.github/workflows/ci.yml` – the CI/CD pipelines.
- Other docs: `ml/README_data.md`, `ml/README_experiments.md`,
  `ml/PROVENANCE.md`, `DEPLOYMENT.md`.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# start the app (it needs models/current.pt)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# run the tests
.venv\Scripts\python.exe -m pytest -q
```

Endpoints: `GET /` (upload page), `POST /predict` (image -> result),
`GET /health`.

## The data

19,860 road images (240x360). The label is in the file name, for example
`...-dry-asphalt-smooth.jpg`, so I wrote a small script to build
`data/manifest.csv`.

The images are too big for Git, so I track them with DVC and keep them on Azure.
Git only has the `.dvc` pointer files and the manifest.

```bash
dvc pull   # download the images
dvc push   # upload a new version
```

More details in `ml/README_data.md`.

## The model

I fine-tune a pretrained CNN from torchvision. Every run is logged to MLflow
(local `mlflow.db`): the git commit, the DVC data hash, the hyper-parameters,
accuracy / macro-F1, the weights and a confusion matrix.

```bash
# baseline
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone resnet18 --epochs 5 --run-name baseline-v1-resnet18

# improved
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 --epochs 8 --lr 5e-4 --run-name improved-v1-efficientnet_b0

# view the runs
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Results and the final model choice are in `ml/README_experiments.md`.

## Tests

There are 11 tests in `tests/`:
- `test_api.py` – checks `/health`, a real image gives a valid class with
  probabilities that sum to 1, a text file gives 400, a broken image gives 400,
  and a missing file gives 422. The model is replaced by a tiny fake one, so the
  tests don't need the real weights and still run on CI.
- `test_model.py` – checks the model output size, the class list and the
  preprocessing.

## Docker

```bash
docker build -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```

The image is kept small: it installs only the inference dependencies
(`requirements-api.txt`) and uses the CPU build of torch. `models/current.pt`
is copied into the image.

## CI/CD

Everything is in `.github/workflows/ci.yml`.

Branches: `feature/* -> dev -> staging -> main`

What runs where:
- **PR to `dev`, `staging` or `main`**: install deps, `ruff check`,
  `ruff format --check`, `pytest`, and a `docker build` to make sure the image
  still builds. If this fails, the PR should not be merged.
- **Push/merge to `staging`**: the same checks, then a staging job writes a
  staging `.env` from the secrets and builds/pushes `road-surface:staging`.
- **Push/merge to `main`**: the same checks, then a production job writes a
  prod `.env` from the secrets and builds/pushes `road-surface:latest` and
  `road-surface:sha-<commit>`.

The path from a commit to prod:

1. commit and push a feature branch, open a PR
2. CI runs (lint, format, tests, docker build)
3. PR is green -> merge
4. merge to staging -> staging `.env` -> build/push `:staging`
5. merge staging to main -> prod `.env` -> build/push `:latest`

Secrets I added in GitHub (Settings -> Secrets and variables -> Actions):
`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`. `STAGING_MODEL_PATH` and
`PROD_MODEL_PATH` are used by the `.env` but have a default.

## Other docs

- `ml/README_data.md` – the dataset, v1 vs v2, DVC versions
- `ml/README_experiments.md` – the runs and the final model
- `ml/PROVENANCE.md` – which code and data made the model
- `DEPLOYMENT.md` – how staging and production are built
