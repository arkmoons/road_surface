# Deployment (DEPLOYMENT.md)

This project builds and publishes Docker images automatically with GitHub Actions.
No live server deployment is required by the project; the delivery step is a
tested, environment-specific Docker image pushed to Docker Hub.

## Pipelines in `.github/workflows/ci.yml`

| Job | Trigger | What it does |
|---|---|---|
| `test` | every PR to `dev`/`staging`/`main`, and every push to `staging`/`main` | install deps, `ruff check`, `ruff format --check`, `pytest`, `docker build` (verify) |
| `deploy-staging` | push / merge to `staging` (after `test` passes) | create `staging` `.env` from staging secrets → build & push `road-surface:staging` |
| `deploy-prod` | push / merge to `main` (after `test` passes) | create `production` `.env` from production secrets → build & push `road-surface:latest` and `road-surface:sha-<commit>` |

CI is a hard gate: `deploy-staging` and `deploy-prod` both declare
`needs: test`, so a failing test never publishes an image.

## Environment-specific `.env`

Each deploy job creates a `.env` from GitHub **environment secrets**:

```
APP_ENV=staging
MODEL_PATH=<STAGING_MODEL_PATH secret>
```

```
APP_ENV=production
MODEL_PATH=<PROD_MODEL_PATH secret>
```

The stage is also passed to the image as a build arg (`APP_ENV`), so the image
knows which environment it was built for.

## Image tags

| Environment | Tag(s) |
|---|---|
| staging | `<dockerhub-user>/road-surface:staging`, `:staging-sha-<commit>` |
| production | `<dockerhub-user>/road-surface:latest`, `:sha-<commit>` |

## Required GitHub configuration

Create the repository secrets:

- `DOCKERHUB_USERNAME` — Docker Hub username
- `DOCKERHUB_TOKEN` — Docker Hub personal access token (Repo Read & Write)
- `STAGING_MODEL_PATH` — model path used by the staging `.env`
- `PROD_MODEL_PATH` — model path used by the production `.env`

Create the GitHub **environments** `staging` and `production` (Settings →
Environments) so the stage secrets and any protection rules apply.

## Running a published image

```bash
docker pull <dockerhub-user>/road-surface:latest
docker run -d --name road-surface \
  -p 127.0.0.1:8000:8000 \
  <dockerhub-user>/road-surface:latest
# open http://localhost:8000/
```

The image contains FastAPI, the raw HTML page, and the production weights
(`models/current.pt`), so it is self-contained.

## Local build (no CI)

```bash
docker build -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```
