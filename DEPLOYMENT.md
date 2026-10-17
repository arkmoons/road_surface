# Deployment

The project doesn't run on a real server. The "deployment" is just GitHub
Actions building the Docker image and pushing it to Docker Hub.

## The jobs

All jobs are in `.github/workflows/ci.yml`.

- **test** – runs on every PR to `dev`/`staging`/`main` and on every push to
  `staging`/`main`. It installs the dependencies, runs ruff, runs pytest, and
  does a `docker build` to check the image still builds.
- **deploy-staging** – only on a push to `staging` (and only if `test` passed).
  It writes a staging `.env`, then builds and pushes `road-surface:staging`.
- **deploy-prod** – only on a push to `main` (and only if `test` passed). It
  writes a production `.env`, then builds and pushes `road-surface:latest` and
  `road-surface:sha-<commit>`.

Both deploy jobs have `needs: test`, so if the tests fail nothing is pushed.

## The .env files

The staging job writes:

    APP_ENV=staging
    MODEL_PATH=<STAGING_MODEL_PATH>

and the production job writes:

    APP_ENV=production
    MODEL_PATH=<PROD_MODEL_PATH>

This is how the project wants "environment-specific config from secrets".

## Image tags

- staging: `road-surface:staging` and `road-surface:staging-sha-<commit>`
- production: `road-surface:latest` and `road-surface:sha-<commit>`

## Secrets

I added these in GitHub under Settings -> Secrets and variables -> Actions:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (a Docker Hub token with Read & Write)
- `STAGING_MODEL_PATH` (optional, has a default)
- `PROD_MODEL_PATH` (optional, has a default)

## Run the published image

```bash
docker pull <dockerhub-user>/road-surface:latest
docker run -d --name road-surface -p 127.0.0.1:8000:8000 <dockerhub-user>/road-surface:latest
```

Then open http://localhost:8000/. The image already has the app, the page and
the model, so nothing else is needed.
