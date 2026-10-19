# 部署说明

这个项目没有部署到真实服务器上。"部署"就是 GitHub Actions 构建 Docker 镜像并推送到 Docker Hub。

## 任务（jobs）

所有任务都在 `.github/workflows/ci.yml` 里。

- **test**：对 `dev`/`staging`/`main` 的每个 PR、以及 push 到 `staging`/`main` 时运行。装依赖、跑 ruff、跑 pytest，再 `docker build` 确认镜像能构建。
- **deploy-staging**：只在 push 到 `staging` 时运行（且要 `test` 先通过）。写 staging `.env`，然后构建并推送 `road-surface:staging`。
- **deploy-prod**：只在 push 到 `main` 时运行（且要 `test` 先通过）。写 production `.env`，然后构建并推送 `road-surface:latest` 和 `road-surface:sha-<commit>`。

两个部署任务都有 `needs: test`，所以测试不过就不会推送镜像。

## `.env` 文件

staging 任务写：

    APP_ENV=staging
    MODEL_PATH=<STAGING_MODEL_PATH>

production 任务写：

    APP_ENV=production
    MODEL_PATH=<PROD_MODEL_PATH>

这就是项目要求的"从 secrets 生成各环境的配置"。

## 镜像 tag

- staging：`road-surface:staging` 和 `road-surface:staging-sha-<commit>`
- production：`road-surface:latest` 和 `road-surface:sha-<commit>`

## Secrets

我在 GitHub 的 Settings → Secrets and variables → Actions 里加了：

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`（Docker Hub 的 Read & Write token）
- `STAGING_MODEL_PATH`（可选，有默认值）
- `PROD_MODEL_PATH`（可选，有默认值）

## 运行已发布的镜像

```bash
docker pull <dockerhub-user>/road-surface:latest
docker run -d --name road-surface -p 127.0.0.1:8000:8000 <dockerhub-user>/road-surface:latest
```

然后打开 http://localhost:8000/。镜像里已经有应用、页面和模型，不需要别的东西。
