# 部署

项目没上真实服务器，所谓"部署"就是 GitHub Actions 构建镜像并推到 Docker Hub。

`.github/workflows/ci.yml` 里有三个任务：

- `test`：PR 到 dev/staging/main、或 push 到 staging/main 时跑。装依赖、ruff、pytest、再 docker build。
- `deploy-staging`：push 到 staging 时跑，生成 staging 的 `.env`，构建并推 `road-surface:staging`。
- `deploy-prod`：push 到 main 时跑，生成生产的 `.env`，构建并推 `road-surface:latest`。

两个部署任务都 `needs: test`，测试不过就不推。

`.env` 内容：

```
APP_ENV=staging
MODEL_PATH=<STAGING_MODEL_PATH>
```

（生产就是 `APP_ENV=production` + `PROD_MODEL_PATH`）

需要配的 secret：`DOCKERHUB_USERNAME`、`DOCKERHUB_TOKEN`，另外 `STAGING_MODEL_PATH` / `PROD_MODEL_PATH` 可选（有默认值）。

跑已经发布的镜像：

```bash
docker pull <用户名>/road-surface:latest
docker run -d --name road-surface -p 127.0.0.1:8000:8000 <用户名>/road-surface:latest
```

打开 http://localhost:8000/ 就能用。
