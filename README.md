# 路面状态识别（MLOps 期末项目）

我的 MLOps 课程项目。App 会看一张路面照片，判断路面状态：`dry`（干燥）、`wet`（潮湿）、`water`（积水）、`fresh_snow`（新雪）、`melted_snow`（融雪）、`ice`（结冰）。

模型是我自己训练的（微调预训练的 EfficientNet-B0），然后在它外面加了课程要求的东西：Git 分支、DVC 管数据、MLflow 记实验、Docker、GitHub Actions 做 CI/CD。

## 文件说明

- `app/main.py`：FastAPI 应用，加载训练好的模型，提供 `/predict`（传图片，返回类别）。
- `ml/train.py`：训练脚本（每次训练都记到 MLflow）。
- `ml/data_pipeline.py`：从原始 v1 图片生成平衡的 v2 数据。
- `ml/registry/current_model.md`：记录当前用的是哪个模型。
- `data/`：`manifest.csv`（文件名 → 标签）和 DVC 指针文件。
- `tests/`：pytest 测试。
- `Dockerfile`、`.dockerignore`：构建镜像。
- `.github/workflows/ci.yml`：CI/CD 流水线。
- 其他文档：`ml/README_data.md`、`ml/README_experiments.md`、`ml/PROVENANCE.md`、`DEPLOYMENT.md`。

## 本地运行

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 启动应用（需要 models/current.pt）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 跑测试
.venv\Scripts\python.exe -m pytest -q
```

接口：`GET /`（上传页面）、`POST /predict`（图片 → 结果）、`GET /health`（健康检查）。

## 数据

19,860 张路面图片（240x360）。标签在文件名里，例如 `...-dry-asphalt-smooth.jpg`，所以我写了个小脚本生成 `data/manifest.csv`。

图片太大不适合放 Git，所以用 DVC 管理、存在 Azure。Git 里只有 `.dvc` 指针文件和 manifest。

```bash
dvc pull   # 下载图片
dvc push   # 上传新版本
```

更多细节见 `ml/README_data.md`。

## 模型

我用 torchvision 的预训练模型做微调。每次训练都记到 MLflow（本地 `mlflow.db`）：git commit、DVC 数据哈希、超参数、accuracy / macro-F1、模型权重和混淆矩阵。

```bash
# 基线
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone resnet18 --epochs 5 --run-name baseline-v1-resnet18

# 改进
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 --epochs 8 --lr 5e-4 --run-name improved-v1-efficientnet_b0

# 看实验
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

结果和最终模型选择见 `ml/README_experiments.md`。

## 测试

`tests/` 里有 11 个测试：
- `test_api.py`：检查 `/health`、真实图片返回合法类别且概率和为 1、文本文件返回 400、坏图返回 400、缺文件返回 422。测试里把模型换成了一个小假模型，所以不需要真实权重，CI 也能跑。
- `test_model.py`：检查模型输出大小、类别列表和预处理。

## Docker

```bash
docker build -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```

镜像特意做小了：只装推理依赖（`requirements-api.txt`），并用 CPU 版 torch。`models/current.pt` 会被复制进镜像。

## CI/CD

全部在 `.github/workflows/ci.yml` 里。

分支：`feature/* -> dev -> staging -> main`

什么时候跑什么：
- **对 `dev`、`staging` 或 `main` 的 PR**：安装依赖、`ruff check`、`ruff format --check`、`pytest`，再 `docker build` 确认镜像还能构建。这一步失败就不应该合并。
- **push/merge 到 `staging`**：同样的检查，然后 staging 任务用 secrets 生成 staging `.env`，构建并推送 `road-surface:staging`。
- **push/merge 到 `main`**：同样的检查，然后生产任务用 secrets 生成 prod `.env`，构建并推送 `road-surface:latest` 和 `road-surface:sha-<commit>`。

从提交到生产的流程：

1. 提交并 push 功能分支，开 PR
2. CI 跑（lint、format、测试、docker build）
3. PR 变绿 → 合并
4. 合并到 staging → 生成 staging `.env` → 构建/推送 `:staging`
5. staging 合并到 main → 生成 prod `.env` → 构建/推送 `:latest`

我在 GitHub 上加的 secrets（Settings → Secrets and variables → Actions）：`DOCKERHUB_USERNAME`、`DOCKERHUB_TOKEN`。`STAGING_MODEL_PATH` 和 `PROD_MODEL_PATH` 也是给 `.env` 用的，但有默认值。

## 其他文档

- `ml/README_data.md`：数据集、v1 vs v2、DVC 版本
- `ml/README_experiments.md`：训练实验和最终模型
- `ml/PROVENANCE.md`：模型是用哪份代码、哪份数据训出来的
- `DEPLOYMENT.md`：staging 和 production 怎么构建
