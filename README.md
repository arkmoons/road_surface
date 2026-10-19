# 路面状态识别

程序输入一张路面照片，输出它的路面状态，一共 6 类：dry（干燥）、wet（潮湿）、water（积水）、fresh_snow（新雪）、melted_snow（融雪）、ice（结冰）。

模型是我自己微调的 EfficientNet-B0。外面套了课程要求的一套流程：Git 分支、DVC 管数据、MLflow 记实验、Docker 打包、GitHub Actions 做 CI/CD。

## 怎么跑

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000/，传一张图就能看结果。接口还有 `POST /predict` 和 `GET /health`。

跑测试：

```bash
.venv\Scripts\python.exe -m pytest -q
```

## 数据

19,860 张路面图，标签写在文件名里（比如 `...-dry-asphalt-smooth.jpg`），我自己写脚本整理成了 `data/manifest.csv`。

图太大不放 Git，用 DVC 存到 Azure。本地要图就 `dvc pull`。

数据分两个版本：v1 是原始的，v2 是我按类抽 820 张做的平衡版（后来发现效果反而差，详见 `ml/README_experiments.md`）。

## 模型

用 torchvision 的预训练模型微调。训练脚本是 `ml/train.py`，每次训练都记到 MLflow（本地 `mlflow.db`）。最后选了 efficientnet_b0 那一次，权重在 `models/current.pt`。

```bash
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 --epochs 8 --lr 5e-4
```

## 测试

`tests/` 里 11 个测试，测接口的（`test_api.py`）和测模型的（`test_model.py`）。跑测试用假模型，不需要真权重。

## Docker

```bash
docker build -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```

镜像只装了推理需要的东西（`requirements-api.txt` + CPU 版 torch），比较小。

## CI/CD

在 `.github/workflows/ci.yml`。分支是 `feature/* -> dev -> staging -> main`。

- 往 dev/staging/main 提 PR：跑 lint、格式检查、测试，再 build 一次镜像。
- 合进 staging：生成 staging 的 `.env`，构建并推 `road-surface:staging`。
- 合进 main：生成生产的 `.env`，构建并推 `road-surface:latest`。

需要在 GitHub 配两个 secret：`DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN`。

## 其他

- `ml/README_data.md`：数据详情
- `ml/README_experiments.md`：实验对比
- `ml/PROVENANCE.md`：模型溯源
- `DEPLOYMENT.md`：部署说明
