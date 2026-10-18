# 手工复现指南 (REPRODUCE.md)

从零把项目复现一遍。命令都在 **Windows PowerShell**、项目根目录下执行。
`.venv\Scripts\python.exe` 是虚拟环境里的 Python，后面所有 python 命令都用它。

需要准备：

- Python 3.12、Git、Docker Desktop（含 Compose）、DVC
- GitHub 账号、Docker Hub 账号
- 课程给的 `vali_20k.zip`（19,860 张路面图片）
- 课程给的 Azure SAS token（storage account `wut2026`，container `dvc`）

---

## 1. 建项目 + 初始化 Git

```powershell
mkdir mlops-project
cd mlops-project
git init -b main
git branch dev
git branch staging
```

**作用**：建仓库，一开始就有 `main / dev / staging` 三个分支（课程 4.Version-control 的分支隔离思想）。

---

## 2. 解压数据 + 生成标签清单

把 `vali_20k.zip` 放进项目根目录，新建 `prepare_data.py`：

```python
import os, csv, shutil, zipfile

src = 'vali_20k.zip'
raw = 'data/raw'
os.makedirs(raw, exist_ok=True)

z = zipfile.ZipFile(src)
for n in z.namelist():
    if n.lower().endswith('.jpg'):
        base = n.split('/', 1)[-1]          # 去掉 zip 里的 vali_20k/ 顶层目录
        with z.open(n) as fsrc, open(os.path.join(raw, base), 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)

rows = []
for fn in sorted(os.listdir(raw)):
    if not fn.lower().endswith('.jpg'):
        continue
    parts = fn[:-4].split('-')              # <时间戳>-<condition>-<material>-<roughness>
    rows.append((fn, parts[1],
                 parts[2] if len(parts) > 2 else '',
                 parts[3] if len(parts) > 3 else ''))

with open('data/manifest.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['filename', 'condition', 'material', 'roughness'])
    w.writerows(rows)

print('images:', len(rows))
```

运行：

```powershell
python prepare_data.py
```

**预期**：`images: 19860`，`data/raw/` 下有 19,860 张 jpg，`data/manifest.csv` 生成。

**作用**：把"文件名里的标签"变成训练能用的标签表（`manifest.csv` 里有 filename 和 6 类 condition）。

---

## 3. DVC + Azure 远程存储

```powershell
dvc init
dvc remote add -d myazure azure://dvc
dvc remote modify myazure account_name wut2026
dvc remote modify --local myazure sas_token "<课程给的 SAS token，保持 %2B/%3D 原样>"
```

**注意**：SAS token 必须用课程 PDF 里的**原始 URL 编码形式**（`%2B`、`%3D`），不要手动解码成 `+`/`=`，否则 Azure 会报 `AuthenticationFailed`。

验证能连上：

```powershell
dvc status --cloud
```

**预期**：列出待推送的文件（说明认证成功），而不是报错。

```powershell
dvc add data/raw
git add data/raw.dvc data/.gitignore data/manifest.csv .dvc .dvcignore
git commit -m "track v1 data with DVC"
dvc push
```

**预期**：`19861 files pushed`（19,860 张图 + 1 个目录记录）。

**作用**：图片不进 Git，只在 Git 里存一个很小的 `.dvc` 指针；以后能回答"哪个数据版本训出了这个模型"。

---

## 4. 生成 v2（类别平衡版）

新建 `ml/data_pipeline.py`（内容见仓库），它按类抽固定张数：

```powershell
.venv\Scripts\python.exe ml\data_pipeline.py --src data\raw --dst data\processed --manifest data\manifest.csv --per-class 820
```

**预期**：每类选 820 张，共 4920 张。

```powershell
dvc add data/processed
git add data/processed.dvc data/processed_manifest.csv ml/data_pipeline.py
git commit -m "add v2 balanced dataset"
dvc push
```

**作用**：v2 是"去偏"实验版；v1 是原始版。两个版本都留在 Azure。

---

## 5. 安装 Python 依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

**作用**：CUDA 版 torch 用于 GPU 训练（RTX 4070）；`requirements-dev.txt` 装 fastapi、mlflow、pytest、ruff 等。

---

## 6. 训练 + MLflow 记录

新建 `ml/train.py`（内容见仓库），然后跑两个实验：

```powershell
# baseline：ResNet18 + v1，5 轮
.venv\Scripts\python.exe ml\train.py --data-dir data/raw --backbone resnet18 --epochs 5 --run-name baseline-v1-resnet18

# improved：EfficientNet-B0 + v1，8 轮，lr 5e-4
.venv\Scripts\python.exe ml\train.py --data-dir data/raw --backbone efficientnet_b0 --epochs 8 --lr 5e-4 --run-name improved-v1-efficientnet_b0
```

**作用**：每次训练会往 MLflow（`mlflow.db`）写：git commit、DVC 数据哈希、超参、accuracy / macro-F1、模型权重、混淆矩阵。

看实验：

```powershell
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

打开 http://127.0.0.1:5000 对比 3 个 run。

选出生产模型（EfficientNet-B0 最好），复制成 `current.pt`：

```powershell
Copy-Item models\improved-v1-efficientnet_b0_best.pt models\current.pt -Force
```

**预期**：`models/current.pt` 生成；`ml/registry/current_model.md` 指向它。

---

## 7. App + 测试

新建 `app/main.py`、`tests/test_api.py`、`tests/test_model.py`（内容见仓库）。

启动服务：

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000/ 上传一张图即可看到预测。

或用命令测（另开终端）：

```powershell
.venv\Scripts\python.exe -c "import httpx, glob; f=open(glob.glob('data/processed/*dry-asphalt-smooth.jpg')[0],'rb'); print(httpx.post('http://127.0.0.1:8000/predict', files={'file':('r.jpg',f,'image/jpeg')}).json())"
```

**预期**：返回 `{'condition': 'dry', ...}`（对 dry 图的预测）。

跑测试和 lint：

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
```

**预期**：`11 passed`、`All checks passed!`、`12 files already formatted`。

---

## 8. Docker 镜像

```powershell
# 国内用课程 Azure 代理当基础镜像，避免 Docker Hub 慢
docker build --build-arg BASE_IMAGE=efreiwut.azurecr.io/dockerhub/library/python:3.12-slim -t road-surface:local .
docker run -d --name road-surface -p 127.0.0.1:8000:8000 road-surface:local
```

验证：

```powershell
.venv\Scripts\python.exe -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
```

**预期**：`{'status': 'ok'}`；镜像约 1.6GB（CPU torch + 只装推理依赖）。

---

## 9. GitHub 仓库 + CI/CD

```powershell
git remote add origin https://github.com/<你的用户名>/road_surface.git
git push -u origin main dev staging
```

**GitHub 设置**：

1. Settings → Secrets and variables → Actions → New repository secret：
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`（Docker Hub 的 Read & Write token）
2. Settings → Environments：建 `staging` 和 `production`

`.github/workflows/ci.yml` 已在仓库里（test + deploy-staging + deploy-prod）。

**看结果**：Actions 页面

- 开 PR 到 dev/staging/main → 跑 `Test, lint and build`
- push 到 staging → 跑 staging 部署
- push 到 main → 跑 production 部署

---

## 10. 验收清单

- [ ] `dvc status` 显示 `Data and pipelines are up to date`
- [ ] `pytest -q` → `11 passed`
- [ ] `ruff check .` → `All checks passed`
- [ ] 启动 uvicorn 后 `/predict` 返回合法类别
- [ ] `docker run` 后 `/health` 返回 ok
- [ ] GitHub Actions：test job 绿
- [ ] staging / main 的部署 job 绿
- [ ] Docker Hub 上能看到 `road-surface:latest` / `:staging` 镜像

---

## 附：常见坑

| 坑 | 现象 | 解决 |
|---|---|---|
| SAS token 解码 | Azure `AuthenticationFailed` | 用原始 `%2B`/`%3D` 编码 |
| Windows `num_workers>0` | `error code 1455` | 训练用 `--num-workers 0` |
| Linux torch 拉 CUDA | 镜像 10GB | 镜像里用 CPU 索引 + 推理依赖 |
| Docker Hub 拉基础镜像慢 | 卡在 `FROM` | 用 `--build-arg BASE_IMAGE=efreiwut...` |
| 测试导入路径 | `No module named 'APP'` | 保持小写 `app`（不是 `APP`） |
