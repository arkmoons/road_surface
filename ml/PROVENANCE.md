# 模型是从哪来的

## 模型

- 文件：`models/current.pt`
- 骨干：EfficientNet-B0（ImageNet 预训练 + 微调）
- MLflow run：`improved-v1-efficientnet_b0`（id `fadfc0188ea04a7c84459ba71abaf305`）
- 测试集：acc 0.881 / macro-F1 0.890

## 代码

- 训练脚本：`ml/train.py`，commit `7afc88e`（之后没再改）
- 每次训练会把 git SHA 记成 `code_version_git`

## 数据

- 版本：v1
- DVC md5：`226e861c295a314f530950af6879f8f3`
- 指针：`data/data_v1.csv.dvc`（commit `23ff9a7`）
- 每次训练会把哈希记成 `data_version_dvc`

## 训练设置

轮数 8，batch 32，lr 5e-4，图片 224，seed 42，GPU RTX 4070 Laptop。

## 重新训练

```bash
git checkout 23ff9a7 -- data/data_v1.csv.dvc
dvc pull
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 --epochs 8 --lr 5e-4 --batch-size 32 --seed 42
```
