# 最终模型是用哪份代码、哪份数据训出来的

这份说明回答"这个模型从哪来"。

## 模型

- 文件：`models/current.pt`
- 骨干网络：EfficientNet-B0（先在 ImageNet 上预训练，再微调）
- MLflow run：`improved-v1-efficientnet_b0`（id `fadfc0188ea04a7c84459ba71abaf305`）
- 测试集准确率：0.881，测试集 macro-F1：0.890
- 也记录在 `ml/registry/current_model.md`

## 代码

- 训练脚本：`ml/train.py`
- git commit：`7afc88e`。之后我没有再改 `ml/train.py`。
- 每次 MLflow 训练都会把 git SHA 记成 `code_version_git`，所以也能从 run 里核对。

## 数据

- 版本：v1（原始）
- DVC md5：`226e861c295a314f530950af6879f8f3`
- 指针文件：`data/data_v1.csv.dvc`（首次在 commit `23ff9a7` 加入）
- 每次 MLflow 训练都会把这个哈希记成 `data_version_dvc`。

## 使用的设置

- 轮数 8，batch 32，学习率 5e-4，图片尺寸 224，随机种子 42
- GPU：RTX 4070 Laptop

## 怎么重新训练

```bash
git checkout 23ff9a7 -- data/data_v1.csv.dvc   # 用当时的数据版本
dvc pull

.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 \
    --epochs 8 --lr 5e-4 --batch-size 32 --seed 42 --run-name improved-v1-efficientnet_b0

cp models/improved-v1-efficientnet_b0_best.pt models/current.pt
```

完整历史（git SHA、数据哈希、参数、指标）都在 `mlflow.db` 里。
