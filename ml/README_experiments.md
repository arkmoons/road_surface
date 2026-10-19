# 我的训练实验

我把每次训练都记到 MLflow（本地 `mlflow.db`）。每次记录：git commit、DVC 数据哈希、超参数、accuracy 和 macro-F1、模型权重、混淆矩阵。

## 为什么看 macro-F1？

6 个类别不平衡（`dry` 大约是 `ice` 的 8 倍）。准确率可能看起来很高，但模型在小类上仍然很差（尤其是 `ice`），所以我在意的是 macro-F1——它对每个类一视同仁地平均。当然我也记准确率。

## 实验

| 实验 | 骨干网络 | 数据 | 轮数 | 学习率 | acc | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| baseline-v1-resnet18 | ResNet18 | v1 | 5 | 1e-3 | 0.765 | 0.771 |
| improved-v2-resnet18 | ResNet18 | v2 | 8 | 1e-3 | 0.748 | 0.743 |
| improved-v1-efficientnet_b0 | EfficientNet-B0 | v1 | 8 | 5e-4 | 0.881 | 0.890 |

### 1. 基线（ResNet18 + v1）

第一个能用的模型。学习率 1e-3 有点高，验证分数跳来跳去（0.70、0.68、0.77、0.71、0.78）。测试集：acc 0.765，macro-F1 0.771。`ice` 是最差的类（F1 0.66）。

### 2. 平衡数据（ResNet18 + v2）

在平衡的 v2 上，小类变好了（`ice` 0.66 → 0.78，`melted_snow` 0.87 → 0.94），但大类变差了（`dry` 0.77 → 0.58）。整体分数下降（0.748 / 0.743）。原因：每类只抽 820 张，训练集只剩 3936 张，数据不够。所以最终模型没有用 v2。

### 3. 改进（EfficientNet-B0 + v1）

更强的预训练模型、更多轮数、更低学习率，用完整的 v1 数据。验证分数很稳定且一路上升（0.80 → 0.88）。测试集：acc 0.881，macro-F1 0.890。每一类都在 0.82 以上。

## 用了哪个模型

用的是 `improved-v1-efficientnet_b0`（run `fadfc0188ea04a7c84459ba71abaf305`）。理由：它的 macro-F1 和准确率都最好，而且用的是完整数据集、不是小的平衡子集。我把它的权重复制到 `models/current.pt`，并在 `ml/registry/current_model.md` 里登记。

## 怎么查看实验

```bash
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
