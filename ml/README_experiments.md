# 训练实验

每次训练都记在 MLflow（本地 `mlflow.db`），记录 git commit、数据哈希、超参数、指标、权重、混淆矩阵。

## 为什么看 macro-F1

6 类不均衡，准确率容易虚高，小类（尤其 ice）可能很差。macro-F1 对每类一视同仁，更能反映真实水平。

## 三次实验

| 实验 | 骨干 | 数据 | acc | macro-F1 |
|---|---|---|---|---|
| baseline-v1-resnet18 | ResNet18 | v1 | 0.765 | 0.771 |
| improved-v2-resnet18 | ResNet18 | v2 | 0.748 | 0.743 |
| improved-v1-efficientnet_b0 | EfficientNet-B0 | v1 | 0.881 | 0.890 |

1. 基线（ResNet18 + v1）：第一个能用的，lr 1e-3 有点高，验证分数来回跳。ice 最差（F1 0.66）。
2. 平衡数据（ResNet18 + v2）：小类变好（ice 0.66→0.78），但大类变差（dry 0.77→0.58），整体下降。原因是每类只剩 820 张、训练数据太少。
3. 改进（EfficientNet-B0 + v1）：更强模型 + 更多轮 + 更低学习率，acc 0.881、macro-F1 0.890，每类都在 0.82 以上。

## 最后用哪个

`improved-v1-efficientnet_b0`。macro-F1 和准确率都最好，而且用全量数据。权重复制成了 `models/current.pt`，登记在 `ml/registry/current_model.md`。

## 看实验

```bash
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
