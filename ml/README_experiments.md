# Experiment Tracking (README_experiments.md)

All training runs are logged to MLflow (local SQLite backend `mlflow.db`). Every
run records the git commit, the DVC data hash, hyper-parameters, validation/test
metrics, per-class F1, the model weights and a confusion matrix.

## What metric we optimise

The dataset has 6 classes and is imbalanced (`dry` has 8× the images of `ice`).
Overall accuracy can look good while the model still fails on minority classes,
so we optimise **macro-F1** (the unweighted mean of per-class F1). It rewards a
model that is good on *every* class, not just the frequent ones. We still report
accuracy for context.

## Runs

| run | backbone | data | epochs | lr | test acc | test macro-F1 |
|---|---|---|---|---:|---:|---:|
| `baseline-v1-resnet18` | ResNet18 | v1 raw (imbalanced) | 5 | 1e-3 | 0.7649 | 0.7706 |
| `improved-v2-resnet18` | ResNet18 | v2 balanced (4,920) | 8 | 1e-3 | 0.7480 | 0.7430 |
| `improved-v1-efficientnet_b0` | EfficientNet-B0 | v1 raw | 8 | 5e-4 | **0.8812** | **0.8898** |

### Baseline — ResNet18 on v1

First usable model. With a fairly high learning rate the validation curve was
noisy (val acc 0.70 → 0.68 → 0.77 → 0.71 → 0.78 across the 5 epochs). Test:
acc 0.7649 / macro-F1 0.7706; class F1: dry 0.77, wet 0.69, water 0.82,
fresh_snow 0.82, melted_snow 0.87, **ice 0.66**.

### Balanced-data experiment — ResNet18 on v2

Trained on the class-balanced v2 (820/class). This **did** help the minority
classes (ice F1 0.66 → 0.78, melted_snow 0.87 → 0.94) but hurt the majority
classes (dry 0.77 → 0.58) and the overall score (0.748 / 0.743). The reason:
undersampling to 820/class leaves only 3,936 training images, so the model loses
too much data. This is why v2 was kept as a documented dataset version and an
honest ablation rather than the production model.

### Improved model — EfficientNet-B0 on v1

A stronger backbone, more epochs and a lower learning rate on the full v1 data.
Validation was stable and steadily improved (val acc 0.80 → 0.88, macro-F1
0.80 → 0.90). Test: **acc 0.8812 / macro-F1 0.8898**; class F1: dry 0.91,
wet 0.82, water 0.89, fresh_snow 0.92, melted_snow 0.95, ice 0.85.

## Production-model decision

**`improved-v1-efficientnet_b0`** (MLflow run `fadfc0188ea04a7c84459ba71abaf305`)
is the production model:

- best macro-F1 on the held-out test set (0.8898), best accuracy (0.8812);
- trained on the full dataset (no data thrown away);
- weights copied to `models/current.pt` and registered in
  [`ml/registry/current_model.md`](registry/current_model.md).

The balanced v2 remains a valid, versioned dataset (see `README_data.md`) and an
instructive negative result: *balancing by undersampling is not worth it here.*

## Reproduce / inspect

```bash
# the runs are in the local SQLite backend
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# reproduce the production model
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 \
    --epochs 8 --lr 5e-4 --run-name improved-v1-efficientnet_b0
```
