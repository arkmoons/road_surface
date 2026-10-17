# My training runs

I logged every run to MLflow (the local `mlflow.db`). Each run stores the git
commit, the DVC data hash, the hyper-parameters, the accuracy and macro-F1, the
model weights and a confusion matrix.

## Why macro-F1?

The 6 classes are not balanced (`dry` has about 8x the images of `ice`).
Accuracy can look high while the model is still bad on the small classes
(`ice` especially), so the number I care about is macro-F1, which averages every
class equally. I still write accuracy too.

## The runs

| run | backbone | data | epochs | lr | acc | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| baseline-v1-resnet18 | ResNet18 | v1 | 5 | 1e-3 | 0.765 | 0.771 |
| improved-v2-resnet18 | ResNet18 | v2 | 8 | 1e-3 | 0.748 | 0.743 |
| improved-v1-efficientnet_b0 | EfficientNet-B0 | v1 | 8 | 5e-4 | 0.881 | 0.890 |

### 1. Baseline (ResNet18 on v1)

First model that worked. Learning rate 1e-3 was a bit high, so the validation
score jumped around (0.70, 0.68, 0.77, 0.71, 0.78). Test: acc 0.765,
macro-F1 0.771. `ice` was the worst class (F1 0.66).

### 2. Balanced data (ResNet18 on v2)

On the balanced v2 the small classes got better (`ice` 0.66 -> 0.78,
`melted_snow` 0.87 -> 0.94) but the big classes got worse (`dry` 0.77 -> 0.58).
The overall score went down (0.748 / 0.743). The reason: taking only 820 per
class left just 3936 training images, so there was not enough data. That's why
I didn't use v2 for the final model.

### 3. Improved (EfficientNet-B0 on v1)

A stronger pretrained model, more epochs and a lower learning rate, on the full
v1 data. The validation score was stable and went up (0.80 -> 0.88). Test:
acc 0.881, macro-F1 0.890. Every class is now 0.82 or above.

## Which model I used

I used `improved-v1-efficientnet_b0` (run `fadfc0188ea04a7c84459ba71abaf305`).
Reasons: it has the best macro-F1 and accuracy, and it trained on the full
dataset instead of a small balanced copy. I copied its weights to
`models/current.pt` and noted it in `ml/registry/current_model.md`.

## How to look at the runs

```bash
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
