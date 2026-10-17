# Model Provenance (PROVENANCE.md)

This file links the **production model** back to the exact code and data that
produced it, so "which code + which dataset produced this model?" always has a
concrete answer.

## Production model

| field | value |
|---|---|
| model | `models/current.pt` |
| backbone | EfficientNet-B0 (ImageNet pretrained, fine-tuned) |
| MLflow run id | `fadfc0188ea04a7c84459ba71abaf305` |
| MLflow run name | `improved-v1-efficientnet_b0` |
| test accuracy | 0.8812 |
| test macro-F1 | 0.8898 |
| registry pointer | `ml/registry/current_model.md` |

## Code version

- training script: [`ml/train.py`](train.py)
- git commit used for the run: `7afc88e` ("feat: add FastAPI app, tests,
  Docker, CI/CD, docs"); the training script was unchanged after that commit.
- `ml/train.py` logs the git SHA into every MLflow run as `code_version_git`.

## Data version

- dataset version: **v1 (raw)**
- DVC md5: `226e861c295a314f530950af6879f8f3`
- DVC pointer: `data/raw.dvc` (first committed at `23ff9a7`)
- `ml/train.py` logs this hash into every MLflow run as `data_version_dvc`.

## Hyper-parameters (as logged in MLflow)

| param | value |
|---|---|
| epochs | 8 |
| batch_size | 32 |
| learning_rate | 5e-4 |
| img_size | 224 |
| seed | 42 |
| device | cuda (RTX 4070 Laptop) |

## Reproduce exactly

```bash
# 1. get the exact data version
git checkout 23ff9a7 -- data/raw.dvc    # or stay on the current commit
dvc pull

# 2. train with the recorded hyper-parameters
.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 \
    --epochs 8 --lr 5e-4 --batch-size 32 --seed 42 --run-name improved-v1-efficientnet_b0

# 3. promote the produced weights
cp models/improved-v1-efficientnet_b0_best.pt models/current.pt
```

The MLflow run history in `mlflow.db` is the authoritative record of every
training run (code SHA, data hash, params, metrics, artifacts).
