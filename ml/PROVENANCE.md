# Which code and data made the final model

This note answers "where did this model come from?".

## The model

- file: `models/current.pt`
- backbone: EfficientNet-B0 (pretrained on ImageNet, then fine-tuned)
- MLflow run: `improved-v1-efficientnet_b0` (id `fadfc0188ea04a7c84459ba71abaf305`)
- test accuracy: 0.881, test macro-F1: 0.890
- also listed in `ml/registry/current_model.md`

## The code

- training script: `ml/train.py`
- git commit: `7afc88e`. I didn't change `ml/train.py` after that commit.
- every MLflow run stores the git SHA as `code_version_git`, so I could check
  this from the run too.

## The data

- version: v1 (raw)
- DVC md5: `226e861c295a314f530950af6879f8f3`
- pointer file: `data/raw.dvc` (first added in commit `23ff9a7`)
- every MLflow run stores this hash as `data_version_dvc`.

## Settings used

- epochs 8, batch size 32, learning rate 5e-4, image size 224, seed 42
- GPU: RTX 4070 Laptop

## How to train it again

```bash
git checkout 23ff9a7 -- data/raw.dvc   # the data version used
dvc pull

.venv\Scripts\python.exe ml/train.py --data-dir data/raw --backbone efficientnet_b0 \
    --epochs 8 --lr 5e-4 --batch-size 32 --seed 42 --run-name improved-v1-efficientnet_b0

cp models/improved-v1-efficientnet_b0_best.pt models/current.pt
```

The full history (git SHA, data hash, params, metrics) is in `mlflow.db`.
