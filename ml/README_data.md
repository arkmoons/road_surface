# Dataset Documentation (README_data.md)

## 1. What the data is

Road-surface images used to train a 6-class surface-condition classifier.

- **Format**: JPEG, RGB, 240×360, ~299 MB total
- **Files**: 19,860 images
- **Labels**: encoded in the filename, three levels:
  `<timestamp>-<condition>-<material>-<roughness>.jpg`

The model target is the **top-level `condition`**, which has 6 classes:

| class | images | meaning |
|---|---:|---|
| `dry`         | 6560 | dry surface |
| `wet`         | 5420 | wet surface |
| `water`       | 5420 | standing water |
| `fresh_snow`  | 820  | fresh snow |
| `melted_snow` | 820  | melted snow |
| `ice`         | 820  | ice |

The second (`material`: asphalt / concrete / mud / gravel) and third
(`roughness`: smooth / slight / severe) levels are kept in `manifest.csv`
for reference but are **not** used as the training target.

## 2. Where the data comes from

Delivered as `vali_20k.zip` (course / instructor-provided dataset).

The raw archive is **not** committed to Git. Only small DVC pointer files are.

## 3. Dataset versions (DVC)

| version | directory | DVC md5 | files | commit |
|---|---|---|---:|---|
| v1 (raw) | `data/raw/` | `226e861c295a314f530950af6879f8f3` | 19,860 | `23ff9a7` |
| v2 (balanced) | `data/processed/` | `e22aa7c0702400f7303f28d2c9ff17ca` | 4,920 | `299588f` |

### What changed from v1 → v2

v1 is the raw, **imbalanced** dataset (`dry` has 8× the images of `ice`).
v2 is a **class-balanced** subset produced by `ml/data_pipeline.py`:
- classes larger than 820 images are downsampled to 820
- classes at/below 820 keep all their images
- result: exactly **820 images per class** (4920 total), seed = 42

The goal is to remove the majority-class bias that hurts macro-F1 on
`fresh_snow` / `melted_snow` / `ice`.

## 4. How to obtain the data

```bash
# requires Azure remote "myazure" configured in .dvc/config (+ local SAS token)
dvc pull
```

`dvc pull` restores `data/raw/` and `data/processed/` exactly as recorded by
the checked-out commit. `git switch` changes *which* version is referenced;
`dvc pull` materializes the corresponding image files.

## 5. Label manifest

`data/manifest.csv` (Git-tracked) maps every raw filename to its three label
levels. It has no image bytes — just the labels — so Git stores the mapping
while DVC stores the images.
