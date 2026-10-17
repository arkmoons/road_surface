# The data

## What it is

19,860 road photos (JPEG, 240x360, about 299 MB in total). The label is in the
file name, for example:

    ...-dry-asphalt-smooth.jpg

The three parts after the timestamp are condition, material and roughness.
I only use the first one (condition) as the target. It has 6 values:

| condition | how many |
|---|---:|
| dry | 6560 |
| wet | 5420 |
| water | 5420 |
| fresh_snow | 820 |
| melted_snow | 820 |
| ice | 820 |

The other two levels (material, roughness) are still in `manifest.csv` but I
don't train on them.

## Where it came from

It was the `vali_20k.zip` file the course gave us. I didn't download it from a
website myself.

## v1 and v2

- **v1** (`data/raw/`) is the data as it was, 19,860 images. It is imbalanced:
  `dry` has about 8 times more images than `ice`.
- **v2** (`data/processed/`) is a balanced copy I made with
  `ml/data_pipeline.py`. I took 820 images from every class (the small classes
  already have 820), so it has 4920 images, 820 per class.

The idea of v2 was to fix the "too much dry" problem. In the end v2 was worse
than v1 (see `README_experiments.md`), so I kept v2 only as an experiment.

## DVC

19k images are too heavy for Git, so the images are tracked with DVC and stored
on Azure. Each version has a small `.dvc` file in Git:

| version | folder | DVC md5 | images | commit that added it |
|---|---|---|---|---|
| v1 | `data/raw/` | `226e861c...` | 19,860 | `23ff9a7` |
| v2 | `data/processed/` | `e22aa7c0...` | 4,920 | `299588f` |

To get the images:

```bash
dvc pull
```

`git switch` only changes which `.dvc` file is checked out; `dvc pull` is what
actually downloads the matching images.

## manifest.csv

`data/manifest.csv` is a small file in Git that maps every file name to its
labels. It has no images, only the labels.
