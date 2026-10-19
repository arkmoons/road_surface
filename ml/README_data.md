# 数据

## 是什么

19,860 张路面照片（JPEG，240x360，总共约 299 MB）。标签在文件名里，例如：

    ...-dry-asphalt-smooth.jpg

时间戳后面的三段分别是 condition（状态）、material（材质）、roughness（粗糙度）。我只用第一段（condition）作为目标，它有 6 个值：

| condition | 数量 |
|---|---:|
| dry | 6560 |
| wet | 5420 |
| water | 5420 |
| fresh_snow | 820 |
| melted_snow | 820 |
| ice | 820 |

另外两级（材质、粗糙度）仍然留在 `manifest.csv` 里，但我不训练它们。

## 数据来源

是课程给的 `vali_20k.zip`。我自己没有从网站上下载。

## v1 和 v2

- **v1**（`data/raw/`）是原样的数据，19,860 张。它不平衡：`dry` 大约是 `ice` 的 8 倍。
- **v2**（`data/processed/`）是我用 `ml/data_pipeline.py` 做的平衡版。每类抽 820 张（小类本来就只有 820），所以一共 4920 张，每类 820。

v2 的想法是解决"dry 太多"的问题。结果 v2 反而比 v1 差（见 `README_experiments.md`），所以 v2 只作为一个实验保留。

## DVC

1.9 万张图片对 Git 来说太重了，所以用 DVC 管理，存在 Azure 上。每个版本在 Git 里有一个小的 `.dvc` 文件：

| 版本 | 目录 | DVC md5 | 图片数 | 提交它的 commit |
|---|---|---|---|---|
| v1 | `data/raw/` | `226e861c...` | 19,860 | `23ff9a7` |
| v2 | `data/processed/` | `e22aa7c0...` | 4,920 | `299588f` |

取回图片：

```bash
dvc pull
```

`git switch` 只是切换 checkout 的 `.dvc` 文件；真正下载对应图片的是 `dvc pull`。

## manifest.csv

`data/manifest.csv` 是 Git 里的小文件，记录每个文件名对应的标签。它不含图片，只有标签。
