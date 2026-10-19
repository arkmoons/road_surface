# 数据说明

19,860 张路面照片（240x360，约 299MB）。标签在文件名里：

```
...-dry-asphalt-smooth.jpg
```

按 `-` 拆开，时间戳后面是 condition、material、roughness，我只用 condition 当标签。6 类：dry 6560、wet 5420、water 5420、fresh_snow 820、melted_snow 820、ice 820。

数据是课程给的 `vali_20k.zip`，我没自己去网站下。

## 两个版本

- v1（`data/raw/`）：原样的，19,860 张，不太平衡（dry 差不多是 ice 的 8 倍）。
- v2（`data/processed/`）：我用 `ml/data_pipeline.py` 每类抽 820 张做的平衡版，共 4920 张。本来想解决不平衡，结果反而更差，所以只当实验留着。

## DVC

图太多太重，用 DVC 管，实际存在 Azure。Git 里只有 `.dvc` 指针文件：

| 版本 | 目录 | md5 | 图数 |
|---|---|---|---|
| v1 | data/raw/ | 226e861c... | 19,860 |
| v2 | data/processed/ | e22aa7c0... | 4,920 |

要图就 `dvc pull`。

## manifest.csv

`data/manifest.csv` 记录每个文件名对应的标签，不含图片。
