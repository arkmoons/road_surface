"""Build the class-balanced v2 dataset from the raw v1 images."""

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

# The 6 target classes we train on (top-level label in the filename).
CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]


def build_balanced(
    src_dir: Path,
    dst_dir: Path,
    manifest: Path,
    per_class: int,
    seed: int = 42,
) -> None:
    """Copy a fixed number of images per class into ``dst_dir``.

    Classes larger than ``per_class`` are downsampled; smaller classes keep
    all of their images. The result is a class-balanced version of the data.
    """
    df = pd.read_csv(manifest)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    selected: list[str] = []
    rng = random.Random(seed)
    for cls in CLASSES:
        cls_files = [
            f for f in df.loc[df["condition"] == cls, "filename"] if (src_dir / f).exists()
        ]
        if len(cls_files) > per_class:
            picked = rng.sample(cls_files, per_class)
        else:
            picked = cls_files
        for f in picked:
            shutil.copy2(src_dir / f, dst_dir / f)
        selected.extend(picked)
        print(f"  {cls:12s} selected {len(picked):>5} images")

    # Write a manifest for the derived version, next to the raw manifest.
    sel_set = set(selected)
    proc_df = df[df["filename"].isin(sel_set)].sort_values("filename")
    proc_manifest = manifest.parent / "processed_manifest.csv"
    proc_df.to_csv(proc_manifest, index=False)
    print(f"v2 total: {len(selected)} images -> {dst_dir}")
    print(f"manifest -> {proc_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dataset version.")
    parser.add_argument("--src", type=Path, default=Path("data/raw"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--per-class", type=int, default=820)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_balanced(args.src, args.dst, args.manifest, args.per_class, args.seed)


if __name__ == "__main__":
    main()
