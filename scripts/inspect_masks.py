"""CPU-only descriptive mask statistics; no concept/action labels or tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import yaml

# Raw IDs, not train IDs. Source: Cityscapes helpers/labels.py.
LABEL_NAMES = (
    "unlabeled", "ego vehicle", "rectification border", "out of roi",
    "static", "dynamic", "ground", "road", "sidewalk", "parking",
    "rail track", "building", "wall", "fence", "guard rail", "bridge",
    "tunnel", "pole", "polegroup", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
)
LABELS = dict(enumerate(LABEL_NAMES)) | {255: "encoded -1 / license plate"}
IGNORED_IDS = {0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 255}
FOCUS_IDS = [7, 24, 25, 26, 27, 28, 31, 32, 33, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = ROOT / cfg["manifest"]
    samples = pd.read_csv(manifest, dtype={"sample_id": str, "sequence_id": str})
    if samples.empty or samples.sample_id.duplicated().any():
        raise ValueError("Manifest must contain unique, nonempty sample records")
    rows = []
    for index, sample in enumerate(samples.itertuples(index=False), 1):
        path = (ROOT / sample.mask_path).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Mask path outside repository: {sample.mask_path}")
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (sample.width, sample.height):
                raise ValueError(f"Invalid raw label-ID mask: {path}")
            mask = np.asarray(image)
        unknown = set(np.unique(mask).tolist()) - LABELS.keys()
        if unknown:
            raise ValueError(f"Unexpected IDs in {path}: {unknown}")
        height, width = mask.shape
        # Illustrative region only, NOT a chosen road/proximity labeling region.
        regions = {"full_image": mask,
                   "exploratory_lower_center": mask[int(.50 * height):int(.90 * height),
                                                      int(.30 * width):int(.70 * width)]}
        for region_name, region in regions.items():
            counts = np.bincount(region.ravel(), minlength=256)
            for label_id, name in LABELS.items():
                rows.append({"sample_id": sample.sample_id,
                             "official_split": sample.official_split,
                             "sequence_id": sample.sequence_id,
                             "region": region_name, "label_id": label_id,
                             "class_name": name, "ignored_in_official_eval": label_id in IGNORED_IDS,
                             "pixel_count": int(counts[label_id]),
                             "region_pixels": region.size,
                             "pixel_fraction": float(counts[label_id] / region.size)})
        if index % 100 == 0 or index == len(samples):
            print(f"Inspected {index}/{len(samples)} masks", flush=True)
    output = ROOT / cfg["output_dir"] / "mask_statistics"
    output.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame(rows)
    stats.to_csv(output / "per_image_class_statistics.csv", index=False)
    summaries = []
    for split in ["all", *sorted(samples.official_split.unique())]:
        subset = stats if split == "all" else stats[stats.official_split == split]
        for (region, label_id, name), group in subset.groupby(["region", "label_id", "class_name"]):
            positive = group.loc[group.pixel_count > 0, "pixel_fraction"]
            summaries.append({"official_split": split, "region": region,
                              "label_id": label_id, "class_name": name,
                              "images_inspected": len(group),
                              "images_with_any_pixels": len(positive),
                              "percent_images_with_any_pixels": 100 * len(positive) / len(group),
                              "total_pixels": int(group.pixel_count.sum()),
                              "mean_pixel_fraction_all_images": group.pixel_fraction.mean(),
                              "median_pixel_fraction_when_present": positive.median(),
                              "max_pixel_fraction": group.pixel_fraction.max()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "class_summary.csv", index=False)
    sequences = samples.groupby(["official_split", "sequence_id"]).size().reset_index(name="image_count")
    sequences.to_csv(output / "sequence_summary.csv", index=False)
    focus = summary[(summary.official_split == "all") & (summary.region == "full_image")
                    & summary.label_id.isin(FOCUS_IDS)].set_index("label_id").loc[FOCUS_IDS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(focus.class_name, focus.percent_images_with_any_pixels)
    ax.set(xlabel="Images with at least one annotated pixel (%)", xlim=(0, 100),
           title="Annotation occurrence, not concept labels or object counts")
    fig.tight_layout()
    fig.savefig(output / "annotation_occurrence.png", dpi=150)
    plt.close(fig)
    report = {"images_inspected": len(samples),
              "unique_sequences": int(samples.sequence_id.nunique()),
              "official_split_counts": {str(k): int(v) for k, v in samples.official_split.value_counts().items()},
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "exploratory_region": {"x_fraction": [0.30, 0.70], "y_fraction": [0.50, 0.90],
                                     "bounds": "start inclusive, end exclusive, floor to integer"},
              "label_source": "https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py",
              "notes": ["Descriptive inspection of the entire subset; no experimental splits assigned.",
                        "Any pixels means pixel_count > 0; this is not a concept threshold.",
                        "No ignored pixels are removed from fractions; denominators include all region pixels.",
                        "Rider and bicycle occurrence does not establish rider-bicycle association.",
                        "Region is illustrative, not a road-clearance or distance estimate.",
                        "Freeze rules using development sequences, not future test examples."]}
    (output / "inspection_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(focus[["class_name", "images_with_any_pixels", "percent_images_with_any_pixels"]].to_string(index=False))
    print(f"Saved inspection outputs to {output}")


if __name__ == "__main__":
    main()
