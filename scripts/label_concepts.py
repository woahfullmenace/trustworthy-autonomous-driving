"""Generate concept CSV, diagnostics, split summaries and an offline training audit."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import yaml

from src.concepts import CONCEPTS, label_concepts, validate_rules
from scripts.prepare_data import COLORS


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "official_split", "image_path", "mask_path", "image_sha256", "mask_sha256", "width", "height"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Subset manifest lacks required columns")
        rows = list(reader)
    if any(None in row or any(not isinstance(row.get(key), str) or not row[key].strip() for key in required) for row in rows):
        raise ValueError("Manifest rows must provide nonempty values for every required column")
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest must contain nonempty, unique sample IDs")
    for row in rows:
        if not row["sample_id"] or row["official_split"] not in {"train", "val"}:
            raise ValueError("Only nonempty IDs and official train/val rows are supported")
        for key in ("width", "height"):
            if int(row[key]) <= 0:
                raise ValueError(f"Invalid {key}: {row['sample_id']}")
        for key in ("image_sha256", "mask_sha256"):
            if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError(f"Invalid {key}: {row['sample_id']}")
        for key in ("image_path", "mask_path"):
            target = (ROOT / row[key]).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                raise ValueError(f"Missing or out-of-repository {key}: {row[key]}")
        if not row["mask_path"].endswith("_gtFine_labelIds.png"):
            raise ValueError("Only original gtFine labelIds masks are supported")
    return rows


def read_verified(path, expected_hash):
    data = path.read_bytes()
    if sha256(data) != expected_hash:
        raise ValueError(f"File hash does not match the subset manifest: {path}")
    return data


def select_audit(records, count, seed):
    train = sorted((r for r in records if r["official_split"] == "train"), key=lambda r: r["sample_id"])
    if len(train) < count:
        raise ValueError(f"Requested {count} audit examples; only {len(train)} training rows exist")
    selected = []
    used = set()
    for concept in CONCEPTS:
        for value in (1, 0, None):
            example = next((r for r in train if r["labels"][concept] == value), None)
            if example is not None and example["sample_id"] not in used:
                selected.append(example)
                used.add(example["sample_id"])
    if len(selected) > count:
        raise ValueError("Audit count is too small to cover all available categories")
    selected.extend(random.Random(seed).sample([r for r in train if r["sample_id"] not in used], count - len(selected)))
    return selected


def summarize(records):
    summaries = {}
    for split in ("train", "val"):
        subset = [r for r in records if r["official_split"] == split]
        counts = {name: {"positive":sum(r["labels"][name] == 1 for r in subset),
                         "negative":sum(r["labels"][name] == 0 for r in subset),
                         "invalid":sum(r["labels"][name] is None for r in subset)} for name in CONCEPTS}
        cooccurrence = {a: {b: {"both_positive":sum(r["labels"][a] == 1 and r["labels"][b] == 1 for r in subset),
                                "both_valid":sum(r["labels"][a] is not None and r["labels"][b] is not None for r in subset)}
                            for b in CONCEPTS} for a in CONCEPTS}
        patterns = Counter("|".join("invalid" if r["labels"][c] is None else str(r["labels"][c]) for c in CONCEPTS) for r in subset)
        summaries[split] = {"rows":len(subset), "counts":counts, "cooccurrence":cooccurrence,
                            "concept_patterns":dict(sorted(patterns.items()))}
    return summaries


def flat_row(record):
    row = {k:record[k] for k in ("sample_id", "official_split", "rule_version")}
    row.update(record["labels"])
    row.update({f"{k}_valid":v for k,v in record["validity"].items()})
    row["audit_flags"] = ";".join(record["audit_flags"])
    for name in ("image_pixels", "presence_min_pixels", "person_pixels", "vehicle_pixels", "matched_cyclist_pair_count"):
        row[name] = record[name]
    row["qualifying_rider_components"] = len(record["riders"])
    row["qualifying_bicycle_components"] = len(record["bicycles"])
    row.update({f"class_{k}_pixels":v for k,v in record["class_pixel_counts"].items()})
    for name, region in record["regions"].items():
        row.update({f"{name}_{key}":value for key,value in region.items() if key != "box"})
    return row


def percent(value):
    return "invalid" if value is None else f"{100*value:.3f}%"


def render_audit(selected, manifest_by_id, output, rules):
    output.mkdir(parents=True, exist_ok=True)
    palette = np.zeros((256,3),dtype=np.uint8)
    for key,color in COLORS.items(): palette[key] = color
    cards = []
    for index, record in enumerate(selected, 1):
        row = manifest_by_id[record["sample_id"]]
        with Image.open(ROOT / row["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((1024,512)))
        with Image.open(ROOT / row["mask_path"]) as image:
            mask = np.asarray(image.resize((1024,512), Image.Resampling.NEAREST))
        fig, axes = plt.subplots(1,2,figsize=(16,5.8))
        axes[0].imshow(rgb)
        axes[1].imshow((0.6*rgb+0.4*palette[mask]).astype(np.uint8))
        scale_x, scale_y = 1024/int(row["width"]), 512/int(row["height"])
        for ax in axes:
            for name,color in (("road","#00ffff"),("near","#ffcc00")):
                x0,y0,x1,y1 = record["regions"][name]["box"]
                ax.add_patch(Rectangle((x0*scale_x,y0*scale_y),(x1-x0)*scale_x,(y1-y0)*scale_y,
                                       fill=False,edgecolor=color,linewidth=1.8))
            for kind,color,key in (("riders","#ff66ff","rider_id"),("bicycles","#00ff66","bicycle_id")):
                matched = {pair[key] for pair in record["cyclist_matches"]}
                for component in record[kind]:
                    if component["id"] in matched:
                        x0,y0,x1,y1=component["box"]
                        ax.add_patch(Rectangle((x0*scale_x,y0*scale_y),(x1-x0)*scale_x,(y1-y0)*scale_y,
                                               fill=False,edgecolor=color,linewidth=1.4))
            ax.axis("off")
        axes[0].set_title("Original + regions and matched components", fontsize=10)
        axes[1].set_title("Semantic overlay", fontsize=10)
        road,near=record["regions"]["road"],record["regions"]["near"]
        labels="  ".join(f"{c}={record['labels'][c] if record['labels'][c] is not None else 'INVALID'}" for c in CONCEPTS)
        measurements=(f"Person {record['person_pixels']}px; vehicle {record['vehicle_pixels']}px; minimum {record['presence_min_pixels']}px; cyclist pairs {record['matched_cyclist_pair_count']}\n"
                      f"Road ROI: valid {percent(road['valid_fraction'])}, road {percent(road['road_fraction'])}, obstacles {percent(road['obstacle_fraction'])}; "
                      f"near ROI: valid {percent(near['valid_fraction'])}, obstacles {percent(near['obstacle_fraction'])}")
        fig.suptitle(f"{index:02d}. {record['sample_id']} (official train)\n{labels}",fontsize=10)
        fig.text(.02,.015,measurements,fontsize=9)
        fig.tight_layout(rect=(0,.10,1,.90))
        filename=f"audit_{index:02d}.png"
        fig.savefig(output/filename,dpi=110)
        plt.close(fig)
        details = {"labels":record["labels"], "validity":record["validity"], "audit_flags":record["audit_flags"],
                   "regions":record["regions"], "riders":record["riders"], "bicycles":record["bicycles"],
                   "cyclist_matches":record["cyclist_matches"]}
        cards.append(f'<article><h2>{index:02d}. {html.escape(record["sample_id"])}</h2>'
                     f'<img src="{filename}" alt="Image and annotated mask for {html.escape(record["sample_id"])}">'
                     f'<p>{html.escape(labels)}</p><p>{html.escape(measurements)}</p>'
                     f'<p>Flags: {html.escape("; ".join(record["audit_flags"]) or "none")}</p>'
                     f'<details><summary>Component geometry and measurements</summary><pre>{html.escape(json.dumps(details,indent=2))}</pre></details></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept label audit</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1500px;padding:0 1rem;color:#202630;background:#f6f7fa}article{background:white;padding:1rem;margin:2rem 0;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}pre{overflow:auto}p{line-height:1.5}</style>
<h1>Concept label audit</h1><p>30 official-training examples. Cyan: road region; yellow: near region; magenta: matched rider; green: matched bicycle. Blank region labels are INVALID, not negative. Fixed regions and cyclist associations are heuristic proxies.</p>'''
    page+=f'<details><summary>Rules: {html.escape(rules["rule_version"])}</summary><pre>{html.escape(json.dumps(rules,indent=2))}</pre></details>'
    (output/"index.html").write_text(page+"\n".join(cards)+"</html>\n",encoding="utf-8")


def run(config_path, rules_path, labels_path, summary_path, output):
    config_bytes, rules_bytes = config_path.read_bytes(), rules_path.read_bytes()
    cfg, rules = yaml.safe_load(config_bytes), yaml.safe_load(rules_bytes)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("manifest"), str) or not cfg["manifest"].strip():
        raise ValueError("Dataset configuration requires a nonempty manifest path")
    validate_rules(rules)
    manifest_path=ROOT/cfg["manifest"]
    manifest_bytes=manifest_path.read_bytes()
    rows=read_manifest(manifest_path)
    records=[]
    for i,row in enumerate(rows,1):
        data=read_verified(ROOT/row["mask_path"],row["mask_sha256"])
        read_verified(ROOT/row["image_path"],row["image_sha256"])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.mode != "L" or image.size != (int(row["width"]),int(row["height"])):
                raise ValueError(f"Unexpected mask mode or dimensions: {row['sample_id']}")
            result=label_concepts(np.asarray(image),rules)
        records.append({"sample_id":row["sample_id"],"official_split":row["official_split"],**result})
        if i%100==0: print(f"Labeled and hash-verified {i}/{len(rows)} pairs",flush=True)
    selected=select_audit(records,rules["audit"]["count"],rules["audit"]["seed"])
    table=io.StringIO(newline="")
    flat=[flat_row(r) for r in records]
    writer=csv.DictWriter(table,fieldnames=list(flat[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    label_bytes=table.getvalue().encode("utf-8")
    summary={"rule_version":rules["rule_version"],"sample_count":len(records),"concept_order":list(CONCEPTS),
             "config_sha256":sha256(config_bytes),"rules_sha256":sha256(rules_bytes),"manifest_sha256":sha256(manifest_bytes),
             "labels_sha256":sha256(label_bytes),"source_hashes_verified":True,"splits":summarize(records),
             "audit_sample_ids":[r["sample_id"] for r in selected],
             "note":"Heuristic labels; official splits only. Audit uses training examples; no action labels or model training."}
    # Render before publishing CSV/summary so corrupt audit assets do not produce a success report.
    render_audit(selected,{r["sample_id"]:r for r in rows},output/"audit",rules)
    labels_path.parent.mkdir(parents=True,exist_ok=True)
    labels_path.write_bytes(label_bytes)
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"diagnostics.json").write_text(json.dumps(records,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v["counts"] for k,v in summary["splits"].items()},indent=2))
    print(f"Labels: {labels_path}\nAudit: {output/'audit/index.html'}")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/default.yaml")
    parser.add_argument("--rules",type=Path,default=ROOT/"configs/concepts.yaml")
    parser.add_argument("--labels",type=Path,default=ROOT/"data/manifests/concept_labels.csv")
    parser.add_argument("--summary",type=Path,default=ROOT/"data/manifests/concept_summary.json")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/concepts")
    args=parser.parse_args()
    try:
        run(args.config,args.rules,args.labels,args.summary,args.output)
    except (ValueError,KeyError,OSError,yaml.YAMLError) as exc:
        parser.exit(1,f"Concept labeling failed: {exc}\n")


if __name__=="__main__":
    main()
