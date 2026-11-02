"""Generate synthetic action targets and an offline training-only audit."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path
import sys

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_eligibility import digest, json_bytes, read_table, run as eligibility_report
from src.actions import ACTION_VERSION, concepts_to_action, validate_policy
from src.concepts import CONCEPTS


def run(root=ROOT):
    root = Path(root).resolve()
    folder = root / "data/manifests"
    paths = {"manifest": folder / "subset.csv", "concept_labels": folder / "concept_labels.csv",
             "concept_summary": folder / "concept_summary.json", "concept_rules": root / "configs/concepts.yaml",
             "eligibility": folder / "eligibility.csv", "eligibility_summary": folder / "eligibility_summary.json",
             "inspection_registry": folder / "inspection_registry.json", "action_policy": root / "configs/actions.yaml"}
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    validate_policy(yaml.safe_load(inputs["action_policy"]))
    expected = eligibility_report(root, write_outputs=False)
    saved = json.loads(inputs["eligibility_summary"])
    if saved != expected or digest(inputs["eligibility"]) != expected["output_hashes"]["eligibility.csv"]:
        raise ValueError("Stale or inconsistent eligibility metadata; rerun prepare_eligibility.py")
    eligibility = read_table(inputs["eligibility"], ("sample_id", "official_split", "sequence_id", "supervised_eligible", "directly_inspected"), "Eligibility")
    labels = {r["sample_id"]: r for r in read_table(inputs["concept_labels"], ("sample_id", "official_split", *CONCEPTS), "Concept labels")}
    manifest = {r["sample_id"]: r for r in read_table(inputs["manifest"], ("sample_id", "official_split", "image_path", "image_sha256"), "Manifest")}
    rows = []
    for entry in eligibility:
        if entry["supervised_eligible"] != "1":
            continue
        sid = entry["sample_id"]
        action, reason = concepts_to_action({c: int(labels[sid][c]) for c in CONCEPTS})
        rows.append({"sample_id": sid, "official_split": entry["official_split"], "sequence_id": entry["sequence_id"],
                     "concept_version": "concepts-v1.0", "eligibility_version": "eligibility-v1.0",
                     "action_version": ACTION_VERSION, "action": action, "reason": reason})
    if not rows:
        raise ValueError("No eligible action rows")
    by_id = {r["sample_id"]: r for r in rows}
    audit_ids = [sid for sid in json.loads(inputs["concept_summary"])["audit_sample_ids"] if sid in by_id]
    inspected = {r["sample_id"] for r in eligibility if r["directly_inspected"] == "1"}
    if not audit_ids or any(by_id[sid]["official_split"] != "train" or sid not in inspected for sid in audit_ids):
        raise ValueError("Audit must contain previously inspected eligible training examples")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()
    counts = {}
    for split in ("all", "train", "val"):
        selected = [r for r in rows if split == "all" or r["official_split"] == split]
        counts[split] = {action: {"count": sum(r["action"] == action for r in selected),
                                 "proportion": sum(r["action"] == action for r in selected) / len(selected) if selected else 0,
                                 "sequences": len({r["sequence_id"] for r in selected if r["action"] == action})}
                         for action in ("STOP", "SLOW", "GO")}
    # Use verified original images: the gallery never depends on ignored concept previews.
    assets, cards = {}, []
    for sid in audit_ids:
        source = manifest[sid]
        path = (root / source["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes repository: {sid}")
        data = path.read_bytes()
        if digest(data) != source["image_sha256"]:
            raise ValueError(f"Audit image hash mismatch: {sid}")
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        filename = f"{sid}.png"
        assets[filename] = data
        row = by_id[sid]
        concepts = "; ".join(f"{c}={labels[sid][c]}" for c in CONCEPTS)
        cards.append(f'<article><h2>{html.escape(sid)} — {row["action"]}</h2>'
                     f'<p>{html.escape(concepts)}</p><p>Reason: {html.escape(row["reason"])}</p>'
                     f'<img src="{filename}" alt="Original training image {html.escape(sid)}"></article>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Action target audit</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}article{margin:2rem 0;border-top:1px solid #ccc}img{width:100%}</style>'
            '<h1>Action target audit — actions-v1.0</h1><p>Synthetic coursework targets, not driving instructions. '
            'Previously inspected official-training examples only. STOP: close obstacle; otherwise SLOW: pedestrian, cyclist or unclear road; otherwise GO.</p>'
            + "\n".join(cards) + "</html>\n").encode()
    report = {"action_version": ACTION_VERSION, "sample_count": len(rows), "counts": counts,
              "audit_sample_ids": audit_ids, "audit_counts": {a: sum(by_id[s]["action"] == a for s in audit_ids) for a in ("STOP", "SLOW", "GO")},
              "input_hashes": {k: digest(v) for k, v in inputs.items()},
              "output_hashes": {"action_labels.csv": digest(csv_bytes), "audit/index.html": digest(page),
                                **{f"audit/{k}": digest(v) for k, v in assets.items()}},
              "limitations": "Synthetic policy agreement, not driving safety; vehicle_present alone has no action effect. Official validation counts are descriptive only."}
    # No outputs are written until all metadata and selected image assets pass checks.
    audit = root / "outputs/actions/audit"
    audit.mkdir(parents=True, exist_ok=True)
    for filename, data in assets.items():
        (audit / filename).write_bytes(data)
    (audit / "index.html").write_bytes(page)
    (folder / "action_labels.csv").write_bytes(csv_bytes)
    (folder / "action_summary.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = run()
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Action labeling failed: {exc}\n")
    print(json.dumps(report["counts"], indent=2))
    print("Audit: outputs/actions/audit/index.html")


if __name__ == "__main__":
    main()
