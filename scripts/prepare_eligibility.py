"""Validate concept metadata and preserve eligibility and inspection history."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.concepts import CONCEPTS

POLICY_VERSION = "eligibility-v1.0"
RULE_VERSION = "concepts-v1.0"
REGISTRY_VERSION = "inspection-v1.0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_table(data, required, name):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(required) <= set(fields):
        raise ValueError(f"{name}: missing or duplicate columns")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{name}: empty table")
    ids = set()
    for row in rows:
        if None in row or any(not isinstance(row.get(k), str) for k in required):
            raise ValueError(f"{name}: malformed CSV row")
        sid = row["sample_id"]
        if not sid.strip() or sid in ids:
            raise ValueError(f"{name}: empty or duplicate sample ID {sid!r}")
        ids.add(sid)
        if row["official_split"] not in {"train", "val"}:
            raise ValueError(f"{name}: invalid official split for {sid}")
    return rows


def check_hash(actual_bytes, expected, name):
    if digest(actual_bytes) != expected:
        raise ValueError(f"{name}: SHA-256 mismatch; restore the matching source metadata")


def check_ids(value, name, lookup):
    if not isinstance(value, list) or not value or any(not isinstance(s, str) for s in value):
        raise ValueError(f"{name}: expected a nonempty list of sample IDs")
    if len(value) != len(set(value)) or not set(value) <= set(lookup):
        raise ValueError(f"{name}: duplicate or unresolved inspection IDs")
    return set(value)


def make_registry(lookup, summary, preview, manifest_hash, summary_hash, preview_hash):
    groups = {"concept_audit": check_ids(summary.get("audit_sample_ids"), "Concept audit", lookup),
              "setup_preview": check_ids(preview.get("preview_sample_ids"), "Setup preview", lookup)}
    return {
        "registry_version": REGISTRY_VERSION,
        "manifest_sha256": manifest_hash,
        "sources": {name: {"report_sha256": summary_hash if name == "concept_audit" else preview_hash,
                           "sample_ids": sorted(ids)} for name, ids in groups.items()},
        "inspections": [{"sample_id": sid, "official_split": lookup[sid]["official_split"],
                         "sequence_id": lookup[sid]["sequence_id"],
                         "sources": sorted(name for name, ids in groups.items() if sid in ids)}
                        for sid in sorted(set.union(*groups.values()))],
    }


def validate_registry(registry, lookup, summary, manifest_hash, summary_hash):
    if registry.get("registry_version") != REGISTRY_VERSION or registry.get("manifest_sha256") != manifest_hash:
        raise ValueError("Inspection registry version or manifest hash does not match")
    sources = registry.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"concept_audit", "setup_preview"}:
        raise ValueError("Inspection registry must preserve both source reports")
    groups = {}
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise ValueError("Malformed registry source")
        checksum = source.get("report_sha256")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid inspection source hash")
        groups[name] = check_ids(source.get("sample_ids"), name, lookup)
    audit_ids = check_ids(summary.get("audit_sample_ids"), "Concept audit", lookup)
    if groups["concept_audit"] != audit_ids or sources["concept_audit"]["report_sha256"] != summary_hash:
        raise ValueError("Inspection registry does not match the concept-audit summary")
    if any(lookup[sid]["official_split"] != "train" for sid in audit_ids):
        raise ValueError("Concept audit must contain official-training images only")
    records = registry.get("inspections")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Malformed inspection registry records")
    seen = set()
    for record in records:
        sid = record.get("sample_id")
        if not isinstance(sid, str) or sid not in lookup or sid in seen:
            raise ValueError("Duplicate or unresolved inspection record")
        seen.add(sid)
        if any(record.get(k) != lookup[sid][k] for k in ("official_split", "sequence_id")):
            raise ValueError(f"Inspection split or sequence mismatch: {sid}")
        expected_sources = sorted(name for name, ids in groups.items() if sid in ids)
        if not expected_sources or record.get("sources") != expected_sources:
            raise ValueError(f"Inspection sources do not match: {sid}")
    if seen != set.union(*groups.values()):
        raise ValueError("Inspection registry is missing source-report sample IDs")
    return seen, {lookup[sid]["sequence_id"] for sid in seen}


def build_rows(manifest, labels, inspected_ids, inspected_sequences):
    by_id = {r["sample_id"]: r for r in labels}
    if len(by_id) != len(labels) or set(by_id) != {r["sample_id"] for r in manifest}:
        raise ValueError("Manifest and labels must have identical, unique sample IDs")
    result = []
    for source in manifest:
        sid = source["sample_id"]
        label = by_id[sid]
        if label["official_split"] != source["official_split"]:
            raise ValueError(f"Original split mismatch: {sid}")
        if label["rule_version"] != RULE_VERSION:
            raise ValueError(f"Unsupported concept rule version: {sid}")
        invalid = []
        for concept in CONCEPTS:
            value, validity = label[concept], label[f"{concept}_valid"]
            if value == "" and validity == "0":
                invalid.append(concept)
            elif value not in {"0", "1"} or validity != "1":
                raise ValueError(f"Contradictory label/validity for {sid}: {concept}")
        result.append({"sample_id": sid, "official_split": source["official_split"],
                       "sequence_id": source["sequence_id"], "rule_version": RULE_VERSION,
                       "policy_version": POLICY_VERSION, "supervised_eligible": int(not invalid),
                       "exclusion_reason": "invalid_concepts:" + ";".join(invalid) if invalid else "",
                       "directly_inspected": int(sid in inspected_ids),
                       "inspected_sequence": int(source["sequence_id"] in inspected_sequences)})
    return result


def summarize(rows, labels):
    label_by_id = {r["sample_id"]: r for r in labels}
    summary = {}
    for split in ("all", "train", "val"):
        selected = [r for r in rows if split == "all" or r["official_split"] == split]
        eligible = [r for r in selected if r["supervised_eligible"]]
        summary[split] = {
            "total": len(selected), "eligible": len(eligible), "excluded": len(selected) - len(eligible),
            "eligible_concept_counts": {c: {"positive": sum(label_by_id[r["sample_id"]][c] == "1" for r in eligible),
                                            "negative": sum(label_by_id[r["sample_id"]][c] == "0" for r in eligible)} for c in CONCEPTS},
            "exclusion_reasons": dict(sorted(Counter(r["exclusion_reason"] for r in selected if not r["supervised_eligible"]).items())),
            "directly_inspected": sum(r["directly_inspected"] for r in selected),
            "inspected_sequences": len({r["sequence_id"] for r in selected if r["inspected_sequence"]}),
            "rows_in_inspected_sequences": sum(r["inspected_sequence"] for r in selected),
        }
    return summary


def run(root=ROOT, output_dir=None, *, write_outputs=True):
    root = Path(root).resolve()
    folder = root / "data/manifests"
    destination = Path(output_dir).resolve() if output_dir else folder
    paths = {"manifest": folder / "subset.csv", "labels": folder / "concept_labels.csv",
             "concept_summary": folder / "concept_summary.json", "rules": root / "configs/concepts.yaml"}
    inputs = {k: p.read_bytes() for k, p in paths.items()}
    summary = json.loads(inputs["concept_summary"])
    rules = yaml.safe_load(inputs["rules"])
    if not isinstance(summary, dict) or not isinstance(rules, dict):
        raise ValueError("Concept summary and rules must be mappings")
    for name, field in (("manifest", "manifest_sha256"), ("labels", "labels_sha256"), ("rules", "rules_sha256")):
        check_hash(inputs[name], summary.get(field), name)
    if summary.get("rule_version") != RULE_VERSION or rules.get("rule_version") != RULE_VERSION:
        raise ValueError(f"This baseline policy requires {RULE_VERSION}")
    if summary.get("concept_order") != list(CONCEPTS):
        raise ValueError("Concept order does not match the five expected concepts")
    manifest = read_table(inputs["manifest"], ("sample_id", "official_split", "sequence_id"), "Manifest")
    for row in manifest:
        expected_sequence = "_".join(row["sample_id"].split("_")[:-1])
        if not row["sequence_id"].strip() or row["sequence_id"] != expected_sequence:
            raise ValueError(f"Sequence does not match sample ID: {row['sample_id']}")
    columns = ("sample_id", "official_split", "rule_version", *CONCEPTS, *(f"{c}_valid" for c in CONCEPTS))
    labels = read_table(inputs["labels"], columns, "Concept labels")
    if summary.get("sample_count") != len(manifest):
        raise ValueError("Concept summary sample count does not match the manifest")
    lookup = {r["sample_id"]: r for r in manifest}
    registry_path = folder / "inspection_registry.json"
    registry_exists = registry_path.exists()
    if registry_exists:
        registry_bytes = registry_path.read_bytes()
        registry = json.loads(registry_bytes)
    else:
        preview_path = root / "outputs/data_verification.json"
        if not preview_path.is_file():
            raise ValueError("Inspection registry is absent and outputs/data_verification.json is missing. Restore the original inspected-preview report; do not generate substitute inspection history.")
        preview_bytes = preview_path.read_bytes()
        preview = json.loads(preview_bytes)
        if not isinstance(preview, dict):
            raise ValueError("Setup preview report must be a mapping")
        check_hash(inputs["manifest"], preview.get("manifest_sha256"), "Setup preview manifest")
        registry = make_registry(lookup, summary, preview, digest(inputs["manifest"]), digest(inputs["concept_summary"]), digest(preview_bytes))
        registry_bytes = json_bytes(registry)
    if not isinstance(registry, dict):
        raise ValueError("Inspection registry must be a mapping")
    inspected, sequences = validate_registry(registry, lookup, summary, digest(inputs["manifest"]), digest(inputs["concept_summary"]))
    rows = build_rows(manifest, labels, inspected, sequences)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    eligibility_bytes = buffer.getvalue().encode("utf-8")
    report = {"policy_version": POLICY_VERSION, "rule_version": RULE_VERSION,
              "input_hashes": {**{k: digest(v) for k, v in inputs.items()}, "inspection_registry": digest(registry_bytes)},
              "output_hashes": {"eligibility.csv": digest(eligibility_bytes)},
              "counts": summarize(rows, labels),
              "inspection_sources": {k: len(v["sample_ids"]) for k, v in registry["sources"].items()},
              "future_test_policy": "Exclude every inspected sequence from an unseen test set; official splits are not experimental assignments."}
    # All metadata is checked before any output is written. Inputs and raw files are read-only.
    if not write_outputs:
        return report
    destination.mkdir(parents=True, exist_ok=True)
    if not registry_exists:
        registry_path.write_bytes(registry_bytes)
    (destination / "eligibility.csv").write_bytes(eligibility_bytes)
    (destination / "eligibility_summary.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Alternate eligibility CSV/summary destination; registry stays in data/manifests")
    args = parser.parse_args()
    try:
        report = run(output_dir=args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Eligibility preparation failed: {exc}\n")
    print(json.dumps(report["counts"], indent=2))
    print("Inspection history preserved in data/manifests/inspection_registry.json")


if __name__ == "__main__":
    main()
