"""Freeze sequence-grouped pilot splits without reading test images."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_eligibility import digest, json_bytes, read_table, run as eligibility_report
from src.actions import concepts_to_action, validate_policy
from src.concepts import CONCEPTS

SPLITS = ("train", "validation", "test")
ACTIONS = ("STOP", "SLOW", "GO")
VERSION = "splits-v1.0"


def validate_config(cfg):
    expected = {"policy_version": VERSION, "seed": 42, "targets": [0.7, 0.15, 0.15],
                "image_bounds": [[310, 370], [60, 90], [60, 90]],
                "action_minimums": {"STOP": [10, 3, 3], "SLOW": [1, 1, 1], "GO": [20, 5, 5]},
                "time_limit_seconds": 60, "primary_tolerance": 1e-7}
    if cfg != expected:
        raise ValueError("Configuration differs from frozen splits-v1.0; use a separately versioned split decision")


def load_inputs(root):
    folder = root / "data/manifests"
    paths = {"manifest": folder / "subset.csv", "concept_labels": folder / "concept_labels.csv",
             "concept_summary": folder / "concept_summary.json", "concept_rules": root / "configs/concepts.yaml",
             "eligibility": folder / "eligibility.csv", "eligibility_summary": folder / "eligibility_summary.json",
             "inspection_registry": folder / "inspection_registry.json", "action_policy": root / "configs/actions.yaml",
             "action_labels": folder / "action_labels.csv", "action_summary": folder / "action_summary.json",
             "split_policy": root / "configs/splits.yaml"}
    data = {k: p.read_bytes() for k, p in paths.items()}
    hashes = {k: digest(v) for k, v in data.items()}
    cfg = yaml.safe_load(data["split_policy"])
    validate_config(cfg)
    validate_policy(yaml.safe_load(data["action_policy"]))
    expected = eligibility_report(root, write_outputs=False)
    if json.loads(data["eligibility_summary"]) != expected or hashes["eligibility"] != expected["output_hashes"]["eligibility.csv"]:
        raise ValueError("Stale or inconsistent eligibility provenance")
    action_summary = json.loads(data["action_summary"])
    source_keys = ("manifest", "concept_labels", "concept_summary", "concept_rules", "eligibility", "eligibility_summary", "inspection_registry", "action_policy")
    if action_summary.get("input_hashes") != {k: hashes[k] for k in source_keys}:
        raise ValueError("Stale action input provenance")
    if action_summary.get("output_hashes", {}).get("action_labels.csv") != hashes["action_labels"]:
        raise ValueError("Action label hash mismatch")
    eligibility = read_table(data["eligibility"], ("sample_id", "official_split", "sequence_id", "supervised_eligible", "inspected_sequence"), "Eligibility")
    eligible = [r for r in eligibility if r["supervised_eligible"] == "1"]
    labels = {r["sample_id"]: r for r in read_table(data["concept_labels"], ("sample_id", "official_split", *CONCEPTS), "Concept labels")}
    actions = read_table(data["action_labels"], ("sample_id", "official_split", "sequence_id", "concept_version", "eligibility_version", "action_version", "action", "reason"), "Action labels")
    if [r["sample_id"] for r in actions] != [r["sample_id"] for r in eligible]:
        raise ValueError("Action IDs/order do not match eligible manifest IDs")
    for row, entry in zip(actions, eligible):
        if any(row[k] != entry[k] for k in ("official_split", "sequence_id")):
            raise ValueError("Action split or sequence mismatch")
        if (row["concept_version"], row["eligibility_version"], row["action_version"]) != ("concepts-v1.0", "eligibility-v1.0", "actions-v1.0"):
            raise ValueError("Action version mismatch")
        if (row["action"], row["reason"]) != concepts_to_action({c: int(labels[row["sample_id"]][c]) for c in CONCEPTS}):
            raise ValueError("Action disagrees with frozen concept policy")
    if action_summary.get("sample_count") != len(actions) or action_summary.get("action_version") != "actions-v1.0":
        raise ValueError("Action summary version/count mismatch")
    for split in ("all", "train", "val"):
        selected = [r for r in actions if split == "all" or r["official_split"] == split]
        counts = {a: {"count": sum(r["action"] == a for r in selected),
                      "proportion": sum(r["action"] == a for r in selected) / len(selected) if selected else 0,
                      "sequences": len({r["sequence_id"] for r in selected if r["action"] == a})} for a in ACTIONS}
        if action_summary.get("counts", {}).get(split) != counts:
            raise ValueError("Action summary counts mismatch")
    audit_ids = [s for s in json.loads(data["concept_summary"])["audit_sample_ids"] if s in {r["sample_id"] for r in actions}]
    if action_summary.get("audit_sample_ids") != audit_ids:
        raise ValueError("Action audit selection mismatch")
    locked = {r["sequence_id"] for r in eligibility if r["inspected_sequence"] == "1"}
    return actions, labels, locked, cfg, hashes


def optimize(actions, locked, cfg):
    sequences = sorted({r["sequence_id"] for r in actions})
    n = len(sequences)
    features = np.array([[sum(r["sequence_id"] == s and (a == "all" or r["action"] == a) for r in actions)
                          for s in sequences] for a in ("all", *ACTIONS)], dtype=float)
    totals = features.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("Infeasible split: an action class is absent")
    # Three assignment binaries per sequence, followed by 12 absolute-deviation variables.
    size = n * 3 + 12
    matrix, lower, upper = [], [], []
    def add(v, low, high):
        matrix.append(v); lower.append(low); upper.append(high)
    for i in range(n):
        v = np.zeros(size); v[3*i:3*i+3] = 1
        add(v, 1, 1)
    primary = np.zeros(size)
    for j in range(3):
        for f in range(4):
            v = np.zeros(size); v[j:n*3:3] = features[f]
            low, high = cfg["image_bounds"][j] if f == 0 else (cfg["action_minimums"][ACTIONS[f-1]][j], np.inf)
            add(v.copy(), low, high)
            target = totals[f] * cfg["targets"][j]
            d = n*3 + j*4 + f
            v[d] = -1; add(v.copy(), -np.inf, target)
            v[d] = 1; add(v.copy(), target, np.inf)
            primary[d] = 1 / totals[f]
    ub = np.full(size, np.inf); ub[:n*3] = 1
    for i, seq in enumerate(sequences):
        if seq in locked:
            ub[3*i+1:3*i+3] = 0
    bounds = Bounds(np.zeros(size), ub)
    integers = np.zeros(size); integers[:n*3] = 1
    options = {"time_limit": cfg["time_limit_seconds"], "mip_rel_gap": 0.0}
    def solve(objective):
        result = milp(objective, integrality=integers, bounds=bounds,
                      constraints=LinearConstraint(csr_matrix(np.array(matrix)), lower, upper), options=options)
        if not result.success or result.status != 0:
            raise ValueError(f"Split optimization did not reach optimum: {result.message}. No constraints relaxed.")
        return result
    first = solve(primary)
    add(primary.copy(), -np.inf, float(first.fun) + cfg["primary_tolerance"])
    secondary = np.zeros(size)
    secondary[:n*3] = np.random.default_rng(cfg["seed"]).random(n*3)
    second = solve(secondary)
    assignments = np.rint(second.x[:n*3]).reshape(n, 3)
    if not np.allclose(assignments.ravel(), second.x[:n*3], atol=1e-6) or not np.all(assignments.sum(axis=1) == 1):
        raise ValueError("Solver returned nonintegral assignments")
    counts = features @ assignments
    actual_primary = float(np.sum(np.abs(counts / totals[:, None] - np.array(cfg["targets"]))))
    if actual_primary > float(first.fun) + cfg["primary_tolerance"] + 1e-8:
        raise ValueError("Secondary objective degraded the primary optimum")
    mapping = {s: SPLITS[int(np.argmax(assignments[i]))] for i, s in enumerate(sequences)}
    from scipy.optimize._highspy import _core
    solver = {"engine": "HiGHS", "highs_version": ".".join(str(getattr(_core, f"HIGHS_VERSION_{part}")) for part in ("MAJOR", "MINOR", "PATCH")),
              "scipy_version": scipy.__version__, "numpy_version": np.__version__,
              "primary_optimum": float(first.fun), "assignment_primary": actual_primary,
              "secondary_objective": float(second.fun), "seed": cfg["seed"]}
    return mapping, solver


def split_rows(actions, mapping):
    return [{"sample_id": r["sample_id"], "sequence_id": r["sequence_id"], "official_split": r["official_split"],
             "experimental_split": mapping[r["sequence_id"]], "policy_version": VERSION} for r in actions]


def validate_assignments(rows, actions, locked, cfg):
    if len(rows) != len(actions):
        raise ValueError("Split row count mismatch")
    mapping = {}
    for row, source in zip(rows, actions):
        if any(row[k] != source[k] for k in ("sample_id", "sequence_id", "official_split")) or row["policy_version"] != VERSION:
            raise ValueError("Split IDs/order/metadata mismatch")
        seq, split = row["sequence_id"], row["experimental_split"]
        if split not in SPLITS or (seq in locked and split != "train"):
            raise ValueError("Invalid split or inspected sequence outside training")
        if seq in mapping and mapping[seq] != split:
            raise ValueError("Recording sequence crosses experimental splits")
        mapping[seq] = split
    for j, split in enumerate(SPLITS):
        selected = [a for r, a in zip(rows, actions) if r["experimental_split"] == split]
        low, high = cfg["image_bounds"][j]
        if not low <= len(selected) <= high:
            raise ValueError("Split image bounds violated")
        if any(sum(r["action"] == a for r in selected) < cfg["action_minimums"][a][j] for a in ACTIONS):
            raise ValueError("Split action minimum violated")


def summarize(rows, actions, labels, locked):
    by_id = {r["sample_id"]: r for r in actions}
    result = {}
    for split in SPLITS:
        selected = [r for r in rows if r["experimental_split"] == split]
        seqs = {r["sequence_id"] for r in selected}
        result[split] = {"images": len(selected), "sequences": len(seqs),
                         "actions": {a: sum(by_id[r["sample_id"]]["action"] == a for r in selected) for a in ACTIONS},
                         "concepts": {c: {"positive": sum(labels[r["sample_id"]][c] == "1" for r in selected),
                                           "negative": sum(labels[r["sample_id"]][c] == "0" for r in selected)} for c in CONCEPTS},
                         "official_splits": {s: sum(r["official_split"] == s for r in selected) for s in ("train", "val")},
                         "inspected_sequences": len(seqs & locked),
                         "images_from_inspected_sequences": sum(r["sequence_id"] in locked for r in selected)}
    return result


def run(root=ROOT):
    root = Path(root).resolve()
    folder = root / "data/manifests"
    csv_path, summary_path = folder / "experimental_splits.csv", folder / "split_summary.json"
    actions, labels, locked, cfg, hashes = load_inputs(root)
    if csv_path.exists() or summary_path.exists():
        if not (csv_path.exists() and summary_path.exists()):
            raise ValueError("Incomplete frozen split: restore both original artifacts; refusing overwrite")
        report = json.loads(summary_path.read_bytes())
        if report.get("input_hashes") != hashes or report.get("configuration") != cfg or report.get("policy_version") != VERSION:
            raise ValueError("Frozen split inputs/configuration changed; refusing overwrite. Restore inputs or make a separately versioned dataset/split decision.")
        raw = csv_path.read_bytes()
        if report.get("output_sha256") != digest(raw):
            raise ValueError("Frozen split CSV hash mismatch; refusing overwrite")
        rows = read_table(raw, ("sample_id", "sequence_id", "official_split", "experimental_split", "policy_version"), "Frozen splits")
        validate_assignments(rows, actions, locked, cfg)
        if report.get("counts") != summarize(rows, actions, labels, locked):
            raise ValueError("Frozen split summary counts mismatch")
        return report
    mapping, solver = optimize(actions, locked, cfg)
    rows = split_rows(actions, mapping)
    validate_assignments(rows, actions, locked, cfg)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    raw = buffer.getvalue().encode()
    report = {"policy_version": VERSION, "configuration": cfg, "input_hashes": hashes,
              "output_sha256": digest(raw), "solver": solver,
              "counts": summarize(rows, actions, labels, locked),
              "checks": {"eligible_ids_exactly_once": True, "sequence_overlap": 0, "inspected_sequences_only_in_train": True},
              "limitations": "Pooled coursework pilot, not official Cityscapes benchmark; three STOP test examples cannot establish reliable safety performance."}
    # Exclusive creation prevents accidental replacement. A partial write is rejected on rerun.
    with csv_path.open("xb") as stream:
        stream.write(raw)
    with summary_path.open("xb") as stream:
        stream.write(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = run()
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Split preparation failed (no existing split overwritten): {exc}\n")
    print(json.dumps({s: {"images": v["images"], "actions": v["actions"]} for s, v in report["counts"].items()}, indent=2))


if __name__ == "__main__":
    main()
