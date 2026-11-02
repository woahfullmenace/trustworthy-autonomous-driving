"""Metadata-only split checks; never load or expose held-out images."""
import csv
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from scripts import prepare_splits as splits
from scripts.prepare_eligibility import digest, json_bytes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for directory, names in {
        "data/manifests": ("subset.csv", "concept_labels.csv", "concept_summary.json", "eligibility.csv",
                           "eligibility_summary.json", "inspection_registry.json", "action_labels.csv", "action_summary.json"),
        "configs": ("concepts.yaml", "actions.yaml", "splits.yaml"),
    }.items():
        (tmp_path / directory).mkdir(parents=True)
        for name in names:
            shutil.copyfile(ROOT / directory / name, tmp_path / directory / name)
    return tmp_path


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_real_metadata_freeze_and_fresh_determinism(project, tmp_path):
    before = snapshot(project)
    report = splits.run(project)
    folder = project / "data/manifests"
    with (folder / "experimental_splits.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    actions, labels, locked, cfg, _ = splits.load_inputs(project)
    assert len(rows) == len({r["sample_id"] for r in rows}) == 490
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in actions]
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_id"], set()).add(row["experimental_split"])
        if row["sequence_id"] in locked:
            assert row["experimental_split"] == "train"
    assert all(len(values) == 1 for values in by_sequence.values())
    assert report["counts"]["validation"]["actions"]["STOP"] == 3
    assert report["counts"]["test"]["actions"]["STOP"] == 3
    assert sum(v["images_from_inspected_sequences"] for v in report["counts"].values()) == 125
    splits.validate_assignments(rows, actions, locked, cfg)
    frozen = snapshot(project)
    assert splits.run(project) == report
    assert snapshot(project) == frozen
    assert all(frozen[name] == data for name, data in before.items())
    # A fresh solve, rather than only reading frozen assignments, must reproduce bytes.
    fresh = tmp_path / "fresh"
    for name, data in before.items():
        path = fresh / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert splits.run(fresh) == report
    assert (fresh / "data/manifests/experimental_splits.csv").read_bytes() == (folder / "experimental_splits.csv").read_bytes()


@pytest.mark.parametrize("name", ["subset.csv", "concept_labels.csv", "concept_summary.json", "eligibility.csv",
                                 "eligibility_summary.json", "inspection_registry.json", "action_labels.csv", "action_summary.json"])
def test_stale_or_malformed_inputs_leave_no_split(project, name):
    path = project / "data/manifests" / name
    if name == "action_summary.json":
        data = json.loads(path.read_bytes()); data["sample_count"] = 1
        path.write_bytes(json_bytes(data))
    else:
        path.write_bytes(b"broken")
    with pytest.raises((ValueError, KeyError, TypeError)):
        splits.run(project)
    assert not (path.parent / "experimental_splits.csv").exists()
    assert not (path.parent / "split_summary.json").exists()


@pytest.mark.parametrize("issue", ["duplicate", "missing", "sequence", "split", "action"])
def test_rehashed_inconsistent_action_table(project, issue):
    folder = project / "data/manifests"
    path = folder / "action_labels.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    if issue == "duplicate":
        rows.append(rows[0].copy())
    elif issue == "missing":
        rows.pop()
    elif issue == "sequence":
        rows[0]["sequence_id"] = "unresolved"
    elif issue == "split":
        rows[0]["official_split"] = "val" if rows[0]["official_split"] == "train" else "train"
    else:
        rows[0]["action"] = "INVALID"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode())
    summary = json.loads((folder / "action_summary.json").read_bytes())
    summary["output_hashes"]["action_labels.csv"] = digest(path.read_bytes())
    (folder / "action_summary.json").write_bytes(json_bytes(summary))
    with pytest.raises(ValueError):
        splits.run(project)
    assert not (folder / "experimental_splits.csv").exists()


@pytest.mark.parametrize("change", ["config", "input", "csv", "summary", "partial", "missing_input"])
def test_frozen_artifacts_never_overwritten(project, change):
    splits.run(project)
    folder = project / "data/manifests"
    if change == "config":
        path = project / "configs/splits.yaml"
        path.write_bytes(path.read_bytes() + b"\n# changed configuration bytes\n")
    elif change == "input":
        path = folder / "action_summary.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "csv":
        path = folder / "experimental_splits.csv"
        path.write_bytes(path.read_bytes() + b"bad")
    elif change == "summary":
        path = folder / "split_summary.json"
        report = json.loads(path.read_bytes()); report["counts"]["train"]["images"] = 0
        path.write_bytes(json_bytes(report))
    elif change == "partial":
        (folder / "split_summary.json").unlink()
    else:
        (folder / "action_labels.csv").unlink()
    before = snapshot(project)
    with pytest.raises((ValueError, OSError)):
        splits.run(project)
    assert snapshot(project) == before


def test_infeasible_constraints_no_artifacts(project, monkeypatch):
    actual = splits.optimize
    def impossible(actions, locked, cfg):
        return actual(actions, {r["sequence_id"] for r in actions}, cfg)
    monkeypatch.setattr(splits, "optimize", impossible)
    with pytest.raises(ValueError, match="did not reach optimum"):
        splits.run(project)
    assert not (project / "data/manifests/experimental_splits.csv").exists()


def test_optimizer_timeout_rejected(project, monkeypatch):
    monkeypatch.setattr(splits, "milp", lambda *a, **kw: SimpleNamespace(success=False, status=1, message="time limit"))
    with pytest.raises(ValueError, match="time limit"):
        splits.run(project)
    assert not (project / "data/manifests/experimental_splits.csv").exists()


def test_assignment_guardrails(project):
    actions, _, locked, cfg, _ = splits.load_inputs(project)
    mapping, _ = splits.optimize(actions, locked, cfg)
    rows = splits.split_rows(actions, mapping)
    i = next(i for i, r in enumerate(rows) if r["sequence_id"] in locked)
    changed = [r.copy() for r in rows]; changed[i]["experimental_split"] = "test"
    with pytest.raises(ValueError, match="inspected"):
        splits.validate_assignments(changed, actions, locked, cfg)
    changed = [dict(r, experimental_split="train") for r in rows]
    with pytest.raises(ValueError, match="bounds"):
        splits.validate_assignments(changed, actions, locked, cfg)
    stricter = {**cfg, "action_minimums": {**cfg["action_minimums"], "STOP": [11, 3, 3]}}
    with pytest.raises(ValueError, match="minimum"):
        splits.validate_assignments(rows, actions, locked, stricter)
    # Force a sequence to straddle two splits, without an inspection constraint.
    i = next(i for i, r in enumerate(rows) if r["sequence_id"] not in locked and sum(x["sequence_id"] == r["sequence_id"] for x in rows) > 1)
    changed = [r.copy() for r in rows]
    changed[i]["experimental_split"] = "test" if rows[i]["experimental_split"] != "test" else "validation"
    with pytest.raises(ValueError, match="crosses"):
        splits.validate_assignments(changed, actions, locked, cfg)
