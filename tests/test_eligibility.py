import csv
import io
import json

import pytest

from scripts.prepare_eligibility import CONCEPTS, digest, json_bytes, run


def table(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "data/manifests"
    folder.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs").mkdir()
    manifest = [dict(sample_id=sid, official_split=split, sequence_id=sid.rsplit("_", 1)[0])
                for sid, split in [("city_000001_000001", "train"),
                                   ("city_000001_000002", "train"),
                                   ("city_000002_000001", "val")]]
    labels = [{"sample_id": row["sample_id"], "official_split": row["official_split"],
               "rule_version": "concepts-v1.0", **{c: "0" for c in CONCEPTS},
               **{f"{c}_valid": "1" for c in CONCEPTS}} for row in manifest]
    labels[1]["road_clear"] = ""
    labels[1]["road_clear_valid"] = "0"
    labels[1]["obstacle_close"] = ""
    labels[1]["obstacle_close_valid"] = "0"

    def save():
        (folder / "subset.csv").write_bytes(table(manifest))
        (folder / "concept_labels.csv").write_bytes(table(list(reversed(labels))))
        rules = b"rule_version: concepts-v1.0\n"
        (tmp_path / "configs/concepts.yaml").write_bytes(rules)
        summary = {"rule_version": "concepts-v1.0", "concept_order": list(CONCEPTS),
                   "sample_count": len(manifest), "manifest_sha256": digest(table(manifest)),
                   "labels_sha256": digest(table(list(reversed(labels)))),
                   "rules_sha256": digest(rules), "audit_sample_ids": [manifest[0]["sample_id"]]}
        (folder / "concept_summary.json").write_bytes(json_bytes(summary))
        preview = {"manifest_sha256": summary["manifest_sha256"],
                   "preview_sample_ids": [manifest[0]["sample_id"], manifest[2]["sample_id"]]}
        (tmp_path / "outputs/data_verification.json").write_bytes(json_bytes(preview))
    save()
    return tmp_path, manifest, labels, save


def test_order_exclusions_inspection_and_repeat_without_preview(dataset):
    root, manifest, _, _ = dataset
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = run(root)
    folder = root / "data/manifests"
    rows = list(csv.DictReader(io.StringIO((folder / "eligibility.csv").read_text())))
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in manifest]
    assert [r["supervised_eligible"] for r in rows] == ["1", "0", "1"]
    assert rows[1]["exclusion_reason"] == "invalid_concepts:road_clear;obstacle_close"
    assert rows[1]["directly_inspected"] == "0"
    assert rows[1]["inspected_sequence"] == "1"
    assert rows[2]["official_split"] == "val"
    assert report["counts"]["all"]["eligible"] == 2
    registry = json.loads((folder / "inspection_registry.json").read_bytes())
    assert registry["inspections"][0]["sources"] == ["concept_audit", "setup_preview"]
    assert all(p.read_bytes() == data for p, data in original.items())
    artifacts = {p: p.read_bytes() for p in folder.glob("*")}
    (root / "outputs/data_verification.json").unlink()
    assert run(root) == report
    assert all(p.read_bytes() == data for p, data in artifacts.items())


@pytest.mark.parametrize("value,valid", [("", "1"), ("0", "0"), ("1", "0"), ("2", "1"), ("0", ""), ("NaN", "0")])
def test_contradictory_labels_fail_before_writes(dataset, value, valid):
    root, _, labels, save = dataset
    labels[0]["pedestrian_present"] = value
    labels[0]["pedestrian_present_valid"] = valid
    save()
    with pytest.raises(ValueError, match="Contradictory"):
        run(root)
    assert not (root / "data/manifests/inspection_registry.json").exists()
    assert not (root / "data/manifests/eligibility.csv").exists()


@pytest.mark.parametrize("issue", ["duplicate_labels", "duplicate_manifest", "missing", "extra", "split", "sequence"])
def test_inconsistent_inputs(dataset, issue):
    root, manifest, labels, save = dataset
    if issue == "duplicate_labels":
        labels.append(labels[0].copy())
    elif issue == "duplicate_manifest":
        manifest.append(manifest[0].copy())
    elif issue == "missing":
        labels.pop()
    elif issue == "extra":
        labels.append({**labels[0], "sample_id": "city_999999_000001"})
    elif issue == "split":
        labels[0]["official_split"] = "val"
    else:
        manifest[0]["sequence_id"] = "wrong"
    save()
    with pytest.raises(ValueError):
        run(root)


@pytest.mark.parametrize("path", ["data/manifests/subset.csv", "data/manifests/concept_labels.csv", "configs/concepts.yaml"])
def test_hash_mismatch(dataset, path):
    root, *_ = dataset
    source = root / path
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run(root)


def test_missing_inspection_history(dataset):
    root, *_ = dataset
    (root / "outputs/data_verification.json").unlink()
    with pytest.raises(ValueError, match="Restore the original"):
        run(root)


@pytest.mark.parametrize("issue", ["unknown_id", "sequence", "split", "duplicate", "missing", "source"])
def test_invalid_registry(dataset, issue):
    root, *_ = dataset
    run(root)
    path = root / "data/manifests/inspection_registry.json"
    registry = json.loads(path.read_bytes())
    records = registry["inspections"]
    if issue == "unknown_id":
        records[0]["sample_id"] = "unknown"
    elif issue == "sequence":
        records[0]["sequence_id"] = "unknown"
    elif issue == "split":
        records[0]["official_split"] = "val"
    elif issue == "duplicate":
        records.append(records[0].copy())
    elif issue == "missing":
        records.pop()
    else:
        registry["sources"]["setup_preview"]["sample_ids"].append("unknown")
    path.write_bytes(json_bytes(registry))
    before = (path.parent / "eligibility.csv").read_bytes()
    with pytest.raises(ValueError):
        run(root)
    assert (path.parent / "eligibility.csv").read_bytes() == before
