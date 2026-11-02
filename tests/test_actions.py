import csv
from itertools import product
import json

from PIL import Image
import pytest
import yaml

from src.actions import POLICY, concepts_to_action, validate_policy
from src.concepts import CONCEPTS
from scripts.label_actions import run
from scripts.prepare_eligibility import digest, run as prepare
from test_eligibility import dataset


@pytest.mark.parametrize("bits", list(product((0, 1), repeat=5)))
def test_all_binary_combinations(bits):
    p, v, c, r, o = bits
    action, reason = concepts_to_action(dict(zip(CONCEPTS, bits)))
    expected = "STOP" if o else "GO" if (p, c, r) == (0, 0, 1) else "SLOW"
    assert action == expected
    if action == "STOP":
        assert reason == "obstacle_close=1"
    if action == "SLOW":
        assert ("pedestrian_present=1" in reason) == bool(p)
        assert ("cyclist_present=1" in reason) == bool(c)
        assert ("road_clear=0" in reason) == (not r)
    changed = dict(zip(CONCEPTS, bits))
    changed["vehicle_present"] = 1-v
    assert concepts_to_action(changed) == (action, reason)


@pytest.mark.parametrize("value", [None, "", "0", 2, -1, 0.5, True])
def test_invalid_concepts(value):
    concepts = dict.fromkeys(CONCEPTS, 0)
    concepts["vehicle_present"] = value
    with pytest.raises(ValueError):
        concepts_to_action(concepts)


def test_missing_concept_and_changed_policy():
    with pytest.raises(ValueError):
        concepts_to_action({})
    with pytest.raises(ValueError):
        validate_policy({**POLICY, "otherwise": "STOP"})


@pytest.fixture
def pipeline(dataset):
    root, manifest, labels, save = dataset
    image = root / "sample.png"
    Image.new("RGB", (8, 4), "gray").save(image)
    for row in manifest:
        row.update(image_path="sample.png", image_sha256=digest(image.read_bytes()))
    save()
    (root / "configs/actions.yaml").write_text(yaml.safe_dump(POLICY))
    prepare(root)
    return root, manifest, labels, save


def test_pipeline_order_selection_hashes_and_determinism(pipeline):
    root, manifest, _, _ = pipeline
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = run(root)
    folder = root / "data/manifests"
    with (folder / "action_labels.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["sample_id"] for r in rows] == [manifest[0]["sample_id"], manifest[2]["sample_id"]]
    assert report["audit_sample_ids"] == [manifest[0]["sample_id"]]
    assert report["sample_count"] == 2
    assert report["counts"]["train"]["SLOW"]["count"] == 1
    for relative, checksum in report["output_hashes"].items():
        path = folder / relative if relative.endswith(".csv") else root / "outputs/actions" / relative
        assert digest(path.read_bytes()) == checksum
    assert all(p.read_bytes() == data for p, data in before.items())
    artifacts = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert run(root) == report
    assert all(p.read_bytes() == data for p, data in artifacts.items())


@pytest.mark.parametrize("name", ["subset.csv", "concept_labels.csv", "concept_summary.json", "eligibility.csv", "eligibility_summary.json", "inspection_registry.json"])
def test_stale_metadata_fails_before_outputs(pipeline, name):
    root, *_ = pipeline
    path = root / "data/manifests" / name
    if name == "eligibility_summary.json":
        data = json.loads(path.read_bytes())
        data["counts"]["all"]["eligible"] += 1
        path.write_text(json.dumps(data))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run(root)
    assert not (path.parent / "action_labels.csv").exists()
    assert not (root / "outputs/actions").exists()


@pytest.mark.parametrize("issue", ["duplicate", "missing", "split", "invalid", "sequence"])
def test_semantic_inconsistency_even_with_updated_hashes(pipeline, issue):
    root, manifest, labels, save = pipeline
    if issue == "duplicate":
        labels.append(labels[0].copy())
    elif issue == "missing":
        labels.pop()
    elif issue == "split":
        labels[0]["official_split"] = "val"
    elif issue == "invalid":
        labels[0]["road_clear"] = ""
    else:
        manifest[0]["sequence_id"] = "unknown"
    save()
    with pytest.raises(ValueError):
        run(root)


def test_changed_image_fails_before_outputs(pipeline):
    root, *_ = pipeline
    (root / "sample.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash mismatch"):
        run(root)
    assert not (root / "data/manifests/action_labels.csv").exists()
