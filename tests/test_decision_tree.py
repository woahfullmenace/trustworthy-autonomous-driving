import csv
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.actions import concepts_to_action
from src.concepts import CONCEPTS
from src import decision_tree as dt
from scripts.predict_action import select_sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic():
    records, actions = [], {}
    for i, bits in enumerate(product((0, 1), repeat=5)):
        sid = str(i)
        records.append({"sample_id": sid, "experimental_split": "train", **dict(zip(CONCEPTS, map(str, bits)))})
        actions[sid] = concepts_to_action(dict(zip(CONCEPTS, bits)))[0]
    return records, actions


def test_concept_order_paths_and_serialization(synthetic, tmp_path):
    records, actions = synthetic
    tree = dt.fit_tree(records, actions)
    x = dt.binary_matrix([[int(r[c]) for c in CONCEPTS] for r in records])
    assert tree.get_depth() <= 4
    assert tree.predict(x).tolist() == [actions[r["sample_id"]] for r in records]
    assert [dt.decision_path(tree, bits)["action"] for bits in x] == tree.predict(x).tolist()
    assert dt.vehicle_dependence(tree)["changed_patterns_of_32"] == 0
    path = tmp_path / "tree.joblib"; joblib.dump(tree, path)
    assert np.array_equal(joblib.load(path).predict(x), tree.predict(x))
    again = dt.fit_tree(records, actions)
    for name in ("feature", "threshold", "children_left", "children_right", "value"):
        assert np.array_equal(getattr(tree.tree_, name), getattr(again.tree_, name))


@pytest.mark.parametrize("bad_split", ["validation", "test"])
def test_fitting_rejects_held_out_rows(synthetic, bad_split):
    records, actions = synthetic
    records[0]["experimental_split"] = bad_split
    with pytest.raises(ValueError, match="training rows only"):
        dt.fit_tree(records, actions)


def test_missing_predictions_metrics():
    m = dt.action_metrics(["STOP", "STOP", "SLOW", "GO"], ["SLOW"]*4)
    assert m["confusion_matrix"] == [[0, 2, 0], [0, 1, 0], [0, 1, 0]]
    assert m["accuracy"] == 0.25
    assert m["classes"]["STOP"]["precision"] is None
    assert m["stop_recall"] == 0 and m["stop_to_go_errors"] == 0
    assert m["macro_f1"] == pytest.approx(0.4/3)
    assert dt.action_metrics(["STOP"], ["GO"])["stop_to_go_errors"] == 1
    absent = dt.action_metrics(["SLOW"], ["SLOW"])
    assert absent["classes"]["STOP"]["recall"] is None
    assert absent["macro_f1"] == pytest.approx(1/3)


@pytest.mark.parametrize("values", [[[0, 1]], [[0, 1, 0, 1, 0.5]], [[0, 1, 0, 1, float('nan')]]])
def test_bad_concept_vectors(values):
    with pytest.raises(ValueError):
        dt.binary_matrix(values)


def test_recovered_metadata_and_test_exclusion():
    records, actions, cfg, hashes, checkpoint, metadata = dt.load_context(ROOT)
    assert len(records["train"]) == 342 and len(records["validation"]) == 74
    with (ROOT/"data/manifests/experimental_splits.csv").open() as f:
        test_ids = {r["sample_id"] for r in csv.DictReader(f) if r["experimental_split"] == "test"}
    assert not test_ids & {r["sample_id"] for rows in records.values() for r in rows}
    with pytest.raises(ValueError, match="test-split"):
        select_sample(records, next(iter(test_ids)))
    tree = dt.fit_tree(records["train"], actions)
    assert tree.n_features_in_ == 5 and cfg["max_depth"] == 4
    assert hashes["concept_run/best.pt"] == dt.sha256(checkpoint.read_bytes())


@pytest.mark.parametrize("problem", ["hash", "order", "configuration"])
def test_stale_tree_rejected(synthetic, tmp_path, monkeypatch, problem):
    records, actions = synthetic
    cfg = {"max_depth": 4}; hashes = {"inputs": "original"}
    tree = dt.fit_tree(records, actions)
    joblib.dump(tree, tmp_path/"tree.joblib")
    report = {"input_hashes": hashes, "concept_order": list(CONCEPTS), "configuration": cfg,
              "tree_classes": tree.classes_.tolist(), "tree_sha256": dt.sha256((tmp_path/"tree.joblib").read_bytes())}
    if problem == "hash": report["tree_sha256"] = "changed"
    if problem == "order": report["concept_order"] = list(reversed(CONCEPTS))
    if problem == "configuration": report["configuration"] = {"max_depth": 5}
    (tmp_path/"metadata.json").write_text(json.dumps(report))
    monkeypatch.setattr(dt, "load_context", lambda root: (records, actions, cfg, hashes, None, None))
    with pytest.raises(ValueError): dt.load_tree_run(tmp_path, tmp_path)


def test_incompatible_checkpoint_provenance(monkeypatch):
    metadata = {"input_hashes": {"a": "b"}, "configuration": {}, "weights": {}}
    monkeypatch.setattr(dt, "load_checkpoint", lambda *args: (None, {**metadata, "input_hashes": {"a": "changed"}}))
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        dt.compatible_model(None, metadata, "cpu")


def test_prediction_comparison_rejects_changes(tmp_path):
    row = {"sample_id": "example", **{c: "0" for c in CONCEPTS}}
    saved = {"sample_id": "example", **{c+suffix: value for c in CONCEPTS for suffix, value in (("_probability", 0.1), ("_prediction", 0), ("_target", 0))}}
    path = tmp_path/"predictions.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(saved)); writer.writeheader(); writer.writerow(saved)
    p = np.full((1, 5), 0.1)
    assert dt.compare_predictions(path, [row], p)["binary_predictions_identical"]
    p[0, 0] = 0.6
    with pytest.raises(ValueError, match="disagree"):
        dt.compare_predictions(path, [row], p)
