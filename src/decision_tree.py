"""Shallow action tree, metrics and compatibility checks for the pilot."""
import csv
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import torch
import yaml

from src.concepts import CONCEPTS
from src.concept_training import PREPROCESSING, load_checkpoint, read_records, sha256

ACTIONS = ("STOP", "SLOW", "GO")
VERSION = "decision-tree-v1.0"
PARAMETERS = dict(max_depth=4, criterion="gini", min_samples_leaf=1, class_weight=None, random_state=42)


def binary_matrix(values):
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != 5 or not len(x) or not np.isin(x, (0, 1)).all():
        raise ValueError("Expected a nonempty N x 5 matrix of binary concepts in the documented order")
    return x.astype(np.int64)


def fit_tree(records, action_by_id):
    if not records or any(r["experimental_split"] != "train" for r in records):
        raise ValueError("Tree fitting requires training rows only")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate training IDs")
    x = binary_matrix([[int(r[c]) for c in CONCEPTS] for r in records])
    y = [action_by_id[r["sample_id"]] for r in records]
    if set(y) != set(ACTIONS):
        raise ValueError("Training must include all three expected actions")
    return DecisionTreeClassifier(**PARAMETERS).fit(x, y)


def decision_path(tree, concepts):
    x = binary_matrix([concepts])[0]
    node, steps = 0, []
    while tree.tree_.children_left[node] != -1:
        feature = int(tree.tree_.feature[node]); threshold = float(tree.tree_.threshold[node])
        left = x[feature] <= threshold
        steps.append(f"{CONCEPTS[feature]}={x[feature]} {'<=' if left else '>'} {threshold:g}")
        node = int(tree.tree_.children_left[node] if left else tree.tree_.children_right[node])
    action = str(tree.classes_[np.argmax(tree.tree_.value[node][0])])
    return {"steps": steps, "leaf": node, "action": action}


def action_metrics(targets, predictions):
    if not len(targets) or len(targets) != len(predictions) or not set(targets) <= set(ACTIONS) or not set(predictions) <= set(ACTIONS):
        raise ValueError("Invalid action targets or predictions")
    cm = confusion_matrix(targets, predictions, labels=ACTIONS)
    per_class = {}
    for i, action in enumerate(ACTIONS):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum())-tp; fn = int(cm[i, :].sum())-tp
        per_class[action] = {"support": tp+fn, "precision": tp/(tp+fp) if tp+fp else None,
                             "recall": tp/(tp+fn) if tp+fn else None,
                             "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    return {"class_order": list(ACTIONS), "accuracy": float(np.trace(cm)/cm.sum()),
            "macro_f1": sum(v["f1"] for v in per_class.values())/3,
            "classes": per_class, "confusion_matrix": cm.tolist(),
            "stop_to_go_errors": int(cm[0, 2]), "stop_recall": per_class["STOP"]["recall"],
            "metric_policy": "Confusion rows=target, columns=prediction. Precision/recall with zero denominator=null; undefined F1=0; macro F1 averages all three actions."}


def vehicle_dependence(tree):
    patterns = np.array(list(product((0, 1), repeat=5)))
    flipped = patterns.copy(); flipped[:, 1] = 1-flipped[:, 1]
    changed = tree.predict(patterns) != tree.predict(flipped)
    return {"used_in_splits": bool(np.any(tree.tree_.feature == 1)),
            "changed_patterns_of_32": int(changed.sum()),
            "scope": "Exhaustive binary inputs, including combinations absent from the dataset"}


def load_context(root):
    root = Path(root)
    config_path = root / "configs/decision_tree.yaml"
    cfg = yaml.safe_load(config_path.read_bytes())
    expected = {"policy_version": VERSION, **PARAMETERS, "concept_threshold": 0.5,
                "concept_run": "outputs/concept_training/pilot_v1"}
    if cfg != expected:
        raise ValueError("Unexpected decision-tree-v1.0 configuration")
    records, hashes = read_records(root)
    hashes["training_config"] = sha256((root / "configs/concept_training.yaml").read_bytes())
    concept_run = root / cfg["concept_run"]
    metadata = json.loads((concept_run / "metadata.json").read_bytes())
    status = json.loads((concept_run / "status.json").read_bytes())
    if status.get("status") != "complete" or status.get("mode") != "pilot":
        raise ValueError("A completed pilot concept run is required")
    if metadata.get("input_hashes") != hashes or metadata.get("concept_order") != list(CONCEPTS) or metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Concept run provenance/order/preprocessing mismatch")
    if metadata.get("training_ids") != [r["sample_id"] for r in records["train"]] or metadata.get("evaluation_ids") != [r["sample_id"] for r in records["validation"]] or metadata.get("evaluation_split") != "validation":
        raise ValueError("Concept run split membership mismatch")
    checkpoint = concept_run / "best.pt"
    hashes = {**hashes, "tree_configuration": sha256(config_path.read_bytes()),
              **{f"concept_run/{name}": sha256((concept_run/name).read_bytes()) for name in ("best.pt", "metadata.json", "status.json", "validation_predictions.csv")}}
    with (root / "data/manifests/action_labels.csv").open(newline="", encoding="utf-8") as stream:
        actions = {r["sample_id"]: r["action"] for r in csv.DictReader(stream)}
    return records, actions, cfg, hashes, checkpoint, metadata


def compatible_model(checkpoint, metadata, device):
    model, saved = load_checkpoint(checkpoint, device)
    if saved.get("input_hashes") != metadata["input_hashes"] or saved.get("configuration") != metadata["configuration"] or saved.get("weights") != metadata["weights"]:
        raise ValueError("Checkpoint provenance/configuration/weights mismatch")
    if saved.get("balance") != metadata.get("balance"):
        raise ValueError("Checkpoint balance weights/provenance mismatch")
    return model, saved


def compare_predictions(path, records, probabilities):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    if [r["sample_id"] for r in saved] != [r["sample_id"] for r in records]:
        raise ValueError("Saved validation IDs/order mismatch")
    previous = np.array([[float(r[c+"_probability"]) for c in CONCEPTS] for r in saved])
    p = np.asarray(probabilities)
    if p.shape != previous.shape or not np.isfinite(p).all() or not np.isfinite(previous).all():
        raise ValueError("Malformed validation probabilities")
    for old, row in zip(saved, records):
        for c in CONCEPTS:
            if old[c+"_target"] != row[c] or int(old[c+"_prediction"]) != int(float(old[c+"_probability"]) >= 0.5):
                raise ValueError("Saved validation targets/predictions mismatch")
    if not np.allclose(p, previous, rtol=1e-6, atol=1e-6) or not np.array_equal(p >= 0.5, previous >= 0.5):
        raise ValueError("Regenerated concept probabilities disagree with saved predictions")
    return {"max_absolute_difference": float(np.max(np.abs(p-previous))), "binary_predictions_identical": True,
            "tolerance": "atol=1e-6, rtol=1e-6; binary predictions must match exactly"}


def load_tree_run(root, run_dir):
    context = load_context(root)
    report = json.loads((Path(run_dir)/"metadata.json").read_bytes())
    if report.get("input_hashes") != context[3] or report.get("concept_order") != list(CONCEPTS) or report.get("configuration") != context[2]:
        raise ValueError("Tree run input provenance/configuration/order mismatch")
    path = Path(run_dir)/"tree.joblib"
    if sha256(path.read_bytes()) != report.get("tree_sha256"):
        raise ValueError("Saved tree hash mismatch")
    # Only load the locally produced, provenance-bound artifact supplied by this project.
    tree = joblib.load(path)
    if not isinstance(tree, DecisionTreeClassifier) or tree.n_features_in_ != 5 or tree.classes_.tolist() != report.get("tree_classes") or set(tree.classes_) != set(ACTIONS):
        raise ValueError("Incompatible saved tree")
    if any(tree.get_params()[key] != value for key, value in PARAMETERS.items()):
        raise ValueError("Saved tree settings differ from frozen policy")
    return tree, context
