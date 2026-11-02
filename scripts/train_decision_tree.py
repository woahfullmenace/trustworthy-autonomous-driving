"""Train an annotation-concept tree and evaluate the recovered image-to-action pipeline."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/.matplotlib"))
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.tree import export_text, plot_tree
import torch
from torch.utils.data import DataLoader

from scripts.train_concepts import seed_everything, write_json
from src.concept_training import ConceptDataset, evaluate, sha256
from src.concepts import CONCEPTS
from src.decision_tree import (ACTIONS, action_metrics, compare_predictions, compatible_model, decision_path,
                               fit_tree, load_context, load_tree_run, vehicle_dependence)


def run(root=ROOT, run_dir=None):
    root = Path(root).resolve()
    records, actions, cfg, hashes, checkpoint, concept_metadata = load_context(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to reproduce the recorded ResNet validation run")
    seed_everything(42); torch.set_num_threads(4)
    directory = (root/"outputs/decision_tree").resolve()
    output = Path(run_dir).resolve() if run_dir else directory/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not output.is_relative_to(directory) or output == directory:
        raise ValueError("Tree outputs must be inside outputs/decision_tree")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        print("Validating and regenerating the 74 validation predictions...", flush=True)
        dataset = ConceptDataset(root, records["validation"], "validation")
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        model, saved = compatible_model(checkpoint, concept_metadata, "cuda")
        loss, probabilities, targets, ids = evaluate(model, loader, "cuda")
        if ids != [r["sample_id"] for r in records["validation"]]:
            raise ValueError("Regenerated validation order mismatch")
        comparison = compare_predictions(checkpoint.parent/"validation_predictions.csv", records["validation"], probabilities.numpy())
        if abs(loss-saved["validation_loss"]) > 1e-6:
            raise ValueError("Regenerated validation loss differs from checkpoint")
        tree = fit_tree(records["train"], actions)
        annotation = targets.numpy().astype(int)
        predicted = (probabilities.numpy() >= 0.5).astype(int)
        truth = [actions[s] for s in ids]
        oracle_actions, predicted_actions = tree.predict(annotation), tree.predict(predicted)
        metrics = {"annotation_concepts": action_metrics(truth, oracle_actions),
                   "predicted_concepts": action_metrics(truth, predicted_actions),
                   "always_slow": action_metrics(truth, ["SLOW"]*len(ids)),
                   "concept_prediction_comparison": comparison,
                   "vehicle_dependence": vehicle_dependence(tree)}
        joblib.dump(tree, output/"tree.joblib")
        metadata = {"configuration": cfg, "input_hashes": hashes, "concept_order": list(CONCEPTS),
                    "tree_classes": tree.classes_.tolist(), "metric_class_order": list(ACTIONS),
                    "tree_sha256": sha256((output/"tree.joblib").read_bytes()),
                    "training_ids": [r["sample_id"] for r in records["train"]], "validation_ids": ids,
                    "tree_depth": int(tree.get_depth()), "tree_leaves": int(tree.get_n_leaves()),
                    "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "torch": str(torch.__version__)}}
        write_json(output/"metadata.json", metadata)
        restored, _ = load_tree_run(root, output)
        if not np.array_equal(restored.predict(predicted), predicted_actions):
            raise ValueError("Saved tree reload changed predictions")
        write_json(output/"metrics.json", metrics)
        rows = []
        for i, sid in enumerate(ids):
            path = decision_path(tree, predicted[i])
            if path["action"] != predicted_actions[i]:
                raise ValueError("Decision path disagrees with tree prediction")
            row = {"sample_id": sid, "target_action": truth[i], "annotation_tree_action": oracle_actions[i],
                   "predicted_tree_action": predicted_actions[i], "decision_path": " -> ".join([*path["steps"], path["action"]])}
            for j, c in enumerate(CONCEPTS):
                row.update({c+"_target": int(annotation[i,j]), c+"_probability": float(probabilities[i,j]), c+"_prediction": int(predicted[i,j])})
            rows.append(row)
        with (output/"validation_actions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output/"tree.txt").write_text(export_text(tree, feature_names=list(CONCEPTS)), encoding="utf-8")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(tree, feature_names=list(CONCEPTS), class_names=tree.classes_.tolist(), filled=True, rounded=True, ax=ax)
        fig.tight_layout(); fig.savefig(output/"tree.png", dpi=150); plt.close(fig)
        lines = ["# First image-to-action pipeline", "", "Validation only: 74 images. Targets come from concepts-v1.0/actions-v1.0 proxy rules.", "",
                 "| Input to decision stage | Accuracy | Macro F1 | STOP recall | STOP-to-GO errors |", "|---|---:|---:|---:|---:|"]
        for name in ("annotation_concepts", "predicted_concepts", "always_slow"):
            m = metrics[name]
            lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['stop_recall']:.4f} | {m['stop_to_go_errors']} |")
        lines.extend(["", "Confusion matrices use target rows and prediction columns in STOP/SLOW/GO order.",
                      "Precision/recall with a zero denominator are null. Undefined class F1 is zero; macro F1 averages all three classes.",
                      "", "```text", export_text(tree, feature_names=list(CONCEPTS)), "```", "",
                      f"Vehicle dependence: {json.dumps(metrics['vehicle_dependence'])}", "",
                      "The tree is fitted to annotation-derived concepts and policy-derived targets. Annotation-concept agreement tests policy reproduction; it is not independent expert driving validation.",
                      "Only three STOP examples are available. Zero STOP-to-GO errors does not establish STOP recognition: inspect STOP recall and STOP-to-SLOW errors as well.",
                      "Concept thresholds, tree settings, checkpoint and splits were held fixed; test images were not loaded."])
        (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        if load_context(root)[3] != hashes:
            raise ValueError("Input files changed during tree evaluation")
        write_json(output/"status.json", {"status": "complete"})
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"Saved: {output}", flush=True)
        return output, metrics
    except Exception as exc:
        write_json(output/"status.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        run(run_dir=args.run_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Decision tree failed: {exc}\n")


if __name__ == "__main__":
    main()
