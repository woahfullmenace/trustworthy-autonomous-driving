"""Explain an action for a training/validation sample using saved local models."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(ROOT))
import torch
from scripts.train_concepts import seed_everything
from src.concept_training import ConceptDataset
from src.concepts import CONCEPTS
from src.decision_tree import compatible_model, decision_path, load_tree_run


def select_sample(records, sample_id):
    for split in ("train", "validation"):
        for row in records[split]:
            if row["sample_id"] == sample_id:
                return row, split
    raise ValueError("Sample is unavailable for development: test-split, excluded and unknown IDs are rejected")


def predict(root, sample_id, tree_run):
    tree, context = load_tree_run(root, tree_run)
    records, _, _, _, checkpoint, metadata = context
    row, split = select_sample(records, sample_id)  # Reject test IDs before loading any image.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot inference command")
    seed_everything(42); torch.set_num_threads(4)
    model, _ = compatible_model(checkpoint, metadata, "cuda")
    image, _, sid = ConceptDataset(root, [row], split)[0]
    with torch.no_grad():
        probabilities = model(image.unsqueeze(0).cuda()).sigmoid().cpu().numpy()[0]
    binary = (probabilities >= 0.5).astype(int)
    return {"sample_id": sid, "experimental_split": split,
            "concepts": {c: {"probability": float(probabilities[j]), "binary": int(binary[j])} for j, c in enumerate(CONCEPTS)},
            "decision": decision_path(tree, binary), "meaning": "Synthetic coursework action, not a driving instruction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--tree-run", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = predict(ROOT, args.sample_id, args.tree_run)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Action prediction failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
