"""Train the ResNet18 concept head or layer4 plus head on frozen pilot splits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
sys.path.insert(0, str(ROOT))

import numpy as np
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import ResNet18_Weights
import yaml

from src.concept_training import (PREPROCESSING, ConceptDataset, FrozenConceptResNet, backbone_digest,
                                  concept_metrics, evaluate, is_better, load_checkpoint, read_records, sha256, frozen_digest, make_optimizer)
from src.concepts import CONCEPTS
from src.balanced_loss import compute_balance, balanced_bce, weight_tensors


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_config(cfg):
    expected = {"training_version": "concept-training-v1.0", "seed": 42, "epochs": 20, "batch_size": 16,
                "learning_rate": 0.001, "weight_decay": 0.0001, "num_workers": 0, "cpu_threads": 4,
                **PREPROCESSING, "weights": "IMAGENET1K_V1", "threshold": 0.5}
    if cfg.get("training_version") == "concept-training-v2.0":
        expected.update(training_version="concept-training-v2.0", strategy="layer4_and_head", backbone_learning_rate=0.0001)
    if cfg.get("training_version") == "concept-training-v3.0":
        expected.update(training_version="concept-training-v3.0", strategy="layer4_and_head", backbone_learning_rate=0.0001, loss="balanced_binary_bce", checkpoint_selection="unweighted_validation_loss")
    if cfg != expected:
        raise ValueError("Configuration differs from the approved v1, v2 or v3 training recipe")


def run(root=ROOT, run_dir=None, smoke=False, config=None):
    root = Path(root).resolve()
    config_path = (root / config).resolve() if config else root / "configs/concept_training.yaml"
    cfg = yaml.safe_load(config_path.read_bytes()); validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this pilot; no CPU fallback")
    device = torch.device("cuda")
    torch.set_num_threads(cfg["cpu_threads"])
    seed_everything(cfg["seed"])
    records, hashes = read_records(root)
    hashes["training_config"] = sha256(config_path.read_bytes())
    balance = compute_balance(records["train"]) if cfg["training_version"] == "concept-training-v3.0" else None
    loss_weights = weight_tensors(balance, device) if balance is not None else None
    output_root = (root / "outputs/concept_training").resolve()
    run_dir = Path(run_dir).resolve() if run_dir else output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ("_smoke" if smoke else "_pilot"))
    if not run_dir.is_relative_to(output_root) or run_dir == output_root:
        raise ValueError("Run directory must be inside ignored outputs/concept_training")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "status.json", {"status": "running", "mode": "smoke" if smoke else "pilot"})
    print(f"Run directory: {run_dir}", flush=True)
    try:
        train_records = records["train"][:16] if smoke else records["train"]
        train = ConceptDataset(root, train_records, "train")
        validation = train if smoke else ConceptDataset(root, records["validation"], "validation")
        generator = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator, num_workers=0)
        val_loader = DataLoader(validation, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        cache = root / "outputs/torch_cache"
        torch.hub.set_dir(str(cache))
        weights = ResNet18_Weights.IMAGENET1K_V1
        try:
            model = FrozenConceptResNet(weights=weights, strategy=cfg.get("strategy", "head_only")).to(device)
        except Exception as exc:
            raise RuntimeError(f"Could not load required ImageNet weights; no random fallback: {exc}") from exc
        weight_path = cache / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        weight_hash = sha256(weight_path.read_bytes())
        if not weight_hash.startswith(weight_path.stem.rsplit("-", 1)[-1]):
            raise ValueError("Cached pretrained weight hash mismatch")
        metadata = {"configuration": cfg, "config_path": str(config_path.relative_to(root)), "mode": "smoke" if smoke else "pilot", "input_hashes": hashes,
                    "concept_order": list(CONCEPTS), "preprocessing": PREPROCESSING,
                    "weights": {"name": "ResNet18_Weights.IMAGENET1K_V1", "url": weights.url, "sha256": weight_hash},
                    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                    "torchvision": str(torchvision.__version__), "numpy": np.__version__, "pillow": PIL.__version__,
                                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)},
                    "determinism": {"algorithms": True, "cudnn_benchmark": False, "tf32": False, "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
                    "training_ids": [r["sample_id"] for r in train.records],
                    "evaluation_ids": [r["sample_id"] for r in validation.records],
                    "evaluation_split": "train_smoke_only" if smoke else "validation"}
        if balance is not None:
            metadata["balance"] = balance
        optimizer = make_optimizer(model, cfg)
        metadata["trainable_groups"] = [{"name": g["name"], "lr": g["lr"], "parameters": [n for n, p in model.named_parameters() if any(p is q for q in g["params"])]} for g in optimizer.param_groups]
        metadata["initial_head_sha256"] = sha256(model.network.fc.weight.detach().cpu().numpy().tobytes())
        metadata["initial_state_digests"] = {"frozen_parameters_and_buffers": frozen_digest(model), "backbone": backbone_digest(model)}
        write_json(run_dir / "metadata.json", metadata)
        original_backbone = frozen_digest(model)
        initial_layer4 = model.network.layer4[0].conv1.weight.detach().cpu().clone()
        initial_head = model.network.fc.weight.detach().cpu().clone()
        # Optimizer groups were recorded above.
        best_loss, history, best_predictions = float("inf"), [], None
        for epoch in range(1, (1 if smoke else cfg["epochs"]) + 1):
            model.train()
            total, elements = 0.0, 0
            for images, targets, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(device)); y = targets.to(device)
                loss = balanced_bce(logits, y, loss_weights) if loss_weights is not None else nn.functional.binary_cross_entropy_with_logits(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                    raise ValueError("Missing or nonfinite head gradients")
                optimizer.step()
                total += loss.item() * y.numel(); elements += y.numel()
            evaluation = evaluate(model, val_loader, device, balance=balance)
            val_loss, probabilities, targets, ids = evaluation[:4]
            history.append({"epoch": epoch, "training_loss": total/elements, "validation_loss": val_loss})
            if balance is not None:
                history[-1].update(weighted_training_loss=total/elements, weighted_validation_loss=evaluation[4])
            if is_better(val_loss, best_loss):
                best_loss = val_loss; best_predictions = probabilities.clone()
                torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": epoch, "validation_loss": val_loss, "concept_order": list(CONCEPTS),
                            "preprocessing": PREPROCESSING, "configuration": cfg, "input_hashes": hashes,
                            "weights": metadata["weights"], **({"balance": balance} if balance is not None else {})}, run_dir / "best.pt")
            write_json(run_dir / "history.json", history)
            print(f"Epoch {epoch:02d}: train={total/elements:.6f} validation={val_loss:.6f} best={best_loss:.6f}", flush=True)
        if frozen_digest(model) != original_backbone or torch.equal(initial_head, model.network.fc.weight.detach().cpu()):
            raise ValueError("Frozen parameters or buffers changed, or head failed to update")
        if model.strategy == "layer4_and_head" and torch.equal(initial_layer4, model.network.layer4[0].conv1.weight.detach().cpu()):
            raise ValueError("Trainable layer4 failed to update")
        reloaded, saved = load_checkpoint(run_dir / "best.pt", device)
        evaluation = evaluate(reloaded, val_loader, device, balance=balance)
        val_loss, probabilities, targets, ids = evaluation[:4]
        if not torch.equal(probabilities, best_predictions) or abs(val_loss-best_loss) > 1e-10:
            raise ValueError("Checkpoint reload did not exactly reproduce validation predictions")
        prevalence = np.array([[int(r[c]) for c in CONCEPTS] for r in train.records], dtype=np.float64).mean(axis=0)
        baseline = np.tile(prevalence, (len(ids), 1))
        result = {"best_epoch": saved["epoch"], "validation_loss": val_loss,
                  "model": concept_metrics(targets.numpy(), probabilities.numpy()),
                  "constant_baseline": concept_metrics(targets.numpy(), baseline), "training_prevalence": dict(zip(CONCEPTS, prevalence.tolist())),
                  "checks": {"frozen_parameters_and_batchnorm_unchanged": True, "layer4_trainable": model.strategy == "layer4_and_head", "head_changed": True, "checkpoint_predictions_identical": True},
                  "limitations": "Validation has no vehicle negatives and only three obstacle_close positives; probabilities are uncalibrated. Smoke uses training images only."}
        if balance is not None:
            result.update(balance=balance, weighted_validation_loss=evaluation[4], checkpoint_selection="unweighted_validation_loss")
        if model.strategy == "head_only":
            result["checks"]["backbone_and_batchnorm_unchanged"] = True
        write_json(run_dir / "metrics.json", result)
        with (run_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["sample_id", *[f"{c}_{kind}" for c in CONCEPTS for kind in ("target", "probability", "prediction")]]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for i, sid in enumerate(ids):
                row = {"sample_id": sid}
                for j, c in enumerate(CONCEPTS):
                    row.update({f"{c}_target": int(targets[i,j]), f"{c}_probability": float(probabilities[i,j]), f"{c}_prediction": int(probabilities[i,j] >= 0.5)})
                writer.writerow(row)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for key in (("weighted_training_loss", "validation_loss", "weighted_validation_loss") if balance is not None else ("training_loss", "validation_loss")):
            ax.plot([h["epoch"] for h in history], [h[key] for h in history], label=key)
        ax.set(xlabel="Epoch", ylabel="Mean binary cross-entropy", title=cfg["training_version"] if not smoke else "CUDA smoke check (training images only)")
        ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "loss.png"); plt.close(fig)
        _, final_hashes = read_records(root)
        final_hashes["training_config"] = sha256(config_path.read_bytes())
        if hashes != final_hashes:
            raise ValueError("Source metadata changed during training")
        write_json(run_dir / "status.json", {"status": "complete", "mode": "smoke" if smoke else "pilot", "best_epoch": saved["epoch"]})
        print(f"Completed: {run_dir}", flush=True)
        return run_dir, result
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Versioned training configuration; defaults to v1")
    parser.add_argument("--smoke", action="store_true", help="One batch/epoch, training images only, separate run")
    args = parser.parse_args()
    try:
        run(run_dir=args.run_dir, smoke=args.smoke, config=args.config)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Concept training failed: {exc}\n")


if __name__ == "__main__":
    main()
