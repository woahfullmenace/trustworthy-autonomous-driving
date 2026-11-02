"""Data, frozen ResNet18 and metrics for the first concept-training pilot."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from src.concepts import CONCEPTS

PREPROCESSING = {"image_size": [224, 448], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_records(root):
    """Validate the entire metadata chain; never create splits or read source images."""
    from scripts.prepare_splits import run as validate_splits
    root = Path(root)
    folder = root / "data/manifests"
    for name in ("experimental_splits.csv", "split_summary.json"):
        if not (folder / name).is_file():
            raise ValueError("Frozen split artifacts are required; training never generates splits")
    report = validate_splits(root)
    def table(name):
        with (folder / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    manifest = {r["sample_id"]: r for r in table("subset.csv")}
    labels = {r["sample_id"]: r for r in table("concept_labels.csv")}
    result = {"train": [], "validation": []}
    for entry in table("experimental_splits.csv"):
        split = entry["experimental_split"]
        if split not in result:
            continue
        sid = entry["sample_id"]
        result[split].append({**manifest[sid], **labels[sid], "experimental_split": split})
    hashes = {**report["input_hashes"], **{name: sha256((folder / name).read_bytes()) for name in ("experimental_splits.csv", "split_summary.json")}}
    return result, hashes


class ConceptDataset(Dataset):
    def __init__(self, root, records, split):
        if split not in ("train", "validation"):
            raise ValueError("This milestone loads only train/validation images")
        self.root, self.records = Path(root).resolve(), records
        self.transform = transforms.Compose([
            transforms.Resize(PREPROCESSING["image_size"], interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])
        seen = set()
        for row in records:
            sid = row["sample_id"]
            if sid in seen or row["experimental_split"] != split:
                raise ValueError("Duplicate ID or incorrect experimental split")
            seen.add(sid)
            if any(row.get(c) not in ("0", "1") or row.get(c+"_valid") != "1" for c in CONCEPTS):
                raise ValueError(f"Invalid concept target: {sid}")
            self._read(row)  # Verify hash and decoding before any training starts.
        if not records:
            raise ValueError("Empty dataset")

    def _read(self, row):
        path = (self.root / row["image_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes repository")
        data = path.read_bytes()
        if sha256(data) != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Image dimensions mismatch: {row['sample_id']}")
            return im.convert("RGB")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return self.transform(self._read(row)), torch.tensor([int(row[c]) for c in CONCEPTS], dtype=torch.float32), row["sample_id"]


class FrozenConceptResNet(nn.Module):
    def __init__(self, weights=None, strategy="head_only"):
        super().__init__()
        if strategy not in ("head_only", "layer4_and_head"):
            raise ValueError("Unknown training strategy")
        self.strategy = strategy
        self.network = models.resnet18(weights=weights)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.network.fc = nn.Linear(self.network.fc.in_features, len(CONCEPTS))
        if strategy == "layer4_and_head":
            for parameter in self.network.layer4.parameters():
                parameter.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()  # Includes BatchNorm running means, variances and counters.
        self.network.fc.train(mode)
        return self

    def forward(self, images):
        return self.network(images)


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("network.fc."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def concept_metrics(targets, probabilities):
    y, p = np.asarray(targets), np.asarray(probabilities)
    if y.ndim != 2 or y.shape[1] != len(CONCEPTS) or y.shape != p.shape or not len(y):
        raise ValueError("Expected matching nonempty N x 5 targets/probabilities")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid targets or probabilities")
    predicted = p >= 0.5
    results = {}
    for j, c in enumerate(CONCEPTS):
        positive, guess = y[:, j] == 1, predicted[:, j]
        tp = int(np.sum(positive & guess)); fn = int(np.sum(positive & ~guess))
        fp = int(np.sum(~positive & guess)); tn = int(np.sum(~positive & ~guess))
        ratios = {"precision": (tp, tp+fp), "recall": (tp, tp+fn), "f1": (2*tp, 2*tp+fp+fn), "specificity": (tn, tn+fp)}
        results[c] = {"positive": tp+fn, "negative": tn+fp, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                      **{name: num/den if den else None for name, (num, den) in ratios.items()},
                      "undefined": {name: "zero denominator" for name, (_, den) in ratios.items() if not den}}
    f1 = [v["f1"] for v in results.values() if v["f1"] is not None]
    return {"concepts": results, "macro_f1": sum(f1)/len(f1) if f1 else None, "macro_f1_concepts": len(f1), "threshold": 0.5}


def is_better(loss, best_loss):
    if not np.isfinite(loss):
        raise ValueError("Nonfinite validation loss")
    return loss < best_loss


def evaluate(model, loader, device, balance=None):
    model.eval()
    from src.balanced_loss import balanced_bce, weight_tensors
    weights = weight_tensors(balance, device) if balance is not None else None
    weighted_total = 0.0
    total, elements, probabilities, targets, ids = 0.0, 0, [], [], []
    with torch.no_grad():
        for images, y, sample_ids in loader:
            logits = model(images.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation loss")
            if weights is not None:
                weighted_total += balanced_bce(logits, y.to(device), weights, reduction="sum").item()
            total += loss.item(); elements += y.numel()
            probabilities.append(logits.sigmoid().cpu()); targets.append(y); ids.extend(sample_ids)
    result = (total/elements, torch.cat(probabilities), torch.cat(targets), ids)
    return (*result, weighted_total/elements) if balance is not None else result


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["concept_order"] != list(CONCEPTS) or saved["preprocessing"] != PREPROCESSING:
        raise ValueError("Checkpoint concept order or preprocessing mismatch")
    if saved.get("configuration", {}).get("training_version") == "concept-training-v3.0":
        from src.balanced_loss import validate_balance
        validate_balance(saved.get("balance"))
    model = FrozenConceptResNet(strategy=saved.get("configuration", {}).get("strategy", "head_only"))
    model.load_state_dict(saved["model_state"])
    return model.to(device).eval(), saved


def frozen_digest(model):
    """Hash frozen parameters and every buffer, including BatchNorm statistics."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, cfg):
    groups = [{"params": list(model.network.fc.parameters()), "lr": cfg["learning_rate"], "name": "head"}]
    if model.strategy == "layer4_and_head":
        groups.insert(0, {"params": list(model.network.layer4.parameters()), "lr": cfg["backbone_learning_rate"], "name": "layer4"})
    return torch.optim.Adam(groups, weight_decay=cfg["weight_decay"])
