import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.concepts import CONCEPTS
from src.concept_training import (PREPROCESSING, ConceptDataset, FrozenConceptResNet, backbone_digest,
                                  concept_metrics, evaluate, is_better, load_checkpoint, read_records, sha256)
from scripts.train_concepts import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 16), (255, 0, 128)).save(path)
    row = {"sample_id": "synthetic", "image_path": "image.png", "image_sha256": sha256(path.read_bytes()),
           "width": "32", "height": "16", "experimental_split": "train",
           **{c: str(i % 2) for i, c in enumerate(CONCEPTS)}, **{c+"_valid": "1" for c in CONCEPTS}}
    return tmp_path, row


def test_image_shape_normalization_targets(example):
    root, row = example
    image, targets, sid = ConceptDataset(root, [row], "train")[0]
    assert image.shape == (3, 224, 448)
    expected = (torch.tensor([1.0, 0.0, 128/255]) - torch.tensor(PREPROCESSING["mean"])) / torch.tensor(PREPROCESSING["std"])
    torch.testing.assert_close(image[:, 0, 0], expected)
    assert targets.tolist() == [0, 1, 0, 1, 0]
    assert sid == "synthetic"


@pytest.mark.parametrize("problem", ["missing", "hash", "decode", "target", "validity", "split", "duplicates"])
def test_bad_inputs_rejected(example, problem):
    root, row = example
    path = root / "image.png"
    if problem == "missing":
        path.unlink()
    elif problem in ("hash", "decode"):
        path.write_bytes(b"broken")
        if problem == "decode":
            row["image_sha256"] = sha256(path.read_bytes())
    elif problem == "target":
        row[CONCEPTS[0]] = ""
    elif problem == "validity":
        row[CONCEPTS[0]+"_valid"] = "0"
    elif problem == "split":
        row["experimental_split"] = "test"
    with pytest.raises((ValueError, OSError)):
        ConceptDataset(root, [row, row] if problem == "duplicates" else [row], "train")


def test_test_split_rejected_before_image_read(example):
    root, row = example
    (root / "image.png").unlink()
    with pytest.raises(ValueError, match="only train/validation"):
        ConceptDataset(root, [row], "test")


def test_metadata_join_no_images_read(monkeypatch):
    original = Path.read_bytes
    def guarded(path):
        assert path.suffix != ".png", "Metadata join attempted to read an image"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    records, hashes = read_records(ROOT)
    assert set(records) == {"train", "validation"}
    assert len(records["train"]) == 342 and len(records["validation"]) == 74
    assert not {r["sequence_id"] for r in records["train"]} & {r["sequence_id"] for r in records["validation"]}
    assert "experimental_splits.csv" in hashes


def test_missing_frozen_splits_never_generated(tmp_path):
    with pytest.raises(ValueError, match="never generates"):
        read_records(tmp_path)
    assert not list(tmp_path.rglob("*"))


def test_frozen_backbone_head_gradients_and_checkpoint(tmp_path):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = FrozenConceptResNet().train()
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = backbone_digest(model)
    head = model.network.fc.weight.detach().clone()
    x, y = torch.randn(2, 3, 224, 448), torch.tensor([[0., 1., 0., 1., 0.], [1., 1., 1., 0., 1.]])
    optimizer = torch.optim.Adam(model.network.fc.parameters(), lr=0.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.network.fc.parameters())
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("network.fc."))
    optimizer.step()
    assert not torch.equal(head, model.network.fc.weight)
    assert before == backbone_digest(model)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "concept_order": list(CONCEPTS), "preprocessing": PREPROCESSING}, path)
    reloaded, _ = load_checkpoint(path, "cpu")
    with torch.no_grad():
        assert torch.equal(model.eval()(x), reloaded(x))


def test_best_checkpoint_ties_and_nonfinite():
    assert is_better(0.4, 0.5)
    assert not is_better(0.5, 0.5)
    assert not is_better(0.6, 0.5)
    with pytest.raises(ValueError):
        is_better(float("nan"), 0.5)


def test_metrics_undefined_and_missed_positives():
    y = np.zeros((4, 5)); y[:, 1] = 1; y[:2, 2] = 1
    p = np.zeros((4, 5)); p[:, 1] = 0.5; p[[0, 2], 2] = 0.7
    result = concept_metrics(y, p)
    empty = result["concepts"][CONCEPTS[0]]
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] is None
    assert empty["specificity"] == 1
    vehicle = result["concepts"][CONCEPTS[1]]
    assert vehicle["specificity"] is None and vehicle["f1"] == 1
    cyclist = result["concepts"][CONCEPTS[2]]
    assert (cyclist["tp"], cyclist["fp"], cyclist["fn"], cyclist["tn"]) == (1, 1, 1, 1)
    assert cyclist["f1"] == 0.5
    assert result["macro_f1"] == 0.75 and result["macro_f1_concepts"] == 2
    missed = concept_metrics(np.ones((2, 5)), np.zeros((2, 5)))
    assert missed["macro_f1"] == 0
    assert missed["concepts"][CONCEPTS[0]]["precision"] is None


def test_evaluation_loss_weights_last_batch():
    logits = torch.tensor([[0.]*5, [1.]*5, [-2.]*5])
    targets = torch.tensor([[1.]*5, [0.]*5, [1.]*5])
    loader = DataLoader(TensorDataset(logits, targets, torch.arange(3)), batch_size=2)
    loss, p, y, ids = evaluate(torch.nn.Identity(), loader, "cpu")
    assert loss == pytest.approx(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item())
    assert torch.equal(p, logits.sigmoid()) and torch.equal(y, targets)
    assert len(ids) == 3


def test_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run(ROOT)
