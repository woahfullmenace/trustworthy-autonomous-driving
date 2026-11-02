"""Direct action prediction with the same image preprocessing and frozen split."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models
from src.concept_training import ConceptDataset, PREPROCESSING, read_records
from src.decision_tree import ACTIONS

CONFIG = dict(training_version='action-training-v1.0', seed=42, epochs=20, batch_size=16,
              learning_rate=.001, backbone_learning_rate=.0001, weight_decay=.0001,
              num_workers=0, cpu_threads=4, weights='IMAGENET1K_V1', strategy='layer4_and_head',
              loss='CrossEntropyLoss', class_order=list(ACTIONS), **PREPROCESSING)


def validate_config(cfg):
    if cfg != CONFIG:
        raise ValueError('Configuration differs from action-training-v1.0')


def read_action_records(root):
    records, hashes = read_records(root)  # Validates full action/split provenance first.
    with (Path(root)/'data/manifests/action_labels.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    actions = {}
    for row in rows:
        if row['sample_id'] in actions or row['action'] not in ACTIONS:
            raise ValueError('Duplicate ID or invalid action label')
        actions[row['sample_id']] = row
    for split in records:
        for row in records[split]:
            action = actions.get(row['sample_id'])
            if action is None or action['official_split'] != row['official_split'] or action['sequence_id'] != row['sequence_id']:
                raise ValueError('Action ID/split/sequence mismatch')
            row['action'] = action['action']
    return records, hashes


class ActionDataset(ConceptDataset):
    def __init__(self, root, records, split):
        if split not in ('train', 'validation') or any(r.get('experimental_split') != split for r in records):
            raise ValueError('Only matching train/validation samples may be loaded')
        if any(r.get('action') not in ACTIONS for r in records):
            raise ValueError('Invalid action target')
        super().__init__(root, records, split)

    def __getitem__(self, index):
        row = self.records[index]
        return self.transform(self._read(row)), torch.tensor(ACTIONS.index(row['action']), dtype=torch.long), row['sample_id']


class ActionResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.strategy = 'layer4_and_head'
        self.network = models.resnet18(weights=weights)
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.network.fc = nn.Linear(self.network.fc.in_features, len(ACTIONS))
        for p in self.network.layer4.parameters():
            p.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()  # ResNet has no dropout; preserves every BatchNorm statistic.
        self.network.fc.train(mode)
        return self

    def forward(self, images):
        return self.network(images)


def evaluate(model, loader, device):
    model.eval()
    total, count, ps, ys, ids = 0., 0, [], [], []
    with torch.no_grad():
        for images, y, sample_ids in loader:
            logits = model(images.to(device))
            loss = nn.functional.cross_entropy(logits, y.to(device), reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite action evaluation loss')
            total += loss.item(); count += len(y)
            ps.append(logits.softmax(dim=1).cpu()); ys.append(y); ids.extend(sample_ids)
    if not count:
        raise ValueError('Empty evaluation dataset')
    return total/count, torch.cat(ps), torch.cat(ys), ids


def load_checkpoint(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    validate_config(saved['configuration'])
    if saved['class_order'] != list(ACTIONS) or saved['preprocessing'] != PREPROCESSING:
        raise ValueError('Incompatible action mapping or preprocessing')
    model = ActionResNet()
    model.load_state_dict(saved['model_state'])
    return model.to(device).eval(), saved


def action_names(probabilities):
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Expected normalized N x 3 action probabilities')
    return [ACTIONS[i] for i in p.argmax(axis=1)]
