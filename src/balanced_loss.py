"""Training-only class balance for the five binary concepts."""
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from src.concepts import CONCEPTS


def compute_balance(records):
    if not records or any(r.get('experimental_split') != 'train' for r in records):
        raise ValueError('Balance requires training rows only')
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training IDs for balance')
    if any(r.get(c) not in ('0','1') or r.get(c+'_valid') != '1' for r in records for c in CONCEPTS):
        raise ValueError('Invalid training concept for balance')
    y = np.array([[int(r[c]) for c in CONCEPTS] for r in records], dtype=np.int64)
    positive = y.sum(axis=0); negative = len(y)-positive
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError('Every concept requires positive and negative training examples')
    source = json.dumps(list(zip(ids,y.tolist())),separators=(',',':')).encode()
    return {'version':'binary-balance-v1.0','concept_order':list(CONCEPTS),'training_count':len(y),
            'training_rows_sha256':hashlib.sha256(source).hexdigest(),
            'positive_counts':positive.tolist(),'negative_counts':negative.tolist(),
            'positive_weights':(len(y)/(2*positive)).tolist(),'negative_weights':(len(y)/(2*negative)).tolist()}


def validate_balance(balance, records=None):
    if records is not None:
        if balance != compute_balance(records):
            raise ValueError('Stale balance weights or training provenance')
        return
    if not isinstance(balance,dict) or balance.get('version') != 'binary-balance-v1.0' or balance.get('concept_order') != list(CONCEPTS):
        raise ValueError('Invalid balance metadata')
    n=balance['training_count']
    pos=np.asarray(balance['positive_counts']); neg=np.asarray(balance['negative_counts'])
    if pos.shape!=(5,) or neg.shape!=(5,) or n<=0 or np.any(pos<=0) or np.any(neg<=0) or np.any(pos+neg != n):
        raise ValueError('Invalid balance counts')
    if not np.array_equal(np.asarray(balance['positive_weights']),n/(2*pos)) or not np.array_equal(np.asarray(balance['negative_weights']),n/(2*neg)):
        raise ValueError('Invalid balance weights')


def weight_tensors(balance, device):
    validate_balance(balance)
    return (torch.tensor(balance['positive_weights'],device=device,dtype=torch.float32),
            torch.tensor(balance['negative_weights'],device=device,dtype=torch.float32))


def balanced_bce(logits, targets, weights, reduction='mean'):
    if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != len(CONCEPTS):
        raise ValueError('Expected matching N x 5 logits and targets')
    losses = F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    losses = losses * torch.where(targets == 1,weights[0],weights[1])
    if reduction == 'sum':
        return losses.sum()
    if reduction == 'mean':
        return losses.mean()
    raise ValueError('Unsupported balanced reduction')
