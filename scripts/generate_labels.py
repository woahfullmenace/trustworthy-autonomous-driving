"""Freeze v1, then apply unchanged proxies to every frozen partition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from audit_concepts import ROOT, CONCEPTS, measure, local_path, training_rows
from audit_actions import action_target

LOCK = ROOT / 'configs/label_protocol_v1.json'
SOURCES = ['configs/concepts.json', 'scripts/audit_concepts.py', 'scripts/audit_actions.py',
           'scripts/generate_labels.py', 'data/manifests/experimental_splits.csv',
           'data/manifests/split_protocol.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(lock):
    if set(lock['sha256']) != set(SOURCES):
        raise ValueError('Frozen source list differs')
    for name, expected in lock['sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}; do not retune v1')


def write_unchanged(files):
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite frozen output: {path}')
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Record protocol before reading held-out masks; no labels generated')
    args = parser.parse_args()
    if args.freeze:
        lock = {'version': 'v1', 'frozen_date': '2026-11-01',
                'sha256': {p: digest(ROOT / p) for p in SOURCES},
                'unknown_policy': 'Blank action if any concept is missing; retain sample and known concept targets.',
                'action_mapping': {'STOP': 0, 'SLOW': 1, 'GO': 2},
                'prior_exposure': 'Whole-subset previews/statistics preceded partitioning; rule development used training only.',
                'held_out_use': 'Apply unchanged rules and report counts only; no retuning from held-out labels.'}
        write_unchanged({LOCK: (json.dumps(lock, indent=2)+'\n').encode()})
        print('Protocol frozen; no held-out masks read.')
        return
    lock = json.loads(LOCK.read_text())
    verify_sources(lock)
    cfg = json.loads((ROOT / 'configs/concepts.json').read_text())
    manifest = ROOT / 'data/manifests/experimental_splits.csv'
    protocol = json.loads((ROOT / 'data/manifests/split_protocol.json').read_text())
    if digest(manifest) != protocol['split_manifest_sha256']:
        raise ValueError('Split manifest mismatch')
    frame = pd.read_csv(manifest)
    training_rows(frame)  # Validate IDs, partition names, and sequence isolation.
    records = []
    for index, row in enumerate(frame.sort_values('sample_id').to_dict('records'), 1):
        path = local_path(row['mask_path'])
        if digest(path) != row['mask_sha256']:
            raise ValueError(f'Mask hash mismatch: {path}')
        with Image.open(path) as mask:
            if mask.mode != 'L' or mask.size != (row['width'], row['height']):
                raise ValueError(f'Invalid mask: {path}')
            values = measure(np.asarray(mask), cfg)
        action, reason, eligible = action_target(values)
        records.append(dict(row, **values, action=action, action_reason=reason,
                            action_eligible=eligible, action_id=lock['action_mapping'].get(action),
                            label_protocol='v1'))
        if index % 100 == 0:
            print(f'Labeled {index}/{len(frame)} masks', flush=True)
    labeled = pd.DataFrame(records)
    for col in [*CONCEPTS, 'action_id']:
        labeled[col] = labeled[col].astype('Int64')
    summary = {'protocol_version': 'v1', 'protocol_sha256': digest(LOCK), 'partitions': {}}
    for split in ['train', 'val', 'test']:
        group = labeled[labeled.experimental_split == split]
        summary['partitions'][split] = {
            'images': len(group), 'sequences': int(group.sequence_id.nunique()),
            'action_eligible': int(group.action_eligible.sum()),
            'action_unknown': int((~group.action_eligible).sum()),
            'actions': {a: int((group.action == a).sum()) for a in ['STOP', 'SLOW', 'GO']},
            'concepts': {c: {'positive': int((group[c] == 1).sum()), 'negative': int((group[c] == 0).sum()),
                             'unknown': int(group[c].isna().sum())} for c in CONCEPTS}}
    content = labeled.to_csv(index=False, lineterminator='\n').encode()
    summary['labels_sha256'] = hashlib.sha256(content).hexdigest()
    destination = ROOT / 'data/manifests'
    write_unchanged({destination/'labels_v1.csv': content,
                     destination/'label_summary_v1.json': (json.dumps(summary, indent=2)+'\n').encode()})
    for split, counts in summary['partitions'].items():
        print(split, {k: v for k, v in counts.items() if k != 'concepts'})
    print('Frozen labels created or verified unchanged.')


if __name__ == '__main__':
    main()
