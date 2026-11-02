"""Freeze sequence-grouped experimental partitions using metadata only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assign_splits(rows, seed=42, validation_images=80):
    """Reserve official val for test; choose whole train sequences totaling val size."""
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError("Require nonempty, unique sample IDs")
    groups = {}
    sources = {}
    for row in rows:
        sequence = row['sequence_id']
        if sequence != '_'.join(row['sample_id'].split('_')[:2]):
            raise ValueError("Sequence ID must match city and recording sequence in sample ID")
        source = row['official_split']
        if source not in {'train', 'val'}:
            raise ValueError("Only official train/val sources are allowed")
        if sequence in sources and sources[sequence] != source:
            raise ValueError("Sequence crosses official source splits")
        sources[sequence] = source
        groups.setdefault(sequence, []).append(row)
    train_groups = sorted(g for g in groups if sources[g] == 'train')
    if not any(s == 'val' for s in sources.values()):
        raise ValueError("Official val images are required for the test partition")
    total = sum(len(groups[g]) for g in train_groups)
    if not 0 < validation_images < total:
        raise ValueError("Validation count must leave nonempty train and validation partitions")
    random.Random(seed).shuffle(train_groups)
    # Subset sum on group sizes only; no label balancing or seed search.
    reachable = {0: ()}
    for sequence in train_groups:
        size = len(groups[sequence])
        for count, chosen in list(reachable.items()):
            target = count + size
            if target <= validation_images and target not in reachable:
                reachable[target] = (*chosen, sequence)
        if validation_images in reachable:
            break
    if validation_images not in reachable:
        raise ValueError("Exact validation count is impossible without breaking sequences")
    validation_groups = set(reachable[validation_images])
    result = []
    for row in sorted(rows, key=lambda r: r['sample_id']):
        partition = ('test' if row['official_split'] == 'val' else
                     'val' if row['sequence_id'] in validation_groups else 'train')
        result.append(dict(row, experimental_split=partition))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/splits.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = ROOT / config['source_manifest']
    with manifest.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    result = assign_splits(rows, config['seed'], config['validation_images'])
    partition_groups = {s: {r['sequence_id'] for r in result if r['experimental_split'] == s}
                        for s in ('train', 'val', 'test')}
    overlap = any(partition_groups[a] & partition_groups[b]
                  for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    if overlap or len(result) != len(rows):
        raise ValueError('Partition completeness or sequence isolation failed')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=[*fields, 'experimental_split'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(result)
    content = buffer.getvalue().encode('utf-8')
    report = {
        'protocol': 'official val -> test; seeded whole-sequence subset of official train -> val; remainder -> train',
        'seed': config['seed'], 'validation_images': config['validation_images'],
        'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'split_manifest_sha256': hashlib.sha256(content).hexdigest(),
        'counts': {s: {'images': sum(r['experimental_split'] == s for r in result),
                          'sequences': len({r['sequence_id'] for r in result if r['experimental_split'] == s})}
                   for s in ('train', 'val', 'test')},
        'sequence_overlap': overlap,
        'limitations': ['No label stratification or performance-based seed selection.',
                       'Entire subset had preliminary previews and aggregate mask inspection before partitioning.',
                       'Small, correlated sequence groups limit effective test sample size; not a real-driving safety benchmark.'],
    }
    files = {ROOT / config['split_manifest']: content,
             ROOT / config['protocol_record']: (json.dumps(report, indent=2) + '\n').encode('utf-8')}
    # Check all frozen outputs before writing either one.
    for path, data in files.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Frozen output differs: {path}. Review a versioned protocol change; refusing overwrite.')
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    print(json.dumps(report, indent=2))
    print('Frozen split files created or verified unchanged.')


if __name__ == '__main__':
    main()
