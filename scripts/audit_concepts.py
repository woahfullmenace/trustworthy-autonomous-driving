"""Generate reproducible training-only concept proxy labels and audit sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

CONCEPTS = ['pedestrian_present', 'vehicle_present', 'cyclist_present', 'road_clear', 'obstacle_close']


def measure(mask, cfg):
    if mask.ndim != 2 or not set(np.unique(mask).tolist()) <= set(range(34)) | {255}:
        raise ValueError('Require a 2D Cityscapes raw label-ID mask')
    h, w = mask.shape
    x0, x1 = [int(x * w) for x in cfg['corridor']['x']]
    y0, y1 = [int(y * h) for y in cfg['corridor']['y']]
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError('Empty corridor')
    valid = ~np.isin(region, cfg['ignored_ids'])
    valid_count = int(valid.sum())
    coverage = valid_count / region.size
    result = {}
    for name, key in [('person_fraction', 'pedestrian_ids'), ('vehicle_fraction', 'vehicle_ids'),
                      ('bicycle_fraction', 'cyclist_bicycle_ids'), ('rider_fraction', 'cyclist_rider_ids')]:
        result[name] = float(np.isin(mask, cfg[key]).mean())
    road = float(((region == 7) & valid).sum() / valid_count) if valid_count else 0.0
    obstacle = float((np.isin(region, cfg['obstacle_ids']) & valid).sum() / valid_count) if valid_count else 0.0
    usable = coverage >= cfg['minimum_valid_corridor_fraction']
    result.update(corridor_valid_fraction=coverage, corridor_road_fraction=road,
                  corridor_obstacle_fraction=obstacle, corridor_usable=int(usable))
    threshold = cfg['presence_min_fraction']
    result.update(pedestrian_present=int(result['person_fraction'] >= threshold),
                  vehicle_present=int(result['vehicle_fraction'] >= threshold),
                  cyclist_present=int(min(result['bicycle_fraction'], result['rider_fraction']) >= threshold),
                  road_clear=int(usable and road >= cfg['road_clear_min_road_fraction']
                                 and obstacle <= cfg['road_clear_max_obstacle_fraction']),
                  obstacle_close=int(usable and obstacle >= cfg['obstacle_close_min_fraction']))
    # Unknown corridor targets must never be silently treated as negatives.
    if not usable:
        result['road_clear'] = None
        result['obstacle_close'] = None
    return result


def training_rows(frame):
    if frame.sample_id.duplicated().any():
        raise ValueError('Duplicate sample IDs')
    if not frame.experimental_split.isin(['train', 'val', 'test']).all():
        raise ValueError('Unknown experimental partition')
    if (frame.groupby('sequence_id').experimental_split.nunique() > 1).any():
        raise ValueError('Recording sequence crosses partitions')
    return frame[frame.experimental_split == 'train'].sort_values('sample_id').copy()


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Path outside repository')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/concepts.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = ROOT / 'data/manifests/experimental_splits.csv'
    protocol = json.loads((ROOT / 'data/manifests/split_protocol.json').read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol['split_manifest_sha256']:
        raise ValueError('Frozen split manifest hash mismatch')
    samples = training_rows(pd.read_csv(manifest))
    if samples.empty:
        raise ValueError('No training samples')
    output = ROOT / 'outputs/concept_audit' / cfg['version']
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(samples.to_dict('records'), 1):
        path = local_path(sample['mask_path'])
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if content_hash != sample['mask_sha256']:
            raise ValueError(f'Mask hash mismatch: {path}')
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (sample['width'], sample['height']):
                raise ValueError(f'Invalid mask: {path}')
            measurements = measure(np.asarray(image), cfg)
        records.append(dict(sample, **measurements))
        if index % 100 == 0 or index == len(samples):
            print(f'Audited {index}/{len(samples)} training masks', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'training_concepts.csv', index=False)
    frame.select_dtypes(include='number').drop(columns=['width', 'height'], errors='ignore').describe().to_csv(output / 'training_statistics.csv')
    counts = {c: {'positive': int((frame[c] == 1).sum()), 'negative': int((frame[c] == 0).sum()),
                  'unknown': int(frame[c].isna().sum())} for c in CONCEPTS}
    # Deterministic positive, negative and near-boundary examples per concept.
    reasons = {}
    boundary = {
        'pedestrian_present': abs(frame.person_fraction - cfg['presence_min_fraction']),
        'vehicle_present': abs(frame.vehicle_fraction - cfg['presence_min_fraction']),
        'cyclist_present': abs(frame[['bicycle_fraction', 'rider_fraction']].min(axis=1) - cfg['presence_min_fraction']),
        'road_clear': pd.concat([abs(frame.corridor_road_fraction - cfg['road_clear_min_road_fraction']),
                                 abs(frame.corridor_obstacle_fraction - cfg['road_clear_max_obstacle_fraction'])], axis=1).min(axis=1),
        'obstacle_close': abs(frame.corridor_obstacle_fraction - cfg['obstacle_close_min_fraction'])}
    for concept in CONCEPTS:
        for value in (0, 1):
            matches = frame[frame[concept] == value]
            if len(matches):
                sample_id = matches.iloc[0].sample_id
                reasons.setdefault(sample_id, []).append(f'{concept}={value}')
        idx = boundary[concept].idxmin()
        reasons.setdefault(frame.loc[idx, 'sample_id'], []).append(f'{concept} boundary')
    for sample_id in frame.loc[frame.corridor_usable == 0, 'sample_id'].head(3):
        reasons.setdefault(sample_id, []).append('insufficient valid corridor')
    chosen = frame.set_index('sample_id').loc[list(reasons)].reset_index()
    chosen['selection_reasons'] = chosen.sample_id.map(lambda s: '; '.join(reasons[s]))
    chosen.to_csv(output / 'audit_examples.csv', index=False)
    colors = {7: (128, 64, 128), 24: (220, 20, 60), 25: (255, 0, 0),
              26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60, 100),
              31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}
    for start in range(0, len(chosen), 4):
        batch = chosen.iloc[start:start + 4]
        fig, axes = plt.subplots(len(batch), 2, figsize=(16, 3.6 * len(batch)), squeeze=False)
        for pair, sample in zip(axes, batch.to_dict('records')):
            with Image.open(local_path(sample['image_path'])) as image:
                rgb = np.asarray(image.convert('RGB').resize((768, 384)))
            with Image.open(local_path(sample['mask_path'])) as image:
                mask = np.asarray(image.resize((768, 384), Image.Resampling.NEAREST))
            overlay = rgb.copy()
            for label_id, color in colors.items():
                pixels = mask == label_id
                overlay[pixels] = (0.5 * rgb[pixels] + 0.5 * np.array(color)).astype(np.uint8)
            for ax, picture in zip(pair, (rgb, overlay)):
                ax.imshow(picture)
                x0, x1 = cfg['corridor']['x']; y0, y1 = cfg['corridor']['y']
                ax.add_patch(Rectangle((x0 * 768, y0 * 384), (x1-x0)*768, (y1-y0)*384,
                                       fill=False, edgecolor='yellow', linewidth=2))
                ax.axis('off')
            pair[0].set_title(sample['sample_id'] + ' (TRAIN)', fontsize=10)
            short = ['ped', 'veh', 'cyc', 'clear', 'close']
            values = ', '.join(f'{s}={sample[c]}' for s, c in zip(short, CONCEPTS))
            pair[1].set_title(values + '\n' + f"road={sample['corridor_road_fraction']:.3f}, obstacle={sample['corridor_obstacle_fraction']:.3f}, valid={sample['corridor_valid_fraction']:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(output / f'audit_{start // 4 + 1:02d}.png', dpi=100)
        plt.close(fig)
    report = {'rule_version': cfg['version'], 'training_images': len(frame),
              'training_sequences': int(frame.sequence_id.nunique()), 'concept_counts': counts,
              'split_manifest_sha256': protocol['split_manifest_sha256'],
              'rules_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'audit_sample_ids': list(reasons), 'status': 'training audit; no validation/test labels generated'}
    (output / 'audit_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts, indent=2))
    print(f'Saved {len(chosen)} training audit examples to {output}')


if __name__ == '__main__':
    main()
