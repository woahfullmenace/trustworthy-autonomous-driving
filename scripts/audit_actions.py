"""Training-only action proxy audit; no model fitting or held-out inspection."""
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from audit_concepts import CONCEPTS, ROOT, training_rows

VERSION = 'v1'


def action_target(row):
    values = [row[c] for c in CONCEPTS]
    if any(pd.notna(v) and v not in (0, 1) for v in values):
        raise ValueError('Concept values must be binary or missing')
    if any(pd.isna(v) for v in values):
        return '', 'unknown_concept', False
    pedestrian, vehicle, cyclist, clear, close = map(bool, values)
    vulnerable = pedestrian or cyclist
    if close:
        return 'STOP', 'close_obstacle', True
    if not clear and vulnerable:
        return 'STOP', 'unclear_corridor_with_vulnerable_user', True
    if vulnerable:
        return 'SLOW', 'vulnerable_user_in_scene', True
    if not clear:
        return 'SLOW', 'corridor_not_clear', True
    return 'GO', 'clear_corridor_no_vulnerable_user_or_close_obstacle', True


def main():
    directory = ROOT / 'outputs/concept_audit/v1-training-audit'
    rules = ROOT / 'configs/concepts.json'
    manifest = ROOT / 'data/manifests/experimental_splits.csv'
    summary = json.loads((directory / 'audit_summary.json').read_text())
    for path, key in [(rules, 'rules_sha256'), (manifest, 'split_manifest_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
            raise ValueError('Concept audit is stale; rerun audit_concepts.py')
    frame = pd.read_csv(directory / 'training_concepts.csv')
    expected = training_rows(pd.read_csv(manifest))
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(expected.sample_id) or not frame.experimental_split.eq('train').all():
        raise ValueError('Action audit requires exactly the frozen training samples')
    results = [action_target(row) for row in frame.to_dict('records')]
    frame[['action', 'action_reason', 'action_eligible']] = pd.DataFrame(results, index=frame.index)
    output = ROOT / 'outputs/action_audit' / VERSION
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'training_actions.csv', index=False)
    truth = []
    for bits in itertools.product((0, 1), repeat=5):
        row = dict(zip(CONCEPTS, bits))
        action, reason, eligible = action_target(row)
        truth.append(dict(row, action=action, action_reason=reason))
    pd.DataFrame(truth).to_csv(output / 'policy_truth_table.csv', index=False)
    chosen = frame.sort_values('sample_id').groupby('action_reason', sort=True).head(2)
    chosen.to_csv(output / 'audit_examples.csv', index=False)
    for start in range(0, len(chosen), 4):
        batch = chosen.iloc[start:start+4]
        sheet = Image.new('RGB', (1000, 300 * len(batch)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(batch.to_dict('records')):
            path = (ROOT / row['image_path']).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError('Image path outside repository')
            with Image.open(path) as image:
                sheet.paste(image.convert('RGB').resize((500, 250)), (0, i*300+40))
            draw.text((8, i*300+8), row['sample_id']+' TRAIN', fill='black')
            title = row['action'] or 'UNLABELED (excluded from action supervision)'
            draw.text((515, i*300+40), title, fill='black')
            draw.text((515, i*300+65), row['action_reason'], fill='black')
            for j, c in enumerate(CONCEPTS):
                draw.text((515, i*300+100+j*24), f'{c}: {row[c]}', fill='black')
        sheet.save(output / f'audit_{start//4+1:02d}.png')
    report = {'policy_version': VERSION, 'training_images': len(frame),
              'eligible_images': int(frame.action_eligible.sum()),
              'excluded_unknown_images': int((~frame.action_eligible).sum()),
              'action_counts': frame.loc[frame.action_eligible, 'action'].value_counts().to_dict(),
              'reason_counts': frame.action_reason.value_counts().to_dict(),
              'concept_rules_sha256': summary['rules_sha256'],
              'split_manifest_sha256': summary['split_manifest_sha256'],
              'input_concepts_sha256': hashlib.sha256((directory/'training_concepts.csv').read_bytes()).hexdigest(),
              'policy_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': 'Annotation-derived policy imitation. Unknown labels are not model abstention. No held-out labels generated.'}
    (output / 'audit_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
