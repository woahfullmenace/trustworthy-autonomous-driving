"""Compare a separately trained concept model through the unchanged baseline tree."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from scripts.train_concepts import seed_everything, validate_config, write_json
from src.concept_training import ConceptDataset, PREPROCESSING, concept_metrics, evaluate, read_records, sha256
from src.concepts import CONCEPTS
from src.balanced_loss import validate_balance
from src.decision_tree import action_metrics, compare_predictions, compatible_model, decision_path, load_tree_run


def validate_run(root, directory, records, hashes):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_bytes())
    status = json.loads((directory/'status.json').read_bytes())
    if status.get('status') != 'complete' or status.get('mode') != 'pilot' or metadata.get('mode') != 'pilot':
        raise ValueError('Comparison requires a completed full pilot, not smoke or failed runs')
    cfg = metadata['configuration']
    validate_config(cfg)
    default = {'concept-training-v1.0':'configs/concept_training.yaml','concept-training-v2.0':'configs/concept_finetuning.yaml','concept-training-v3.0':'configs/concept_balanced.yaml'}[cfg['training_version']]
    config = Path(root)/metadata.get('config_path', default)
    if yaml.safe_load(config.read_bytes()) != cfg:
        raise ValueError('Run configuration differs from its source configuration')
    expected = {**hashes, 'training_config': sha256(config.read_bytes())}
    if metadata['input_hashes'] != expected:
        raise ValueError('Stale run provenance')
    if metadata['concept_order'] != list(CONCEPTS) or metadata['preprocessing'] != PREPROCESSING:
        raise ValueError('Incompatible concept order or preprocessing')
    if metadata['training_ids'] != [r['sample_id'] for r in records['train']] or metadata['evaluation_ids'] != [r['sample_id'] for r in records['validation']] or metadata['evaluation_split'] != 'validation':
        raise ValueError('Run split membership mismatch')
    if cfg["training_version"] == "concept-training-v3.0":
        validate_balance(metadata.get("balance"), records["train"])
    return metadata


def prediction_table(path, records):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    p = np.array([[float(r[c+'_probability']) for c in CONCEPTS] for r in rows])
    compare_predictions(path, records, p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability range')
    return p


def snapshot(directories):
    return {str(p.resolve()): sha256(p.read_bytes()) for d in directories for p in sorted(Path(d).iterdir()) if p.is_file()}


def run(candidate_run, tree_run, root=ROOT, run_dir=None):
    root = Path(root).resolve()
    candidate, tree_dir = root/ candidate_run, root/ tree_run
    tree, context = load_tree_run(root, tree_dir)
    records, actions, _, _, baseline_checkpoint, _ = context
    if json.loads((tree_dir/'status.json').read_bytes()).get('status') != 'complete':
        raise ValueError('Completed tree run required')
    _, hashes = read_records(root)
    baseline = baseline_checkpoint.parent
    before = snapshot([baseline, candidate, tree_dir])
    bm = validate_run(root, baseline, records, hashes)
    cm = validate_run(root, candidate, records, hashes)
    if cm['configuration']['training_version'] not in ('concept-training-v2.0', 'concept-training-v3.0'):
        raise ValueError('Candidate must use v2 or v3')
    is_balanced = cm['configuration']['training_version'] == 'concept-training-v3.0'
    directories = [baseline,candidate,tree_dir]
    fine = root/'outputs/concept_training/finetune_v2'
    if is_balanced:
        # Verify the frozen v2 comparison and its original source hashes before reusing predictions.
        previous = root/'outputs/concept_comparison/finetune_v2'
        old_report = json.loads((previous/'metadata.json').read_bytes())
        if json.loads((previous/'status.json').read_bytes()).get('status') != 'complete' or old_report['input_hashes'] != hashes or old_report['artifact_hashes'] != snapshot([baseline,fine,tree_dir]):
            raise ValueError('Stale v2 comparison/source artifacts')
        fm = validate_run(root,fine,records,hashes)
        directories.extend([fine,previous])
        before = snapshot(directories)
    if cm['weights'] != bm['weights']:
        raise ValueError('Pretrained weight identities differ')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for validation reproduction')
    seed_everything(42)
    torch.set_num_threads(4)
    output_root = root/'outputs/concept_comparison'
    output = (root/run_dir).resolve() if run_dir else output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not output.is_relative_to(output_root) or output == output_root:
        raise ValueError('Comparison output must be inside outputs/concept_comparison')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', {'status': 'running'})
    try:
        dataset = ConceptDataset(root, records['validation'], 'validation')
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        model, saved = compatible_model(candidate/'best.pt', cm, 'cuda')
        evaluation = evaluate(model, loader, 'cuda', balance=cm.get('balance'))
        loss, probabilities, targets, ids = evaluation[:4]
        if ids != cm['evaluation_ids'] or abs(loss-saved['validation_loss']) > 1e-6:
            raise ValueError('Candidate validation reproduction mismatch')
        comparison = compare_predictions(candidate/'validation_predictions.csv', records['validation'], probabilities.numpy())
        # Validate baseline checkpoint as well; its predictions were regenerated by the frozen tree run.
        _, bs = compatible_model(baseline_checkpoint, bm, 'cpu')
        old = prediction_table(baseline/'validation_predictions.csv', records['validation'])
        probabilities = probabilities.numpy()
        truth = [actions[s] for s in ids]
        metrics = {'candidate_prediction_check': comparison, 'always_slow': action_metrics(truth, ['SLOW']*len(ids))}
        guesses = {}
        evaluations = [('baseline', old, bs['validation_loss']), ('candidate', probabilities, loss)]
        if is_balanced:
            _, fs = compatible_model(fine/'best.pt', fm, 'cpu')
            if fm['weights'] != cm['weights']:
                raise ValueError('v2/v3 pretrained identities differ')
            evaluations.insert(1, ('finetuned_v2', prediction_table(fine/'validation_predictions.csv',records['validation']), fs['validation_loss']))
        for name, p, val_loss in evaluations:
            guesses[name] = tree.predict((p >= .5).astype(int))
            metrics[name] = {'validation_loss': val_loss, 'concepts': concept_metrics(targets.numpy(), p), 'actions': action_metrics(truth, guesses[name])}
        if is_balanced:
            metrics['candidate']['weighted_validation_loss'] = evaluation[4]
            for name in (*[e[0] for e in evaluations], 'always_slow'):
                action_result = metrics[name] if name == 'always_slow' else metrics[name]['actions']
                action_result['slow_to_go_errors'] = action_result['confusion_matrix'][1][2]
        write_json(output/'metrics.json', metrics)
        rows = []
        for i, sid in enumerate(ids):
            row = {'sample_id': sid, 'target_action': truth[i]}
            for name, p, _ in evaluations:
                path = decision_path(tree, (p[i] >= .5).astype(int))
                if path['action'] != guesses[name][i]:
                    raise ValueError('Path and prediction disagree')
                row[name+'_action'] = guesses[name][i]
                row[name+'_decision_path'] = ' -> '.join([*path['steps'], path['action']])
                for j, c in enumerate(CONCEPTS):
                    row[c+'_target'] = int(targets[i,j])
                    row[name+'_'+c+'_probability'] = float(p[i,j])
                    row[name+'_'+c+'_prediction'] = int(p[i,j] >= .5)
            rows.append(row)
        with (output/'validation_comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if before != snapshot(directories) or hashes != read_records(root)[1]:
            raise ValueError('Source artifacts changed during comparison')
        write_json(output/'metadata.json', {'input_hashes': hashes, 'artifact_hashes': before, 'validation_ids': ids, 'concept_order': list(CONCEPTS), 'action_order': ['STOP','SLOW','GO'], 'tree_refitted': False})
        write_json(output/'status.json', {'status':'complete'})
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Comparison saved: {output}', flush=True)
        return output, metrics
    except Exception as exc:
        write_json(output/'status.json', {'status':'failed', 'error':str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--tree-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    try:
        run(args.candidate_run, args.tree_run, run_dir=args.run_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'Comparison failed: {exc}\n')
