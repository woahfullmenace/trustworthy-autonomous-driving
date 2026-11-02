"""Compare the direct-action baseline and two saved concept pipelines on validation only."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_concepts import seed_everything, write_json
from scripts.compare_concept_runs import prediction_table, snapshot, validate_run as validate_concept_run
from src.concept_training import PREPROCESSING, sha256
from src.action_training import ActionDataset, action_names, evaluate, load_checkpoint, read_action_records, validate_config
from src.decision_tree import ACTIONS, action_metrics, decision_path, load_tree_run
from src.concepts import CONCEPTS
import yaml


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def check_ids_targets(rows, records, target_key='target'):
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in records]:
        raise ValueError('Validation IDs/order mismatch')
    if [r[target_key] for r in rows] != [r['action'] for r in records]:
        raise ValueError('Validation action targets mismatch')


def concept_sources(root, records, hashes):
    root = Path(root)
    baseline = root/'outputs/concept_training/pilot_v1'
    candidate = root/'outputs/concept_training/finetune_v2'
    tree_dir = root/'outputs/decision_tree/pilot_v1_complete'
    comparison = root/'outputs/concept_comparison/finetune_v2'
    meta = json.loads((comparison/'metadata.json').read_bytes())
    if json.loads((comparison/'status.json').read_bytes()).get('status') != 'complete':
        raise ValueError('Completed concept comparison required')
    if meta['input_hashes'] != hashes or meta['validation_ids'] != [r['sample_id'] for r in records['validation']]:
        raise ValueError('Stale concept comparison provenance or IDs')
    if meta.get('concept_order') != list(CONCEPTS) or meta.get('action_order') != list(ACTIONS) or meta.get('tree_refitted') is not False:
        raise ValueError('Concept comparison mapping or protocol mismatch')
    if meta['artifact_hashes'] != snapshot([baseline,candidate,tree_dir]):
        raise ValueError('Modified concept/tree source artifacts')
    validate_concept_run(root,baseline,records,hashes)
    validate_concept_run(root,candidate,records,hashes)
    tree,_ = load_tree_run(root,tree_dir)
    rows = read_csv(comparison/'validation_comparison.csv')
    check_ids_targets(rows,records['validation'],'target_action')
    saved_metrics = json.loads((comparison/'metrics.json').read_bytes())
    predictions = {}
    truth = [r['action'] for r in records['validation']]
    for name, directory in [('baseline',baseline),('candidate',candidate)]:
        probabilities = prediction_table(directory/'validation_predictions.csv',records['validation'])
        prediction = tree.predict((probabilities >= .5).astype(int)).tolist()
        if prediction != [r[name+'_action'] for r in rows] or action_metrics(truth,prediction) != saved_metrics[name]['actions']:
            raise ValueError('Saved concept comparison differs from verified source predictions')
        for i,row in enumerate(rows):
            for j,c in enumerate(CONCEPTS):
                if int(row[c+'_target']) != int(records['validation'][i][c]) or float(row[name+'_'+c+'_probability']) != probabilities[i,j] or int(row[name+'_'+c+'_prediction']) != int(probabilities[i,j] >= .5):
                    raise ValueError('Concept comparison probabilities/targets mismatch')
            path = decision_path(tree,(probabilities[i] >= .5).astype(int))
            if row[name+'_decision_path'] != ' -> '.join([*path['steps'],path['action']]):
                raise ValueError('Concept comparison decision path mismatch')
        predictions[name] = prediction
    return predictions, [baseline,candidate,tree_dir,comparison]


def validate_action_run(root, directory, records, hashes):
    meta = json.loads((directory/'metadata.json').read_bytes())
    status = json.loads((directory/'status.json').read_bytes())
    if status.get('status') != 'complete' or status.get('mode') != 'pilot' or meta.get('mode') != 'pilot':
        raise ValueError('Completed full action run required; smoke runs are excluded')
    config = Path(root)/'configs/action_training.yaml'
    validate_config(meta['configuration'])
    if yaml.safe_load(config.read_bytes()) != meta['configuration'] or meta['input_hashes'] != {**hashes,'training_config':sha256(config.read_bytes())}:
        raise ValueError('Stale action configuration or source provenance')
    if meta['class_order'] != list(ACTIONS) or meta['preprocessing'] != PREPROCESSING:
        raise ValueError('Action mapping/preprocessing mismatch')
    if meta['training_ids'] != [r['sample_id'] for r in records['train']] or meta['evaluation_ids'] != [r['sample_id'] for r in records['validation']] or meta['evaluation_split'] != 'validation':
        raise ValueError('Action split membership mismatch')
    return meta


def check_action_predictions(path, records, probabilities):
    rows = read_csv(path)
    check_ids_targets(rows,records)
    old = np.array([[float(r[a+'_probability']) for a in ACTIONS] for r in rows])
    names = action_names(old)
    if names != [r['prediction'] for r in rows] or names != action_names(probabilities) or not np.allclose(old, probabilities, atol=1e-6, rtol=1e-6):
        raise ValueError('Action checkpoint predictions disagree with saved table')
    return float(np.max(np.abs(old-probabilities)))


def run(action_run, root=ROOT, run_dir=None):
    root = Path(root).resolve(); directory = root/action_run
    records, hashes = read_action_records(root)
    concept_predictions, source_dirs = concept_sources(root,records,hashes)
    meta = validate_action_run(root,directory,records,hashes)
    before = snapshot([*source_dirs,directory])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for prediction reproduction')
    seed_everything(42); torch.set_num_threads(4)
    output_root = root/'outputs/action_comparison'
    output = (root/run_dir).resolve() if run_dir else output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not output.is_relative_to(output_root) or output == output_root:
        raise ValueError('Comparison outputs must be under outputs/action_comparison')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',{'status':'running'})
    try:
        model,saved = load_checkpoint(directory/'best.pt','cuda')
        for key in ('configuration','input_hashes','weights','class_order','preprocessing'):
            if saved[key] != meta[key]:
                raise ValueError('Checkpoint provenance mismatch: '+key)
        loader = DataLoader(ActionDataset(root,records['validation'],'validation'),batch_size=16,shuffle=False,num_workers=0)
        loss,p,y,ids = evaluate(model,loader,'cuda')
        if ids != meta['evaluation_ids'] or abs(loss-saved['validation_loss']) > 1e-6:
            raise ValueError('Reloaded validation IDs/loss mismatch')
        difference = check_action_predictions(directory/'validation_predictions.csv',records['validation'],p.numpy())
        truth = [ACTIONS[int(t)] for t in y]
        predictions = {'original_concept_pipeline':concept_predictions['baseline'],
                       'finetuned_concept_pipeline':concept_predictions['candidate'],
                       'direct_action':action_names(p.numpy()),'always_slow':['SLOW']*len(ids)}
        metrics = {name:action_metrics(truth,pred) for name,pred in predictions.items()}
        if metrics['direct_action'] != json.loads((directory/'metrics.json').read_bytes())['model']:
            raise ValueError('Saved action metrics differ from regenerated predictions')
        write_json(output/'metrics.json',metrics)
        rows = [{'sample_id':sid,'target':truth[i],**{name:pred[i] for name,pred in predictions.items()}} for i,sid in enumerate(ids)]
        with (output/'validation_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = [{'model':name,'accuracy':v['accuracy'],'macro_f1':v['macro_f1'],'stop_recall':v['stop_recall'],'stop_to_go_errors':v['stop_to_go_errors']} for name,v in metrics.items()]
        with (output/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        if before != snapshot([*source_dirs,directory]) or hashes != read_action_records(root)[1]:
            raise ValueError('Sources changed during comparison')
        write_json(output/'metadata.json',{'input_hashes':hashes,'artifact_hashes':before,'validation_ids':ids,'class_order':list(ACTIONS),'checkpoint_prediction_max_difference':difference,'test_images_loaded':False,'tree_refitted':False})
        write_json(output/'status.json',{'status':'complete'})
        print(json.dumps(summary,indent=2),flush=True)
        print(f'Saved: {output}',flush=True)
        return output,metrics
    except Exception as exc:
        write_json(output/'status.json',{'status':'failed','error':str(exc)})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    try:
        run(args.action_run,run_dir=args.run_dir)
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        parser.exit(1,f'Action comparison failed: {exc}\n')
