"""Simulated concept corrections using saved validation predictions only."""
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from scripts.compare_concept_runs import prediction_table, snapshot, validate_run
from src.concept_training import read_records, sha256
from src.concepts import CONCEPTS
from src.decision_tree import ACTIONS, action_metrics, binary_matrix, decision_path, load_tree_run

PRIORITY = ('obstacle_close','pedestrian_present','cyclist_present','road_clear','vehicle_present')
POLICY = {'version':'intervention-v1.0','concept_run':'outputs/concept_training/balanced_v3',
          'tree_run':'outputs/decision_tree/pilot_v1_complete','comparison_run':'outputs/concept_comparison/balanced_v3',
          'split':'validation','threshold':0.5,'priority':list(PRIORITY),
          'success_denominator':'all_initially_incorrect_actions'}


def metrics_for(truth, prediction):
    m = action_metrics(truth,prediction)
    m['slow_to_go_errors'] = m['confusion_matrix'][1][2]
    return m


def load_inputs(root):
    root = Path(root).resolve()
    config_path = root/'configs/interventions.yaml'
    policy = yaml.safe_load(config_path.read_bytes())
    if policy != POLICY:
        raise ValueError('Configuration differs from intervention-v1.0')
    tree, context = load_tree_run(root,root/policy['tree_run'])
    records, actions = context[:2]
    _, hashes = read_records(root)
    candidate=root/policy['concept_run']; comparison=root/policy['comparison_run']
    directories = [root/'outputs/concept_training/pilot_v1',candidate,root/policy['tree_run'],
                   root/'outputs/concept_training/finetune_v2',root/'outputs/concept_comparison/finetune_v2']
    report = json.loads((comparison/'metadata.json').read_bytes())
    if json.loads((comparison/'status.json').read_bytes()).get('status') != 'complete':
        raise ValueError('Completed balanced comparison required')
    if report['input_hashes'] != hashes or report['artifact_hashes'] != snapshot(directories):
        raise ValueError('Stale comparison provenance or modified model artifacts')
    validation=records['validation']; ids=[r['sample_id'] for r in validation]
    if report['validation_ids'] != ids or report['concept_order'] != list(CONCEPTS) or report['action_order'] != list(ACTIONS) or report.get('tree_refitted') is not False:
        raise ValueError('Comparison IDs/order/protocol mismatch')
    metadata = validate_run(root,candidate,records,hashes)
    if metadata['configuration']['training_version'] != 'concept-training-v3.0' or metadata['configuration']['threshold'] != .5:
        raise ValueError('Balanced v3 predictions at threshold 0.5 required')
    probabilities=prediction_table(candidate/'validation_predictions.csv',validation)
    predicted=(probabilities >= .5).astype(int)
    targets=binary_matrix([[int(r[c]) for c in CONCEPTS] for r in validation])
    truth=[actions[sid] for sid in ids]; original=tree.predict(predicted).tolist()
    with (comparison/'validation_comparison.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    if [r['sample_id'] for r in rows] != ids or [r['target_action'] for r in rows] != truth or [r['candidate_action'] for r in rows] != original:
        raise ValueError('Saved comparison IDs/targets/actions mismatch')
    for i,row in enumerate(rows):
        for j,c in enumerate(CONCEPTS):
            if int(row[c+'_target']) != targets[i,j] or int(row['candidate_'+c+'_prediction']) != predicted[i,j] or float(row['candidate_'+c+'_probability']) != probabilities[i,j]:
                raise ValueError('Saved comparison concepts/probabilities mismatch')
        if row['candidate_decision_path'] != path_text(tree,predicted[i]):
            raise ValueError('Saved decision path mismatch')
    saved=json.loads((comparison/'metrics.json').read_bytes())['candidate']['actions']
    if metrics_for(truth,original) != saved:
        raise ValueError('Reproduced original metrics disagree')
    hashes={**hashes,'intervention_config':sha256(config_path.read_bytes())}
    directories.append(comparison)
    return {'tree':tree,'ids':ids,'probabilities':probabilities,'predicted':predicted,'targets':targets,
            'truth':truth,'original':original,'policy':policy,'input_hashes':hashes,
            'artifact_hashes':snapshot(directories)}


def path_text(tree, values):
    p=decision_path(tree,values)
    if p['action'] != tree.predict(binary_matrix([values]))[0]:
        raise ValueError('Decision path/action disagreement')
    return ' -> '.join([*p['steps'],p['action']])


def correct_concepts(predicted, targets, mode):
    original=binary_matrix(predicted); target=binary_matrix(targets)
    if original.shape != target.shape:
        raise ValueError('Concept arrays differ in shape')
    result=original.copy()
    if mode=='original':
        return result
    if mode=='all':
        return target.copy()
    if mode=='one':
        for i in range(len(result)):
            for c in PRIORITY:
                j=CONCEPTS.index(c)
                if result[i,j] != target[i,j]:
                    result[i,j]=target[i,j]
                    break
        return result
    if mode in CONCEPTS:
        j=CONCEPTS.index(mode); result[:,j]=target[:,j]
        return result
    raise ValueError('Unknown correction mode')


def intervention_metrics(truth, before, after, changed):
    # action_metrics validates labels and lengths before deriving denominators.
    metrics=metrics_for(truth,after)
    metrics_for(truth,before)
    changed=np.asarray(changed,dtype=bool)
    if changed.shape != (len(truth),len(CONCEPTS)):
        raise ValueError('Invalid change mask')
    wrong=np.asarray(before)!=np.asarray(truth); now_correct=np.asarray(after)==np.asarray(truth)
    denominator=int(wrong.sum()); fixed=int((wrong & now_correct).sum()); harmed=int((~wrong & ~now_correct).sum())
    return {'action_metrics':metrics,'initially_incorrect':denominator,'initially_correct':len(truth)-denominator,
            'samples_corrected':int(changed.any(axis=1).sum()),'concept_values_corrected':int(changed.sum()),
            'initially_incorrect_samples_corrected':int((wrong & changed.any(axis=1)).sum()),
            'fixed_errors':fixed,'unresolved_original_errors':denominator-fixed,'harmed_correct_actions':harmed,
            'success_rate':fixed/denominator if denominator else None,
            'success_rate_undefined':'no initially incorrect actions' if not denominator else None}


def evaluate_interventions(context):
    tree=context['tree']; predicted=context['predicted']; targets=context['targets']; before=context['original']
    results={}; rows=[]
    for mode in ('original','one',*CONCEPTS,'all'):
        corrected=correct_concepts(predicted,targets,mode)
        after=tree.predict(corrected).tolist(); changed=corrected!=predicted
        results[mode]=intervention_metrics(context['truth'],before,after,changed)
        for i,sid in enumerate(context['ids']):
            changes=[{'concept':c,'original':int(predicted[i,j]),'replacement':int(corrected[i,j])} for j,c in enumerate(CONCEPTS) if changed[i,j]]
            was=before[i]==context['truth'][i]; now=after[i]==context['truth'][i]
            outcome='fixed' if not was and now else 'harmed' if was and not now else 'still_wrong' if not now else 'still_correct'
            rows.append({'sample_id':sid,'mode':mode,'target_action':context['truth'][i],
                         'selected_concepts':','.join(x['concept'] for x in changes),'changes':json.dumps(changes,separators=(',',':')),
                         'original_concepts':json.dumps(predicted[i].tolist()),'corrected_concepts':json.dumps(corrected[i].tolist()),
                         'before_action':before[i],'after_action':after[i],'outcome':outcome,
                         'before_path':path_text(tree,predicted[i]),'after_path':path_text(tree,corrected[i])})
    return results,rows


def parse_overrides(values):
    overrides={}
    for text in values:
        parts=text.split('=')
        if len(parts)!=2 or parts[0] not in CONCEPTS or parts[1] not in ('0','1') or parts[0] in overrides:
            raise ValueError('Use unique known CONCEPT=0|1 overrides')
        overrides[parts[0]]=int(parts[1])
    if not overrides:
        raise ValueError('At least one override is required')
    return overrides


def manual_intervention(context, sample_id, overrides):
    if sample_id not in context['ids']:
        raise ValueError('Only saved validation IDs are allowed; training, test, excluded and unknown IDs are rejected')
    overrides=parse_overrides(overrides)
    i=context['ids'].index(sample_id); before=context['predicted'][i]; after=before.copy()
    for c,value in overrides.items():
        after[CONCEPTS.index(c)]=value
    return {'sample_id':sample_id,'mode':'manual_override','concept_order':list(CONCEPTS),
            'probabilities':dict(zip(CONCEPTS,context['probabilities'][i].tolist())),
            'original_concepts':dict(zip(CONCEPTS,before.tolist())), 'overridden_concepts':dict(zip(CONCEPTS,after.tolist())),
            'before':decision_path(context['tree'],before),'after':decision_path(context['tree'],after),
            'meaning':'User-supplied override; not necessarily an annotation-correct concept or a valid driving decision.'}
