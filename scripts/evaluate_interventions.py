"""Evaluate annotation-simulated interventions on saved validation concepts."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.interventions import load_inputs, evaluate_interventions
from scripts.train_concepts import write_json


def report_text(results,rows):
    lines=['# Simulated concept intervention', '',
           'Balanced v3 predictions, saved original tree, 74 validation samples. Annotation-derived concepts simulate correct feedback; no human participants were studied.', '',
           'Policy intervention-v1.0 selects the first incorrect concept in fixed order: obstacle_close, pedestrian_present, cyclist_present, road_clear, vehicle_present. It uses annotation knowledge to identify mistakes and is not a deployable human-query policy. The order was fixed before evaluation. All samples are evaluated, including initially correct actions.', '',
           '| Trial | Samples corrected | Concept values corrected | Fixed / initial errors | Harmed correct actions | Accuracy | Macro F1 | STOP recall |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,m in results.items():
        a=m['action_metrics']
        lines.append(f"| {mode} | {m['samples_corrected']} | {m['concept_values_corrected']} | {m['fixed_errors']}/{m['initially_incorrect']} | {m['harmed_correct_actions']} | {a['accuracy']:.6f} | {a['macro_f1']:.6f} | {a['stop_recall']:.6f} |")
    lines += ['', 'Success rate always divides fixed original action errors by all original action errors, including examples where the trial made no correction. Independent per-concept trials each begin from the original concepts. The all-corrections trial supplies annotation concepts as a reference. It is not evidence that every partial correction helps.', '',
              'Vehicle presence does not occur in this tree, so changing it cannot independently change the action. The learned STOP branch depends on pedestrian presence; it does not implement the reference policy for every possible binary concept combination.', '',
              '## Representative primary-trial examples', '']
    for label,predicate in [('Successful',lambda r:r['outcome']=='fixed'),('Ineffective',lambda r:r['outcome']=='still_wrong' and r['selected_concepts']),('Harmful',lambda r:r['outcome']=='harmed')]:
        matches=sorted([r for r in rows if r['mode']=='one' and predicate(r)],key=lambda r:r['sample_id'])
        if matches:
            r=matches[0]
            lines += [f"{label}: `{r['sample_id']}`, target {r['target_action']}, {r['before_action']} -> {r['after_action']}.",f"Correction: `{r['changes']}`.",f"Before: `{r['before_path']}`.",f"After: `{r['after_path']}`.",'']
        else:
            lines += [f'{label}: no example occurred in the primary trial.','']
    lines += ['Only three STOP validation targets are available. These results concern synthetic action targets and annotation-based feedback. They do not establish real-world driving correctness, human annotation reliability or unseen-test performance.', '',
              'Full confusion matrices, STOP-to-GO/SLOW-to-GO counts and all before/after paths are saved with this report. Original probabilities and source annotations are preserved. No dataset images are loaded, no model is trained and the tree is not refitted.','']
    return '\n'.join(lines)


def run(root=ROOT,run_dir=None):
    root=Path(root).resolve(); context=load_inputs(root)
    output_root=root/'outputs/interventions'
    output=(root/run_dir).resolve() if run_dir else output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not output.is_relative_to(output_root) or output==output_root:
        raise ValueError('Outputs must be inside outputs/interventions')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',{'status':'running'})
    try:
        results,rows=evaluate_interventions(context)
        write_json(output/'metrics.json',results)
        write_json(output/'policy.json',context['policy'])
        write_json(output/'metadata.json',{'input_hashes':context['input_hashes'],'artifact_hashes':context['artifact_hashes'],
                                         'validation_ids':context['ids'],'policy':context['policy'],
                                         'tree_refitted':False,'images_loaded':False})
        with (output/'interventions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output/'report.md').write_text(report_text(results,rows),encoding='utf-8')
        after=load_inputs(root)
        if context['input_hashes']!=after['input_hashes'] or context['artifact_hashes']!=after['artifact_hashes']:
            raise ValueError('Source artifacts changed during evaluation')
        write_json(output/'status.json',{'status':'complete'})
        print(json.dumps({k:{q:v[q] for q in ('samples_corrected','fixed_errors','unresolved_original_errors','harmed_correct_actions','success_rate')} for k,v in results.items()},indent=2))
        print(f'Saved: {output}')
        return output,results
    except Exception as exc:
        write_json(output/'status.json',{'status':'failed','error':str(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    try:
        run(run_dir=args.run_dir)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,f'Intervention evaluation failed: {exc}\n')
