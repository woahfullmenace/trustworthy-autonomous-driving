# Simulated concept intervention

Balanced v3 predictions, saved original tree, 74 validation samples. Annotation-derived concepts simulate correct feedback; no human participants were studied.

Policy intervention-v1.0 selects the first incorrect concept in fixed order: obstacle_close, pedestrian_present, cyclist_present, road_clear, vehicle_present. It uses annotation knowledge to identify mistakes and is not a deployable human-query policy. The order was fixed before evaluation. All samples are evaluated, including initially correct actions.

| Trial | Samples corrected | Concept values corrected | Fixed / initial errors | Harmed correct actions | Accuracy | Macro F1 | STOP recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0 | 0 | 0/11 | 0 | 0.851351 | 0.657026 | 1.000000 |
| one | 53 | 53 | 10/11 | 2 | 0.959459 | 0.925459 | 1.000000 |
| pedestrian_present | 22 | 22 | 6/11 | 2 | 0.905405 | 0.808714 | 1.000000 |
| vehicle_present | 1 | 1 | 0/11 | 0 | 0.851351 | 0.657026 | 1.000000 |
| cyclist_present | 24 | 24 | 0/11 | 0 | 0.851351 | 0.657026 | 1.000000 |
| road_clear | 20 | 20 | 4/11 | 1 | 0.891892 | 0.729167 | 1.000000 |
| obstacle_close | 2 | 2 | 2/11 | 0 | 0.878378 | 0.745975 | 1.000000 |
| all | 53 | 69 | 11/11 | 0 | 1.000000 | 1.000000 | 1.000000 |

Success rate always divides fixed original action errors by all original action errors, including examples where the trial made no correction. Independent per-concept trials each begin from the original concepts. The all-corrections trial supplies annotation concepts as a reference. It is not evidence that every partial correction helps.

Vehicle presence does not occur in this tree, so changing it cannot independently change the action. The learned STOP branch depends on pedestrian presence; it does not implement the reference policy for every possible binary concept combination.

## Representative primary-trial examples

Successful: `bremen_000245_000019`, target GO, SLOW -> GO.
Correction: `[{"concept":"pedestrian_present","original":1,"replacement":0}]`.
Before: `pedestrian_present=1 > 0.5 -> obstacle_close=0 <= 0.5 -> SLOW`.
After: `pedestrian_present=0 <= 0.5 -> road_clear=1 > 0.5 -> cyclist_present=0 <= 0.5 -> GO`.

Ineffective: `bremen_000179_000019`, target GO, SLOW -> SLOW.
Correction: `[{"concept":"pedestrian_present","original":1,"replacement":0}]`.
Before: `pedestrian_present=1 > 0.5 -> obstacle_close=0 <= 0.5 -> SLOW`.
After: `pedestrian_present=0 <= 0.5 -> road_clear=0 <= 0.5 -> SLOW`.

Harmful: `bremen_000199_000019`, target SLOW, SLOW -> GO.
Correction: `[{"concept":"pedestrian_present","original":1,"replacement":0}]`.
Before: `pedestrian_present=1 > 0.5 -> obstacle_close=0 <= 0.5 -> SLOW`.
After: `pedestrian_present=0 <= 0.5 -> road_clear=1 > 0.5 -> cyclist_present=0 <= 0.5 -> GO`.

Only three STOP validation targets are available. These results concern synthetic action targets and annotation-based feedback. They do not establish real-world driving correctness, human annotation reliability or unseen-test performance.

Full confusion matrices, STOP-to-GO/SLOW-to-GO counts and all before/after paths are saved with this report. Original probabilities and source annotations are preserved. No dataset images are loaded, no model is trained and the tree is not refitted.

## Interpretation of the primary result

One correct concept was supplied for each of 53 examples with at least one incorrect concept; 21 examples needed no correction. This includes originally correct actions. The fixed ordering repaired 10 of the original 11 action errors (90.91% intervention success rate), left one original error unresolved and introduced two errors among the 63 originally correct actions. Final accuracy is 71/74 (95.95%), compared with 63/74 (85.14%); macro F1 rises from 0.657026 to 0.925459. The net gain is eight correct actions, not ten, because harms must also count.

Correcting all concepts repairs all 11 errors with no new errors, giving 74/74 agreement with synthetic targets. This reference result supplies annotation answers directly and must not be presented as autonomous model accuracy. Likewise, the 95.95% one-correction result assumes access to correct concept labels and knowledge of which prediction is wrong. ResNet's autonomous accuracy remains 85.14%.

An accurate concept correction can expose another wrong concept farther down the learned tree. In the harmful example above, removing a false pedestrian prediction exposes a missed cyclist: the annotation says cyclist_present=1, while ResNet still supplies 0. The tree then chooses GO instead of SLOW. Thus accurate feedback need not improve each individual action. Independent cyclist corrections did not change overall action correctness in this validation set; this is a property of these paths and inputs, not proof that cyclist information is universally irrelevant.

## Commands and verification

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_interventions.py
.\.venv\Scripts\python.exe scripts/intervene_concepts.py --sample-id bremen_000245_000019 --set pedestrian_present=0
```

The evaluator accepts --run-dir inside ignored outputs/interventions and refuses overwrites. The manual command accepts repeated --set CONCEPT=0|1 arguments and operates on saved validation examples only; arbitrary overrides are explicitly distinguished from annotation-correct simulated feedback. The original predicted probabilities remain visible and are not replaced by fabricated confidence scores.

Policy is stored in configs/interventions.yaml. The completed handoff is outputs/interventions/pilot_v1_verified; pilot_v1_repeat has byte-identical files. The earlier pilot_v1 evaluation is also retained. Artifacts include policy.json, metadata.json, metrics.json, interventions.csv, report.md and status.json. The CSV has 592 rows: 74 examples across eight trials. Change lists record concept names and original/replacement values; complete concept vectors use pedestrian_present, vehicle_present, cyclist_present, road_clear, obstacle_close order.

All 223 tests passed. Tests cover fixed priority, no-ops, independent corrections, zero-denominator metrics, harmful effects, manual override validation, preserved inputs, path agreement, stale hashes/IDs, byte-identical outputs and overwrite rejection. The first image-access guard also rejected hashing a saved loss plot; it was corrected to forbid dataset-image reads while allowing artifact hashing. No dataset images are decoded by these commands, and no tree fitting or ResNet inference occurs. Input hashes and saved model/tree artifacts were rechecked after both evaluations. The manual example above was also executed successfully.

The next milestone is uncertainty-based abstention, tuned on validation only. This experiment performed no training, test evaluation, commits or pushes.
