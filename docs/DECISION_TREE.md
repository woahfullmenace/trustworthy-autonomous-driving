# First image-to-action pipeline

Validation only: 74 images. Targets come from concepts-v1.0/actions-v1.0 proxy rules.

| Input to decision stage | Accuracy | Macro F1 | STOP recall | STOP-to-GO errors |
|---|---:|---:|---:|---:|
| annotation_concepts | 1.0000 | 1.0000 | 1.0000 | 0 |
| predicted_concepts | 0.8649 | 0.3092 | 0.0000 | 0 |
| always_slow | 0.8649 | 0.3092 | 0.0000 | 0 |

Confusion matrices use target rows and prediction columns in STOP/SLOW/GO order.
Precision/recall with a zero denominator are null. Undefined class F1 is zero; macro F1 averages all three classes.

```text
|--- pedestrian_present <= 0.50
|   |--- road_clear <= 0.50
|   |   |--- class: SLOW
|   |--- road_clear >  0.50
|   |   |--- cyclist_present <= 0.50
|   |   |   |--- class: GO
|   |   |--- cyclist_present >  0.50
|   |   |   |--- class: SLOW
|--- pedestrian_present >  0.50
|   |--- obstacle_close <= 0.50
|   |   |--- class: SLOW
|   |--- obstacle_close >  0.50
|   |   |--- class: STOP

```

Vehicle dependence: {"used_in_splits": false, "changed_patterns_of_32": 0, "scope": "Exhaustive binary inputs, including combinations absent from the dataset"}

The tree is fitted to annotation-derived concepts and policy-derived targets. Annotation-concept agreement tests policy reproduction; it is not independent expert driving validation.
Only three STOP examples are available. Zero STOP-to-GO errors does not establish STOP recognition: inspect STOP recall and STOP-to-SLOW errors as well.
Concept thresholds, tree settings, checkpoint and splits were held fixed; test images were not loaded.


## Interpretation and reproducible commands

Completed 2026-10-22 in the recovered Desktop repository. Configuration is `configs/decision_tree.yaml`; the completed run is `outputs/decision_tree/pilot_v1_complete`. Two earlier incomplete run directories retain failure status records and must not be used as completed results.

```powershell
.\.venv\Scripts\python.exe scripts/train_decision_tree.py
.\.venv\Scripts\python.exe scripts/predict_action.py --sample-id aachen_000002_000019 --tree-run outputs/decision_tree/pilot_v1_complete
```

The training command uses 342 annotation-concept training rows, Gini, maximum depth four, minimum leaf size one, no class weights and seed 42. It creates a new ignored run directory and refuses an existing directory. The fitted tree has depth 3 and 5 leaves. Tree class order is GO/SLOW/STOP (also used in the tree diagram's value arrays); metric matrices deliberately use STOP/SLOW/GO. Both mappings are recorded.

All 74 annotation-concept validation actions match their proxy targets. With predicted concepts, all 74 outputs are SLOW. This gives 64/74 accuracy (86.49%) and macro F1 0.309179, exactly matching always-SLOW. All three STOP examples become SLOW and all seven GO examples become SLOW. There are zero STOP-to-GO errors because GO is never predicted; STOP recall is nevertheless zero. The high accuracy reflects class imbalance, not successful recognition of all actions.

The tree's root asks about pedestrian presence. ResNet predicts pedestrian presence on 73/74 validation images and close-obstacle presence on none, which drives those 73 images to SLOW. The remaining example also reaches SLOW. The trained tree does not use vehicle_present, and flipping it changes none of the 32 binary input patterns.

The tree matches the supplied validation examples, but it does not reproduce the reference policy for every conceivable concept combination. A metadata-only exhaustive check finds 8/32 disagreements, all in combinations with obstacle_close=1 and pedestrian_present=0. All training STOP examples have pedestrian_present=1, so the fitted tree only learns its STOP branch on that side. This is a learned association from limited training support. No tree settings or labels were changed in response. Human concept interventions later must use the learned tree and report this limitation rather than assume the reference policy is reproduced everywhere.

The run saves `tree.joblib`, `metadata.json`, `metrics.json`, `validation_actions.csv`, `tree.txt`, `tree.png`, `report.md` and `status.json`. Prediction paths name each tested concept, comparison, leaf and final action. Sample inspection accepts only training/validation IDs. Checkpoint, input hashes, concept order, preprocessing and tree configuration must match. Load only locally generated trusted joblib artifacts; hashes check accidental changes, not authenticity against malicious replacement.

## Recovery and verification

The moved virtual environment lacked its packages. It was restored using the recorded requirements lock; pip check found no broken dependencies. The extracted RGB folder was also missing. Exactly 416 training/validation RGB images were restored from the existing Cityscapes archive and checked against their original manifest hashes. Test images were not extracted or loaded; masks were not needed. Source archives, labels, splits and the trained checkpoint were preserved.

All 168 tests passed, including concept order, action-class mapping, training-only fitting, held-out ID rejection, stale provenance, checkpoint incompatibility, tree reload equivalence, decision paths, absent predicted classes and repeatable fitting. Windows system temporary-folder permissions required an explicit ignored pytest directory: `python -m pytest -q --basetemp outputs/pytest_tree_01 --tb=short` (use a fresh name for another run).

## Next step

Plan one controlled concept-model improvement using training and validation only, keeping this tree and checkpoint as the first baseline. The largest observed problems are pedestrian false positives, poor cyclist recall and zero close-obstacle recall. The end-to-end baseline, intervention experiment, abstention and corruption evaluation remain pending. Test evaluation, dataset expansion, commits and pushes were not performed.
