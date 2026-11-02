# Balanced concept-loss experiment

Completed 2026-10-23 in the recovered Desktop repository. This is one predefined training experiment using 342 training and 74 validation images. All 74 test images remain reserved. Previous checkpoints and the tree remain preserved; no candidate is automatically promoted.

## Protocol

Version concept-training-v3.0 changes the training loss from v2. For each concept, positive weight is N/(2*positive_count) and negative weight is N/(2*negative_count), using the full training set only. Binary cross-entropy is computed for each image/concept, multiplied by its label weight, then averaged over all entries. Positive and negative outcomes have equal total training weight within a concept. No clipping, validation-derived weights or threshold changes are used. Missing outcomes cause an error.

| Concept | Positive | Negative | Positive weight | Negative weight |
|---|---:|---:|---:|---:|
| pedestrian_present | 271 | 71 | 0.630996 | 2.408451 |
| vehicle_present | 325 | 17 | 0.526154 | 10.058824 |
| cyclist_present | 107 | 235 | 1.598131 | 0.727660 |
| road_clear | 209 | 133 | 0.818182 | 1.285714 |
| obstacle_close | 10 | 332 | 17.100000 | 0.515060 |

Weights, counts, concept order and a hash of training IDs/labels are saved in both run metadata and the checkpoint. The 16-image training-only smoke check uses these same full-training weights and initializes independently from the full run.

Fresh ImageNet IMAGENET1K_V1 weights and seed-42 output-layer initialization match v2. Only layer4 and the five-output head train; earlier parameters and all BatchNorm running statistics stay frozen. Adam learning rates are 0.0001 for layer4 and 0.001 for the head, with weight decay 0.0001. Other settings remain 20 epochs, batch size 16, float32, zero loader workers, four CPU threads, deterministic CUDA, full-image 224 x 448 resize and ImageNet normalization. There is no augmentation, scheduler, early stopping or class resampling. Binary predictions use probability >= 0.5.

Checkpoint selection still uses lowest UNWEIGHTED mean validation loss, retaining earlier epochs on ties. Weighted validation loss uses the fixed training weights and is diagnostic only. History records weighted training loss and both validation losses. The training_loss history field is an alias for weighted_training_loss in v3; it is not directly comparable with v2 training loss.

```powershell
.\.venv\Scripts\python.exe scripts/train_concepts.py --config configs/concept_balanced.yaml --smoke
.\.venv\Scripts\python.exe scripts/train_concepts.py --config configs/concept_balanced.yaml
.\.venv\Scripts\python.exe scripts/compare_concept_runs.py --candidate-run outputs/concept_training/balanced_v3 --tree-run outputs/decision_tree/pilot_v1_complete
```

Completed artifacts are in outputs/concept_training/balanced_v3 and outputs/concept_comparison/balanced_v3. Both are ignored by Git. Commands accept --run-dir inside the appropriate output folder and refuse existing directories. Existing v1/v2 commands and checkpoints remain supported.

## Validation results

Best checkpoint: epoch 9. Unweighted validation loss 0.349363; weighted validation loss 0.478709.

| Model | Concept macro F1 | Action accuracy | Action macro F1 | STOP recall | STOP-to-GO | SLOW-to-GO |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.495728 | 0.864865 | 0.309179 | 0.000000 | 0 | 0 |
| finetuned_v2 | 0.516741 | 0.810811 | 0.349262 | 0.000000 | 0 | 5 |
| candidate | 0.714210 | 0.851351 | 0.657026 | 1.000000 | 0 | 4 |
| always SLOW | n/a | 0.864865 | 0.309179 | 0.000000 | 0 | 0 |

Here baseline means v1, finetuned_v2 means the previous unweighted fine-tuning run, and candidate means balanced v3.

| Concept | v2 TP / FP / FN / TN | v3 TP / FP / FN / TN | v2 F1 | v3 F1 |
|---|---|---|---:|---:|
| pedestrian_present | 39 / 9 / 13 / 13 | 46 / 16 / 6 / 6 | 0.780000 | 0.807018 |
| vehicle_present | 74 / 0 / 0 / 0 | 73 / 0 / 1 / 0 | 1.000000 | 0.993197 |
| cyclist_present | 1 / 4 / 14 / 55 | 6 / 15 / 9 / 44 | 0.100000 | 0.333333 |
| road_clear | 19 / 5 / 11 / 39 | 22 / 12 / 8 / 32 | 0.703704 | 0.687500 |
| obstacle_close | 0 / 0 / 3 / 71 | 3 / 2 / 0 / 69 | 0.000000 | 0.750000 |

Candidate action confusion matrix: target rows and predicted columns in STOP/SLOW/GO order.
```text
[3, 0, 0]
[2, 58, 4]
[0, 5, 2]
```

All per-concept precision/recall/specificity and action per-class metrics are saved in metrics.json. Undefined concept ratios are null; concept macro F1 averages defined concepts. Action macro F1 follows the existing three-class policy. The per-image table contains probabilities, concepts, target actions, tree actions and paths for v1, v2 and v3.

## Interpretation

The candidate improves concept macro F1 from v2's 0.516741 to 0.714210 and action macro F1 from 0.349262 to 0.657026. Action accuracy rises from 60/74 (81.08%) to 63/74 (85.14%). Always-SLOW still gets one more image correct (64/74), but misses every STOP and GO target. The candidate recognizes all three STOP targets, 58/64 SLOW and 2/7 GO. It predicts STOP five times, including two SLOW-target false alarms. Four SLOW targets become GO, down from five in v2. None of the three STOP targets becomes GO.

The improvement comes with more concept false alarms. Pedestrian misses fall from 13 to 6, but false positives rise from 9 to 16. Cyclists detected rise from 1/15 to 6/15, while false positives rise from 4 to 15. Close obstacles detected rise from 0/3 to 3/3, with two false positives. Road-clear F1 falls from 0.703704 to 0.687500, and one vehicle-positive image is now missed. All of these changes are reported rather than hiding regressions behind macro F1.

Epoch 9 is selected by unweighted validation loss (0.349363), which is slightly worse than v2's 0.336631 despite better thresholded concept and action metrics. Epoch 20 has weighted training loss 0.000526 and unweighted validation loss 0.603645, showing later overfitting. No checkpoint was chosen using the reported F1 scores. This is encouraging pilot evidence for the weighting experiment, not proof of robust STOP detection: only three positive validation examples are available.

Weighting can trade fewer missed rare concepts for additional false alarms. Assess both sides of each confusion count, and check downstream actions rather than assuming better concept F1 implies better decisions. The saved tree is never refitted. Its learned STOP branch requires pedestrian presence, which can limit the action benefit of improved close-obstacle recognition.

Final epoch weighted training loss is 0.000526; unweighted validation loss is 0.603645 and weighted validation loss is 1.000720. The selected checkpoint, not the last epoch, supplies the reported predictions.

Validation has only three close-obstacle/STOP positives and no vehicle-negative examples. Vehicle F1 therefore cannot demonstrate discrimination. Weighted training changes the learning objective; sigmoid outputs remain uncalibrated and are not reliable confidence guarantees. Results use one seed and a repeatedly used development split, not an untouched test estimate. Action targets are synthetic proxy rules, not expert driving decisions.

## Verification and next step

All 207 tests passed. Coverage includes training-only counts, equal total class influence, weighted-loss reduction and gradients, missing-class rejection, frozen parameters and BatchNorm buffers, stale metadata and weights, old checkpoint compatibility, reload predictions and existing unchanged-tree checks. The CUDA smoke/full run succeeded, and checkpoint predictions were reproduced exactly. Source metadata and existing model/tree/comparison artifacts were checked by hashes.

The next milestone is concept intervention. There was no dataset expansion, test evaluation, direct-model retraining, tree replacement, commit or push.
