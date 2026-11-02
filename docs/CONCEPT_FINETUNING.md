# Partial ResNet18 fine-tuning comparison

Completed on the recovered Desktop repository using 342 training and 74 validation images. No test images were loaded. The original model and tree remain the first baseline; this experiment does not automatically replace them.

## Protocol

Version concept-training-v2.0 starts from fresh ResNet18 IMAGENET1K_V1 weights and the same seed-42 five-output layer initialization as v1. Only layer4 and the final layer train. Earlier parameters and all BatchNorm running statistics remain frozen; layer4 BatchNorm affine parameters train. Adam uses learning rates 0.0001 for layer4 and 0.001 for the head, weight decay 0.0001, batch size 16 and 20 epochs. Loss is unweighted BCEWithLogitsLoss. Resolution is 224 x 448 with existing ImageNet normalization, float32 and no augmentation. CUDA deterministic settings, four CPU threads and zero loader workers match v1. The threshold remains 0.5.

The selected checkpoint minimizes mean validation loss; ties keep the earlier epoch. No scheduler, early stopping, weighting or threshold tuning was added. The training-only smoke run and full run initialized independently. This compares the complete predefined partial-fine-tuning recipe with the head-only recipe; it does not isolate the effect of every optimizer setting.

```powershell
.\.venv\Scripts\python.exe scripts/train_concepts.py --config configs/concept_finetuning.yaml --smoke
.\.venv\Scripts\python.exe scripts/train_concepts.py --config configs/concept_finetuning.yaml
.\.venv\Scripts\python.exe scripts/compare_concept_runs.py --candidate-run outputs/concept_training/finetune_v2 --tree-run outputs/decision_tree/pilot_v1_complete
```

The default training command still selects v1. Existing output directories are refused. Completed artifacts are under outputs/concept_training/finetune_v2 and outputs/concept_comparison/finetune_v2; the smoke directory is separate. These ignored directories contain checkpoints, probabilities, losses, metrics, provenance and per-image action paths.

## Validation results

Candidate best epoch: 3 of 20.

| Model | Validation loss | Concept macro F1 | Action accuracy | Action macro F1 | STOP recall | STOP-to-GO errors |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.359139 | 0.495728 | 0.864865 | 0.309179 | 0.000000 | 0 |
| candidate | 0.336631 | 0.516741 | 0.810811 | 0.349262 | 0.000000 | 0 |
| always SLOW | n/a | n/a | 0.864865 | 0.309179 | 0.000000 | 0 |

| Concept | Baseline F1 | Candidate F1 | Baseline TP / FP / FN / TN | Candidate TP / FP / FN / TN |
|---|---:|---:|---|---|
| pedestrian_present | 0.816000 | 0.780000 | 51 / 22 / 1 / 0 | 39 / 9 / 13 / 13 |
| vehicle_present | 1.000000 | 1.000000 | 74 / 0 / 0 / 0 | 74 / 0 / 0 / 0 |
| cyclist_present | 0.105263 | 0.100000 | 1 / 3 / 14 / 56 | 1 / 4 / 14 / 55 |
| road_clear | 0.557377 | 0.703704 | 17 / 14 / 13 / 30 | 19 / 5 / 11 / 39 |
| obstacle_close | 0.000000 | 0.000000 | 0 / 0 / 3 / 71 | 0 / 0 / 3 / 71 |

Confusion matrices have target rows and prediction columns in STOP/SLOW/GO order:

baseline:
```text
[0, 3, 0]
[0, 64, 0]
[0, 7, 0]
```

candidate:
```text
[0, 3, 0]
[0, 59, 5]
[0, 6, 1]
```

The same saved annotation-trained tree was loaded for both models; it was never refitted. Constant concept probabilities still come only from training prevalence. All per-class precision/recall/F1, specificity and support are in the JSON metrics. Concept macro F1 averages defined concept F1 values; undefined ratios are null. Action macro F1 includes all three classes, assigning zero to undefined class F1 as in the established tree report.

## Interpretation and limits

The result is mixed, not an overall replacement recommendation. Concept macro F1 rises from 0.495728 to 0.516741, mainly with road-clear F1 improving from 0.557377 to 0.703704. Pedestrian false alarms drop from 22 to 9, but missed pedestrians increase from 1 to 13, reducing pedestrian F1 from 0.816 to 0.780. Cyclists detected remain 1/15 while cyclist false alarms rise from 3 to 4. Close obstacles detected remain 0/3. Vehicle predictions are unchanged.

The candidate produces 68 SLOW and six GO actions, with no STOP predictions. One GO target is now correct, but five SLOW targets become GO. Action macro F1 increases from 0.309179 to 0.349262 while accuracy decreases from 64/74 (86.49%) to 60/74 (81.08%). All three STOP targets still become SLOW. The zero STOP-to-GO count does not indicate successful STOP recognition.

Epoch 3 is best (validation loss 0.336631 versus baseline 0.359139). By epoch 20 training loss is 0.000183 while validation loss rises to 0.716847, showing substantial overfitting. The prescribed best-validation-loss selection protects against using that final checkpoint, but selecting on this small validation set limits generalization claims. No later checkpoint was selected to improve the reported F1 or actions.

Use the per-concept confusion counts to assess the tradeoff between missed objects and false alarms; improvement in one aggregate metric need not improve STOP recognition. Vehicle F1 is not evidence of discrimination because validation has no vehicle-negative images. There are only three close-obstacle/STOP positives, so recall changes in increments of one third. Probabilities remain uncalibrated.

These are validation-selected results from one seed, not final test estimates or evidence of driving safety. Actions reproduce annotation-derived proxy rules. The tree itself has a known limitation: its STOP branch requires predicted pedestrian presence, so correct close-obstacle predictions alone need not yield STOP. No labels, thresholds, splits or tree settings were changed to improve these results.

## Verification

All 178 tests passed. Tests verify trainable parameter groups and learning rates, exact seeded pretrained initialization, finite gradients, unchanged earlier parameters and BatchNorm buffers, trainable-layer updates, checkpoint reload, original-model compatibility, stale metadata and split rejection, prediction order, and unchanged tree use. The CUDA smoke and full training checks passed. Candidate checkpoint reload reproduced its saved predictions exactly; the comparison independently regenerated validation predictions. Source metadata and baseline artifacts were checked by hashes.

Next, review this comparison and move to the end-to-end action baseline using the same splits. Intervention, abstention and corruption experiments remain pending. No automatic model promotion, test evaluation, dataset expansion, commits or pushes were performed.
