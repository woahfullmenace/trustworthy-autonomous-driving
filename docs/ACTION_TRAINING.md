# Direct-action ResNet18 baseline

The direct model maps a photograph to three action logits, bypassing concept predictions and the decision tree. Targets remain the existing synthetic actions-v1.0 labels. This is a validation comparison, not a real-world driving-safety evaluation.

## Protocol and commands

Version action-training-v1.0 uses the same 342 training and 74 validation images as the concept pipelines. No test images are loaded. Images use the existing full 224 x 448 resize and ImageNet normalization. The action mapping is STOP=0, SLOW=1, GO=2. Softmax probabilities sum to one; argmax selects an action, with ties resolved in that order. There is no 0.5 threshold.

Training starts independently from ImageNet IMAGENET1K_V1 weights and a seed-42 three-output layer. Only layer4 and the head train. Earlier parameters and every BatchNorm running statistic remain frozen; layer4 affine parameters train. Adam uses learning rates 0.0001 and 0.001 respectively, weight decay 0.0001, batch size 16 and unweighted cross-entropy. All 20 epochs run with float32, four CPU threads, zero loader workers and the existing CUDA deterministic settings. There is no augmentation, scheduler, early stopping, class weighting or tuning. The lowest mean validation cross-entropy selects the checkpoint; ties keep the earlier epoch.

The partial-fine-tuning concept pipeline is the closest training-budget comparison. The original head-only model is additional context. This is one seed and one predefined recipe, not an exhaustive search. Direct-action cross-entropy and concept BCE losses have different targets and scales and must not be ranked against one another.

```powershell
.\.venv\Scripts\python.exe scripts/train_actions.py --smoke
.\.venv\Scripts\python.exe scripts/train_actions.py
.\.venv\Scripts\python.exe scripts/compare_action_models.py --action-run outputs/action_training/pilot_v1
```

Use --run-dir with a new path inside the corresponding ignored output folder to name a run. Existing directories are refused. The training-only smoke run and the full run initialize independently. Completed run directories are outputs/action_training/pilot_v1 and outputs/action_comparison/pilot_v1. Both save explicit completion status. The former saves checkpoint, metadata, loss history/plot, probabilities and metrics; the latter saves combined metrics and per-image comparisons. Load only locally trusted checkpoint artifacts.

## Results

Best action checkpoint: epoch 3, mean validation cross-entropy 0.436605.

| Model | Accuracy | Macro F1 | STOP recall | STOP-to-GO errors |
|---|---:|---:|---:|---:|
| original_concept_pipeline | 0.864865 | 0.309179 | 0.000000 | 0 |
| finetuned_concept_pipeline | 0.810811 | 0.349262 | 0.000000 | 0 |
| direct_action | 0.864865 | 0.309179 | 0.000000 | 0 |
| always_slow | 0.864865 | 0.309179 | 0.000000 | 0 |

Direct model confusion matrix: target rows, prediction columns, both STOP/SLOW/GO.
```text
[0, 3, 0]
[0, 64, 0]
[0, 7, 0]
```

| Action | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| STOP | 3 | undefined | 0.000000 | 0.000000 |
| SLOW | 64 | 0.864865 | 1.000000 | 0.927536 |
| GO | 7 | undefined | 0.000000 | 0.000000 |

The direct model correctly identifies 0/3 STOP, 64/64 SLOW and 0/7 GO targets. It predicts STOP 0 times, SLOW 74 times and GO 0 times.

Final-epoch training loss is 0.000064, and validation loss is 1.413537. Interpret this trajectory against the selected earlier checkpoint in loss.png; no checkpoint was selected to maximize the reported F1 or accuracy.

Accuracy is dominated by the 64 SLOW targets. Always-SLOW gets 86.49%, yet detects no STOP or GO. Use macro F1, class recall and confusion counts alongside accuracy. There are only three STOP validation examples, so a single prediction changes STOP recall by one third. A zero STOP-to-GO error count alone does not establish STOP recognition. Probabilities are uncalibrated.

## Interpretation

The direct model predicts SLOW for every validation photograph. Its 86.49% accuracy comes entirely from the dominant class, matching the original concept pipeline and always-SLOW. It recognizes neither STOP nor GO. Compared with the fine-tuned concept pipeline, it has higher accuracy but lower macro F1 (0.309179 versus 0.349262), reflecting the latter's single correctly recognized GO and additional SLOW-to-GO errors. This does not establish either system as reliably useful for driving.

The loss trajectory shows substantial overfitting: epoch 3 has the best validation loss (0.436605), while epoch 20 reaches training loss 0.000064 and validation loss 1.413537. The best-loss checkpoint is used, not the final one. Rare STOP support and a small, imbalanced dataset limit interpretation. No class weights or thresholds were adjusted after observing the result.

## Verification and next milestone

All 196 tests passed, including preprocessing and mapping, disallowed splits, invalid data, optimizer groups, finite gradients, frozen buffers, checkpoint reload, loss weighting across batches, argmax ties, absent predicted classes, stale provenance and comparison mismatch rejection. The CUDA smoke and full run passed. Selected-checkpoint reload reproduced validation predictions, and the comparison independently regenerated them. Original concept checkpoints, tree artifacts, labels and splits remain unchanged. No model was automatically promoted.

The next milestone is concept intervention: correct annotation-disagreeing predicted concepts and measure how often the existing tree action improves. Abstention, corruption experiments and final test evaluation follow. No dataset expansion, balanced-loss experiment, test evaluation, commits or pushes were performed.
