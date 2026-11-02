# First concept-model experiment

Configuration: `concept-training-v1.0`, in `configs/concept_training.yaml`. This milestone learns image -> five concepts, using concepts-v1.0 and frozen splits-v1.0. It does not train the action tree, end-to-end baseline or abstention rule.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/train_concepts.py --smoke
.\.venv\Scripts\python.exe scripts/train_concepts.py
```

Commands create separate timestamped directories under ignored `outputs/concept_training/`. An optional `--run-dir outputs/concept_training/unique_name` chooses a name; an existing directory is rejected. Failed directories retain a status/error record. Runs are not resumed or silently overwritten. The smoke run uses only the first 16 training examples for one update and checks its predictions on those same examples; its loss is not validation evidence. The full run starts again from pretrained weights with a freshly seeded head.

The CUDA GPU is required. ResNet18 `IMAGENET1K_V1` weights are downloaded from the official PyTorch URL into ignored `outputs/torch_cache/` and their filename hash prefix is verified. A missing or failed download is an error, never a fallback to random initialization.

## Data and optimization

The loader validates the full metadata provenance chain and requires both frozen split artifacts to exist. It joins by sample ID and reads only 342 training and 74 validation images. It verifies image SHA-256, decoding, dimensions and concept validity before training; hashes are checked again on each load. The 74 test images and all segmentation masks are unused. No validation images are visually inspected and no concept rules are revised.

RGB images are resized in full to 224 high x 448 wide with PIL bilinear interpolation and antialiasing, scaled to [0,1], then normalized by ImageNet mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. There is no crop or augmentation. Concepts use the established order: pedestrian_present, vehicle_present, cyclist_present, road_clear, obstacle_close.

Only the new 512-to-5 linear layer is optimized. Backbone parameters are frozen, and every backbone BatchNorm layer stays in evaluation mode, preserving its buffers. Training uses unweighted binary cross-entropy with logits, Adam with learning rate 0.001 and weight decay 0.0001, batch size 16, float32, and 20 epochs. Seed 42 controls Python, NumPy, Torch, CUDA and the shuffled training loader. Validation order is fixed. Windows data-loader workers are zero; CPU preprocessing uses four Torch threads.

Strict deterministic algorithms are enabled, cuDNN benchmarking and TF32 are disabled, and CUBLAS_WORKSPACE_CONFIG is :4096:8. Unsupported deterministic operations fail explicitly. Reproducibility is scoped to the recorded software/hardware environment; a seed alone does not guarantee identical results across platforms or package versions.

There is no scheduler, early stopping, class weighting, fine-tuning or threshold tuning. Epoch loss is averaged across every sample and all five concepts, including the smaller final batch. Training loss describes batches during optimization; validation loss describes the model at epoch end. The checkpoint with lowest validation loss is selected; ties preserve the earlier epoch. Model selection makes validation metrics development results rather than held-out test estimates.

## Artifacts and metric definitions

Each run saves `metadata.json`, `history.json`, `best.pt`, `metrics.json`, `validation_predictions.csv`, `loss.png` and `status.json`. Metadata records input hashes, configuration, ImageNet weight identity/hash, library/GPU versions and exact training/evaluation IDs. The checkpoint stores the state dictionary, epoch, preprocessing and concept order for reuse. The full run checks that backbone parameters and BatchNorm buffers remain byte-identical, the head changes, and reloading the selected checkpoint exactly reproduces its validation probabilities and loss.

For each concept, probabilities >= 0.5 are positive. Metrics include TP/TN/FP/FN, positive/negative support, precision, recall, specificity and F1 = 2TP/(2TP+FP+FN). Zero denominators produce JSON null and an explicit explanation. Macro F1 averages defined concept F1 values and records how many were included; false negatives still produce F1=0 when no positives are predicted.

The constant baseline outputs each concept's training-set positive fraction for every validation example, thresholded at 0.5. It never estimates frequencies from validation. Compare this baseline with the trained model to reveal how much comes from class prevalence.

Validation has no vehicle-negative examples, so vehicle specificity is undefined. There are only three obstacle_close positives, making recall sensitive to a single prediction. Probabilities are uncalibrated. Accuracy, favorable F1 or low loss cannot establish driving safety. Recovered pilot results are recorded below.


## Recovered pilot results (verified 2026-10-22)

The completed run is `outputs/concept_training/pilot_v1`. Its original checkpoint was reused without retraining. All 20 epochs completed; epoch 18 was selected with validation loss 0.3591389166. The new tree evaluation regenerated all 74 validation probability vectors with maximum absolute difference 0.0 from the saved predictions.

Concept macro F1 is 0.495728, compared with 0.480464 for constant predictions based on training prevalence. Pedestrian recall is 51/52, but all 22 pedestrian-negative images are false positives. Cyclist recall is 1/15; obstacle_close recall is 0/3. Road-clear F1 is 0.557377. Vehicle F1 is 1.0 on 74 positives and zero negatives, so it does not establish vehicle discrimination. This is a weak first baseline, preserved for comparison.

The original run's checks confirm that backbone parameters and BatchNorm buffers were unchanged, the head changed, and checkpoint reload reproduced predictions. The saved checkpoint, original run files and frozen source metadata remain unchanged. The next milestone report, DECISION_TREE.md, measures the consequences of concept errors on actions.

## Subsequent controlled experiment

The separate layer4 fine-tuning experiment is complete. See [CONCEPT_FINETUNING.md](CONCEPT_FINETUNING.md) for the v2 configuration, commands, mixed validation findings and comparison through the unchanged tree. The recovered v1 run remains preserved and is still the default training configuration.
