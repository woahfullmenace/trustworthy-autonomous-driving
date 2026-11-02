# Implementation checklist

## Milestone 1: setup and inspection

- [x] Create the isolated Python 3.14 environment and install CUDA PyTorch.
- [x] Preserve the supplied proposal and document the coursework scope.
- [x] Record dependencies and verify imports with pip check.
- [x] Verify ResNet18 forward/backward computation on the NVIDIA GPU.
- [x] Prepare 500 reproducible image/mask pairs and save hashes in a manifest.
- [x] Inspect 20 semantic overlay examples and record findings.
- [x] Run selection/path checks and verify Git exclusions.

Completed 2026-10-08. See SETUP_VERIFICATION.md for evidence and inspection notes.

## Milestone 2A: concept labels and visual audit

- [x] Define exact class mappings and pixel thresholds for presence concepts.
- [x] Define cyclist handling using qualifying rider/bicycle component geometry.
- [x] Define lower-center region and road_clear / obstacle_close proxy thresholds.
- [x] Generate five labels, explicit invalidity flags, diagnostics, counts and co-occurrences for 500 images.
- [x] Inspect 30 deterministic training-only audit examples and record individual observations.
- [x] Verify deterministic reruns, source integrity, offline gallery assets, and boundary tests.

Completed 2026-10-08 with rules concepts-v1.0. See CONCEPT_AUDIT.md. Draft rules remain unchanged; fixed-region semantics, rare classes and invalid labels need decisions before actions/training.

## Milestone 2B: action labels and experimental splits

- [x] Accept concepts-v1.0 as the initial baseline with documented limitations.
- [x] Exclude incomplete concept rows from first action labeling and supervised comparison.
- [x] Preserve inspection IDs, sources and sequence flags in persistent metadata.
- [x] Generate 500 eligibility decisions: 490 eligible and 10 excluded.
- [x] Verify metadata integrity, deterministic regeneration and focused eligibility tests.
- [x] Define actions-v1.0 STOP / SLOW / GO proxy targets and their limitations.
- [x] Audit 29 eligible training examples and report action-class support.
- [x] Confirm pilot split feasibility; defer expansion while documenting limited rare-action support.
- [x] Freeze splits-v1.0: 342 train / 74 validation / 74 test; inspected sequences only in training.

Eligibility chunk completed 2026-10-14. See DATASET_POLICY.md. The 49 directly inspected images span 45 sequences (128 subset images); those sequences cannot enter a future unseen test set. Action targets completed 2026-10-14: 490 rows (16 STOP, 428 SLOW, 46 GO), with 29 training examples visually audited. See ACTION_AUDIT.md. Pilot splits completed 2026-10-14; see SPLIT_POLICY.md. Expansion is deferred for this pilot, with only three STOP examples in validation/test and no vehicle-negative validation examples.

## Milestone 3: core models

- [x] Train five-output pretrained ResNet18 with a frozen backbone first; recovered results verified 2026-10-22.
- [x] Fit the shallow tree and compare annotation/predicted concepts on validation; see DECISION_TREE.md. Predicted pipeline returns SLOW for all 74 examples.
- [x] Train the direct-action ResNet18 baseline and compare with both concept pipelines; see ACTION_TRAINING.md.
- [x] Fine-tune layer4 and the head in a controlled v2 experiment; compare through the unchanged tree. See CONCEPT_FINETUNING.md.

- [x] Run the balanced concept-loss v3 experiment and compare against v1/v2 through the same tree; see CONCEPT_BALANCED.md.

## Milestone 4: trustworthy-AI experiments

- [x] Measure action improvements and harms after simulated concept corrections; see CONCEPT_INTERVENTION.md.
- [x] Freeze intervention-v1.0 priority and report success over all 11 initial errors; 10 fixed, 2 newly harmed.
- [ ] Tune safety-concept uncertainty ranges using validation data only.
- [ ] Report coverage, selective error, and non-abstained STOP-to-GO errors.
- [ ] Evaluate blur, darkness, and noise on held-out images with fixed severities/seeds.
- [ ] Report per-concept precision/recall/F1 and action accuracy/macro F1.
- [ ] Discuss proxy-target bias, probability calibration, and small-sample limitations.
- [ ] Save reproducible configurations, checkpoints, plots, and final report.
