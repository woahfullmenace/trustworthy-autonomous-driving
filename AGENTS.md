# Coursework project instructions

Read PROJECT_PROPOSAL.md before implementing project behavior.

Build a coursework-sized prototype: road image -> five concepts -> shallow decision tree -> STOP / SLOW / GO.
Concepts: pedestrian_present, vehicle_present, cyclist_present, road_clear, obstacle_close.
Include concept intervention, validation-tuned abstention (REQUEST HUMAN CONTROL), corruption evaluation, and an end-to-end ResNet18 baseline in later milestones.

Use the existing local Python 3.14 virtual environment and NVIDIA GPU. Run commands from the repository root with `.venv/Scripts/python.exe`.
Keep source datasets, environments, and model artifacts out of Git. Preserve source archives.
Do not add CARLA, LiDAR, reinforcement learning, depth networks, new detectors, or real vehicle control.
Keep test data out of threshold selection and model tuning. Document proxy labels and their limitations.
Current milestone: frozen pilot experimental splits (splits-v1.0). Read docs/SPLIT_POLICY.md and use data/manifests/experimental_splits.csv for both models. Keep all inspected sequences in training. Do not overwrite the frozen split or expose test images in development galleries; dataset expansion needs a separate versioned decision. Read docs/ACTION_POLICY.md and docs/ACTION_AUDIT.md before changing action behavior. Keep actions-v1.0 frozen; vehicle presence alone has no reference-policy action effect. Read docs/CONCEPT_RULES.md, docs/CONCEPT_AUDIT.md and docs/DATASET_POLICY.md before changing label or eligibility behavior. Keep concepts-v1.0 as the baseline and version policy changes. Keep invalid concepts distinct from negative labels and predicted uncertainty; do not tune rules on official validation examples. Both models must use identical eligible images and frozen experimental splits. Preserve the inspection registry and exclude inspected recording sequences from a future unseen test set. Action labels and experimental splits are implemented; model training remains a later milestone. Validation has no vehicle-negative examples, and validation/test each have only three STOP examples: report these limits without changing the split.
