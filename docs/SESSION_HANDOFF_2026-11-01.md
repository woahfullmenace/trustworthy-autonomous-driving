# Session handoff: 2026-11-01

## Where to start

Milestone 2 (concept/action labels and experimental partitions) is complete.
No models have been trained. Read PROJECT_PROPOSAL.md, AGENTS.md,
docs/FROZEN_LABELS.md, docs/LABEL_DEFINITIONS.md, and docs/ACTION_LABELS.md first.
The next milestone is a five-output pretrained ResNet18, shallow decision tree,
and direct-action ResNet18 baseline. Advance AGENTS.md to the training milestone
when that work is requested; its current scope is still labeling and inspection.

Use data/manifests/labels_v1.csv as the labeled sample inventory and its
experimental_split column for all experiments. official_split records Cityscapes
provenance only. Source images and masks are external/ignored assets.

## What this session completed

1. Reviewed the proposal, teammate's setup scripts, and verification notes.
2. Reproduced the 500-image inspection subset; its manifest hash matches the
   teammate's original. Inspected raw class occurrence and sequence coverage.
3. Created sequence-disjoint experimental partitions using metadata only, seed 42.
   Official validation images became experimental test images; official training
   images were grouped into training and experimental validation.
4. Defined five explicit mask-derived concept proxies and generated training-only
   statistics/labels. Reviewed 13 targeted examples across four concept sheets.
5. Researched official UK Highway Code guidance for qualitative motivation,
   defined a coursework STOP/SLOW/GO policy, and reviewed ten training examples
   across three action sheets. Sources and assumptions are in ACTION_LABELS.md.
6. Froze source/config/split hashes BEFORE generating held-out labels, then applied
   the same functions unchanged to all 500 samples. No held-out threshold tuning.
7. Verified training labels match the earlier audit, metadata and sequence
   membership remain unchanged, and all 12 tests pass.

## Current data

| Partition | Images | Sequences | STOP | SLOW | GO | Unknown action |
|---|---:|---:|---:|---:|---:|---:|
| train | 320 | 225 | 33 | 213 | 67 | 7 |
| val | 80 | 47 | 7 | 58 | 14 | 1 |
| test | 100 | 47 | 9 | 84 | 5 | 2 |

There are 490 action-eligible images. Unknowns remain in their original partitions.
Blank targets must not be filled with zero. Mask missing concept targets in future
losses; use action_eligible identically for CBM and baseline action supervision and
metrics, reporting exclusions. Action IDs: STOP=0, SLOW=1, GO=2.

## Reproduction on this machine

Commands run from the repository root with Python 3.14:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/create_splits.py
.\.venv\Scripts\python.exe scripts/generate_labels.py
```

The last command needs original extracted masks. On a new checkout, obtain the
licensed Cityscapes source archives and place them in ../dataset/:
leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip. Preserve the archives.
Recreate the environment using README dependency instructions, then run
`.\.venv\Scripts\python.exe scripts/prepare_data.py` to restore the selected files.
The existing pretrained weights have not been downloaded by our setup checks.

Optional training-only audit regeneration:

```powershell
.\.venv\Scripts\python.exe scripts/audit_concepts.py
.\.venv\Scripts\python.exe scripts/audit_actions.py
```

Outputs are under outputs/concept_audit/v1-training-audit/ and
outputs/action_audit/v1/. They are generated/ignored and not included in Git.
scripts/inspect_masks.py is the historical whole-subset descriptive inspection;
do not use its all-subset results to redesign label rules now.

## Hardware and environment

This laptop has Intel Core Ultra 5 225U, 16 GB RAM, and Intel integrated graphics.
The teammate's RTX 5070 Ti verification in SETUP_VERIFICATION.md describes their
machine, not this one. scripts/check_environment.py still explicitly requires
CUDA and fails here as expected. CPU data preparation, labeling, and tests work.
Before training, implement a CPU-compatible model check and time a small batch.
Assess whether to train locally or on the teammate's GPU without promising a
training duration. Keep environments, weights, checkpoints, archives, raw data,
and .idea IDE state out of Git.

## Freeze and scientific limitations

configs/label_protocol_v1.json locks concept config, source scripts, generator,
and split metadata. Generation refuses changed sources or conflicting outputs.
Do not delete or regenerate the lock to bypass a mismatch. A deliberate change
needs a documented versioned protocol. .gitattributes preserves bytes for hashed
files so Windows line-ending conversion does not silently break checksums.

The rules use scene-wide presence and a fixed near-field rectangle. Cyclist is
bicycle/rider co-occurrence, not associated instances. Close means rectangle
occupancy, not physical depth. Signals, motion, right of way, and intended path
are absent. Vehicle presence alone does not affect action targets. A pedestrian
outside the path can cause SLOW/STOP, and a red light can still receive GO.
These are policy-imitation labels, not expert driving ground truth. Intervention
results likewise measure improvements relative to this synthetic policy.

Initial previews and aggregate statistics included the whole subset before splits
were frozen; do not claim completely unseen test data. Rule development thereafter
used training only. One test sequence contributes 45 frames; test GO has only five
samples. Report correlation, class imbalance, and small-sample limitations. Do not
rebalance test or select a new seed because of these counts.

## Next implementation order

1. Advance the milestone when starting training and add CPU/device-aware checks.
2. Build image loaders using frozen labels, train-only augmentation, consistent
   preprocessing, and missing-target masks; never feed mask-derived measurements
   or labels into the direct-image baseline as input features.
3. Train pretrained ResNet18's five-output head with backbone frozen first.
4. Fit a shallow tree; compare its behavior on annotated and predicted concepts.
5. Train the direct-action baseline on the same action-eligible samples.
6. Use validation only for model selection, concept prediction thresholds, and
   abstention. Later add interventions, corruption tests, and final test metrics.

No training, model checkpoints, abstention implementation, corruption evaluation,
or performance claims were produced in this session.
