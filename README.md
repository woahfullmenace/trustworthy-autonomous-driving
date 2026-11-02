# Interpretable and Risk-Aware Autonomous Driving Decisions

Coursework prototype: road image -> five human-readable concepts -> shallow decision tree -> STOP / SLOW / GO, with uncertainty-aware REQUEST HUMAN CONTROL.

The project provides a local Python environment, GPU verification, a reproducible Cityscapes inspection subset, and versioned heuristic concept labels with a training-only visual audit. Proxy action labels, experimental splits, model training, and evaluation are subsequent milestones. The supplied proposal is preserved in `docs/PROJECT_PROPOSAL.pdf` and transcribed in `PROJECT_PROPOSAL.md`.

## Local setup (Windows PowerShell)

Run all commands from the repository root. Python 3.14 is used; no activation is required.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After initial verification, `requirements-lock.txt` records exact versions. To reproduce this setup, install the CUDA pair above first, then use `-r requirements-lock.txt`. GPU wheels come from https://download.pytorch.org/whl/cu128; other packages come from PyPI. No standalone CUDA toolkit is needed for these packaged operations.

In VS Code, use **Python: Select Interpreter** and select `.venv/Scripts/python.exe`.

## Data preparation

`configs/default.yaml` resolves paths relative to this repository, including the source archive directory `../dataset`. Expected source files:

```text
Project/dataset/leftImg8bit_trainvaltest.zip
Project/dataset/gtFine_trainvaltest.zip
```

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
```

The script samples sorted filenames with seed 42: 400 official training images and 100 official validation images. This is an inspection subset, not a final train/validation/test assignment. Selection is not class-balanced. Class coverage and recording-sequence grouping must be checked before the later experimental split is finalized. Official test images are excluded.

Selected RGB images and raw semantic label-ID masks are extracted under ignored `data/raw/cityscapes_subset/`. Original resolution and archive contents are preserved. ZIP CRC checks occur on selected member reads, every selected pair is decoded, dimensions and label IDs are checked, and SHA-256 hashes are saved in `data/manifests/subset.csv`. Reruns retain unchanged extracted files and reproduce the manifest with the same inputs/configuration. Only selected archive members are checked, not the entire archive.

The manifest records sample ID, original split, recording sequence, repository-relative paths, dimensions, and image/mask hashes. It contains no concept or action labels. Four PNG sheets in `outputs/` display 20 image/semantic-overlay pairs; `data_verification.json` records their IDs. Raw label IDs must not be confused with Cityscapes train IDs or color masks. Images remain governed by the original Cityscapes license; extracted source notices are preserved with the subset.

## Concept labels and audit (milestone 2A)

```powershell
.\.venv\Scripts\python.exe scripts/label_concepts.py
```

The command reads the existing subset manifest, verifies all source hashes, and applies `configs/concepts.yaml` to the original raw semantic masks. It writes:

- `data/manifests/concept_labels.csv`: five labels, separate validity flags, original split, rule version, and numeric diagnostics. Empty concept cells mean invalid, not false.
- `data/manifests/concept_summary.json`: counts and co-occurrences by original split, provenance hashes, and selected audit IDs.
- `outputs/concepts/diagnostics.json`: per-image component boxes, matching geometry and regional measurements.
- `outputs/concepts/audit/index.html`: offline gallery of 30 training examples with regions, matches, labels and evidence. Open this file in a browser; no server is required.

See `docs/CONCEPT_RULES.md` for exact thresholds and `docs/CONCEPT_AUDIT.md` for inspection findings and limitations. These are baseline heuristic labels, not model predictions. The audit informed the frozen action policy. The source archives, extracted masks, and subset manifest are not modified.

For reproducibility checks, `--labels`, `--summary`, and `--output` accept alternate output destinations. `--config` and `--rules` select alternate YAML files; run commands from the repository root.

## Dataset eligibility

```powershell
.\.venv\Scripts\python.exe scripts/prepare_eligibility.py
```

This validates existing metadata and keeps `concepts-v1.0` unchanged. It writes `data/manifests/eligibility.csv` (500 decisions: 490 eligible, 10 excluded), `eligibility_summary.json` (counts and hashes), and a persistent `inspection_registry.json`. Both future models will use the same eligible images and frozen experimental splits. Incomplete examples remain available for inspection; they are not automatically human-control targets.

The registry preserves 49 inspected images across 45 sequences. All 128 images in those sequences are flagged for exclusion from a future unseen test set. Official split names remain unchanged; experimental assignments are stored separately by the split command below. Initial registry creation requires the original ignored `outputs/data_verification.json`; subsequent runs use the registry alone. See [dataset policy](docs/DATASET_POLICY.md) for definitions, counts and limitations.

## Action targets and audit

```powershell
.\.venv\Scripts\python.exe scripts/label_actions.py
```

The frozen `actions-v1.0` policy assigns STOP for a close obstacle; otherwise SLOW for a pedestrian, cyclist or unclear road; otherwise GO. Vehicle presence alone has no effect. The command validates eligibility provenance and writes `data/manifests/action_labels.csv` and `action_summary.json`: 490 targets (16 STOP, 428 SLOW, 46 GO). Ten ineligible examples receive no action.

Open `outputs/actions/audit/index.html` for the 29 already-inspected eligible training examples, their concepts and reasons. See [action policy](docs/ACTION_POLICY.md) and [visual audit](docs/ACTION_AUDIT.md). These are synthetic targets, not expert driving decisions; traffic lights and motion are omitted. STOP support is limited. Experimental splits are frozen below; model training remains pending.

## Frozen pilot experimental splits

```powershell
.\.venv\Scripts\python.exe scripts/prepare_splits.py
```

`data/manifests/experimental_splits.csv` freezes 342 training, 74 validation and 74 test images. STOP counts are 10/3/3; SLOW 300/64/64; GO 32/7/7. Recording sequences never cross splits, and all inspected sequences are restricted to training. The original Cityscapes split remains separate metadata: this is a pooled coursework pilot, not the official benchmark protocol.

`split_summary.json` records provenance, solver versions and action/concept counts. Reruns validate and preserve the frozen artifacts; changed inputs or configuration are rejected. Both models must consume the same assignments. Validation tunes settings and abstention; test is reserved for final evaluation. See [split policy](docs/SPLIT_POLICY.md), including limited STOP support and the absence of vehicle-negative validation examples. No test images were opened for this step.

## First concept-model training run

```powershell
.\.venv\Scripts\python.exe scripts/train_concepts.py --smoke
.\.venv\Scripts\python.exe scripts/train_concepts.py
```

The trainer uses the NVIDIA GPU, ImageNet-pretrained ResNet18 and the frozen 342/74 training/validation split. It trains only the new five-output head for 20 epochs; backbone parameters and BatchNorm statistics remain frozen. Full images are resized to 224 x 448 and normalized with ImageNet statistics. Test images are not loaded.

Each command creates a separate ignored run directory under `outputs/concept_training/`. Use `--run-dir outputs/concept_training/unique_name` to name a run; existing directories are refused. The smoke command uses training images only and starts separately from the full experiment. The official weights are cached under ignored `outputs/torch_cache/`.

Runs save the best validation-loss checkpoint, loss history/plot, validation predictions, metrics and provenance. See [training protocol and results](docs/CONCEPT_TRAINING.md) for metric definitions, reproducibility settings and rare-class limitations. These five outputs feed the saved action tree described below.

## Image-to-action decision tree

```powershell
.\.venv\Scripts\python.exe scripts/train_decision_tree.py
.\.venv\Scripts\python.exe scripts/predict_action.py --sample-id aachen_000002_000019 --tree-run outputs/decision_tree/pilot_v1_complete
```

The first complete pipeline reuses the recovered ResNet checkpoint and fits a shallow tree on training annotation concepts. Validation agreement is 100% with annotation concepts; using ResNet predictions produces SLOW for every image (86.49% accuracy, macro F1 0.3092, STOP recall 0%). It matches the always-SLOW baseline and needs better concept recognition.

See [decision-tree report](docs/DECISION_TREE.md) for class metrics, limitations and reproduction. The completed run contains `tree.png`, `tree.txt`, per-image decisions and model/provenance files under ignored `outputs/decision_tree/pilot_v1_complete/`. Test images remain reserved. The controlled fine-tuning comparison is complete; see [fine-tuning results](docs/CONCEPT_FINETUNING.md).

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
```

The GPU check uses randomly initialized ResNet18 and a synthetic batch, with five output logits. It verifies actual CUDA forward/backward execution and finite gradients without downloading pretrained weights or training a dataset model. Its report is `outputs/environment_verification.json`.

## Structure and next milestones

- `configs/`: dataset paths and reproducibility settings.
- `scripts/`: runnable setup/inspection checks.
- `src/`: concept labeling, action policy, image loading, concept model and metrics; future tree/intervention modules.
- `data/manifests/`: small tracked subset metadata; raw data is ignored.
- `docs/`: source proposal and implementation checklist.
- `notebooks/`: future results presentation.
- `tests/`: meaningful selection and archive-path checks.
- `outputs/`: ignored reports, figures, and future checkpoints.

See `docs/IMPLEMENTATION_CHECKLIST.md` for milestone order. The initial concept definitions are implemented, audited and accepted as the baseline under `eligibility-v1.0`. Eligibility and inspection history are recorded. Action targets are now generated and audited. Pilot splits are frozen. The first concept model and image-to-action tree are implemented; the controlled concept fine-tuning comparison is complete; the direct-action baseline is complete; the simulated concept-intervention experiment is complete; next comes uncertainty-based abstention; broader evaluation may require a separately versioned dataset expansion. Proximity is an image-space heuristic, not measured distance; action evaluation will measure agreement with proxy rules, not real-world driving safety.

## Partial fine-tuning experiment

Use `scripts/train_concepts.py --config configs/concept_finetuning.yaml` for the separate v2 run; omit `--config` to retain v1. Add `--smoke` for a separate training-only CUDA check. Compare with `scripts/compare_concept_runs.py --candidate-run outputs/concept_training/finetune_v2 --tree-run outputs/decision_tree/pilot_v1_complete`. Run these commands with `.\.venv\Scripts\python.exe` from the repository root.

See [the comparison report](docs/CONCEPT_FINETUNING.md) for results, preserved baselines and limitations. Test images remain reserved.

## Direct-action baseline

Run `.\.venv\Scripts\python.exe scripts/train_actions.py` (or add `--smoke` for a separate training-only check). Compare with `.\.venv\Scripts\python.exe scripts/compare_action_models.py --action-run outputs/action_training/pilot_v1`. Both commands accept `--run-dir` and refuse overwrites. This model predicts STOP/SLOW/GO directly from the image using the same frozen split; it does not receive concept labels as model inputs.

See [direct-action training and comparison](docs/ACTION_TRAINING.md). Test images remain reserved; see the completed concept-intervention experiment below.

## Balanced concept-loss experiment

Run `.\.venv\Scripts\python.exe scripts/train_concepts.py --config configs/concept_balanced.yaml` (add `--smoke` for the training-only check). Compare using `.\.venv\Scripts\python.exe scripts/compare_concept_runs.py --candidate-run outputs/concept_training/balanced_v3 --tree-run outputs/decision_tree/pilot_v1_complete`.

See [balanced-loss results](docs/CONCEPT_BALANCED.md) for the fixed training-derived weights, validation tradeoffs and limitations. Existing models remain preserved.

## Concept intervention

Run `.\.venv\Scripts\python.exe scripts/evaluate_interventions.py` to evaluate saved balanced-v3 validation predictions with annotation-simulated corrections. Inspect an explicit override with `.\.venv\Scripts\python.exe scripts/intervene_concepts.py --sample-id bremen_000245_000019 --set pedestrian_present=0`.

The fixed one-correction policy repairs 10/11 original action errors but introduces two new errors, raising validation agreement from 85.14% to 95.95% with correct feedback. This is assisted performance; autonomous predictions remain unchanged. See [the intervention report](docs/CONCEPT_INTERVENTION.md) for protocol, harms and examples. Uncertainty-based abstention is next.
