# Interpretable and Risk-Aware Autonomous Driving Decisions

Current status: milestone 2 labels and sequence-grouped partitions are complete.
The frozen dataset metadata is `data/manifests/labels_v1.csv`; reproduce/verify
with `.\.venv\Scripts\python.exe scripts/generate_labels.py`.
See `docs/FROZEN_LABELS.md` for counts, unknown-target handling, and reproducibility.
Earlier setup and training-audit sections below describe the preceding stages.

Coursework prototype: road image -> five human-readable concepts -> shallow decision tree -> STOP / SLOW / GO, with uncertainty-aware REQUEST HUMAN CONTROL.

The current milestone provides a local Python environment, GPU verification, and a reproducible Cityscapes inspection subset. Concept labels, proxy action labels, model training, and evaluation are subsequent milestones. The supplied proposal is preserved in `docs/PROJECT_PROPOSAL.pdf` and transcribed in `PROJECT_PROPOSAL.md`.

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
- `src/`: future concept model, tree, intervention, baseline, and evaluation modules.
- `data/manifests/`: small tracked subset metadata; raw data is ignored.
- `docs/`: source proposal and implementation checklist.
- `notebooks/`: future results presentation.
- `tests/`: meaningful selection and archive-path checks.
- `outputs/`: ignored reports, figures, and future checkpoints.

The experimental partitions are now frozen: 320 training images (225 sequences), 80 validation images (47 sequences), and 100 test images (47 sequences). Use `experimental_split` in `data/manifests/experimental_splits.csv`; the original `official_split` only records Cityscapes provenance. Reproduce or verify with `.\.venv\Scripts\python.exe scripts/create_splits.py`. See `docs/EXPERIMENTAL_SPLITS.md` for the protocol, prior inspection exposure, and usage restrictions.

Training-only concept proxies and audit sheets can be reproduced with `.\.venv\Scripts\python.exe scripts/audit_concepts.py`. Rules are in `configs/concepts.json`, definitions and reviewed limitations in `docs/LABEL_DEFINITIONS.md`, and generated artifacts in `outputs/concept_audit/v1-training-audit/`. Seven training images have unknown corridor targets; these must not be filled with zero. No held-out concept labels, action targets, or trained models are produced by this audit.

See `docs/IMPLEMENTATION_CHECKLIST.md` for milestone order. In particular, define cyclist presence, lower-center road clearance and obstacle proximity, and conservative proxy action rules before generating training labels. Proximity is an image-space heuristic, not measured distance; action evaluation measures agreement with proxy rules, not real-world driving safety.
