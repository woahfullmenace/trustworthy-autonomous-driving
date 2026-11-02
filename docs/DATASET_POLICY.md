# Dataset eligibility policy

Policy version: `eligibility-v1.0`. Concept baseline: `concepts-v1.0`.

## First supervised experiment

An image is eligible only when all five concepts have binary values (`0` or `1`) and validity flags of `1`. A blank value with validity `0` excludes the image; contradictory records are errors. Exclusion reasons list every invalid concept in the established concept order. Invalid means missing annotation information, not a negative label.

Both the concept model and end-to-end baseline must use identical eligible images and eventual experimental splits. Exclude incomplete examples from action-label generation and this first supervised comparison. Preserve their source images, annotations and exclusion reasons for inspection. Missing annotation information is distinct from predicted uncertainty: excluded examples do not automatically become REQUEST HUMAN CONTROL targets. Abstention will be tuned using model predictions on experimental validation data in a later milestone.

Retain the current subset membership and all concept thresholds. The fixed road and near regions are image-space proxies: they may miss relevant objects outside the rectangles, and `obstacle_close` measures occupancy rather than physical distance. Presence anywhere in the image does not establish conflict with the ego vehicle. Vehicle presence is common and close obstacles scarce. Consider expansion only after examining action-class support; do not manufacture balance by tuning thresholds on official validation images. See [concept rules](CONCEPT_RULES.md) and [audit findings](CONCEPT_AUDIT.md).

## Inspection history and future test protection

`data/manifests/inspection_registry.json` preserves the union of the 30 concept-audit IDs and 20 setup-preview IDs. Each entry records its original split, recording sequence and all inspection sources. The overlap is one image: 49 unique inspected images span 45 sequences, covering 128 subset images. The setup previews included one official validation image; its `val` name is preserved. The concept audit contains official-training images only.

Every image in an inspected sequence is flagged, including images not directly viewed. Future splitting must exclude all those sequences from an unseen test set and keep recording sequences together. Eligibility alone does not make an image suitable for unseen testing. This milestone assigns no experimental splits and does not reinterpret the official split names.

The registry becomes the persistent input after its first creation. Regeneration does not require ignored preview files. If both registry and original `outputs/data_verification.json` are absent, restore the original report; do not regenerate previews as a substitute for past inspection history. The registry retains source ID lists and report hashes for provenance. Record any future inspections before freezing an unseen test set.

## Reproduction and outputs

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_eligibility.py
```

The command verifies manifest, label-table and rule hashes against the concept summary, validates IDs, splits, sequences, versions and label consistency, and preserves manifest row order. All input checks precede output writes. It creates:

- `data/manifests/eligibility.csv`: one decision per original image, versions, exclusion reasons and inspection flags.
- `data/manifests/inspection_registry.json`: persistent inspection history, created only when absent.
- `data/manifests/eligibility_summary.json`: counts by official split, eligible concept frequencies, inspection counts and SHA-256 input/output provenance.

These small metadata files belong in version control. Raw data and generated image galleries remain ignored. `--output-dir` changes only the eligibility CSV/summary destination; the registry stays in `data/manifests`. Outputs contain no timestamps, so unchanged inputs reproduce identical bytes.

## Verified baseline counts

Generated on 2026-10-14:

| Official split | Eligible | Excluded |
|---|---:|---:|
| train | 392 | 8 |
| val | 98 | 2 |
| Total | 490 | 10 |

Nine rows have invalid `road_clear` and `obstacle_close`; one has invalid `road_clear` alone. All 500 original rows are retained in eligibility metadata.

| Concept among eligible images | Positive | Negative |
|---|---:|---:|
| pedestrian_present | 376 | 114 |
| vehicle_present | 471 | 19 |
| cyclist_present | 147 | 343 |
| road_clear | 279 | 211 |
| obstacle_close | 16 | 474 |

These counts describe the fixed subset, not an independently evaluated driving system. No action targets, model training or driving-safety claims are established by this policy.
