# Frozen pilot splits: splits-v1.0

Frozen on 2026-10-14 for the 490 eligible images, using concepts-v1.0, eligibility-v1.0 and actions-v1.0. Both future models must use these exact assignments. This is a pooled coursework experiment, not the official Cityscapes benchmark protocol. Original `train`/`val` names remain provenance metadata; experimental names are `train`, `validation`, `test`.

## Constraints and objective

Keep every recording sequence entirely in one experimental split. All inspected sequences are restricted to training. Ten ineligible images remain excluded, including their original annotations; no examples were added. The registry has 45 inspected sequences overall, but only 44 have eligible images: all 125 eligible images from these sequences are in training. Inspection flags were computed from the full subset, so an excluded inspected image still protects other eligible images from its sequence.

Target image fractions are 70%/15%/15%. Training must contain 310–370 images, validation and test 60–90 each. Action minimums are:

| Action | Training | Validation | Test |
|---|---:|---:|---:|
| STOP | 10 | 3 | 3 |
| SLOW | 1 | 1 | 1 |
| GO | 20 | 5 | 5 |

The SciPy mixed-integer program assigns one of three binary split indicators per sorted sequence ID. The primary objective sums `abs(split_count / overall_count - target_fraction)` over the three splits and four categories: all images, STOP, SLOW, GO. Concept frequencies are not optimization inputs.

After proving the primary optimum, a second solve minimizes seed-42 NumPy random assignment costs, constrained to the primary optimum plus 1e-7 numerical tolerance. Both solves require optimal status with zero relative MIP gap; each has a 60-second limit. Failure or infeasibility produces no split outputs and never relaxes constraints. Integer assignments, counts and the primary objective are checked again after solving.

Generation used SciPy 1.18.1, its bundled HiGHS 1.12.0 and NumPy 2.5.2. Both primary and assigned objectives were approximately 0.164646443705479. Reproduction from scratch is tested in this environment; different solver versions need not choose identical tied optima. The frozen files, not a new optimization after a dependency upgrade, are the experiment's authority.

## Assignments and support

| Experimental split | Images | Sequences | STOP | SLOW | GO |
|---|---:|---:|---:|---:|---:|
| Training | 342 | 163 | 10 | 300 | 32 |
| Validation | 74 | 74 | 3 | 64 | 7 |
| Test | 74 | 74 | 3 | 64 | 7 |

There is zero sequence overlap and no inspected sequence in validation or test. Original Cityscapes train/val contributions are 264/78 to training, 63/11 to validation and 65/9 to test. These counts reflect sequence constraints and the pooled protocol; they are not official benchmark results.

Per-concept support is reported below as positive / negative:

| Concept | Training | Validation | Test |
|---|---|---|---|
| pedestrian_present | 271 / 71 | 52 / 22 | 53 / 21 |
| vehicle_present | 325 / 17 | 74 / 0 | 72 / 2 |
| cyclist_present | 107 / 235 | 15 / 59 | 25 / 49 |
| road_clear | 209 / 133 | 30 / 44 | 40 / 34 |
| obstacle_close | 10 / 332 | 3 / 71 | 3 / 71 |

This supports a pilot pipeline, not reliable rare-event claims. One STOP test error changes STOP recall by 33.3 percentage points. SLOW is 64/74 (86.49%) of test targets, so accuracy alone can conceal rare-class failure. Validation has no vehicle-negative examples; specificity for that concept cannot be assessed there, and calibration/threshold conclusions would be limited. Test has only two vehicle-negative examples. Do not change the split based on these concept distributions; report unsupported metrics explicitly and avoid test-guided tuning.

## Reproduction and freeze behavior

```powershell
.\.venv\Scripts\python.exe scripts/prepare_splits.py
```

Configuration: `configs/splits.yaml`. Output: `data/manifests/experimental_splits.csv`, in eligible manifest order, containing sample ID, sequence, original split, experimental split and policy version. `data/manifests/split_summary.json` records configuration, input/output SHA-256 values, solver versions/objectives, action/concept/source counts and inspection checks. These small artifacts belong in Git; this milestone does not commit them.

The command verifies the eligibility provenance chain, action input/output hashes and action-policy consistency using metadata only. It never reads or displays test images. On first generation, both outputs use exclusive creation. On rerun, it verifies existing assignments and counts and leaves the exact bytes untouched rather than solving again. Changed inputs/configuration, corrupt outputs or a missing member of the frozen pair cause a clear error. Restore the authoritative files after accidental damage; an intentional dataset expansion requires a separately versioned decision, not deleting the freeze to try another seed.

## Usage and next chunk

Verification: 139 tests passed, including fresh-solve determinism, sequence and inspection restrictions, metadata inconsistencies, class/size bounds, infeasibility, optimizer timeout, and frozen-artifact protection. Two CLI reruns preserved every source table, configuration and frozen output byte-for-byte. All ten excluded IDs are absent. Frozen CSV SHA-256: `3f8296dc6c48889c8ad2cda44f72d677fc58e46f9d9314fb1ff7e452649da2c7`.

- Training: fit the concept predictor, shallow decision tree and end-to-end baseline.
- Validation: choose model settings and later abstention thresholds; never optimize those choices on test results.
- Test: evaluate after model and threshold choices are frozen. Keep test images out of development galleries and rule refinement.

No training, expansion or new image inspection occurred. The next chunk can implement the image loader and first ResNet18 concept-training run, using these assignments and reporting the limitations above. Dataset expansion remains a future versioned option rather than a prerequisite to exercising this pilot pipeline.
