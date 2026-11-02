# Frozen experimental partitions (version 1)

Created on 2026-11-01 from the existing 500-row inspection manifest.

| Experimental partition | Images | Recording sequences | Official source |
|---|---:|---:|---|
| train | 320 | 225 | train |
| val | 80 | 47 | train |
| test | 100 | 47 | val |

All 500 samples occur exactly once. No recording sequence crosses partitions.
`sequence_id` combines city and recording sequence, not individual frame number.
Cities may overlap between train and validation: this is sequence isolation, not
a guarantee of independent routes, locations, or city-level generalization.

## Reproduction and freezing

```powershell
.\.venv\Scripts\python.exe scripts/create_splits.py
.\.venv\Scripts\python.exe -m pytest -q
```

`configs/splits.json` fixes seed 42 and a target of 80 validation images.
The script sorts official training sequence IDs, shuffles them once using a local
seeded random generator, and uses a deterministic subset-sum search on sequence
sizes to obtain exactly 80 validation images. Remaining official training images
become training data. All official validation images become experimental test
data; official test images remain excluded. This metadata-only algorithm is not
uniform sampling of all possible group partitions and does not balance labels.
It does not read images, masks, labels, or statistics reports or try other seeds.

The source manifest is unchanged. `data/manifests/experimental_splits.csv` copies
its metadata and adds `experimental_split`. All later code must use this new
column rather than `official_split` to decide experimental membership.
`data/manifests/split_protocol.json` records counts and source/output SHA-256
hashes. Both are small tracked metadata artifacts; raw data stays ignored.
Reruns verify identical output and refuse to overwrite different frozen files.
Any later change needs an explicitly documented, versioned protocol revision.

## Allowed use

- Train: develop/audit proxy label rules and fit the concept model, decision tree,
  and direct-action baseline. Fit preprocessing statistics on training only.
- Validation: select model settings, concept prediction thresholds, and abstention
  settings. Do not use it to fit model weights or repeatedly redesign label rules.
- Test: apply the frozen label generator and evaluate the frozen pipeline at the
  end. Do not adjust rules, thresholds, seeds, or model choices from test results.

Future label generation can process all partitions using the identical frozen
rules. Future label inspection sheets and rule-development statistics must be
restricted to training samples. Use the same partitions for all compared models.
Corruptions inherit the source image's partition.

## Limitations and prior exposure

Before partition creation, 20 subset examples had been previewed and aggregate
statistics had been inspected across all 500 masks. The test set therefore has
preliminary inspection exposure and must not be described as completely unseen.
No concepts, action targets, or models were tuned before this partition was made.
Stop using the all-subset inspection outputs to develop rules from this point.

No label stratification is claimed. Report missing/rare concepts or actions after
rules are frozen; do not search for a favorable seed based on test labels.
One test sequence contains 45 of the 100 images, so correlated frames reduce the
effective sample size. Report sequence counts alongside image metrics and avoid
treating 100 frames as 100 independent road scenes.
