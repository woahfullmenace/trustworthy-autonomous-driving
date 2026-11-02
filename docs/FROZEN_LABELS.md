# Frozen labeling protocol v1

On 2026-11-01, configs/label_protocol_v1.json recorded SHA-256 hashes of the
concept configuration, concept/action implementations, generator, and experimental
split metadata before the generator opened any held-out masks. The concept and
action functions are the same ones audited on training data. Definitions and
limitations remain in LABEL_DEFINITIONS.md and ACTION_LABELS.md.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/generate_labels.py
.\.venv\Scripts\python.exe -m pytest -q
```

The lock already exists; --freeze is only for establishing/checking that original
record and must not be used to reset it. Source hash mismatches stop generation.
Existing different output files are never overwritten. A genuine future change
requires a new documented protocol version; do not delete the lock to retune v1.

Files under data/manifests:

- labels_v1.csv: all original sample metadata, experimental membership, measured
  fractions, five nullable concept targets, action, action_reason, action_eligible,
  nullable action_id, and label_protocol.
- label_summary_v1.json: per-partition concept positive/negative/unknown counts,
  action counts, image/sequence counts, eligibility, and protocol/output hashes.

The action encoding is STOP=0, SLOW=1, GO=2. Blank concepts/actions are unknown,
not zero. Retain all samples in their frozen partitions. For concept supervision,
mask missing targets; for action supervision and action metrics, select rows with
action_eligible=true identically for both compared models. Report excluded counts.
Unknown annotation eligibility is separate from model abstention. Never expose
mask paths, annotation fractions, or target columns as input features for the
direct-image baseline. CBM inputs are images; the tree receives concepts only.

| Partition | STOP | SLOW | GO | Unknown action | Total |
|---|---:|---:|---:|---:|---:|
| train | 33 | 213 | 67 | 7 | 320 |
| val | 7 | 58 | 14 | 1 | 80 |
| test | 9 | 84 | 5 | 2 | 100 |

Training labels were checked against the prior training audit and match exactly.
All original metadata and sequence assignments are preserved. There are 490
action-eligible samples overall. Test GO has only five examples, limiting precision
of per-class performance estimates; no rule or split was changed in response.

The generator reads semantic masks to calculate labels and validate their hashes;
it does not display held-out images or choose thresholds from their labels.
Aggregate held-out class counts are a descriptive report, not a tuning signal.
Do not resplit, adjust thresholds, or expand the subset based on these counts.
Prior preview/statistics exposure remains documented in EXPERIMENTAL_SPLITS.md.

All class imbalance and unknown labels are retained. The 500-image subset is
fixed for this prototype; imbalanced-class methods, if needed later, must use
training data with model selection on validation. Test remains for final model
evaluation. The next milestone is model implementation/training, which is not
authorized by the current setup/labeling milestone in AGENTS.md.
