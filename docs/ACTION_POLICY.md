# Synthetic action policy: actions-v1.0

The frozen configuration is `configs/actions.yaml`; the reusable function is `src.actions.concepts_to_action(concepts)`. It accepts all five concepts as Python integer 0/1 values and returns `(action, reason)`. Missing, blank or nonbinary inputs are errors. The command validates validity flags before converting CSV values.

Apply rules in this order:

1. `obstacle_close=1`: STOP, overriding every SLOW trigger.
2. Otherwise, any of `pedestrian_present=1`, `cyclist_present=1`, `road_clear=0`: SLOW. Record all applicable triggers in that order.
3. Otherwise: GO. Record the four conditions permitting GO.

Vehicle presence alone has no effect: it includes parked and distant vehicles anywhere in the image. It remains a concept prediction target. Correcting vehicle_present alone cannot change this reference policy's action; a learned tree could nevertheless acquire a spurious dependence, which later experiments must distinguish from policy behavior.

These targets are synthetic annotations, not observed expert actions or real driving instructions. Traffic lights, motion, physical distance and right of way are absent. GO does not establish that proceeding is safe. Evaluation measures agreement with a concept-derived policy; this inherently favors a concept-structured model and is not independent evidence of driving safety. The later shallow tree is still to be trained; this function generates its supervision rather than replacing that model.

## Command and artifacts

```powershell
.\.venv\Scripts\python.exe scripts/label_actions.py
```

Inputs are the subset manifest, concept labels/summary/rules, persistent inspection registry, eligibility CSV/summary and action configuration. Eligibility is recomputed in memory and compared with saved provenance and output hash; it is never regenerated on disk by this command. Changes to the frozen action configuration require a new implemented and audited version.

`data/manifests/action_labels.csv` contains eligible rows only, in manifest order, with sample ID, official split, sequence, concept/eligibility/action versions, action and reason. The ten incomplete rows retain their reasons in eligibility metadata and receive no action. `action_summary.json` records all/train/val counts, proportions, distinct sequence counts, audit IDs and input/output hashes. A sequence may contribute to multiple action classes, so per-class sequence counts need not sum to the total number of sequences.

`outputs/actions/audit/index.html` is an offline gallery of the 29 eligible examples from the existing training concept audit, in that audit's order. It reuses verified source images, copied without modification into the ignored gallery, and shows all concepts and action reasons. It does not require ignored concept-preview files. Existing region/overlay contact sheets were also used for visual review; see ACTION_AUDIT.md. All selected image hashes and decoding are checked before writing outputs. Reruns are byte-deterministic.

No concept rules, eligibility policy, subset membership or inspection registry changes are needed. Both later models must use the same targets and eventual splits. REQUEST HUMAN CONTROL is not an action target; it remains a later validation-tuned response to predicted uncertainty.
