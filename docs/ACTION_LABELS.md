# STOP / SLOW / GO proxy policy v1

Implemented 2026-11-01 in scripts/audit_actions.py. Rules are evaluated in order:

1. Any unknown concept: leave action blank, action_eligible=false.
2. obstacle_close=1: STOP.
3. road_clear=0 and (pedestrian_present=1 or cyclist_present=1): STOP.
4. pedestrian_present=1 or cyclist_present=1: SLOW.
5. road_clear=0: SLOW.
6. Otherwise: GO.

Vehicle presence alone has no effect: parked or distant vehicles may coexist with
an open corridor. The concept remains useful for inspection, but is redundant for
this target policy; a fitted tree need not use all five concepts. STOP takes
precedence over SLOW. GO requires clear=1, close=0, pedestrian=0, cyclist=0.
SLOW and GO are categorical targets, not numerical speed commands.

## Research basis and limits

Sources consulted 2026-11-01:

- UK Highway Code rule 126 describes maintaining the ability to stop within the
  visible clear distance: https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158
- Rules H2/H3 discuss giving way to pedestrians and cyclists in particular
  junction situations: https://www.gov.uk/guidance/the-highway-code/introduction
- Rule 170 adds junction observations and yielding guidance:
  https://www.gov.uk/guidance/the-highway-code/using-the-road-159-to-203

These are official human-driving guidance, used for qualitative motivation only.
They are UK guidance, not German Cityscapes traffic-law labels or an autonomous
driving standard. They do not specify our pixel thresholds or Boolean policy.
Our five concepts cannot establish speed, stopping distance, traffic-light state,
right of way, planned path, cyclist association, or whether a pedestrian is
crossing. The resulting policy is a project design assumption.

For instance, a crossing cyclist beyond the corridor can receive SLOW where a
real driver should stop. A red-light scene can receive GO because signals are not
represented. A sidewalk pedestrian can cause SLOW unnecessarily. Do not present
these targets as human expert ground truth or evidence of real-world safety.

Actions are deterministic functions of annotation-derived concepts. Evaluation
therefore measures policy imitation and concept recovery; a tree trained on
ground-truth concepts can recover a simple label generator. Report this circularity
explicitly. The direct-image baseline must use exactly the same eligible samples
and targets. Concept intervention can demonstrate improvements relative to this
policy, not establish correct real driving decisions.

## Unknown annotations and future evaluation

Seven training samples have unknown corridor targets. They remain in their frozen
partitions and retain known concept targets, but have no supervised action label.
Do not impute zero, STOP, or REQUEST HUMAN CONTROL as their target. Future concept
training must mask missing targets. Future action training and action metrics
must use action_eligible rows identically for the CBM and baseline and report
excluded counts per partition. This exclusion is annotation eligibility, not
confidence-based abstention. Later REQUEST HUMAN CONTROL is a model output tuned
on validation data; its coverage denominator must explicitly distinguish eligible
action-evaluation images from all images.

## Training audit

Command: `.\.venv\Scripts\python.exe scripts/audit_actions.py`

Outputs: `outputs/action_audit/v1/`. training_actions.csv retains all 320 training
rows and adds action, action_reason and action_eligible. audit_summary.json records
counts and source/config/policy hashes. policy_truth_table.csv enumerates all 32
binary concept combinations (including combinations not realized by the current
mask rules). audit_examples.csv selects up to two examples for each observed
reason in sorted sample order. audit_01.png through audit_03.png display the ten
selected training examples for review.

Counts: STOP 33, SLOW 213, GO 67, unlabeled 7; 313 action-eligible samples.
All three sheets (ten training examples) were visually reviewed. The front car in
bremen_000010_000019 yields STOP; roadside cars in aachen_000060_000019 coexist
with a GO proxy. The unclear corridor plus scene-level pedestrian in
bremen_000022_000019 yields STOP even though the pedestrian is not established
to block the planned path. This illustrates over-conservatism rather than expert
action correctness. Unknown cobblestone/paved cases remain unlabeled. This small,
targeted audit checks interpretability, not an accuracy estimate.
The generic non-clear/no-vulnerable/no-close SLOW branch has no training examples.
This and class imbalance should be reported rather than fixed by test-driven
resampling. Tests enumerate binary inputs to verify STOP priority, GO restrictions,
vehicle-only invariance, and handling of unknown targets.

No validation/test images or action labels are inspected/generated here. Rules
were chosen from concept semantics and qualitative guidance, not fitted to action
balance. Subsequent changes require a new documented version developed on training
only. Next, freeze the complete labeling protocol and apply it unchanged to all
partitions, reporting eligibility and balance without retuning against test data.
