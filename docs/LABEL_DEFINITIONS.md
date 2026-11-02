# Concept proxy definitions: v1 training audit

Status: implemented and spot-audited on training images only. These are explicit
coursework proxy targets, not manually verified truth or real driving judgments.
Action targets are not defined yet. Do not train a model at this milestone.

The executable settings are in `configs/concepts.json`. The audit command is:

```powershell
.\.venv\Scripts\python.exe scripts/audit_concepts.py
```

It checks the frozen split-manifest hash, filters to `experimental_split=train`
before opening any image or mask, validates training mask hashes, and reads masks
at their original resolution. It writes generated artifacts only under
`outputs/concept_audit/v1-training-audit/`. No validation/test labels are produced.

## Presence concepts

All presence fractions use the full image area, including ignored pixels, as the
denominator. The rule is `fraction >= 0.0001` (0.01%, at least 210 pixels in a
2048 x 1024 image). This modest, explicit starting cutoff suppresses tiny annotation
fragments; it is not an optimized threshold and has not been selected to maximize
class balance or model accuracy. Pixel counts sum over the specified classes and
all objects; they do not count instances or require one object of that size.

| Concept | Raw semantic IDs and rule |
|---|---|
| `pedestrian_present` | Person (24) fraction >= 0.0001. Rider (25) is separate. |
| `vehicle_present` | Combined car (26), truck (27), bus (28), train (31), motorcycle (32) fraction >= 0.0001. Parked vehicles count. Bicycles, ego vehicle, ignored caravan/trailer classes do not count. |
| `cyclist_present` | Bicycle (33) fraction >= 0.0001 AND rider (25) fraction >= 0.0001 in the same image. |

The cyclist rule is explicitly a bicycle/rider co-occurrence proxy. It cannot
associate riders with particular bicycles; a motorcycle rider plus an unrelated
parked bicycle can be a false positive. Occluded riders or bicycles can be false
negatives. It does not establish riding, motion, or presence in the road corridor.

## Forward corridor proxies

The fixed rectangle spans x=35%-65% of width and y=55%-85% of height from the
top-left. Coordinates are floored to integers and endpoints are excluded. This
narrower near-field rectangle aims to reduce roadside-vehicle contamination and
the bottom hood. It is not lane detection and can miss curved roads, crossings,
and objects just beyond the rectangle.

Ignore IDs 0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,255 when calculating corridor
fractions. This includes ego vehicle and other classes excluded by the standard
Cityscapes evaluation convention. Removing them does not establish free space.
First measure the fraction of all rectangle pixels that remain valid. If less
than 50% are valid, both corridor targets are **unknown**, saved as blank CSV
values. `corridor_usable=0` identifies those rows. Never fill unknown targets with
zero. Future training must mask these target losses, and future action labeling
must specify how to handle these samples without changing frozen partitions.

For usable rectangles, fractions below use only valid rectangle pixels:

- Road fraction: proportion with road ID 7.
- Obstacle fraction: proportion with IDs 24,25,26,27,28,31,32,33.
- `road_clear=1` iff road fraction >= 0.80 AND obstacle fraction <= 0.05.
- `obstacle_close=1` iff obstacle fraction >= 0.10.

These are deliberately different concepts: both may be zero. `road_clear` means
the specified near-field rectangle is predominantly annotated road, not that GO
is safe. `obstacle_close` means substantial annotated human/vehicle occupancy in
that rectangle, not measured depth. Buildings, poles, unrecognized debris, and
ignored objects are not included in the obstacle list. These are important false
negative sources. Do not interpret either rule as a collision-risk estimate.

## Training results and visual audit (2026-11-01)

| Concept | Positive | Negative | Unknown |
|---|---:|---:|---:|
| pedestrian_present | 239 | 81 | 0 |
| vehicle_present | 310 | 10 | 0 |
| cyclist_present | 86 | 234 | 0 |
| road_clear | 280 | 33 | 7 |
| obstacle_close | 17 | 296 | 7 |

All 320 training samples and 225 recording sequences are retained. Thresholds
were not changed to force balance. Vehicle negatives and close-obstacle positives
are scarce; later models need per-concept metrics rather than accuracy alone.

Thirteen unique examples were selected reproducibly: first positive/negative
sample per concept in sorted sample order, nearest numerical rule boundaries,
and three insufficient-coverage cases. All four audit sheets were visually
reviewed. The sample is targeted, not a representative estimate of label accuracy.

Findings:

- `aachen_000013_000019`: roadside cars count as vehicles while the central
  rectangle remains clear, illustrating the difference between scene presence
  and forward occupancy.
- `bremen_000010_000019`: a close front car occupies almost all valid rectangle
  pixels; close=1 and clear=0, consistent with the intended occupancy proxy.
- `bremen_000148_000019`: cyclists crossing just above the rectangle yield
  cyclist=1 but clear=1 and close=0. This is a corridor-proxy limitation; action
  rules must not equate clear=1 alone with GO.
- `cologne_000058_000019`: crossing people produce obstacle fraction about 0.052,
  enough for clear=0 but not close=1. Threshold boundaries are approximate.
- `stuttgart_000164_000019`: a crossing pedestrian near the rectangle edge yields
  obstacle fraction about 0.100, illustrating boundary sensitivity.
- `aachen_000162_000019`, `darmstadt_000083_000019`, and `jena_000084_000019`
  have very low valid corridor coverage on paved/cobblestone areas; their local
  concepts remain unknown despite apparently open foregrounds.

These observations support retaining the definitions as a documented first
version, with the stated limitations. A larger manual audit would be needed to
estimate proxy-label accuracy. Subsequent rule revisions must use training data
only and receive a new version. Validation is reserved for model/abstention
selection, and test for final evaluation. Earlier whole-subset inspection exposure
is documented separately in `docs/EXPERIMENTAL_SPLITS.md`.

## Files to inspect

- `training_concepts.csv`: training metadata, measurements, five targets, and
  corridor usability. Concept fractions are 0-1, not percentages.
- `training_statistics.csv`: descriptive statistics using only training rows.
- `audit_examples.csv`: selected sample IDs, labels, and selection reasons.
- `audit_01.png` through `audit_04.png`: RGB/mask pairs, yellow corridor, labels.
  Titles use ped/veh/cyc/clear/close in the concept order above; `nan` means unknown.
- `audit_summary.json`: positive/negative/unknown counts and configuration/split
  hashes. Outputs are ignored generated artifacts, not source data.

STOP/SLOW/GO proxy targets and unknown-target handling are now implemented and
audited on training only; see ACTION_LABELS.md. Freeze the complete labeling
protocol before applying it unchanged to held-out partitions. No models are trained.
