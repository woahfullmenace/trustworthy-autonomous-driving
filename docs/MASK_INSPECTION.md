# Descriptive mask inspection

Run from the repository root using PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_masks.py
```

This CPU-only script reads the existing configured manifest and raw semantic
label-ID masks. It does not alter source data, generate concept/action labels,
assign experimental splits, train models, or choose thresholds.

Outputs are ignored generated files under `outputs/mask_statistics/`:

- `inspection_summary.json`: sample/sequence counts, input manifest hash,
  illustrative region coordinates, and interpretation limits.
- `class_summary.csv`: class occurrence and pixel-fraction summaries for all
  images and separately by official Cityscapes source split.
- `per_image_class_statistics.csv`: one row per image, region, and raw class ID,
  including zeros. Use this to locate particular examples.
- `sequence_summary.csv`: number of selected images per source split/sequence.
- `annotation_occurrence.png`: full-image annotation occurrence for selected
  classes relevant to later concepts.

`images_with_any_pixels` counts images with at least one pixel of a class. It is
not an object count and is not a finalized concept label. `pixel_fraction` ranges
from 0 to 1; multiply by 100 for percentage area. All pixels, including ignored
classes and ego vehicle, remain in the denominator. A blank conditional median
means the class never occurred in that group.

The illustrative lower-center rectangle spans x=30%-70%, y=50%-90% of image
dimensions, measured from the top-left. Integer boundaries are floored; end
coordinates are excluded. This rectangle is not a chosen labeling rule and is
not guaranteed to exclude the hood or roadside objects. Compare its class
fractions with full-image fractions before designing development-only audits.

Class names and evaluation-ignore flags follow the official raw ID definitions:
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Raw IDs and train IDs must not be interchanged. Byte value 255 is kept separately
as encoded -1/license plate. Rider/bicycle co-occurrence cannot establish which
rider belongs to which bicycle. Semantic pixel totals do not measure distance.

This report describes the entire inspection subset, whose examples have already
been previewed. Document that exposure when reporting experiments. Establish
sequence-grouped development/validation/test assignments before adjusting label
rules or repeatedly auditing examples; do not tune on future test sequences.

Next, examine class frequency and recording-sequence coverage, then reserve
experimental partitions before choosing thresholds on development data.
