# Concept labels: concepts-v1.0

Rules are configured in `configs/concepts.yaml` and implemented by `src.concepts.label_concepts(mask, rules)`. This is a semantic-mask labeling process, not a learned perception model. The function returns five labels (0, 1, or None), separate validity flags, audit flags, and supporting measurements. CLI CSV output represents None as an empty field, not zero.

## Raw classes and presence

Use original-resolution `*_gtFine_labelIds.png` masks, not train-ID or color masks. Class mapping source: [official Cityscapes label definitions](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

| Concept | Inclusive presence threshold |
|---|---|
| pedestrian_present | Total person pixels (24) >= 0.0001 times full image area |
| vehicle_present | Combined car (26), truck (27), bus (28), train (31), motorcycle (32) pixels >= 0.0001 times full image area |
| cyclist_present | At least one qualifying rider (25) / bicycle (33) component pair |

At 2048 x 1024, the pixel threshold is 210. It uses the full image area including ignored pixels as the denominator. Person and vehicle counts are aggregated across the image, so multiple small objects can contribute to a positive label. Bicycle pixels alone are not a cyclist and are not included in vehicle_present. Caravan/trailer classes are ignored under the chosen official evaluation-class mapping.

## Cyclist association

Find 8-connected components separately for riders and bicycles with SciPy. Retain components having at least 0.0001 times the full image area. Bounding boxes are half-open `[left, top, right, bottom]`.

A pair qualifies when its horizontal intersection is at least 25% of the narrower bounding-box width, the rider box's vertical center is strictly above the bicycle box's center, and the nonnegative vertical gap between boxes is no more than 3% of image height. Overlapping vertical boxes have zero gap.

All qualifying pairs are recorded; pair count is not a count of distinct cyclists. Semantic components may merge adjacent people or fragment occluded objects. A motorcycle rider can still be incorrectly paired with an unrelated nearby bicycle. Geometry reduces, but does not eliminate, that ambiguity.

## Road and near regions

Coordinates are normalized fractions of width/height. Floor each boundary and slice half-open:

- Road region: x = [0.35, 0.65), y = [0.50, 0.90).
- Near region: x = [0.35, 0.65), y = [0.65, 0.90).

Valid raw IDs are `7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33`. Ego-vehicle and other ignored classes do not contribute to the denominator. The remaining valid count must be at least 50% of the rectangular region area; otherwise its concept is invalid with an `insufficient_valid_region` flag. No-valid-pixel measurements are null, never NaN or infinite.

Traffic obstacle IDs are `24,25,26,27,28,31,32,33`. These include parked bicycles and unmatched riders as obstacles even when cyclist_present is false. They do not cover arbitrary debris, buildings, or every static obstruction.

- road_clear = 1 when road pixels (7) are at least 60% of valid road-region pixels AND traffic obstacle pixels are strictly below 1% of those pixels.
- obstacle_close = 1 when traffic obstacle pixels are at least 5% of valid near-region pixels.

Road clearance and proximity are fixed image-space proxies, not lane tracking, physical distance, or a safety certificate. Road_clear may be true with people or parked vehicles outside the region. Obstacle_close may be false for a relevant object above the near region. The function does not infer absent information from an invalid region.

## Reproducibility and audit protocol

The CLI checks every source mask and RGB hash against the subset manifest before publishing its label table. It validates mask shape, raw IDs and dimensions. It saves configuration/rule/manifest hashes, label-table hash, class pixel counts, region measurements, and matching geometry. Changing rules requires a new rule version and a renewed training-only audit.

The 30-image audit is selected only from official train rows, sorted by sample ID. For each concept in the listed order, include the lowest-ID example for positive, negative, and invalid categories when available, deduplicating IDs. Fill remaining positions by seed-42 random sampling from the sorted unused training rows. Validation summaries are descriptive only and do not drive rule adjustments. Existing official split names are preserved; experimental train/validation/test assignments remain a later milestone.

Review the offline gallery and `CONCEPT_AUDIT.md` before deciding whether these heuristic labels are adequate for action-label generation. Per-image manual overrides and learned uncertainty/abstention are not implemented in this milestone.
