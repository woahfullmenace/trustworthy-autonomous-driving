# Concept audit — concepts-v1.0

Completed 2026-10-08. All 500 existing image/mask pairs were labeled, and all 30 selected official-training examples were visually inspected using the original image, mask overlay, regions, component boxes, and numeric diagnostics. These observations are an agent visual audit, not independently annotated accuracy measurements or user approval of the rules.

## Generated artifacts and reproducibility

- Labels and measurements: `data/manifests/concept_labels.csv`.
- Split counts, co-occurrences, label patterns and provenance: `data/manifests/concept_summary.json`.
- Full component geometry: `outputs/concepts/diagnostics.json`.
- Offline gallery: `outputs/concepts/audit/index.html` (30 individual PNGs).
- Rules: `configs/concepts.yaml`; exact definitions: `CONCEPT_RULES.md`.

Two full labeling runs produced byte-identical CSV, summary JSON, diagnostic JSON and gallery HTML. All 500 RGB and mask SHA-256 values still match the original subset manifest after execution, and the subset manifest itself is unchanged. Every label is 0, 1, or explicitly invalid with a matching validity flag. All 30 gallery image references resolve locally. The 40 tests pass, including original setup tests; `pip check` reports no broken requirements.

Label-table SHA-256: `1f852f71790940183ec9bf0808828c7fd4a560b956d863b9c91d53e77c5c7b85`.

## Label counts

Each cell is **positive / negative / invalid**. These are heuristic target counts, not prediction metrics. Validation counts are reported descriptively; no validation images were used to refine rules in this milestone.

| Concept | Official train (400) | Official val (100) | Total (500) |
|---|---|---|---|
| pedestrian_present | 302 / 98 / 0 | 82 / 18 / 0 | 384 / 116 / 0 |
| vehicle_present | 387 / 13 / 0 | 93 / 7 / 0 | 480 / 20 / 0 |
| cyclist_present | 101 / 299 / 0 | 50 / 50 / 0 | 151 / 349 / 0 |
| road_clear | 218 / 174 / 8 | 61 / 37 / 2 | 279 / 211 / 10 |
| obstacle_close | 15 / 378 / 7 | 1 / 97 / 2 | 16 / 475 / 9 |

Ten images have at least one invalid concept. Keep these rows and flags visible; do not silently fill them with zero. Their later treatment must be specified before training or proxy-action assignment.

## Findings and decisions needed before action labels

**Presence labels are reasonable as whole-image annotation proxies.** Vehicle presence is positive for 387/400 training examples, so it will carry little discrimination unless evaluated carefully. A parked car or distant car still makes vehicle_present positive. Pedestrian presence includes sidewalk people and excludes riders by definition. Neither concept alone indicates a collision risk.

**Cyclist geometry gives interpretable matches in the audited positives.** Examples include `bremen_000188_000019`, `aachen_000013_000019`, and `bremen_000133_000019`. No obvious incorrect pair was observed among these selected positives, but the audit does not establish precision or recall. Small/far objects and overlapping riders can be missed or wrongly paired. `aachen_000029_000019` contains rider and bicycle components, but none meet the spatial pairing rule: their boxes are separated horizontally. Keep this edge case visible when assessing the intended meaning of cyclist presence.

**Road clearance needs a semantic decision before it drives actions.** A small part of a parked car at the rectangular region edge can make road_clear false even when the visible center corridor appears open. Examples: `bremen_000225_000019`, `bremen_000113_000019`, and `stuttgart_000039_000019`. The implementation follows the agreed 1% threshold; the limitation is the fixed region's approximation of the driving corridor. Do not automatically translate every road_clear=0 into STOP without addressing this behavior.

**Close-obstacle positives are plausible but scarce and spatially restrictive.** The close car in `bremen_000010_000019` and the car intersecting the near region in `bremen_000086_000019` trigger the rule. Crossing pedestrians in `dusseldorf_000070_000019` are above the near region: road_clear=0, obstacle_close=0. This is consistent with the geometry, but obstacle_close=0 cannot mean no relevant hazard. There are only 15 positives in the 400 training images, and only one in the official-validation subset; later evaluation must disclose limited support rather than infer reliability from accuracy alone.

**Invalid coverage is handled correctly and must remain explicit.** In `aachen_000162_000019`, most of the central paved area has ignored annotations. The road/near regions have approximately 22.49%/30.02% valid coverage, so both concepts are invalid. Inferring clearance from just the remaining pixels would be misleading.

The five-concept system also omits traffic-signal state and motion. For example, `bochum_000000_001097` has an open region with a visible traffic signal; road_clear is not an instruction to GO. The later policy remains a coursework proxy.

**Handoff:** the software implementation and audit are complete. Retain v1.0 unchanged as a reproducible baseline. Before action-label generation, explicitly decide whether to accept or revise the fixed-region meaning of road_clear/obstacle_close, specify invalid-label handling, and address rare-class support. No thresholds, masks, individual labels, or subset membership were changed after inspecting the results.

## Inspection log: all 30 training examples

Values below are ordered **pedestrian, vehicle, cyclist, road_clear, obstacle_close**; `?` denotes invalid. Notes describe visible evidence and proxy limitations, not certified scene truth.

| # | Sample | Values | Inspection note |
|---|---|---|---|
| 01 | aachen_000002_000019 | 1,1,0,1,0 | People and parked cars outside a visibly open center region. |
| 02 | aachen_000060_000019 | 0,1,0,0,0 | Cars reach the road-region edge; 1.585% obstacles, while near region is empty. |
| 03 | bremen_000188_000019 | 1,0,1,1,0 | Three plausible rider/bicycle matches; no annotated motor vehicles; center open. |
| 04 | aachen_000013_000019 | 1,1,1,1,0 | Matched cyclist above the region; road obstacle fraction remains 0.539%. |
| 05 | aachen_000029_000019 | 1,1,0,0,0 | Rider and bicycles exist but no boxes meet association; car at upper region edge. |
| 06 | aachen_000162_000019 | 1,1,0,?,? | Ignored central surface leaves too little valid coverage; both flags appropriate. |
| 07 | bremen_000010_000019 | 1,1,1,0,1 | Car fills near region; two plausible side rider/bicycle matches. |
| 08 | tubingen_000023_000019 | 1,1,0,1,0 | Sidewalk people and parked vehicles coexist with an open central rectangle. |
| 09 | bremen_000133_000019 | 1,1,1,0,0 | Right-side cyclist matched; cars overlap road region, not near region. |
| 10 | aachen_000131_000019 | 1,1,1,0,0 | Small distant cyclist matched; traffic occupies upper/side road-region pixels. |
| 11 | zurich_000067_000019 | 1,1,0,1,0 | People and vehicles surround an open near rectangle; clearance is local only. |
| 12 | dusseldorf_000178_000019 | 1,1,0,1,0 | Car ahead mostly above region; 0.393% obstacle overlap still allows road_clear. |
| 13 | dusseldorf_000108_000019 | 1,1,0,1,0 | Sidewalk people and a small distant vehicle; center open. |
| 14 | dusseldorf_000048_000019 | 0,1,0,1,0 | Tree-lined road with side/distant vehicles and no person annotation. |
| 15 | bremen_000225_000019 | 0,1,0,0,0 | Parked cars intrude into rectangle; 6.319% road obstacles, only 0.697% near. |
| 16 | zurich_000021_000019 | 1,1,0,1,0 | Nearby people remain outside central region; presence alone is not road blockage. |
| 17 | bremen_000113_000019 | 0,1,0,0,0 | Narrow street: parked-car edges make clearance false despite open near center. |
| 18 | ulm_000045_000019 | 0,1,0,1,0 | Car and bus above region; clearance is image-region occupancy, not following distance. |
| 19 | strasbourg_000001_039231 | 1,1,0,1,0 | Opposite/side traffic and distant people; only 0.012% road-region obstacles. |
| 20 | bremen_000058_000019 | 1,1,0,1,0 | Sidewalk person and distant cars outside mostly clear central paving. |
| 21 | stuttgart_000039_000019 | 1,1,0,0,0 | Parked-car edge yields 1.457% road obstacles while near region stays empty. |
| 22 | jena_000043_000019 | 1,1,0,1,0 | Person at curb/upper edge; rectangle includes sidewalk but remains >60% road. |
| 23 | bochum_000000_013209 | 0,1,0,1,0 | Curved road with vehicles outside the fixed region; region is not lane tracking. |
| 24 | bochum_000000_006746 | 1,1,0,1,0 | Sidewalk pedestrian and parked/distant cars; center region open. |
| 25 | bremen_000086_000019 | 1,1,0,0,1 | Car intersects near region: 11.772% near obstacles, appropriate proxy positive. |
| 26 | dusseldorf_000039_000019 | 0,1,0,1,0 | Distant traffic above an open central rectangle. |
| 27 | dusseldorf_000070_000019 | 1,1,0,0,0 | Crossing pedestrians overlap road region but lie above near region; key proximity limitation. |
| 28 | strasbourg_000000_023854 | 0,0,0,1,0 | Open road without annotated people or vehicles; expected negative presence controls. |
| 29 | stuttgart_000095_000019 | 0,1,0,0,0 | Rider annotation has no qualifying bicycle component; rider is not counted as person. |
| 30 | bochum_000000_001097 | 0,1,0,1,0 | Open region and car to the right; traffic-signal meaning is outside the concept set. |

Preserve these audit IDs and their recording sequences as development data when designing later experimental splits. They have been inspected during rule development and should not become a nominally unseen test set.
