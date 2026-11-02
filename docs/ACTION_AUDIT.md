# Action audit: actions-v1.0

Completed 2026-10-14. Agent visual inspection covered all 29 eligible images in the existing training concept audit using its original/semantic-overlay contact sheets, with generated action rows checked against displayed concepts. This is a policy-consistency review, not independent expert action annotation. No new image IDs or sequences were inspected. The invalid example `aachen_000162_000019` is excluded and has no action target. Official validation images were not visually reviewed or used for rule refinement.

## Counts and findings

| Original split | STOP | SLOW | GO | Total |
|---|---:|---:|---:|---:|
| train | 15 | 335 | 42 | 392 |
| val | 1 | 93 | 4 | 98 |
| All | 16 | 428 | 46 | 490 |

Training STOP examples span 15 sequences, SLOW 229 and GO 31. The audit contains 2 STOP, 21 SLOW and 6 GO examples. All 29 assigned actions follow the frozen policy. The all-data SLOW share is 87.35%; training SLOW share is 85.46%. A majority-class predictor can therefore appear strong on accuracy while missing STOP entirely. Report class support and per-class metrics later; these are not model results.

The two audited STOP examples have cars occupying the near region. SLOW often follows sidewalk or distant people even when the central road is open, and parked-car edges can trigger SLOW through road_clear. Conversely, crossing pedestrians above the near region receive SLOW rather than STOP (`dusseldorf_000070_000019`). GO at `bochum_000000_001097` coexists with a visibly red overhead signal: traffic-signal state is not represented. These are substantive limitations of the policy, not labeling-code mismatches. No individual labels or rules were adjusted.

Vehicle presence alone cannot change the reference action. Interventions on that concept cannot improve reference-policy actions on their own; later learned-tree behavior must be assessed separately. Action labels are derived from the same concepts supplied to the tree, so performance measures policy reproduction rather than independent human driving judgment.

## Per-image review

Every row below passed the policy-consistency check. Exact concept values and all trigger reasons are in the gallery and action CSV.

| Sample ID | Action | Visual observation and implication |
|---|---|---|
| aachen_000002_000019 | SLOW | Sidewalk people trigger slowing although central region is open. |
| aachen_000060_000019 | SLOW | Car edges reach road rectangle; unclear-road trigger despite open near center. |
| bremen_000188_000019 | SLOW | People and matched cyclists beside a largely open intersection. |
| aachen_000013_000019 | SLOW | Cyclist ahead and people; neither requires near-region occupancy for SLOW. |
| aachen_000029_000019 | SLOW | People at intersection and car in upper road rectangle; no qualifying cyclist match. |
| bremen_000010_000019 | STOP | Close car fills near region; STOP overrides people/cyclist/unclear-road triggers. |
| tubingen_000023_000019 | SLOW | People beside parked cars, open center; whole-image presence drives target. |
| bremen_000133_000019 | SLOW | Side cyclist and traffic above near region; several SLOW triggers. |
| aachen_000131_000019 | SLOW | Distant cyclist, people and road-edge traffic; near region remains empty. |
| zurich_000067_000019 | SLOW | People surround open central rectangle; proximity is not inferred. |
| dusseldorf_000178_000019 | SLOW | People and a car ahead above near region; person presence triggers SLOW. |
| dusseldorf_000108_000019 | SLOW | Sidewalk group with open lane; presence alone triggers SLOW. |
| dusseldorf_000048_000019 | GO | No person/cyclist positives and open region; side cars alone have no effect. |
| bremen_000225_000019 | SLOW | Parked cars overlap road rectangle, with little near occupancy. |
| zurich_000021_000019 | SLOW | Nearby sidewalk people at crossing; future movement is unknown. |
| bremen_000113_000019 | SLOW | Narrow parked-car corridor; rectangle edges make road unclear. |
| ulm_000045_000019 | GO | Car and bus ahead lie above near region; GO ignores following-distance dynamics. |
| strasbourg_000001_039231 | SLOW | Distant people trigger SLOW despite clear central road. |
| bremen_000058_000019 | SLOW | Sidewalk pedestrian drives SLOW on open central paving. |
| stuttgart_000039_000019 | SLOW | Parked-car intrusion and person label; near region remains clear. |
| jena_000043_000019 | SLOW | Person near curb; fixed rectangle includes sidewalk and misses intent. |
| bochum_000000_013209 | GO | Curved road, vehicles outside region; rectangle does not track intended lane. |
| bochum_000000_006746 | SLOW | Person beside parked vehicle; center open. |
| bremen_000086_000019 | STOP | White car overlaps near region; image occupancy triggers STOP without motion context. |
| dusseldorf_000039_000019 | GO | Distant vehicles and clear center; vehicle presence does not force SLOW. |
| dusseldorf_000070_000019 | SLOW | Crossing people are above near rectangle; relevant hazard does not trigger STOP. |
| strasbourg_000000_023854 | GO | Open road with no presence positives; still no signal/right-of-way reasoning. |
| stuttgart_000095_000019 | SLOW | Cars overlap road region; unmatched rider does not supply cyclist presence. |
| bochum_000000_001097 | GO | Visible red overhead signal is omitted from concept set; GO is only policy agreement. |

## Handoff

Verification: all 116 tests passed, including all 32 binary concept combinations, STOP precedence, SLOW reasons, vehicle invariance, invalid inputs and stale metadata. Two runs produced identical action CSV, summary, HTML and 29 copied images. The 490 unique output IDs exactly match eligible IDs in manifest order; ten excluded IDs are absent. Existing metadata and concept rules were unchanged, and all 500 original RGB/mask pairs still match their manifest SHA-256 values.

Keep actions-v1.0 unchanged. The next chunk must assess sequence-grouped split feasibility while excluding inspected sequences from an unseen test set. STOP has only 16 examples overall and official val has only one; do not treat an arbitrary split as statistically adequate. Decide whether dataset expansion is necessary after that assessment. No splits or training were performed here.
