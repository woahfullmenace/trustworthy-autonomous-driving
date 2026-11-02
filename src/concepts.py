"""Versioned, deterministic heuristics on Cityscapes raw semantic label IDs.

Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
No RGB information, learned detector, or distance estimate is used here.
"""
from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
from scipy import ndimage

CONCEPTS = ("pedestrian_present", "vehicle_present", "cyclist_present", "road_clear", "obstacle_close")
VALID_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
VEHICLE_IDS = (26, 27, 28, 31, 32)
OBSTACLE_IDS = (24, 25, *VEHICLE_IDS, 33)
RAW_IDS = (*range(34), 255)


def at_least(count: int, total: int, fraction: float) -> bool:
    """Exact decimal threshold comparison, avoiding floating-point boundary drift."""
    return Decimal(int(count)) >= Decimal(str(fraction)) * int(total)


def validate_rules(rules: dict) -> None:
    if not isinstance(rules, dict) or not isinstance(rules.get("rule_version"), str) or not rules["rule_version"]:
        raise ValueError("A nonempty rule_version is required")
    try:
        fractions = [rules["presence_min_fraction"], rules["min_valid_region_fraction"],
                     rules["cyclist"]["component_min_fraction"], rules["cyclist"]["min_horizontal_overlap"],
                     rules["cyclist"]["max_vertical_gap_fraction"], rules["road_clear"]["min_road_fraction"],
                     rules["road_clear"]["max_obstacle_fraction_exclusive"], rules["obstacle_close"]["min_obstacle_fraction"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1 for v in fractions):
            raise ValueError("Rule fractions must be finite numbers in (0, 1]")
        for name in ("road", "near"):
            box = rules["regions"][name]
            if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError("Region coordinates must be four finite numbers")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("Regions must be nonempty normalized rectangles")
        if type(rules["audit"]["count"]) is not int or rules["audit"]["count"] < 1 or type(rules["audit"]["seed"]) is not int:
            raise ValueError("Audit count must be positive and seed must be an integer")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed concept rules: {exc}") from exc


def region_box(shape: tuple[int, int], normalized: list[float]) -> list[int]:
    h, w = shape
    x0, y0, x1, y1 = [math.floor(Decimal(str(v)) * scale) for v, scale in zip(normalized, (w, h, w, h))]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Image is too small for the configured regions")
    return [x0, y0, x1, y1]


def _components(mask: np.ndarray, label_id: int, minimum_fraction: float) -> list[dict]:
    labels, _ = ndimage.label(mask == label_id, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())
    components = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is not None and at_least(areas[index], mask.size, minimum_fraction):
            ys, xs = slices
            components.append({"id": index, "pixels": int(areas[index]),
                               "box": [xs.start, ys.start, xs.stop, ys.stop]})
    return components


def label_concepts(mask: np.ndarray, rules: dict) -> dict:
    """Return labels (0/1/None), validity, flags and measurements for one raw mask."""
    validate_rules(rules)
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a nonempty 2-D integer raw label-ID mask")
    if not np.isin(mask, RAW_IDS).all():
        raise ValueError("Mask contains unsupported Cityscapes raw label IDs")
    h, w = mask.shape
    histogram = np.bincount(mask.ravel().astype(np.int64), minlength=256)
    person_pixels = int(histogram[24])
    vehicle_pixels = int(histogram[list(VEHICLE_IDS)].sum())
    riders = _components(mask, 25, rules["cyclist"]["component_min_fraction"])
    bicycles = _components(mask, 33, rules["cyclist"]["component_min_fraction"])
    matches = []
    for rider in riders:
        rx0, ry0, rx1, ry1 = rider["box"]
        for bike in bicycles:
            bx0, by0, bx1, by1 = bike["box"]
            overlap = max(0, min(rx1, bx1) - max(rx0, bx0))
            narrower = min(rx1 - rx0, bx1 - bx0)
            gap = max(0, by0 - ry1, ry0 - by1)
            gap_ok = Decimal(gap) <= Decimal(str(rules["cyclist"]["max_vertical_gap_fraction"])) * h
            if at_least(overlap, narrower, rules["cyclist"]["min_horizontal_overlap"]) and ry0 + ry1 < by0 + by1 and gap_ok:
                matches.append({"rider_id": rider["id"], "bicycle_id": bike["id"],
                                "horizontal_overlap_fraction": overlap / narrower, "vertical_gap_pixels": gap})
    labels = {"pedestrian_present": int(at_least(person_pixels, mask.size, rules["presence_min_fraction"])),
              "vehicle_present": int(at_least(vehicle_pixels, mask.size, rules["presence_min_fraction"])),
              "cyclist_present": int(bool(matches))}
    regions = {}
    flags = []
    for name, concept in (("road", "road_clear"), ("near", "obstacle_close")):
        box = region_box(mask.shape, rules["regions"][name])
        x0, y0, x1, y1 = box
        roi = mask[y0:y1, x0:x1]
        valid_pixels = int(np.isin(roi, VALID_IDS).sum())
        road_pixels = int((roi == 7).sum())
        obstacle_pixels = int(np.isin(roi, OBSTACLE_IDS).sum())
        valid = at_least(valid_pixels, roi.size, rules["min_valid_region_fraction"])
        regions[name] = {"box": box, "total_pixels": int(roi.size), "valid_pixels": valid_pixels,
                         "valid_fraction": valid_pixels / roi.size, "road_pixels": road_pixels,
                         "obstacle_pixels": obstacle_pixels,
                         "road_fraction": road_pixels / valid_pixels if valid_pixels else None,
                         "obstacle_fraction": obstacle_pixels / valid_pixels if valid_pixels else None}
        if not valid:
            labels[concept] = None
            flags.append(f"{concept}:insufficient_valid_region")
        elif name == "road":
            labels[concept] = int(at_least(road_pixels, valid_pixels, rules["road_clear"]["min_road_fraction"])
                                  and not at_least(obstacle_pixels, valid_pixels, rules["road_clear"]["max_obstacle_fraction_exclusive"]))
        else:
            labels[concept] = int(at_least(obstacle_pixels, valid_pixels, rules["obstacle_close"]["min_obstacle_fraction"]))
    return {"rule_version": rules["rule_version"], "labels": labels,
            "validity": {name: int(labels[name] is not None) for name in CONCEPTS}, "audit_flags": flags,
            "class_pixel_counts": {str(i): int(histogram[i]) for i in RAW_IDS},
            "image_pixels": int(mask.size), "presence_min_pixels": math.ceil(Decimal(str(rules["presence_min_fraction"])) * mask.size),
            "person_pixels": person_pixels, "vehicle_pixels": vehicle_pixels,
            "riders": riders, "bicycles": bicycles, "cyclist_matches": matches,
            "matched_cyclist_pair_count": len(matches), "regions": regions}
