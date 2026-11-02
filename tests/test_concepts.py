from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.concepts import label_concepts, region_box


@pytest.fixture
def rules():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/concepts.yaml").read_text())


@pytest.mark.parametrize("pixels,expected", [(99, 0), (100, 1), (101, 1)])
@pytest.mark.parametrize("label,concept", [(24, "pedestrian_present"), (26, "vehicle_present")])
def test_presence_boundary(rules, pixels, expected, label, concept):
    mask = np.full((1000, 1000), 7, dtype=np.uint8)
    mask[0, :pixels] = label
    assert label_concepts(mask, rules)["labels"][concept] == expected


def test_vehicle_classes_combine_but_bicycle_is_separate(rules):
    mask = np.full((1000, 1000), 7, dtype=np.uint8)
    mask[0, :50], mask[1, :50] = 26, 28
    assert label_concepts(mask, rules)["labels"]["vehicle_present"] == 1
    mask[:2, :50] = 33
    labels = label_concepts(mask, rules)["labels"]
    assert labels["vehicle_present"] == labels["cyclist_present"] == 0


@pytest.mark.parametrize("case,expected", [("parked",0),("nearby",1),("separate",0),("below",0),("tiny",0)])
def test_cyclist_geometry(rules, case, expected):
    mask = np.full((200, 200), 7, dtype=np.uint8)
    mask[90:105, 70:100] = 33
    if case == "nearby": mask[65:85, 75:85] = 25
    if case == "separate": mask[65:85, 140:150] = 25
    if case == "below": mask[110:130, 75:85] = 25
    if case == "tiny": mask[85, 75:78] = 25
    assert label_concepts(mask, rules)["labels"]["cyclist_present"] == expected


def test_cyclist_inclusive_overlap_and_gap(rules):
    mask = np.full((200, 200), 7, dtype=np.uint8)
    mask[70:84, 60:80] = 25
    mask[90:100, 75:95] = 33  # 25% overlap, 6px = 3% of height gap
    assert label_concepts(mask, rules)["labels"]["cyclist_present"] == 1
    mask[90:100, 75:95] = 7
    mask[91:101, 75:95] = 33
    assert label_concepts(mask, rules)["labels"]["cyclist_present"] == 0


def test_eight_connected_components(rules):
    mask = np.full((200, 200), 7, dtype=np.uint8)
    for k in range(4): mask[80+k, 80+k] = 25
    mask[85:90, 80:88] = 33
    assert label_concepts(mask, rules)["labels"]["cyclist_present"] == 1


def test_road_and_obstacles_inside_outside(rules):
    mask = np.full((100, 100), 7, dtype=np.uint8)
    assert label_concepts(mask, rules)["labels"]["road_clear"] == 1
    mask[70:85, 0:15] = 26
    result = label_concepts(mask, rules)["labels"]
    assert result["road_clear"] == 1 and result["obstacle_close"] == 0
    mask[70:85, 40:55] = 26
    result = label_concepts(mask, rules)["labels"]
    assert result["road_clear"] == 0 and result["obstacle_close"] == 1


@pytest.mark.parametrize("pixels,expected", [(9,1),(10,0),(11,0)])
def test_road_obstacle_strict_boundary(rules, pixels, expected):
    rules = deepcopy(rules)
    rules["regions"]["road"] = [0,0,1,1]
    mask = np.full((20,50), 7, dtype=np.uint8)
    mask[0,:pixels] = 26
    assert label_concepts(mask,rules)["labels"]["road_clear"] == expected


@pytest.mark.parametrize("pixels,expected", [(49,0),(50,1),(51,1)])
def test_close_obstacle_inclusive_boundary(rules, pixels, expected):
    rules = deepcopy(rules)
    rules["regions"]["near"] = [0,0,1,1]
    mask = np.full((10,100), 7, dtype=np.uint8)
    mask[0,:pixels] = 26
    assert label_concepts(mask,rules)["labels"]["obstacle_close"] == expected


@pytest.mark.parametrize("road_pixels,expected", [(599,0),(600,1),(601,1)])
def test_road_fraction_boundary(rules,road_pixels,expected):
    rules=deepcopy(rules)
    rules["regions"]["road"]=[0,0,1,1]
    mask=np.full((10,100),8,dtype=np.uint8)
    mask.ravel()[:road_pixels]=7
    assert label_concepts(mask,rules)["labels"]["road_clear"]==expected


@pytest.mark.parametrize("valid_pixels,expected", [(499,None),(500,1),(501,1)])
def test_validity_and_ignored_denominator(rules,valid_pixels,expected):
    rules=deepcopy(rules)
    rules["regions"]={"road":[0,0,1,1],"near":[0,0,1,1]}
    mask=np.full((10,100),1,dtype=np.uint8)
    mask.ravel()[:valid_pixels]=7
    result=label_concepts(mask,rules)
    assert result["labels"]["road_clear"]==expected
    assert result["regions"]["road"]["road_fraction"]==1
    assert result["validity"]["road_clear"]==int(expected is not None)
    mask[:]=255
    result=label_concepts(mask,rules)
    assert result["labels"]["road_clear"] is None
    assert result["labels"]["obstacle_close"] is None
    assert result["regions"]["near"]["obstacle_fraction"] is None


def test_scale_equivalence_and_floor(rules):
    mask=np.full((100,100),7,dtype=np.uint8)
    mask[70:80,40:50]=26
    assert label_concepts(mask,rules)["labels"]==label_concepts(np.repeat(np.repeat(mask,2,axis=0),2,axis=1),rules)["labels"]
    assert region_box((101,203),rules["regions"]["road"])==[71,50,131,90]


@pytest.mark.parametrize("mask", [np.zeros((10,10,3),dtype=np.uint8),np.zeros((0,10),dtype=np.uint8),np.zeros((10,10)),np.full((10,10),34)])
def test_malformed_masks(rules,mask):
    with pytest.raises(ValueError): label_concepts(mask,rules)
