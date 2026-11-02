import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_concepts', ROOT / 'scripts/audit_concepts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CONFIG = json.loads((ROOT / 'configs/concepts.json').read_text())


def test_clear_road_hood_and_ignored_corridor():
    mask = np.full((100, 100), 7, dtype=np.uint8)
    mask[90:] = 1
    result = module.measure(mask, CONFIG)
    assert result['road_clear'] == 1 and result['obstacle_close'] == 0
    mask[55:85, 35:65] = 1
    result = module.measure(mask, CONFIG)
    assert result['corridor_usable'] == 0
    assert result['road_clear'] is None and result['obstacle_close'] is None


def test_presence_boundary_and_cyclist_requires_both_classes():
    mask = np.full((100, 100), 7, dtype=np.uint8)
    mask[0, 0] = 24  # exactly 0.0001 of the full image
    mask[0, 1] = 33
    assert module.measure(mask, CONFIG)['pedestrian_present'] == 1
    assert module.measure(mask, CONFIG)['cyclist_present'] == 0
    mask[0, 2] = 25
    assert module.measure(mask, CONFIG)['cyclist_present'] == 1


def test_side_vehicle_does_not_make_central_obstacle():
    mask = np.full((100, 100), 7, dtype=np.uint8)
    mask[55:85, :20] = 26
    result = module.measure(mask, CONFIG)
    assert result['vehicle_present'] == 1 and result['road_clear'] == 1
    assert result['obstacle_close'] == 0
    mask[55:65, 35:65] = 26
    result = module.measure(mask, CONFIG)
    assert result['obstacle_close'] == 1 and result['road_clear'] == 0


def test_only_training_metadata_reaches_audit_selection():
    frame = pd.DataFrame([dict(sample_id=str(i), sequence_id=str(i), experimental_split=s)
                          for i, s in enumerate(['test', 'train', 'val'])])
    selected = module.training_rows(frame)
    assert selected.sample_id.tolist() == ['1']
