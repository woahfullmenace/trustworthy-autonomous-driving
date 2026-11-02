import sys
from pathlib import Path
import itertools
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_actions import action_target, CONCEPTS


def test_policy_invariants_over_all_binary_inputs():
    for bits in itertools.product((0, 1), repeat=5):
        row = dict(zip(CONCEPTS, bits))
        action, _, eligible = action_target(row)
        assert eligible
        if row['obstacle_close']:
            assert action == 'STOP'
        if action == 'GO':
            assert row['road_clear'] and not row['obstacle_close']
            assert not row['pedestrian_present'] and not row['cyclist_present']
        row['vehicle_present'] = 1-row['vehicle_present']
        assert action_target(row)[0] == action  # parked vehicles alone cannot determine action


def test_unknown_is_not_a_driving_or_abstention_target():
    row = dict.fromkeys(CONCEPTS, 0)
    row['road_clear'] = float('nan')
    assert action_target(row) == ('', 'unknown_concept', False)
    row['pedestrian_present'] = 2
    with pytest.raises(ValueError):
        action_target(row)
