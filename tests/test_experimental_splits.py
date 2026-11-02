import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('create_splits', Path(__file__).resolve().parents[1] / 'scripts/create_splits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def samples():
    return [dict(sample_id=f'{city}_{sequence:06d}_{frame:06d}',
                 sequence_id=f'{city}_{sequence:06d}', official_split=source)
            for city, sequence, size, source in [('a', 0, 3, 'train'), ('a', 1, 2, 'train'),
                                                  ('a', 2, 2, 'train'), ('b', 0, 3, 'val')]
            for frame in range(size)]


def test_sequences_are_disjoint_complete_and_order_independent():
    rows = samples()
    result = module.assign_splits(rows, validation_images=4)
    assert result == module.assign_splits(list(reversed(rows)), validation_images=4)
    assert {r['sample_id'] for r in result} == {r['sample_id'] for r in rows}
    for sequence in {r['sequence_id'] for r in result}:
        assert len({r['experimental_split'] for r in result if r['sequence_id'] == sequence}) == 1
    assert sum(r['experimental_split'] == 'val' for r in result) == 4
    assert all((r['experimental_split'] == 'test') == (r['official_split'] == 'val') for r in result)


def test_invalid_or_impossible_partitions_fail():
    rows = samples()
    with pytest.raises(ValueError, match='impossible'):
        module.assign_splits(rows, validation_images=1)
    with pytest.raises(ValueError, match='unique'):
        module.assign_splits(rows + [rows[0]], validation_images=4)
    rows[0]['sequence_id'] = 'wrong'
    with pytest.raises(ValueError, match='Sequence ID'):
        module.assign_splits(rows, validation_images=4)
