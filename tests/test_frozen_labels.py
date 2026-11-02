import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import generate_labels as module


def test_freeze_rejects_change_before_writing_any_output(tmp_path):
    existing = tmp_path / 'existing.csv'
    new = tmp_path / 'new.csv'
    existing.write_bytes(b'original')
    with pytest.raises(ValueError, match='overwrite'):
        module.write_unchanged({new: b'new', existing: b'changed'})
    assert not new.exists()
    assert existing.read_bytes() == b'original'
    module.write_unchanged({existing: b'original'})


def test_frozen_source_tampering_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'SOURCES', ['rule.json'])
    source = tmp_path / 'rule.json'
    source.write_text('original')
    lock = {'sha256': {'rule.json': module.digest(source)}}
    module.verify_sources(lock)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Frozen source changed'):
        module.verify_sources(lock)
