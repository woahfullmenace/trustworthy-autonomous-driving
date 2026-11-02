import pytest

from scripts.label_concepts import read_manifest, read_verified, select_audit, summarize
from src.concepts import CONCEPTS


def test_audit_is_training_only_stable_and_covers_categories():
    records=[{"sample_id":f"sample_{i:03d}","official_split":"train" if i<40 else "val",
              "labels":{c:(i+j)%2 for j,c in enumerate(CONCEPTS)}} for i in range(50)]
    records[3]["labels"]["road_clear"]=None
    records[5]["labels"]["obstacle_close"]=None
    chosen=select_audit(records,30,42)
    assert chosen==select_audit(list(reversed(records)),30,42)
    assert len(chosen)==len({r["sample_id"] for r in chosen})==30
    assert all(r["official_split"]=="train" for r in chosen)
    for concept in CONCEPTS:
        for value in (1,0,None):
            first=next((r for r in records[:40] if r["labels"][concept]==value),None)
            if first is not None: assert first in chosen


def test_invalid_values_are_not_counted_as_negatives():
    record={"sample_id":"a","official_split":"train","labels":dict.fromkeys(CONCEPTS,None)}
    summary=summarize([record])["train"]
    assert summary["counts"]["road_clear"]=={"positive":0,"negative":0,"invalid":1}
    assert summary["cooccurrence"]["road_clear"]["obstacle_close"]=={"both_positive":0,"both_valid":0}


def test_corrupted_or_missing_source_fails(tmp_path):
    file=tmp_path/"mask.png"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash"): read_verified(file,"0"*64)
    with pytest.raises(FileNotFoundError): read_verified(tmp_path/"missing.png","0"*64)


@pytest.mark.parametrize("content,message", [
    ("sample_id\na\n", "required columns"),
    ("sample_id,official_split,image_path,mask_path,image_sha256,mask_sha256,width,height\na,train\n", "nonempty values"),
    ("sample_id,official_split,image_path,mask_path,image_sha256,mask_sha256,width,height\n", "nonempty, unique")])
def test_malformed_manifest_fails_clearly(tmp_path,content,message):
    path=tmp_path/"subset.csv"
    path.write_text(content,encoding="utf-8")
    with pytest.raises(ValueError,match=message): read_manifest(path)
