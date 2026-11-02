import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.concepts import CONCEPTS
from src.interventions import (PRIORITY, POLICY, correct_concepts, evaluate_interventions, intervention_metrics,
                               load_inputs, manual_intervention, parse_overrides)

ROOT=Path(__file__).resolve().parents[1]


def test_priority_single_edit_and_no_op():
    p=np.zeros((2,5),dtype=int); y=np.ones((2,5),dtype=int); y[1]=p[1]
    out=correct_concepts(p,y,'one')
    assert out[0].tolist()==[0,0,0,0,1]
    assert np.array_equal(out[1],p[1]) and not p.any()
    for c in PRIORITY:
        j=CONCEPTS.index(c); y=np.zeros((1,5),dtype=int); y[0,j]=1
        assert np.array_equal(correct_concepts(p[:1],y,'one'),y)


def test_independent_trials_and_full_correction():
    p=np.array([[0,1,0,1,0]]); y=1-p
    for c in CONCEPTS:
        result=correct_concepts(p,y,c); j=CONCEPTS.index(c)
        assert np.sum(result!=p)==1 and result[0,j]==y[0,j]
    assert np.array_equal(p,[[0,1,0,1,0]])
    assert np.array_equal(correct_concepts(p,y,'all'),y)
    assert np.array_equal(correct_concepts(p,y,'original'),p)


def test_success_denominator_harm_and_undefined():
    changed=np.zeros((3,5),dtype=bool); changed[:2,0]=True
    m=intervention_metrics(['STOP','GO','SLOW'],['SLOW','GO','GO'],['STOP','SLOW','GO'],changed)
    assert m['initially_incorrect']==2 and m['fixed_errors']==1
    assert m['success_rate']==.5 and m['harmed_correct_actions']==1 and m['unresolved_original_errors']==1
    assert m['samples_corrected']==2 and m['initially_incorrect_samples_corrected']==1
    n=intervention_metrics(['SLOW'],['SLOW'],['GO'],np.ones((1,5),dtype=bool))
    assert n['success_rate'] is None and n['success_rate_undefined'] and n['harmed_correct_actions']==1


@pytest.mark.parametrize('overrides',[[],['unknown=1'],['road_clear=2'],['road_clear=true'],['road_clear=1=0'],['road_clear=1','road_clear=0']])
def test_invalid_overrides(overrides):
    with pytest.raises(ValueError): parse_overrides(overrides)


def test_saved_context_no_images_no_fitting_and_paths(monkeypatch):
    from sklearn.tree import DecisionTreeClassifier
    original=Path.open
    def guard(path,*args,**kwargs):
        assert not (path.is_relative_to(ROOT/'data') and path.suffix.lower() in ('.png','.jpg','.jpeg')), 'Dataset image loaded during interventions'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    monkeypatch.setattr(DecisionTreeClassifier,'fit',lambda *a,**k:pytest.fail('Tree refitted'))
    context=load_inputs(ROOT); before=context['predicted'].copy()
    results,rows=evaluate_interventions(context)
    assert len(context['ids'])==74 and results['original']['initially_incorrect']==11
    assert len(rows)==74*8
    assert results['all']['fixed_errors']==11
    assert results['vehicle_present']['fixed_errors']==0 and results['vehicle_present']['harmed_correct_actions']==0
    assert evaluate_interventions(context)==(results,rows)
    assert np.array_equal(context['predicted'],before)
    manual=manual_intervention(context,context['ids'][0],['pedestrian_present=0','road_clear=1'])
    assert manual['overridden_concepts']['pedestrian_present']==0
    assert manual['overridden_concepts']['road_clear']==1
    assert np.array_equal(context['predicted'],before)
    assert manual['after']['action']==context['tree'].predict([[manual['overridden_concepts'][c] for c in CONCEPTS]])[0]
    for sid in ('aachen_000002_000019','unknown'):
        with pytest.raises(ValueError,match='validation IDs'): manual_intervention(context,sid,['road_clear=0'])


@pytest.mark.parametrize('problem',['hash','ids','order','modified_model'])
def test_stale_source_rejected(monkeypatch,problem):
    import src.interventions as module
    if problem=='modified_model':
        monkeypatch.setattr(module,'snapshot',lambda dirs:{'changed':'hash'})
    else:
        original=Path.read_bytes
        target=ROOT/'outputs/concept_comparison/balanced_v3/metadata.json'
        def changed(path):
            raw=original(path)
            if path==target:
                meta=json.loads(raw)
                if problem=='hash': meta['input_hashes']['manifest']='changed'
                if problem=='ids': meta['validation_ids'].reverse()
                if problem=='order': meta['concept_order'].reverse()
                return json.dumps(meta).encode()
            return raw
        monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError): load_inputs(ROOT)


def test_invalid_binary_shapes():
    with pytest.raises(ValueError): correct_concepts([[0]*5],[[0]*5,[1]*5],'one')
    with pytest.raises(ValueError): correct_concepts([[.5]*5],[[0]*5],'one')
    with pytest.raises(ValueError): correct_concepts([[0]*5],[[0]*5],'other')


def test_deterministic_files_and_refuse_overwrite(tmp_path,monkeypatch):
    import scripts.evaluate_interventions as command
    context=load_inputs(ROOT)
    monkeypatch.setattr(command,'load_inputs',lambda root:context)
    a,_=command.run(tmp_path,Path('outputs/interventions/a'))
    b,_=command.run(tmp_path,Path('outputs/interventions/b'))
    assert {p.name:p.read_bytes() for p in a.iterdir()}=={p.name:p.read_bytes() for p in b.iterdir()}
    with pytest.raises(FileExistsError): command.run(tmp_path,Path('outputs/interventions/a'))
    with pytest.raises(ValueError): command.run(tmp_path,Path('outside'))
