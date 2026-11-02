import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.action_training import (CONFIG, ActionDataset, ActionResNet, action_names, evaluate, load_checkpoint,
                                 read_action_records, validate_config)
from src.concept_training import PREPROCESSING, frozen_digest, make_optimizer, sha256, is_better
from src.concepts import CONCEPTS
from src.decision_tree import ACTIONS, action_metrics
from scripts.compare_action_models import (check_action_predictions, check_ids_targets, concept_sources, validate_action_run)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def example(tmp_path):
    p=tmp_path/'image.png'; Image.new('RGB',(32,16),(255,0,128)).save(p)
    row={'sample_id':'synthetic','image_path':'image.png','image_sha256':sha256(p.read_bytes()),'width':'32','height':'16','experimental_split':'train','action':'STOP',**{c:'0' for c in CONCEPTS},**{c+'_valid':'1' for c in CONCEPTS}}
    return tmp_path,row


def test_action_loader_mapping_shape_normalization(example):
    root,row=example
    for i,action in enumerate(ACTIONS):
        row['action']=action
        x,y,sid=ActionDataset(root,[row],'train')[0]
        assert x.shape==(3,224,448) and y.dtype==torch.long and y.item()==i and sid=='synthetic'
        expected=(torch.tensor([1.,0.,128/255])-torch.tensor(PREPROCESSING['mean']))/torch.tensor(PREPROCESSING['std'])
        torch.testing.assert_close(x[:,0,0],expected)


@pytest.mark.parametrize('problem',['test','split','invalid','duplicate','missing','hash'])
def test_invalid_data(example,problem):
    root,row=example
    if problem=='split': row['experimental_split']='test'
    if problem=='invalid': row['action']='REQUEST HUMAN CONTROL'
    if problem in ('test','missing'): (root/'image.png').unlink()
    if problem=='hash': row['image_sha256']='changed'
    with pytest.raises((ValueError,OSError)):
        ActionDataset(root,[row,row] if problem=='duplicate' else [row], 'test' if problem=='test' else 'train')


def test_action_model_updates_reload_and_groups(tmp_path):
    torch.set_num_threads(4); torch.manual_seed(42)
    model=ActionResNet().train(); optimizer=make_optimizer(model,CONFIG)
    assert [g['lr'] for g in optimizer.param_groups]==[.0001,.001]
    params=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(params)==len(set(params))
    assert set(params)=={id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.requires_grad==n.startswith(('network.layer4.','network.fc.')) for n,p in model.named_parameters())
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    before=frozen_digest(model); layer=model.network.layer4[0].conv1.weight.detach().clone(); head=model.network.fc.weight.detach().clone()
    x=torch.randn(3,3,64,128); logits=model(x)
    assert logits.shape==(3,3)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2])); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert frozen_digest(model)==before
    assert not torch.equal(layer,model.network.layer4[0].conv1.weight)
    assert not torch.equal(head,model.network.fc.weight)
    saved={'configuration':CONFIG,'class_order':list(ACTIONS),'preprocessing':PREPROCESSING,'model_state':model.state_dict()}
    p=tmp_path/'model.pt'; torch.save(saved,p)
    restored,_=load_checkpoint(p,'cpu')
    with torch.no_grad(): assert torch.equal(model.eval()(x),restored(x))
    saved['class_order']=['GO','SLOW','STOP']; torch.save(saved,p)
    with pytest.raises(ValueError): load_checkpoint(p,'cpu')


def test_action_evaluation_and_argmax():
    logits=torch.tensor([[0.,0.,0.],[1.,2.,3.],[-1.,3.,0.]])
    y=torch.tensor([0,2,1]); loader=DataLoader(TensorDataset(logits,y,torch.arange(3)),batch_size=2)
    loss,p,targets,ids=evaluate(torch.nn.Identity(),loader,'cpu')
    assert loss==pytest.approx(torch.nn.functional.cross_entropy(logits,y).item())
    assert torch.allclose(p.sum(1),torch.ones(3))
    assert action_names(p)==['STOP','GO','SLOW']
    assert torch.equal(y,targets)
    assert not is_better(.5,.5) and is_better(.4,.5)
    with pytest.raises(ValueError): action_names(np.ones((2,3)))
    m=action_metrics(['STOP','SLOW','GO'],['SLOW']*3)
    assert m['stop_recall']==0 and m['classes']['STOP']['precision'] is None


def test_metadata_and_concept_comparison_integrity(monkeypatch):
    from sklearn.tree import DecisionTreeClassifier
    records,hashes=read_action_records(ROOT)
    assert [len(records[s]) for s in ('train','validation')]==[342,74]
    monkeypatch.setattr(DecisionTreeClassifier,'fit',lambda *a,**k:pytest.fail('Must not refit'))
    predictions,_=concept_sources(ROOT,records,hashes)
    assert predictions['baseline']==['SLOW']*74
    changed=dict(hashes); changed['manifest']='stale'
    with pytest.raises(ValueError,match='Stale'): concept_sources(ROOT,records,changed)
    import scripts.compare_action_models as comparison
    monkeypatch.setattr(comparison,'snapshot',lambda dirs:{'modified':'hash'})
    with pytest.raises(ValueError,match='Modified'): concept_sources(ROOT,records,hashes)


@pytest.mark.parametrize('problem',['hash','ids','test','mapping','smoke'])
def test_action_run_rejection(tmp_path,problem):
    records,hashes=read_action_records(ROOT)
    meta={'mode':'pilot','configuration':CONFIG,'input_hashes':{**hashes,'training_config':sha256((ROOT/'configs/action_training.yaml').read_bytes())},'class_order':list(ACTIONS),'preprocessing':PREPROCESSING,'training_ids':[r['sample_id'] for r in records['train']],'evaluation_ids':[r['sample_id'] for r in records['validation']],'evaluation_split':'validation'}
    status={'status':'complete','mode':'pilot'}
    if problem=='hash': meta['input_hashes']['manifest']='changed'
    if problem=='ids': meta['evaluation_ids'].reverse()
    if problem=='test': meta['evaluation_split']='test'
    if problem=='mapping': meta['class_order'].reverse()
    if problem=='smoke': status['mode']='smoke'
    (tmp_path/'metadata.json').write_text(json.dumps(meta)); (tmp_path/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError): validate_action_run(ROOT,tmp_path,records,hashes)


def test_prediction_table_mismatch(tmp_path):
    records=[{'sample_id':'a','action':'GO'}]
    rows=[{'sample_id':'a','target':'GO','prediction':'GO','STOP_probability':.1,'SLOW_probability':.2,'GO_probability':.7}]
    path=tmp_path/'predictions.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert check_action_predictions(path,records,np.array([[.1,.2,.7]]))==0
    with pytest.raises(ValueError): check_action_predictions(path,records,np.array([[.2,.1,.7]]))
    with pytest.raises(ValueError): check_ids_targets(rows,[{'sample_id':'b','action':'GO'}])
    with pytest.raises(ValueError): check_ids_targets(rows,[{'sample_id':'a','action':'STOP'}])


def test_configuration_and_cpu_rejection(monkeypatch):
    from scripts.train_actions import run
    validate_config(CONFIG)
    with pytest.raises(ValueError): validate_config({**CONFIG,'epochs':21})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no CPU fallback'): run(ROOT)


def test_modified_concept_comparison_table_rejected(monkeypatch):
    import scripts.compare_action_models as comparison
    records,hashes=read_action_records(ROOT)
    original=comparison.read_csv
    def changed(path):
        rows=original(path)
        rows[0]['baseline_pedestrian_present_probability']='0.123456'
        return rows
    monkeypatch.setattr(comparison,'read_csv',changed)
    with pytest.raises(ValueError,match='probabilities/targets'):
        concept_sources(ROOT,records,hashes)
