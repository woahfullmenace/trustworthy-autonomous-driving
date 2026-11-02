import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from src.balanced_loss import compute_balance, validate_balance, weight_tensors, balanced_bce
from src.concept_training import FrozenConceptResNet, frozen_digest, make_optimizer, read_records, evaluate, load_checkpoint, PREPROCESSING
from src.concepts import CONCEPTS
from scripts.compare_concept_runs import validate_run
from scripts.train_concepts import validate_config

ROOT=Path(__file__).resolve().parents[1]


def test_training_balance_counts_and_total_influence():
    records,_=read_records(ROOT)
    b=compute_balance(records['train'])
    assert b['positive_counts']==[271,325,107,209,10]
    assert b['negative_counts']==[71,17,235,133,332]
    np.testing.assert_allclose(np.array(b['positive_counts'])*b['positive_weights'],171)
    np.testing.assert_allclose(np.array(b['negative_counts'])*b['negative_weights'],171)
    assert b['positive_weights'][-1]==17.1
    validate_balance(b,records['train'])
    smoke=json.loads((ROOT/'outputs/concept_training/balanced_v3_smoke/metadata.json').read_bytes())
    assert len(smoke['training_ids'])==16 and smoke['balance']==b
    with pytest.raises(ValueError,match='training rows only'): compute_balance(records['validation'])


@pytest.mark.parametrize('problem',['missing_class','duplicate','invalid','validity'])
def test_bad_training_balance(problem):
    records,_=read_records(ROOT); rows=copy.deepcopy(records['train'])
    if problem=='missing_class':
        for r in rows: r['obstacle_close']='0'
    if problem=='duplicate': rows.append(rows[0])
    if problem=='invalid': rows[0][CONCEPTS[0]]=''
    if problem=='validity': rows[0][CONCEPTS[0]+'_valid']='0'
    with pytest.raises(ValueError): compute_balance(rows)


def test_weighted_loss_mean_gradients_and_batch_reduction():
    logits=torch.tensor([[0.,1.,-2.,3.,-1.],[2.,-1.,0.,1.,-2.],[0.,2.,-3.,1.,0.]],requires_grad=True)
    y=torch.tensor([[1.,0.,1.,0.,1.],[0.,1.,0.,1.,0.],[1.,1.,0.,0.,1.]])
    weights=(torch.tensor([2.,3.,4.,5.,6.]),torch.tensor([.5,.6,.7,.8,.9]))
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
    expected=(raw*(y*weights[0]+(1-y)*weights[1])).mean()
    loss=balanced_bce(logits,y,weights)
    assert torch.equal(loss,expected)
    loss.backward(); assert torch.isfinite(logits.grad).all()
    assert balanced_bce(logits,y,weights,'sum')/y.numel()==loss
    records,_=read_records(ROOT); balance=compute_balance(records['train'])
    loader=DataLoader(TensorDataset(logits.detach(),y,torch.arange(3)),batch_size=2)
    ordinary=evaluate(torch.nn.Identity(),loader,'cpu')
    weighted=evaluate(torch.nn.Identity(),loader,'cpu',balance=balance)
    assert ordinary[0]==weighted[0] and torch.equal(ordinary[1],weighted[1])
    assert weighted[4]==pytest.approx(balanced_bce(logits,y,weight_tensors(balance,'cpu')).item())


def test_balanced_model_updates_and_checkpoint(tmp_path):
    torch.set_num_threads(4); torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/concept_balanced.yaml').read_bytes()); validate_config(cfg)
    records,hashes=read_records(ROOT); balance=compute_balance(records['train'])
    model=FrozenConceptResNet(strategy='layer4_and_head').train()
    before=frozen_digest(model)
    head=model.network.fc.weight.detach().clone(); layer=model.network.layer4[0].conv1.weight.detach().clone()
    optimizer=make_optimizer(model,cfg)
    x=torch.randn(2,3,64,128); y=torch.tensor([[1.,0.,1.,0.,1.],[0.,1.,0.,1.,0.]])
    loss=balanced_bce(model(x),y,weight_tensors(balance,'cpu')); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert frozen_digest(model)==before
    assert not torch.equal(head,model.network.fc.weight) and not torch.equal(layer,model.network.layer4[0].conv1.weight)
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    saved={'configuration':cfg,'balance':balance,'concept_order':list(CONCEPTS),'preprocessing':PREPROCESSING,'model_state':model.state_dict()}
    path=tmp_path/'best.pt'; torch.save(saved,path)
    restored,_=load_checkpoint(path,'cpu')
    with torch.no_grad(): assert torch.equal(model.eval()(x),restored(x))
    saved['balance']['positive_weights'][0]=99; torch.save(saved,path)
    with pytest.raises(ValueError): load_checkpoint(path,'cpu')


@pytest.mark.parametrize('problem',['weights','source','ids'])
def test_stale_balance_run_rejected(tmp_path,problem):
    records,hashes=read_records(ROOT)
    meta=json.loads((ROOT/'outputs/concept_training/balanced_v3_smoke/metadata.json').read_bytes())
    meta.update(mode='pilot',training_ids=[r['sample_id'] for r in records['train']],evaluation_ids=[r['sample_id'] for r in records['validation']],evaluation_split='validation')
    if problem=='weights': meta['balance']['negative_weights'][0]=8
    if problem=='source': meta['balance']['training_rows_sha256']='stale'
    if problem=='ids': meta['evaluation_ids'].reverse()
    (tmp_path/'metadata.json').write_text(json.dumps(meta)); (tmp_path/'status.json').write_text(json.dumps({'status':'complete','mode':'pilot'}))
    with pytest.raises(ValueError): validate_run(ROOT,tmp_path,records,hashes)


def test_old_versions_still_validate():
    records,hashes=read_records(ROOT)
    for name in ('pilot_v1','finetune_v2'):
        validate_run(ROOT,ROOT/'outputs/concept_training'/name,records,hashes)
