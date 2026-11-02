import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.concept_training import FrozenConceptResNet, frozen_digest, make_optimizer, load_checkpoint, PREPROCESSING, read_records
from src.concepts import CONCEPTS
from scripts.train_concepts import validate_config
from scripts.compare_concept_runs import validate_run, prediction_table

ROOT = Path(__file__).resolve().parents[1]


def test_initialization_update_groups_and_reload(tmp_path):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    baseline = FrozenConceptResNet()
    torch.manual_seed(42)
    model = FrozenConceptResNet(strategy='layer4_and_head').train()
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in baseline.state_dict().items())
    cfg = yaml.safe_load((ROOT/'configs/concept_finetuning.yaml').read_bytes())
    validate_config(cfg)
    opt = make_optimizer(model,cfg)
    assert [g['lr'] for g in opt.param_groups] == [.0001,.001]
    actual = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.requires_grad == n.startswith(('network.layer4.', 'network.fc.')) for n,p in model.named_parameters())
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    before = frozen_digest(model)
    layer = model.network.layer4[0].conv1.weight.detach().clone()
    head = model.network.fc.weight.detach().clone()
    x,y = torch.randn(2,3,64,128), torch.randint(0,2,(2,5)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        assert (p.grad is not None and torch.isfinite(p.grad).all()) if p.requires_grad else p.grad is None
    opt.step()
    assert frozen_digest(model) == before
    assert not torch.equal(layer,model.network.layer4[0].conv1.weight)
    assert not torch.equal(head,model.network.fc.weight)
    saved = {'configuration':cfg,'model_state':model.state_dict(),'concept_order':list(CONCEPTS),'preprocessing':PREPROCESSING}
    path = tmp_path/'best.pt'; torch.save(saved,path)
    restored,_ = load_checkpoint(path,'cpu')
    with torch.no_grad():
        assert torch.equal(model.eval()(x),restored(x))


@pytest.mark.parametrize('change',['hash','order','split','smoke','config'])
def test_comparison_rejects_incompatible_runs(tmp_path,change):
    records, hashes = read_records(ROOT)
    source = ROOT/'outputs/concept_training/pilot_v1'
    metadata = json.loads((source/'metadata.json').read_bytes())
    status = json.loads((source/'status.json').read_bytes())
    if change == 'hash': metadata['input_hashes']['manifest'] = 'stale'
    if change == 'order': metadata['concept_order'].reverse()
    if change == 'split': metadata['evaluation_split'] = 'test'
    if change == 'smoke': status['mode'] = 'smoke'
    if change == 'config': metadata['configuration']['learning_rate'] = .2
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    (tmp_path/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError): validate_run(ROOT,tmp_path,records,hashes)


def test_original_run_compatible_and_bad_prediction_ids(tmp_path):
    records, hashes = read_records(ROOT)
    source = ROOT/'outputs/concept_training/pilot_v1'
    validate_run(ROOT,source,records,hashes)
    p = prediction_table(source/'validation_predictions.csv',records['validation'])
    assert p.shape == (74,5)
    with pytest.raises(ValueError): prediction_table(source/'validation_predictions.csv',records['validation'][::-1])


def test_training_configs_and_unknown_strategy():
    cfg = yaml.safe_load((ROOT/'configs/concept_training.yaml').read_bytes())
    validate_config(cfg)
    model = FrozenConceptResNet()
    optimizer = make_optimizer(model,cfg)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == .001
    with pytest.raises(ValueError): FrozenConceptResNet(strategy='all')


def test_pretrained_initialization_matches_recorded_run():
    from torchvision.models import ResNet18_Weights
    from src.concept_training import sha256
    torch.hub.set_dir(str(ROOT/'outputs/torch_cache'))
    torch.manual_seed(42)
    baseline = FrozenConceptResNet(weights=ResNet18_Weights.IMAGENET1K_V1)
    torch.manual_seed(42)
    candidate = FrozenConceptResNet(weights=ResNet18_Weights.IMAGENET1K_V1, strategy='layer4_and_head')
    assert all(torch.equal(v, candidate.state_dict()[k]) for k,v in baseline.state_dict().items())
    metadata = json.loads((ROOT/'outputs/concept_training/finetune_v2_smoke/metadata.json').read_bytes())
    assert sha256(baseline.network.fc.weight.detach().numpy().tobytes()) == metadata['initial_head_sha256']


def test_comparison_reports_use_one_tree_without_fit(monkeypatch):
    from sklearn.tree import DecisionTreeClassifier
    from src.decision_tree import load_tree_run, action_metrics, decision_path
    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison must not fit a tree')
    monkeypatch.setattr(DecisionTreeClassifier,'fit',forbidden)
    tree, context = load_tree_run(ROOT, ROOT/'outputs/decision_tree/pilot_v1_complete')
    records, actions = context[:2]
    p = prediction_table(ROOT/'outputs/concept_training/pilot_v1/validation_predictions.csv',records['validation'])
    bits = (p >= .5).astype(int)
    prediction = tree.predict(bits)
    metrics = action_metrics([actions[r['sample_id']] for r in records['validation']],prediction)
    assert metrics['confusion_matrix'] == [[0,3,0],[0,64,0],[0,7,0]]
    assert all(decision_path(tree,x)['action'] == a for x,a in zip(bits,prediction))
