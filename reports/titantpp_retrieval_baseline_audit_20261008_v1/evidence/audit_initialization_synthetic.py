import sys, json, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
root = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
source = root / 'search_artifacts/titantpp_retrieval_test_20261008_v1/source'
sys.path.insert(0, str(source))
import torch
from paper.scripts import run_titantpp_contextual_retrieval as engine
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
c = json.loads((source.parent / 'execution_contract.json').read_text())
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
engine.install_hooks()
torch.manual_seed(42)
parent, _ = engine.cnn.build_model(c['data_spec'], 'titantpp_cnn_gru54')
rng_parent = torch.get_rng_state().clone()
torch.manual_seed(42)
mean, _ = engine.build_model(c['data_spec'], 'titantpp_retrieval_mean')
rng_mean = torch.get_rng_state().clone()
ps, ms = parent.state_dict(), mean.state_dict()
assert set(ms) - set(ps) == {'contextual_retrieval_identity'}
assert all(torch.equal(value, ms[name]) for name, value in ps.items())
assert torch.equal(rng_parent, rng_mean)
parent.eval().requires_grad_(False)
mean.eval().requires_grad_(False)
dt = torch.tensor([[0,1,2,3,1,7,2,3,1,5,1,2],[0,0,0,0,1,2,1,7,3,2,1,3]], dtype=torch.float32)
qty = torch.tensor([[0,4,0,8,100,20,2200,3,17,60,7,5],[0,0,0,0,0,17,4,30,1,2200,2,8]], dtype=torch.float32)
mask = torch.tensor([[True]*12,[False]*4+[True]*8])
with torch.no_grad():
    po = target_outputs(parent, dt, mask, qty, lambda_log_qty=1.)
    mo = target_outputs(mean, dt, mask, qty, lambda_log_qty=1.)
assert all(torch.equal(po[key], mo[key]) for key in po)
print(json.dumps({'parent': canonical_state_dict_sha256(ps), 'mean': canonical_state_dict_sha256(ms), 'projected_mean': canonical_state_dict_sha256({key:ms[key] for key in ps}), 'common_parameters_and_RNG_equal': True, 'synthetic_outputs_equal': True}))
