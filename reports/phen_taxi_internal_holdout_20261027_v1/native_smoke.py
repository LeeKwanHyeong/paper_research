"""Synthetic-only native model/optimizer/evaluation check; no real targets."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from paper.scripts import run_shared_history_mlp_control as old
from paper.scripts.count_aware_tpp_backbone import core, training
from paper.scripts.run_taxi_internal_holdout_diagnostic import preserved_rng

p=argparse.ArgumentParser()
p.add_argument('--prepared', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
assert not a.output.exists()
data=json.loads((a.prepared/'DATA.json').read_text())
assert data['statistics_source_scope']=='fit_only'
torch.manual_seed(42); np.random.seed(42); random.seed(42)
q=torch.tensor([[1.,2.,3.,4.,5.,6.],[5.,4.,3.,2.,1.,2.],
                [2.,2.,2.,2.,2.,2.],[1.,3.,1.,3.,1.,3.]],dtype=torch.float32)
dt=torch.tensor([[1.,1.,2.,1.,3.,2.],[1.,2.,1.,2.,1.,1.],
                 [1.,1.,1.,1.,1.,1.],[1.,2.,3.,2.,1.,3.]],dtype=torch.float32)
mask=torch.ones_like(q,dtype=torch.bool)
loader=[(torch.zeros_like(q,dtype=torch.long),dt,mask,None,q)]
with old._engine_hooks(data,'single_mlp'):
    model,_=training.build_model()
    model=model.to(a.device)
    optimizer=training.build_optimizer(model,lr=.001)
    telemetry=training.train_epoch_with_telemetry(model=model,loader=loader,optimizer=optimizer,
              device=a.device,lambda_log_qty=1.,grad_clip=1.,max_batches=None)
    generator=torch.Generator().manual_seed(42)
    before=(torch.get_rng_state().clone(),generator.get_state().clone())
    with preserved_rng(model,(generator,)):
        torch.rand(3);torch.rand(3,generator=generator)
        metrics=core.evaluate(model=model,loader=loader,quantity_contract={'boundaries':[],'strata':[]},
              device=a.device,lambda_log_qty=1.,max_batches=None,include_breakdowns=False)
    assert torch.equal(before[0],torch.get_rng_state())
    assert torch.equal(before[1],generator.get_state())
    assert model.training
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    for value in metrics.values():
        if isinstance(value,(float,int)):assert np.isfinite(value)
result={'status':'PASS','synthetic_events':4,'optimizer_steps':1,'real_data_training':False,
        'Test_access':False,'device':a.device,'torch_version':torch.__version__,
        'model_parameters':sum(v.numel() for v in model.parameters()),'rng_restore':True,
        'telemetry':telemetry,'evaluation':metrics}
with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':'PASS','device':a.device,'synthetic_events':4,'real_data_training':False}))
