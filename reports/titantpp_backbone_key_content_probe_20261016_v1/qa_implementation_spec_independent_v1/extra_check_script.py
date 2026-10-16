import sys,math,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,'/Users/igwanhyeong/PycharmProjects/paper_research')
import torch
from paper.scripts import export_titantpp_key_content_probe as w
from paper.scripts import compare_titantpp_key_content_probe as c
from models.TPPs import CountAwareTitanCNNGRU as b
from models.TPPs import CountAwareTitanQKLocationCNN as branch
from paper.scripts import analyze_titantpp_value_difference_rows as pure
from unittest.mock import patch
from paper.tests.test_compare_titantpp_key_content_probe import ProbeComparison
checks=[]
g=torch.Generator().manual_seed(1731)
for rep in range(40):
 v=torch.randn(3,11,7,generator=g,dtype=torch.float64); k=torch.randn(3,7,generator=g,dtype=torch.float64)
 valid=torch.rand(3,11,generator=g)>.3; valid[:,0]=True
 writable=valid.clone();writable[:,6]=False
 _,s,a=b.observed_context(valid,memory_write_mask=writable)
 originals=[x.clone() for x in (v,k,s,a)]; old=branch.causal_depthwise_residual
 on=b.causal_depthwise_residual(v,k,s,a)
 with w.branch_scope(branch,0,torch) as audit: off=branch.causal_depthwise_residual(v,k,s,a)
 expected=torch.zeros_like(v)
 for lag in range(3):expected+=torch.where(a[...,lag,None],v,torch.zeros_like(v))*k[lag]
 torch.testing.assert_close(off,expected,rtol=1e-12,atol=1e-12)
 assert branch.causal_depthwise_residual is old and audit['restored']
 for x,y in zip((v,k,s,a),originals):assert torch.equal(x,y)
 # Valid but unavailable histories retain masks; one-observation output is exact original.
 one=a[...,0] & ~a[...,1]
 assert torch.equal(on[one],off[one])
 constant=torch.ones_like(v)*.4
 with w.branch_scope(branch,0,torch):dc=branch.causal_depthwise_residual(constant,k,s,a)
 assert torch.equal(dc,b.causal_depthwise_residual(constant,k,s,a))
checks.append({'random_boundary_and_state_cases':40,'status':'PASS'})
# Common-entity weighting must retain equal entity mass despite unequal row counts.
f=ProbeComparison();cohort={}
for seed in c.SEEDS:
 for split in c.SPLITS:
  rows=[];ons=[];hist=[]
  for i,(ent,pred) in enumerate([('a',7.),('a',7.),('a',7.),('b',11.)]):
   row,h=f.row(pred=pred,seed=seed,split=split);on,_=f.row(pred=6.,seed=seed,split=split,condition='on')
   for obj in (row,on,h):obj.update(entity_id=ent,seq=4+i)
   for obj in (row,on):obj['target_id']=f'{ent}:{i}'
   rows.append(row);ons.append(on);hist.append(h)
  with patch.dict(c.COUNTS,{split:4}):cohort[seed,split]=c.paired(rows,ons,hist,pure,seed,split)
summary=c.rename_roles(pure.summarize(cohort,(.2,1.)))
assert len(summary['per_seed'])==180 and len(summary['support'])==90 and len(summary['additive_closure'])==12
for row in summary['per_seed']:
 if row['family']=='whole':
  expect=3. if row['weighting']=='row' else 4.
  assert row['metrics']['off']['mae']==expect
  assert row['metrics']['on']['mae']==1.
  assert row['metrics']['difference']['mae']==expect-1.
assert all(x['passed'] for x in summary['additive_closure'])
checks.append({'independent_unequal_entity_weights_and_all_cells':True,'status':'PASS'})
# Numeric tolerance does not relax exact observed history or time-loss replay.
x,h=f.row(condition='on');x.update(stage='on_validation');y=copy.deepcopy(x)
assert w.validate_rowwise([x],[y],1)['status']=='PASS'
for field,value in [('observed_history_sha256','changed'),('time_nll',2.1),('condition','off'),('stage','off_validation')]:
 z={**y,field:value}
 try:w.validate_rowwise([x],[z],1)
 except ValueError:pass
 else:raise AssertionError(field)
checks.append({'replay_negative_cases':4,'status':'PASS'})
print(json.dumps({'status':'PASS','checks':checks,'torch_version':torch.__version__,'real_data_or_checkpoint_loaded':False}))
