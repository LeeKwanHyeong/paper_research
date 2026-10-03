"""Independent equations, causal masking and state identity checks."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from models.TPPs.CountAwareTitanMLPCandidates import ARMS, MODES, CandidateCorrection
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def build(arm,device='cpu'):
    model,_=build_count_aware_model(arm,hidden_dim=64,train_log_mean=1.2,train_log_std=.8,max_seq_len=256,
        quantity_variant='count_only_log_regression',time_head_mode='heteroscedastic_lognormal_duration',
        time_scale=7.,time_initial_location=.2,time_initial_scale=.8,
        time_observation_contract={'mode':'positive_integer_round_clamp_v1','top_code':None,'unit':'week'})
    return model.to(device)


def check(device='cpu'):
    result={}
    for mode in MODES.values():
        torch.manual_seed(42);m=CandidateCorrection(64,mode).to(device)
        assert sum(p.numel() for p in m.parameters())==6144
        h=torch.randn(2,18,64,device=device,requires_grad=True)
        valid=torch.ones(2,18,device=device,dtype=torch.bool);valid[0,[2,7]]=False
        write=valid.clone();write[1,6]=False
        assert torch.count_nonzero(m(h,valid,memory_write_mask=write))==0
        for p in m.output_projections:torch.nn.init.normal_(p.weight)
        reference=torch.zeros_like(h)
        for b in range(2):
            prior=[]
            for t in range(18):
                if not bool(valid[b,t]):continue
                if not bool(write[b,t]):prior=[];continue
                for j,lag in enumerate((1,2,4,8,16,32,64,128)):
                    if len(prior)<lag:continue
                    weight=m.input_projections[j].weight
                    current=F.linear(h[b,t],weight[:,:64])
                    context=h[b,prior[-1]] if mode=='cross_product' else h[b,prior[-4:]].mean(0)
                    previous=F.linear(context,weight[:,64:])
                    f=F.gelu(current)*previous if mode=='cross_product' else F.gelu(current+previous)
                    reference[b,t]=reference[b,t]+m.output_projections[j](f)/8
                prior.append(t)
        actual=m(h,valid,memory_write_mask=write)
        torch.testing.assert_close(actual,reference,atol=3e-6,rtol=4e-5)
        ga=torch.autograd.grad(actual.sum(),h,retain_graph=True)[0]
        gr=torch.autograd.grad(reference.sum(),h)[0]
        torch.testing.assert_close(ga,gr,atol=5e-6,rtol=4e-5)
        changed=h.detach().clone();changed[:,11:]=1000
        torch.testing.assert_close(actual[:,:11],m(changed,valid,memory_write_mask=write)[:,:11],rtol=0,atol=0)
        result[mode+'_equations_gradients_padding_reset_causality']='passed'
    for arm in ARMS:
        torch.manual_seed(42);m=build(arm,device);state=deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        torch.manual_seed(42);baseline=build('titantpp_history_mlp',device)
        assert torch.equal(rng,torch.get_rng_state())
        for k,v in baseline.state_dict().items():assert torch.equal(v,state[k]),k
        assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in baseline.parameters())
        dt=torch.randint(1,8,(3,12),device=device).float();q=torch.randint(1,20,dt.shape,device=device).float();mask=torch.ones_like(dt,dtype=torch.bool)
        m.eval();baseline.eval()
        a=target_outputs(m,dt,mask,q,lambda_log_qty=1.)
        torch.testing.assert_close(a['pred_qty'],target_outputs(baseline,dt,mask,q,lambda_log_qty=1.)['pred_qty'],rtol=0,atol=0)
        # Test leakage again after nonzero corrections, not only at zero initialization.
        for p in m.multilag_detail.output_projections:torch.nn.init.normal_(p.weight,std=.1)
        a=target_outputs(m,dt,mask,q,lambda_log_qty=1.)
        changed=q.clone();changed[:,-1]=4000
        torch.testing.assert_close(a['pred_qty'],target_outputs(m,dt,mask,changed,lambda_log_qty=1.)['pred_qty'],rtol=0,atol=0)
        changed=dt.clone();changed[:,-1]=90
        torch.testing.assert_close(a['pred_qty'],target_outputs(m,changed,mask,q,lambda_log_qty=1.)['pred_qty'],rtol=0,atol=0)
        for left in (True,False):
            def pad(x,z):return torch.cat((z,x),1) if left else torch.cat((x,z),1)
            out=target_outputs(m,pad(dt,torch.full_like(dt[:,:3],float('nan'))),pad(mask,torch.zeros_like(mask[:,:3])),pad(q,torch.full_like(q[:,:3],float('nan'))),lambda_log_qty=1.)
            torch.testing.assert_close(a['pred_qty'],out['pred_qty'],rtol=3e-5,atol=3e-6)
        restored=build(arm,device);restored.load_state_dict(m.state_dict(),strict=True)
        for other in ARMS:
            if other==arm:continue
            try:build(other,device).load_state_dict(m.state_dict(),strict=True)
            except ValueError:pass
            else:raise AssertionError('Candidate relabelling accepted')
        result[arm]={'status':'passed','parameters':sum(p.numel() for p in m.parameters()),'base_initialization_identical':True}
    return {'status':'passed','checks':result,'device':device,'held_out_evaluated':False}
