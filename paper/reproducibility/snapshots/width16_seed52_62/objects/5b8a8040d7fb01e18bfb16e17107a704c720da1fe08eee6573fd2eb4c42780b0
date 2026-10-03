"""Synthetic CPU/CUDA checks for the approved PAKDD validation extension."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTitanHistoryControls import ARMS as CONTROLS, MODES, HistoryControl
from models.TPPs.DeepRenewalEvent import ARM as DRP, VARIANT as DRP_VARIANT, TIME_MODE as DRP_TIME, nb_log_survival, nb_log_mass
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
ARMS=(*CONTROLS,DRP)

def build(arm,device='cpu',top_code=30):
    native=arm==DRP
    m,meta=build_count_aware_model(arm,hidden_dim=64,train_log_mean=1.2,train_log_std=.8,max_seq_len=256,
       quantity_variant=DRP_VARIANT if native else 'count_only_log_regression',
       time_head_mode=DRP_TIME if native else 'heteroscedastic_lognormal_duration',
       time_scale=7.,time_initial_location=.2,time_initial_scale=.8,
       time_observation_contract={'mode':'positive_integer_round_clamp_v1','top_code':top_code,'unit':'day'})
    return m.to(device),meta

def manual_lstm(model,inputs):
    """Independent Gluon documented i/f/c/o equations, dropout disabled."""
    x=inputs
    for layer in range(2):
        h=x.new_zeros(x.shape[0],64);c=h.clone();out=[]
        w=model.rnn[layer].weight_ih_l0;u=model.rnn[layer].weight_hh_l0
        b=model.rnn[layer].bias_ih_l0;v=model.rnn[layer].bias_hh_l0
        for t in range(x.shape[1]):
            i,f,g,o=(F.linear(x[:,t],w,b)+F.linear(h,u,v)).chunk(4,-1)
            c=f.sigmoid()*c+i.sigmoid()*g.tanh();h=o.sigmoid()*c.tanh();out.append(h)
        x=torch.stack(out,1)
    return x

def check(device='cpu'):
    torch.set_num_threads(4);results={}
    # Independent loop equation, mixed padding/withheld, causal gradients.
    for mode in MODES.values():
        torch.manual_seed(42);m=HistoryControl(64,mode).to(device)
        assert sum(p.numel() for p in m.parameters())==6144
        h=torch.randn(2,14,64,device=device,requires_grad=True)
        mask=torch.ones(2,14,device=device,dtype=torch.bool);mask[0,2]=False
        write=mask.clone();write[1,6]=False
        assert torch.count_nonzero(m(h,mask,memory_write_mask=write))==0
        for p in m.output_projections:torch.nn.init.normal_(p.weight)
        ref=torch.zeros_like(h)
        for b in range(2):
            n=0;prev=None
            for t in range(14):
                if not bool(mask[b,t]):continue
                if not bool(write[b,t]):n=0;prev=None;continue
                n+=1
                for j,threshold in enumerate((1,2,4,8,16,32,64,128)):
                    if n <= (threshold if mode=='current_only' else 1):continue
                    feature=h[b,t] if mode=='current_only' else torch.cat((h[b,t],h[b,prev]))
                    ref[b,t]=ref[b,t]+m.output_projections[j](F.gelu(m.input_projections[j](feature)))/8
                prev=t
        actual=m(h,mask,memory_write_mask=write)
        torch.testing.assert_close(actual,ref,atol=2e-6,rtol=3e-5)
        ga=torch.autograd.grad(actual.sum(),h,retain_graph=True)[0];gr=torch.autograd.grad(ref.sum(),h)[0]
        torch.testing.assert_close(ga,gr,atol=3e-6,rtol=3e-5)
        results[mode+'_formula_gradient_mask']='passed'
    for arm in ARMS:
        torch.manual_seed(42);m,meta=build(arm,device);m.eval()
        dt=torch.randint(1,31,(3,12),device=device).float();q=torch.randint(1,15,dt.shape,device=device).float()
        dt[:,-1]=30;mask=torch.ones_like(dt,dtype=torch.bool)
        a=target_outputs(m,dt,mask,q,lambda_log_qty=1.)
        changed=q.clone();changed[:,-1]=4000
        b=target_outputs(m,dt,mask,changed,lambda_log_qty=1.)
        torch.testing.assert_close(a['pred_qty'],b['pred_qty'],rtol=0,atol=0)
        changed_dt=dt.clone();changed_dt[:,-1]=1
        b=target_outputs(m,changed_dt,mask,q,lambda_log_qty=1.)
        torch.testing.assert_close(a['pred_qty'],b['pred_qty'],rtol=0,atol=0)
        # The loss itself changes with target; predictions must not.
        for left in (True,False):
            def pad(x,z):return torch.cat((z,x),1) if left else torch.cat((x,z),1)
            b=target_outputs(m,pad(dt,torch.full_like(dt[:,:3],float('nan'))),
               pad(mask,torch.zeros_like(mask[:,:3])),pad(q,torch.full_like(q[:,:3],float('nan'))),lambda_log_qty=1.)
            torch.testing.assert_close(a['pred_qty'],b['pred_qty'],rtol=2e-6,atol=2e-6)
        enc=m.encode_task_states(dt,q,mask)[0];dd=dt.clone();qq=q.clone();dd[:,7:]=1;qq[:,7:]=100
        torch.testing.assert_close(enc[:,:7],m.encode_task_states(dd,qq,mask)[0][:,:7],rtol=2e-5,atol=2e-6)
        if arm in CONTROLS:
            torch.manual_seed(42);base,_=build('titantpp_history_mlp',device);base.eval()
            for k,v in base.state_dict().items():
                if not k.startswith('multilag_detail.'):
                    assert torch.equal(v,m.state_dict()[k]),k
            if arm==CONTROLS[1]:
                for k,v in base.multilag_detail.state_dict().items():assert torch.equal(v,m.multilag_detail.state_dict()[k])
            torch.testing.assert_close(a['pred_qty'],target_outputs(base,dt,mask,q,lambda_log_qty=1.)['pred_qty'],rtol=0,atol=0)
        else:
            assert all(torch.count_nonzero(p)==0 for n,p in m.rnn.named_parameters() if 'bias' in n)
            assert torch.equal(m.dense.bias,torch.full_like(m.dense.bias,5.))
            sequence=torch.cat((torch.zeros(3,1,2,device=device),torch.stack((dt,q),-1)),1)
            torch.testing.assert_close(enc,manual_lstm(m,sequence)[:,1:],rtol=2e-5,atol=2e-6)
        m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        def step(model,optimizer):
            optimizer.zero_grad(set_to_none=True);loss=target_outputs(model,dt,mask,q,lambda_log_qty=1.)['joint_loss'].mean()
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm);optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            return loss.detach()
        for _ in range(3):step(m,opt)
        saved=deepcopy(m.state_dict());savedopt=deepcopy(opt.state_dict());rng=torch.get_rng_state()
        crng=torch.cuda.get_rng_state() if device.startswith('cuda') else None
        expected=step(m,opt);nextstate=deepcopy(m.state_dict());nextopt=deepcopy(opt.state_dict())
        restored,_=build(arm,device);restored.load_state_dict(saved,strict=True);ropt=torch.optim.AdamW(restored.parameters(),lr=.001);ropt.load_state_dict(savedopt)
        torch.set_rng_state(rng)
        if crng is not None:torch.cuda.set_rng_state(crng)
        torch.testing.assert_close(step(restored,ropt),expected,rtol=0,atol=0)
        for k,v in nextstate.items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
        for k,v in nextopt['state'].items():
            for key,value in v.items():torch.testing.assert_close(value,ropt.state_dict()['state'][k][key],rtol=0,atol=0)
        results[arm]={'causality_padding_target_exclusion':'passed','initialization':'passed',
                     'model_optimizer_rng_next_loss_roundtrip':'passed','parameters':sum(p.numel() for p in m.parameters())}
    # Moderate grid: trusted torch NB mass; tail probability partition sums to 1.
    mu=torch.logspace(-2,3,20,device=device,dtype=torch.float64,requires_grad=True)
    alpha=torch.tensor(.7,device=device,dtype=torch.float64,requires_grad=True)
    k=torch.arange(29,device=device,dtype=torch.float64)
    pm=nb_log_mass(k,mu[:,None],alpha)
    oracle=torch.distributions.NegativeBinomial(total_count=1/alpha,logits=torch.log(alpha*mu[:,None])).log_prob(k)
    torch.testing.assert_close(pm,oracle,rtol=1e-9,atol=1e-9)
    sf=nb_log_survival(29,mu,alpha)
    torch.testing.assert_close(pm.exp().sum(-1)+sf.exp(),torch.ones_like(mu),rtol=1e-10,atol=1e-10)
    sf.sum().backward();assert torch.isfinite(mu.grad).all() and torch.isfinite(alpha.grad)
    results['NB_native_mass_topcode_normalization_gradients']='passed'
    return {'status':'passed','device':device,'checks':results,'synthetic_only':True,'MXNet_runtime_executed':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
    r=check(a.device);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
