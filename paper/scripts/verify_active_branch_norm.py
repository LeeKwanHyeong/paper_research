"""Small synthetic algorithm checks shared by CPU tests and native CUDA preflight."""
import torch
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCoreAblation import HistoryCorrection
from models.TPPs.CountAwareTitanActiveBranchNorm import ActiveBranchCorrection
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices


def check(device='cpu'):
    torch.manual_seed(42)
    m=ActiveBranchCorrection(8).to(device)
    h=torch.randn(2,131,8,device=device,requires_grad=True)
    valid=torch.ones(2,131,dtype=torch.bool,device=device);valid[1,3]=False
    observed=valid.clone();observed[1,7]=False
    assert torch.equal(m(h,valid,memory_write_mask=observed),torch.zeros_like(h))
    loss=(m(h,valid,memory_write_mask=observed)-1).square().sum();loss.backward()
    assert all(p.weight.grad.abs().sum()>0 for p in m.output_projections)
    assert all(p.weight.grad.abs().sum()==0 for p in m.input_projections)
    for p in m.output_projections:torch.nn.init.normal_(p.weight)
    m.zero_grad();h.grad=None
    source,available=lag_source_indices(valid,memory_write_mask=observed,mode='local')
    ref=torch.zeros_like(h)
    for b in range(2):
        for t in range(131):
            count=available[b,t].sum().item()
            for k in range(8):
                if available[b,t,k]:
                    x=torch.cat((h[b,t],h[b,source[b,t,k]]))
                    ref[b,t]=ref[b,t]+m.output_projections[k](F.gelu(m.input_projections[k](x)))/count
    out=m(h,valid,memory_write_mask=observed)
    torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)
    g=torch.autograd.grad(out.sum(),h,retain_graph=True)[0]
    gr=torch.autograd.grad(ref.sum(),h)[0]
    torch.testing.assert_close(g,gr,rtol=2e-5,atol=3e-6)
    counts=available.sum(-1)
    assert set((0,1,8)).issubset(set(counts.flatten().tolist()))
    assert torch.equal(out[counts==0],torch.zeros_like(out[counts==0]))
    old=HistoryCorrection(8,'mlp').to(device);old.load_state_dict(m.state_dict())
    torch.testing.assert_close(out[counts==8],old(h,valid,memory_write_mask=observed)[counts==8],rtol=0,atol=0)
    changed=h.detach().clone();changed[:,100:]+=100;changed[~observed]=float('nan')
    torch.testing.assert_close(m(changed,valid,memory_write_mask=observed)[:,:100],out[:,:100])
    out=m(h.detach(),valid,memory_write_mask=observed);out.square().sum().backward()
    assert all(p.weight.grad.abs().sum()>0 for p in m.input_projections)
    return {'explicit_formula_and_gradient':True,'zero_one_eight_branches':True,'padding_withheld_causality':True,
        'zero_initial_output':True,'staged_gradient':True,'all_eight_equals_baseline':True,'device':device}
