import sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.TPPs.CountAwareTitanPostAttentionFFN import ObservedFFNResidual,ARMS,CURRENT
from models.TPPs.CountAwareTitanCNNGRU import observed_context
torch.set_num_threads(1); torch.manual_seed(718)
v=torch.tensor([[1,0,1,1,1,1,1],[0,1,1,0,1,1,1]],dtype=torch.bool);w=v.clone();w[0,3]=False;w[1,4]=False
observed,source,available=observed_context(v,memory_write_mask=w)
si=torch.zeros_like(source);av=torch.zeros_like(available)
for b in range(2):
    hist=[]
    for t in range(7):
        if v[b,t] and not w[b,t]:hist=[]
        if v[b,t] and w[b,t]:
            hist.append(t)
            for l,j in enumerate(hist[::-1][:3]):si[b,t,l]=j;av[b,t,l]=True
assert torch.equal(source,si) and torch.equal(available,av)
u=torch.randn(2,7,128,dtype=torch.double);results={}
for arm in ARMS:
    m=ObservedFFNResidual(arm=arm).double()
    with torch.no_grad():m.a.uniform_(.5,1.5);m.b.uniform_(-1,1);m.c.uniform_(-.2,.2)
    actual=m(u,source,available);expected=torch.zeros_like(u)
    for b in range(2):
        for t in range(7):
            for l in range(3):
                if av[b,t,l]:
                    j=t if arm==CURRENT else int(si[b,t,l]);z=m.a[l]*u[b,j]+m.b[l]
                    expected[b,t]+=m.c[l]*(.5*z*(1+torch.erf(z/math.sqrt(2))))/3
    assert torch.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    bad=u.clone();bad[~observed]=float('nan');assert torch.equal(actual,m(bad,source,available))
    bad=u.clone();bad[0,0,0]=float('nan')
    try:m(bad,source,available);raise AssertionError('observed NaN accepted')
    except ValueError:pass
    future=u.clone();future[:,6]+=100;assert torch.equal(actual[:,:6],m(future,source,available)[:,:6])
    assert m(u[:,:0],source[:,:0],available[:,:0]).shape==(2,0,128)
    # Separate fabricated two-update test; no scientific data or baseline optimizer.
    torch.manual_seed(151);x=torch.randn(2,9,128,dtype=torch.double);target=torch.randn_like(x)
    _,s,a=observed_context(torch.ones(2,9,dtype=torch.bool));fresh=ObservedFFNResidual(arm=arm).double();opt=torch.optim.AdamW(fresh.parameters(),lr=.001,weight_decay=.01)
    gradients=[]
    for step in range(2):
        opt.zero_grad();((fresh(x,s,a)-target)**2).sum().backward()
        gradients.append({name:int((p.grad!=0).sum()) for name,p in fresh.named_parameters()});opt.step()
    assert gradients==[{'a':0,'b':0,'c':384},{'a':384,'b':384,'c':384}]
    results[arm]={'max_formula_error':float((actual-expected).abs().max().detach()),'gradient_counts':gradients}
x=torch.linspace(-6,6,121,dtype=torch.double)
p=torch.tensor([1.,1.,1.,-1.,0.,1.,.2,.3,.4],dtype=torch.double,requires_grad=True)
def f(p):
    z=x[:,None]*p[:3]+p[3:6]
    return (p[6:]*.5*z*(1+torch.erf(z/math.sqrt(2)))).sum(1)/3
jac=torch.autograd.functional.jacobian(f,p);sv=torch.linalg.svdvals(jac);rank=int(torch.linalg.matrix_rank(jac));assert rank==9
print(json.dumps({'status':'PASS_independent_formula_and_gradient_oracle','torch':torch.__version__,'arms':results,'current_scalar_jacobian_rank':rank,'singular_values':sv.tolist(),'rank_scope':'one chosen generic point; not global identifiability','synthetic_optimizer_updates':4,'real_data_loaded':False,'historical_cnn_inference':0,'remote_calls':0}))
