"""Native shifted-NB Deep Renewal adapted to frozen next-event windows.

GluonTS reference revision 889a3df86a89a365880b4bc1488bcf4c039f265e.
Retains raw interval/size, causal LSTM, softplus means and global dispersions.
Padding removal and target-only supervision are explicit adapter choices.
"""
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state

ARM='deep_renewal_event_native_nb'
ROLE='native_deep_renewal_event_v1'
VARIANT='shifted_nb_quantity_nll'
TIME_MODE='shifted_nb_duration'

def nb_log_mass(k,mu,alpha):
    k,mu,alpha=k.double(),mu.double(),alpha.double()
    if ((k<0)|(k!=k.round())|~torch.isfinite(k)).any():raise ValueError('NB target outside integer support')
    if ((mu<=0)|(alpha<=0)|~torch.isfinite(mu)|~torch.isfinite(alpha)).any():raise ValueError('Invalid NB parameters')
    r=alpha.reciprocal();logratio=torch.log(alpha)+torch.log(mu);normalizer=F.softplus(logratio)
    return torch.lgamma(k+r)-torch.lgamma(k+1)-torch.lgamma(r)+k*(logratio-normalizer)-r*normalizer

def _beta_fraction(a,b,x):
    """Lentz continued fraction for incomplete beta, differentiable in b/x."""
    tiny=1e-100
    def nz(v):return torch.where(v.abs()<tiny,torch.where(v<0,-tiny,tiny),v)
    qab=a+b;qap=a+1;qam=a-1;c=torch.ones_like(x)
    d=1/nz(1-qab*x/qap);h=d
    for m in range(1,201):
        aa=m*(b-m)*x/((qam+2*m)*(a+2*m))
        d=1/nz(1+aa*d);c=nz(1+aa/c);h=h*d*c
        aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m))
        d=1/nz(1+aa*d);c=nz(1+aa/c);delta=d*c;h=h*delta
        if bool(((delta-1).abs()<2e-13).all()):return h
    raise FloatingPointError('NB log-survival continued fraction did not converge')

def nb_log_survival(threshold,mu,alpha):
    """log P(NB >= threshold); stable small tails without subtracting from 1.

    Integer finite CDF sum is used when its complement is well conditioned.
    The small-tail branch computes I_p(threshold,1/alpha) directly.
    """
    if not isinstance(threshold,int) or threshold<1:raise ValueError('Positive integer threshold required')
    mu,alpha=torch.broadcast_tensors(mu.double(),alpha.double())
    k=torch.arange(threshold,device=mu.device,dtype=torch.float64)
    lcdf=torch.logsumexp(nb_log_mass(k,mu[...,None],alpha[...,None]),-1)
    direct=lcdf>-1e-5
    # Never evaluate log(-expm1(0)) in an unused branch: its gradient is singular.
    answer=torch.empty_like(mu)
    if bool((~direct).any()):answer[~direct]=torch.log(-torch.expm1(lcdf[~direct]))
    if bool(direct.any()):
        r=1/alpha[direct];a=torch.full_like(r,float(threshold))
        l=torch.log(alpha[direct])+torch.log(mu[direct]);logp=-F.softplus(-l);log1p=-F.softplus(l)
        p=torch.exp(logp)
        logbeta=torch.lgamma(a)+torch.lgamma(r)-torch.lgamma(a+r)
        answer[direct]=a*logp+r*log1p-logbeta+torch.log(_beta_fraction(a,r,p))-torch.log(a)
    return answer

def metadata():
    return {'candidate_name':ARM,'backbone_contract_id':'deep_renewal_event_native_nb_v1',
      'model_role':ROLE,'d_model':64,'n_layers':2,'dropout':.1,'input':'raw_recorded_interval_quantity',
      'initialization':'Xavier_uniform_full_gate_matrices; all_LSTM_biases_zero; dense_bias5; alpha_bias2',
      'lstm_gate_order':'input_forget_candidate_output','dropout_implementation':'explicit_between_layers_checkpointed_torch_RNG','time_distribution':'1+NB(mu,alpha)',
      'quantity_distribution':'1+NB(mu,alpha)','quantity_point':'1+mu',
      'dispersion':'two_global_softplus_scalars','padding':'remove; single_leading_zero_token',
      'loss':'last_target_time_NLL_plus_quantity_NLL'}

def identity():
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(),sort_keys=True).encode()).digest()),dtype=torch.uint8)

class DeepRenewalEvent(nn.Module):
    def __init__(self,*,time_observation_contract):
        super().__init__()
        if time_observation_contract.get('mode')!='positive_integer_round_clamp_v1':raise ValueError('Missing recorded-time contract')
        if time_observation_contract.get('top_code') not in (None,30):raise ValueError('Unsupported top-code')
        self.observation=dict(time_observation_contract)
        # Explicit inter-layer dropout uses the saved PyTorch RNG. Fused cuDNN
        # multi-layer dropout carries an opaque state outside state_dict/RNG.
        self.rnn=nn.ModuleList([nn.LSTM(2,64,batch_first=True),nn.LSTM(64,64,batch_first=True)])
        self.dense=nn.Linear(64,2)
        self.alpha_bias=nn.Parameter(torch.full((2,),2.))
        # MXNet Gluon LSTM uses zero i2h/h2h biases (not LSTMCell's default forget=1).
        for name,p in self.rnn.named_parameters():
            if 'weight' in name:nn.init.xavier_uniform_(p)
            else:nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.dense.weight);nn.init.constant_(self.dense.bias,5.)
        self.register_buffer('deep_renewal_identity',identity())

    def encode_task_states(self,dts,quantities,mask,*,memory_write_mask=None):
        observed=mask if memory_write_mask is None else mask & memory_write_mask
        if mask.dtype!=torch.bool or observed.shape!=dts.shape:raise ValueError('Invalid renewal masks')
        lengths=observed.sum(1)
        if bool((lengths<1).any()):raise ValueError('Renewal requires observed history')
        safe=torch.stack((dts,quantities),-1).float()
        safe=torch.where(observed[...,None],safe,0.)
        if not torch.isfinite(safe).all():raise ValueError('Nonfinite renewal input')
        # Compact padding/withheld entries, preserving chronological observed order.
        pos=torch.arange(mask.shape[1],device=mask.device).expand_as(mask)
        order=torch.argsort((~observed).long()*mask.shape[1]+pos,dim=1)
        compact=safe.gather(1,order[...,None].expand(-1,-1,2))
        sequence=torch.cat((torch.zeros_like(compact[:,:1]),compact),1)
        packed=pack_padded_sequence(sequence,(lengths+1).cpu(),batch_first=True,enforce_sorted=False)
        packed_out,_=self.rnn[0](packed)
        packed_out=PackedSequence(F.dropout(packed_out.data,p=.1,training=self.training),
            packed_out.batch_sizes,packed_out.sorted_indices,packed_out.unsorted_indices)
        packed_out,_=self.rnn[1](packed_out)
        encoded,_=pad_packed_sequence(packed_out,batch_first=True,total_length=sequence.shape[1])
        indices=observed.long().cumsum(1)
        hidden=encoded.gather(1,indices[...,None].expand(-1,-1,64))
        hidden=torch.where(observed[...,None],hidden,0.)
        return hidden,hidden

    def parameters_of_distribution(self,hidden):
        return F.softplus(self.dense(hidden))+1e-5,F.softplus(self.alpha_bias)+1e-5

    def log_observation_dt(self,hidden,true_dt):
        mu,alpha=self.parameters_of_distribution(hidden)
        result=nb_log_mass(true_dt-1,mu[...,0],alpha[0])
        code=self.observation['top_code']
        if code is not None:
            if bool((true_dt>code).any()):raise ValueError('Recorded duration above top-code')
            tail=true_dt==code
            if bool(tail.any()):result[tail]=nb_log_survival(code-1,mu[...,0][tail],alpha[0])
        return result

    def quantity_outputs(self,hidden,true_qty):
        mu,alpha=self.parameters_of_distribution(hidden);pred=1+mu[...,1]
        loss=-nb_log_mass(true_qty-1,mu[...,1],alpha[1]);zero=torch.zeros_like(loss)
        return {'train_loss':loss,'distribution_nll':loss,'point_prediction':pred,
                'log_mse':(torch.log1p(pred)-torch.log1p(true_qty)).square(),
                'location_huber':zero,'scale':alpha[1].expand_as(loss),
                'tail_aux_loss':zero,'tail_indicator':zero}

    def time_head_contract(self):
        return {'mode':TIME_MODE,'observation_likelihood':self.observation,'native_distribution':'shifted_negative_binomial'}

    def time_head_telemetry(self):
        return {'train_time_nb_alpha':float((F.softplus(self.alpha_bias[0])+1e-5).detach().cpu())}

    def load_state_dict(self,state_dict,strict=True,assign=False):
        v=state_dict.get('deep_renewal_identity')
        if not isinstance(v,torch.Tensor) or not torch.equal(v.cpu(),identity()):raise ValueError('Renewal identity mismatch')
        _complete_state(self,state_dict)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

def validate_checkpoint(payload,expected_backbone):
    meta=payload.get('encoder_config',{});states=[payload[k] for k in ('model_state_dict','best_state_dict') if k in payload]
    identified=payload.get('backbone')==ARM or meta.get('backbone_contract_id')=='deep_renewal_event_native_nb_v1' or any('deep_renewal_identity' in s for s in states)
    if expected_backbone!=ARM:
        if identified:raise ValueError('Renewal cannot be relabelled')
        return False
    resume=payload.get('resume_identity',{});interface=payload.get('interface_meta',{})
    if (payload.get('backbone')!=ARM or any(meta.get(k)!=v for k,v in metadata().items())
        or payload.get('variant')!=VARIANT or payload.get('evaluation_scope')!='validation_only'
        or payload.get('held_out_test_evaluated') is not False or resume.get('backbone')!=ARM
        or resume.get('interface_meta')!=interface or resume.get('arguments',{}).get('model_role')!=ROLE
        or resume.get('checkpoint_monitor')!='validation_raw_quantity_rmse'
        or meta.get('time_head',{}).get('mode')!=TIME_MODE
        or meta.get('time_head',{}).get('observation_likelihood')!=interface.get('time_head',{}).get('observation_likelihood')):
        raise ValueError('Renewal checkpoint contract mismatch')
    for s in states:
        if not isinstance(s.get('deep_renewal_identity'),torch.Tensor) or not torch.equal(s['deep_renewal_identity'].cpu(),identity()):raise ValueError('Renewal state identity mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256',''))!=64 or payload.get('best_epoch',0)<1):raise ValueError('Unbound renewal summary')
    return True
