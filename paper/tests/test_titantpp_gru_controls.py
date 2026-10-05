import unittest
from copy import deepcopy
import torch
from models.TPPs.CountAwareTitanGRUControls import PairGRUHistoryCorrection,AllActivePairMLP16,PAIR,MLP1,MLP8,ANCHOR,validate_checkpoint,metadata,role_for_arm
from models.TPPs.CountAwareTitanCNNGRU import GRUHistoryCorrection
from paper.scripts import run_titantpp_gru_controls as e
from paper.scripts import titantpp_gru_control_checks as checks
from paper.scripts.count_aware_tpp_backbone.core import target_outputs

class Controls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
    def nonzero(self,m):
        if hasattr(m,'output_projection'):torch.nn.init.normal_(m.output_projection.weight,std=.02)
        else:
            for p in m.output_projections:torch.nn.init.normal_(p.weight,std=.02)
        return m
    def test_pair_matches_manual_two_step_and_active_recurrence(self):
        m=self.nonzero(PairGRUHistoryCorrection());h=torch.randn(2,5,64);valid=torch.ones(2,5,dtype=torch.bool)
        actual=m(h,valid)
        for t in range(1,5):
            output,_=m.gru(m.input_projection(h[:,t-1:t+1]))
            torch.testing.assert_close(actual[:,t],m.output_projection(output[:,-1]),rtol=1e-6,atol=1e-7)
        actual.square().sum().backward()
        self.assertGreater(float(m.gru.weight_hh_l0.grad.abs().sum()),0.)
    def test_pair_rejects_extra_adapter_context_prefix_retains_it(self):
        pair=self.nonzero(PairGRUHistoryCorrection());prefix=GRUHistoryCorrection();prefix.load_state_dict(pair.state_dict())
        h=torch.randn(2,5,64);v=torch.ones(2,5,dtype=torch.bool);changed=h.clone();changed[:,0]+=20.
        self.assertTrue(torch.equal(pair(h,v)[:,-1],pair(changed,v)[:,-1]))
        self.assertFalse(torch.equal(prefix(h,v)[:,-1],prefix(changed,v)[:,-1]))
    def test_withheld_reset_padding_and_call_local_state(self):
        for m in (self.nonzero(PairGRUHistoryCorrection()),self.nonzero(AllActivePairMLP16())):
            h=torch.randn(1,7,64);v=torch.tensor([[True,False,True,True,True,True,True]])
            w=torch.tensor([[True,True,True,False,True,True,True]])
            out=m(h,v,memory_write_mask=w)
            self.assertTrue(torch.equal(out[:,[0,1,3,4]],torch.zeros_like(out[:,[0,1,3,4]])))
            changed=h.clone();changed[:,[1,3]]=float('nan')
            self.assertTrue(torch.equal(out,m(changed,v,memory_write_mask=w)))
            m(h+7,v,memory_write_mask=w)
            self.assertTrue(torch.equal(out,m(h,v,memory_write_mask=w)))
    def test_fixed_alpha_and_equal_parameter_initialization(self):
        one=self.nonzero(AllActivePairMLP16(divisor=1));eight=AllActivePairMLP16(divisor=8);eight.load_state_dict(one.state_dict())
        h=torch.randn(2,12,64);v=torch.ones(2,12,dtype=torch.bool)
        self.assertTrue(torch.equal(one(h,v)/8,eight(h,v)))
        self.assertEqual(sum(p.numel() for p in one.parameters()),24576)
        self.assertEqual(sum(p.numel() for p in PairGRUHistoryCorrection().parameters()),24732)
    def test_all_eight_branches_active_after_two_events(self):
        m=self.nonzero(AllActivePairMLP16());h=torch.randn(2,2,64);v=torch.ones(2,2,dtype=torch.bool)
        m(h,v).square().sum().backward()
        self.assertTrue(all(float(p.weight.grad.abs().sum())>0 for p in m.input_projections))
    def test_common_state_and_matched_init(self):
        data=checks.synthetic_data(84)
        for seed in (42,52,62):
            e.initial_states(data,seed)
            def state(a):
                torch.manual_seed(seed);return e.build_model(data,a)[0].multilag_detail.state_dict()
            for a,b in ((ANCHOR,PAIR),(MLP1,MLP8)):
                sa,sb=state(a),state(b);self.assertTrue(all(torch.equal(sa[k],sb[k]) for k in sa))
    def test_zero_output_and_strict_restoration(self):
        data=checks.synthetic_data(84);dt=torch.ones(2,6);q=torch.ones_like(dt)*4;v=torch.ones_like(dt,dtype=torch.bool);outputs=[]
        for a in e.SUPPORTED_ARMS:
            torch.manual_seed(42);model,_=e.build_model(data,a);model.eval()
            outputs.append(target_outputs(model,dt,v,q,lambda_log_qty=1.))
            restored,_=e.build_model(data,a);restored.load_state_dict(model.state_dict())
        self.assertTrue(all(all(torch.equal(o[k],outputs[0][k]) for k in o) for o in outputs))
        model,_=e.build_model(data,MLP1);foreign,_=e.build_model(data,MLP8)
        with self.assertRaises(ValueError):foreign.load_state_dict(model.state_dict())
    def test_full_model_no_target_future_or_cross_call_leak(self):
        for arm in (PAIR,MLP1,MLP8):checks.causality_check(e,arm,lambda:None,device='cpu')
    def test_metadata_is_bound_and_no_legacy_relabel(self):
        model,meta=e.build_model(checks.synthetic_data(84),MLP1)
        self.assertEqual(metadata(arm=MLP1)['residual_divisor'],1)
        with self.assertRaises(ValueError):validate_checkpoint({'backbone':MLP1,'model_state_dict':model.state_dict()},ANCHOR)
        with self.assertRaises(ValueError):validate_checkpoint({'backbone':MLP1,'encoder_config':{}},MLP1)

if __name__=='__main__':unittest.main()
