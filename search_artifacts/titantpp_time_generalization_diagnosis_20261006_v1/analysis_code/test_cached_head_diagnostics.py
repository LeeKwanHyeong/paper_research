"""Synthetic CPU fixtures for native cached time-head aggregate diagnostics."""
from pathlib import Path
import importlib.util
import math
import tempfile
import unittest
import torch

spec=importlib.util.spec_from_file_location('cached_head_diagnostics',Path(__file__).with_name('cached_head_diagnostics.py'))
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class Diagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.native=diag.load_native_class()

    def fixtures(self):
        state={'v_t.weight':torch.zeros(1,64),'b_t':torch.tensor([0.]),'w_raw':torch.tensor([0.]),'time_scale_weight.weight':torch.zeros(1,64)}
        config={'time_head_mode':'heteroscedastic_lognormal_duration','time_scale':1.,'time_sigma_floor':.001,
            'time_observation_contract':{'mode':'positive_integer_round_clamp_v1','top_code':None,'unit':'hour'}}
        return {'model_state_dict':state},config

    def test_native_head_matches_independent_interval_mass_reference(self):
        source,config=self.fixtures();head=diag.head_view(source,config,native_class=self.native)
        dt=torch.tensor([1.,2.,3.,4.,7.,8.,20.]);cache={'dt':dt,'hidden':torch.zeros(len(dt),64)}
        loss=diag.native_nll(cache,head,batch_size=3)
        sigma=.001+math.log(2.)
        normal=lambda z:.5*(1+math.erf(z/math.sqrt(2)))
        expected=[]
        for d in dt.tolist():
            upper=normal(math.log(d+.5)/sigma)
            lower=0. if d==1 else normal(math.log(d-.5)/sigma)
            expected.append(-math.log(upper-lower))
        torch.testing.assert_close(loss,torch.tensor(expected,dtype=torch.float64),rtol=1e-9,atol=1e-9)
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        self.assertEqual(sum(p.numel() for p in head.parameters()),130)

    def test_head_override_and_source_immutability(self):
        source,config=self.fixtures();before=diag.tensor_sha(source['model_state_dict'])
        state={k:v.clone() for k,v in source['model_state_dict'].items()};state['b_t']+=1.
        h=diag.head_view(source,config,state,self.native)
        cache={'dt':torch.tensor([1.,2.]),'hidden':torch.zeros(2,64)}
        head_before=diag.tensor_sha(h.state_dict());diag.native_nll(cache,h);diag.native_mu_sigma(cache,h)
        self.assertEqual(before,diag.tensor_sha(source['model_state_dict']));self.assertEqual(head_before,diag.tensor_sha(h.state_dict()))
        self.assertEqual(h.b_t.item(),1.)

    def test_wrong_head_names_shapes_and_nonfinite_rejected(self):
        source,config=self.fixtures()
        for mutation in ('name','shape','nan'):
            state={k:v.clone() for k,v in source['model_state_dict'].items()}
            if mutation=='name':state['v_sigma.weight']=state.pop('time_scale_weight.weight')
            elif mutation=='shape':state['v_t.weight']=torch.zeros(1,63)
            else:state['b_t'][0]=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):diag.head_view(source,config,state,self.native)

    def test_cohorts_are_exact_partition_at_boundaries(self):
        dt=torch.tensor([1.,2.,3.,4.,7.,8.,99.]);masks=diag.cohort_masks(dt)
        self.assertEqual([int(v.sum()) for v in masks.values()],[1,2,2,2])
        self.assertTrue(torch.equal(sum(x.long() for x in masks.values()),torch.ones(7,dtype=torch.int64)))

    def test_cohort_means_and_weighted_contributions_reconcile(self):
        dt=torch.tensor([1.,2.,3.,4.,7.,8.,99.]);loss=torch.tensor([1.,2.,3.,4.,5.,6.,7.],dtype=torch.float64)
        s=diag.summarize_losses(loss,dt,torch.zeros(7),torch.ones(7))
        self.assertEqual(s['mean_nll'],4.)
        self.assertAlmostEqual(sum(c['contribution_to_overall_mean_nll'] for c in s['cohorts'].values()),4.)
        self.assertAlmostEqual(sum(c['share_of_total_nll'] for c in s['cohorts'].values()),1.)
        self.assertEqual(s['cohorts']['dt_2_3']['mean_nll'],2.5)
        self.assertEqual(s['loss_concentration']['top_1_percent']['count'],1)
        self.assertEqual(s['loss_concentration']['top_1_percent']['share_of_total_nll'],.25)

    def test_exact_top_one_and_five_percent_concentration(self):
        loss=torch.arange(1,101,dtype=torch.float64);dt=torch.ones(100)
        s=diag.summarize_losses(loss,dt,torch.zeros(100),torch.ones(100))
        self.assertEqual(s['loss_concentration']['top_1_percent']['count'],1)
        self.assertEqual(s['loss_concentration']['top_5_percent']['count'],5)
        self.assertAlmostEqual(s['loss_concentration']['top_5_percent']['share_of_total_nll'],490/5050)
        self.assertIsNone(s['cohorts']['dt_8_plus']['mean_nll'])

    def test_zero_loss_has_no_fabricated_percentage(self):
        s=diag.summarize_losses(torch.zeros(3),torch.ones(3),torch.zeros(3),torch.ones(3))
        self.assertIsNone(s['cohorts']['dt_1']['share_of_total_nll'])
        self.assertIsNone(s['loss_concentration']['top_5_percent']['share_of_total_nll'])

    def test_cpu_native_tolerance_is_visible_and_not_silent(self):
        reference=torch.tensor([1.,2.,3.],dtype=torch.float64)
        same=diag.replay_difference(reference,reference)
        self.assertEqual(same['outside_diagnostic_tolerance_count'],0)
        changed=diag.replay_difference(reference+.1,reference)
        self.assertEqual(changed['outside_diagnostic_tolerance_count'],3)
        self.assertFalse(changed['within_campaign_endpoint_tolerance'])
        self.assertAlmostEqual(changed['mean_signed_difference'],.1)

    def test_immutable_file_hash_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'input.pt';p.write_bytes(b'original');expected=diag.sha(p)
            self.assertEqual(diag.file_checked(root,'input.pt',expected),root.resolve()/'input.pt')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):diag.file_checked(root,'input.pt',expected)
            with self.assertRaises(ValueError):diag.file_checked(root,'../input.pt',expected)

    def test_symmetric_two_block_contributions_reconcile_with_interaction(self):
        loss00=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        loss10=loss00+torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        loss01=loss00+torch.tensor([4.,3.,2.,1.],dtype=torch.float64)
        loss11=loss00+torch.tensor([7.,7.,7.,7.],dtype=torch.float64)
        result=diag.parameter_block_decomposition(loss00,loss10,loss01,loss11,torch.tensor([1.,2.,4.,8.]))
        self.assertEqual(result['overall']['mean_nll_delta'],7.)
        self.assertEqual(result['overall']['location_symmetric_contribution_to_mean_delta'],3.5)
        self.assertEqual(result['overall']['scale_symmetric_contribution_to_mean_delta'],3.5)
        self.assertEqual(result['maximum_absolute_row_reconciliation_residual'],0.)
        self.assertAlmostEqual(sum(v['location_contribution_to_overall_mean_delta']+v['scale_contribution_to_overall_mean_delta'] for v in result['cohorts'].values()),7.)
        self.assertEqual(result['dt_ge_2']['count'],3)
        self.assertFalse(result['candidate_checkpoint_saved'])

    def test_symmetric_two_block_zero_and_signed_counteraction(self):
        a=torch.tensor([1.,2.],dtype=torch.float64)
        same=diag.parameter_block_decomposition(a,a,a,a,torch.ones(2))
        self.assertEqual(same['overall']['mean_nll_delta'],0.)
        result=diag.parameter_block_decomposition(a,a-1,a+3,a+2,torch.ones(2))
        self.assertEqual(result['overall']['location_symmetric_contribution_to_mean_delta'],-1.)
        self.assertEqual(result['overall']['scale_symmetric_contribution_to_mean_delta'],3.)
        self.assertEqual(result['overall']['mean_nll_delta'],2.)

    def test_mu_sigma_are_event_conditioned_not_zero_hidden_telemetry(self):
        source,config=self.fixtures();source['model_state_dict']['v_t.weight'][0,0]=1.
        source['model_state_dict']['time_scale_weight.weight'][0,0]=1.
        head=diag.head_view(source,config,native_class=self.native)
        hidden=torch.zeros(2,64);hidden[1,0]=2.
        mu,sigma=diag.native_mu_sigma({'hidden':hidden,'dt':torch.ones(2)},head)
        self.assertEqual(mu.tolist(),[0.,2.]);self.assertGreater(sigma[1].item(),sigma[0].item())


if __name__=='__main__':unittest.main()
