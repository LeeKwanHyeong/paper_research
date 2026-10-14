"""Offline public replay integration; synthetic data only, no native qualification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from paper.scripts import run_titantpp_gru_residual_cap as r
from paper.scripts.count_aware_tpp_backbone import training
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
PROJECT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution/execution_contract.json'
RESULTS = []

class PublicReplayIntegration(unittest.TestCase):
    def test_nine_fixed_metadata_public_replays_and_identity_boundaries(self):
        torch.set_num_threads(1)
        contract = json.loads(CONTRACT.read_text())
        self.assertEqual(len(contract['datasets']), 3)
        with tempfile.TemporaryDirectory() as temporary:
            for original in contract['datasets']:
                for seed in (42, 52, 62):
                    with self.subTest(dataset=original['dataset_id'], seed=seed):
                        data = copy.deepcopy(original)
                        # Only the target count is replaced for the synthetic two-target loader.
                        data['inherited_data_identity']['populations']['validation']['target_count'] = 2
                        args = r.training_args(contract, data, Path(temporary))
                        args.seeds = str(seed)
                        args.execution_role = 'fresh_contribution_validation'  # frozen run_fit override
                        self.assertEqual((args.epochs,args.min_epochs,args.early_stopping_patience),(300,40,40))
                        torch.manual_seed(seed)
                        model, metadata = r.build_model(data, r.CAPPED)
                        parent, _ = r.build_model(data, r.FULL)
                        capacity = sum(parameter.numel() for parameter in model.parameters())
                        self.assertEqual(capacity, sum(parameter.numel() for parameter in parent.parameters()))
                        state = model.state_dict()
                        state_sha = canonical_state_dict_sha256(state)
                        # Does not run time_interface; this is NOT proof of real interface completeness.
                        interface = {'time_head': model.time_head_contract()}
                        identity = training._resume_identity(args=args, backbone=r.CAPPED,
                            quantity_variant=r.VARIANT, seed=seed, monitor=args.checkpoint_monitor,
                            quantity_contract={'boundaries':data['quantity_boundaries_all_train_rows']},
                            interface_meta=interface)
                        spec = training.checkpoint_monitor_spec(args.checkpoint_monitor)
                        payload = {'backbone':r.CAPPED,'encoder_config':metadata,'model_state_dict':state,
                            'model_state_sha256':state_sha,'evaluation_scope':'validation_only',
                            'held_out_test_evaluated':False,'variant':r.VARIANT,'seed':seed,
                            'interface_meta':interface,'resume_identity':identity,'initial_state_sha256':state_sha,
                            'source_revision':args.source_revision,'source_revision_history':[args.source_revision],
                            'checkpoint_monitor':args.checkpoint_monitor,'checkpoint_monitor_history_key':spec['history_key'],
                            'checkpoint_selection':spec['selection'],'best_epoch':1}
                        path = Path(temporary) / 'synthetic.pt'
                        torch.save(payload,path)
                        dt = torch.tensor([[0.,1.,3.,5.],[0.,4.,2.,6.]])
                        qty = torch.tensor([[0.,2.,4.,8.],[1.,0.,5.,2.]])
                        mask = torch.ones_like(dt,dtype=torch.bool)
                        expected = dict(device='cpu',expected_arm=r.CAPPED,expected_identity=identity,
                            expected_initial=state_sha,expected_epoch=1,expected_seed=seed)
                        with patch('paper.scripts.run_taxi_quantity_interface_ablation.make_loader',
                                return_value=[(None,dt,mask,None,qty)]) as loader:
                            receipt = r.replay_checkpoint(path,data,None,lambda:None,**expected)
                            self.assertEqual(receipt['count'],2)
                            self.assertEqual(loader.call_args.kwargs['target_split'],'validation')
                            self.assertEqual(receipt['state_sha256'],state_sha)
                        rejected = []
                        # These reject before any loader or actual prediction can be invoked.
                        with patch('paper.scripts.run_taxi_quantity_interface_ablation.make_loader',
                                side_effect=AssertionError('Rejected identity must not reach loader')):
                            for key, value in [('expected_arm',r.CNN),('expected_seed',seed+1),
                                               ('expected_epoch',2),('expected_initial','0'*64)]:
                                with self.assertRaises(ValueError):
                                    r.replay_checkpoint(path,data,None,lambda:None,**{**expected,key:value})
                                rejected.append(key)
                            corrupted = {**payload,'source_revision':'wrong-source'}
                            torch.save(corrupted,path)
                            with self.assertRaises(ValueError):
                                r.replay_checkpoint(path,data,None,lambda:None,**expected)
                            rejected.append('source_revision')
                        RESULTS.append({'dataset':data['dataset_id'],'seed':seed,'status':'passed',
                            'parameter_count':capacity,'parent_FULL_parameter_count_equal':True,
                            'loader_target_count':2,'positive_public_replay':True,'rejected_identity_fields':rejected,
                            'expected_epoch':1,'source_revision':args.source_revision})
                        path.unlink()
        self.assertEqual(len(RESULTS),9)

if __name__ == '__main__':
    unittest.main()
