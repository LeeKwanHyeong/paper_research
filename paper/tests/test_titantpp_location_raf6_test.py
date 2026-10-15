"""Offline RAF6 orchestration contracts; no SSH and no real Test inference."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import advance_titantpp_location_raf6_test as a
from paper.scripts import evaluate_titantpp_location_raf6_frozen as e


class RAF6EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='nonlinear-raf6-eval-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_reference_maps_only_verified_validation_and_preserves_original(self):
        endpoint = {'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
                    'qty_rmse': 1.25, 'tail': {'qty_rmse': 3.5}}
        before = deepcopy(endpoint)
        result = a.validation_reference(endpoint)
        self.assertEqual(result, {**before, 'split': 'validation'})
        result['tail']['qty_rmse'] = 99.
        self.assertEqual(endpoint, before)
        for change in ({'evaluation_scope': 'test'}, {'held_out_test_evaluated': True},
                       {'held_out_test_evaluated': None}, {'split': 'test'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Validation-only'):
                a.validation_reference({**endpoint, **change})

    def test_waiting_performs_no_remote_or_binding_and_does_not_claim(self):
        remote, bind = Mock(), Mock()
        with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
             patch.object(a, 'collection_gate', return_value=None), patch.object(a, 'transport', remote), \
             patch.object(a, 'bind', bind):
            result = a.advance('a' * 64)
        self.assertEqual(result['remote_calls'], 0)
        remote.assert_not_called(); bind.assert_not_called()
        self.assertFalse((self.root / 'advance.claim').exists())

    def test_integrity_failure_precedes_collection_and_transport(self):
        gate, remote = Mock(), Mock()
        with patch.object(a, 'verify_bootstrap', side_effect=ValueError('seal differs')), \
             patch.object(a, 'collection_gate', gate), patch.object(a, 'transport', remote):
            with self.assertRaisesRegex(ValueError, 'seal'):
                a.advance('a' * 64)
        gate.assert_not_called(); remote.assert_not_called()

    def test_failed_or_started_attempt_never_retries(self):
        for name in ('advance.claim', 'failure.json'):
            p = self.root / name; p.write_text('{}')
            with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
                 patch.object(a, 'collection_gate') as gate:
                with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                    a.advance('a' * 64)
                gate.assert_not_called()
            p.unlink()

    def test_all_validation_precedes_test_and_failure_prevents_test(self):
        order = []
        with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
             patch.object(a, 'collection_gate', return_value={'status': 'complete'}), \
             patch.object(a, 'bind', return_value={'bound_files': {}}), \
             patch.object(a, 'read', return_value={'files': {}}), patch.object(a, 'deploy'), \
             patch.object(a, 'evaluate_stage', side_effect=lambda c, t, s: order.append(s)), \
             patch.object(a, 'make_gate', side_effect=ValueError('Validation mismatch')):
            with self.assertRaisesRegex(ValueError, 'Validation mismatch'):
                a.advance('a' * 64)
        self.assertEqual(order, ['validation'])
        self.assertTrue((self.root / 'failure.json').exists())

    def test_strict_earliest_finite_minimum(self):
        history = [{'epoch': 1, 'val_qty_rmse': None}, {'epoch': 2, 'val_qty_rmse': 2.},
                   {'epoch': 3, 'val_qty_rmse': 2.}, {'epoch': 4, 'val_qty_rmse': 3.}]
        self.assertEqual(e.strict_selected_epoch(history), (2, 2.))
        with self.assertRaises(ValueError):
            e.strict_selected_epoch(history[::-1])

    def test_unbound_checkpoint_rejected_before_data_access(self):
        with patch.object(e, 'read', return_value={'binding_complete': False}), \
             patch.object(e, 'canonical', return_value='x'), patch.object(e, 'verify_source') as source:
            with self.assertRaises(ValueError):
                e.worker(self.root, 'foreign', 'test', 'cuda')
            source.assert_not_called()

    def test_missing_global_gate_blocks_before_science(self):
        c = {'jobs': [{'id': 'job', 'evaluation_host': '5080'}], 'runtimes': {'5080': {}}}
        with patch.object(e, 'verify_contract', return_value=c), patch.object(e, 'verify_permit', return_value={}), \
             patch.object(e, 'verify_source') as source:
            with self.assertRaises(FileNotFoundError):
                e.worker(self.root, 'job', 'test', 'cuda', permit_path='x', permit_sha='a' * 64)
            source.assert_not_called()

    def test_gate_requires_all_six_explicit_worker_exit0(self):
        jobs = [{'id': str(i)} for i in range(6)]
        c = {'jobs': jobs, 'contract_sha256': 'a' * 64}
        receipts, exits = {}, {}
        for j in jobs:
            stem = self.root / 'results' / j['id']
            a.write(stem / 'validation_replay.json', {'completed_unix': 10.})
            a.write(stem / 'validation_worker_exit.json', {'job_id': j['id'], 'contract_sha256': c['contract_sha256'],
                    'actual_child_exit_observed': True, 'exit_code': 0, 'completed_unix': 10.})
            receipts[j['id']] = a.sha(stem / 'validation_replay.json')
            exits[j['id']] = a.sha(stem / 'validation_worker_exit.json')
        gate = {'schema': 'titantpp_location_raf6_validation_gate_v1', 'cohort': e.COHORT,
                'status': 'PASS', 'contract_sha256': c['contract_sha256'], 'job_count': 6,
                'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': 11.}
        path = self.root / 'validation_gate.json'; a.write(path, gate)
        with patch.object(e, 'verify_validation_receipt', side_effect=lambda r, c, j: r):
            e.verify_global_validation_gate(self.root, c)
            for change in ({'job_count': 5}, {'validation_worker_exits': {}}, {'created_unix': 9.}):
                path.write_text(json.dumps({**gate, **change}))
                with self.assertRaises(ValueError):
                    e.verify_global_validation_gate(self.root, c)
            bad = self.root / 'results/0/validation_worker_exit.json'
            record = a.read(bad); record['exit_code'] = 1; bad.write_text(json.dumps(record))
            gate['validation_worker_exits']['0'] = a.sha(bad); path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, 'actual worker exit0'):
                e.verify_global_validation_gate(self.root, c)

    def test_permit_rejects_wrong_host_stage_and_expired_or_extended_budget(self):
        p = {'schema': 'titantpp_location_raf6_evaluation_permit_v1', 'authorized': True, 'new_training': False,
             'frozen_selection': True, 'stage': 'validation', 'host': '5080', 'job_ids': ['job'],
             'contract_sha256': 'a' * 64, 'evaluator_sha256': e.sha(e.__file__),
             'not_before_unix': 100., 'deadline_unix': 200.}
        job = {'id': 'job', 'evaluation_host': '5080'}
        c = {'contract_sha256': 'a' * 64, 'jobs': [job]}
        def check(value, stage='validation'):
            path = self.root / 'permit.json'; path.write_text(json.dumps(value))
            with patch.object(e, 'verify_partial_metadata', return_value={'remote_roots': {'5080': str(self.root)}}):
                return e.verify_permit(self.root, c, job, stage, 'permit.json', a.sha(path), now=150.)
        check(p)
        for change in ({'host': '5090'}, {'new_training': True}, {'job_ids': ['other']},
                       {'deadline_unix': 120.}, {'deadline_unix': 50000.}):
            with self.assertRaises(ValueError):
                check({**p, **change})
        with self.assertRaises(ValueError):
            check(p, 'test')

    def rows(self):
        rows = []
        for d in a.DATASETS:
            for arm, shift in ((a.CNN, 0.), (a.QUERY, 1.), (a.VALUE, 2.), (a.STORED_CHANNEL, 3.)):
                for seed, x in zip(a.SEEDS, (1., 2., 3.)):
                    metrics = {'qty_rmse': x + shift, 'qty_mae': (x + shift) / 2., 'time_nll': x + shift}
                    rows.append({'dataset': d, 'arm': arm, 'seed': seed, 'metrics': {**metrics, 'tail': metrics.copy()},
                                 'target_identity_sha256': 'a', 'truth_sha256': 'b', 'time_sha256': 'c'})
        return rows

    def test_aggregation_sample_sd_and_unfavorable_seed_differences(self):
        r = a.aggregate(self.rows())
        cnn = r['grouped'][0]['metrics']['whole_qty_rmse']
        self.assertEqual(cnn, {'mean': 2., 'sample_sd': 1.})
        self.assertEqual(len(r['seed_differences']), 15)
        self.assertEqual(r['seed_differences'][0]['metrics']['whole_qty_rmse']['candidate_minus_baseline'], 2.)
        self.assertFalse(r['test_independent'])

    def test_aggregation_rejects_missing_duplicate_or_unmatched_population(self):
        rows = self.rows()
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                a.aggregate(bad)
        bad = deepcopy(rows); bad[0]['truth_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'population'):
            a.aggregate(bad)

    def test_archive_rejects_unexpected_members_before_writing(self):
        archive = self.root / 'bad.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            raw = json.dumps({'../outside': 'a' * 64}).encode()
            info = tarfile.TarInfo('result_manifest.json'); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        with patch.object(a, 'BUNDLE', self.root / 'target'):
            with self.assertRaisesRegex(ValueError, 'membership'):
                a.extract_results(archive, '5080', 'validation', {'jobs': []})
        self.assertFalse((self.root / 'target').exists())


    def test_exact_raf_cohort_excludes_taxi_and_gru(self):
        self.assertEqual(e.expected_jobs(), {('raf_spare_parts', arm, seed) for arm in (a.VALUE,a.QUERY) for seed in (42,52,62)})

    def test_pending_training_is_local_only_and_incomplete_retrieval_blocks(self):
        with patch.object(a,'TRAIN',self.root/'training'),patch.object(a,'REPORT',self.root/'report/evaluation_v1'):
            self.assertIsNone(a.collection_gate({}))
            (self.root/'training/retrieval').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'Incomplete training retrieval'):
                a.collection_gate({})

    def test_training_action_required_blocks_before_receipt_lookup(self):
        report=self.root/'report';report.mkdir()
        (report/'ACTION_REQUIRED.json').write_text('{}')
        with patch.object(a,'TRAIN',self.root/'training'),patch.object(a,'REPORT',report/'evaluation_v1'):
            with self.assertRaisesRegex(ValueError,'user action'):
                a.collection_gate({})

    def test_preparation_wait_does_not_create_bundle_or_inspect_real_results(self):
        target=self.root/'evaluation'
        with patch.object(a,'BUNDLE',target),patch.object(a,'training_contract',return_value={}), \
             patch.object(a,'collection_gate',return_value=None),patch.object(a,'copy_checked') as copying, \
             patch.object(a,'transport') as remote:
            result=a.prepare('a'*64,'b'*64)
        self.assertEqual(result['remote_calls'],0);self.assertFalse(result['Test'])
        copying.assert_not_called();remote.assert_not_called();self.assertFalse(target.exists())

    def test_exact_training6_contract_rejects_a_fresh_cnn_job(self):
        jobs=[{'id':f'{d}__{s}__{arm}'} for d in a.DATASETS for s in a.SEEDS for arm in (a.QUERY,a.VALUE)]
        for changes in (jobs+[{'id':'raf_spare_parts__42__titantpp_cnn'}],jobs[:-1]):
            with patch.object(a,'TRAIN_INTEGRITY_SHA','a'*64),patch.object(a,'sha',return_value='a'*64), \
                 patch.object(a,'read',side_effect=[{'files':{},'contract_raw_sha256':'a'*64},{'jobs':changes}]):
                with self.assertRaisesRegex(ValueError,'Foreign training cohort'):
                    a.training_contract()

    def test_terminal_snapshot_requires_all_bindings_binary_and_owner_absence(self):
        c={'jobs':[{'id':str(i)} for i in range(6)]}
        snapshot={'contract_sha256':a.canonical(c),'all_training_terminal_verified':True,'failure':False,
            'proof_bindings':{k:True for k in ('pipeline','start','qualification','training_permit','dispatcher')},
            'jobs':[{'job':j,'state':'complete','selected_binary_present':True,'terminal_binary_verified':True} for j in c['jobs']],
            'owner_probes':{'123':{'state':'absent'}},'gpu_compute_pids':[]}
        a.verify_terminal_snapshot(c,snapshot)
        changes=[{'owner_probes':{}},{'gpu_compute_pids':[123]},{'proof_bindings':{}},
                 {'jobs':snapshot['jobs'][:-1]},{'failure':True},{'contract_sha256':'b'*64}]
        bad=deepcopy(snapshot['jobs']);bad[0]['terminal_binary_verified']=False;changes.append({'jobs':bad})
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                a.verify_terminal_snapshot(c,{**snapshot,**change})


    def reference_scope(self):
        baseline=[{'seed':seed,'arm':a.LEGACY_CNN,'comparison_arm':a.CNN,'dataset':a.DATASETS[0],
            'role':'historical_immutable_reference','retrained':False,'reevaluated':False,
            'receipt_sha256':'a'*64,'checkpoint_sha256':'b'*64} for seed in a.SEEDS]
        stored=[{'seed':seed,'arm':a.STORED_CHANNEL,'dataset':a.DATASETS[0],'host':'5090',
            'role':'stored_loop1_channel_immutable_reference','retrained':False,'reevaluated':False,
            'added_parameters':3072,'original_evidence_verified':True,
            'final_receipt_sha256':a.STORED_RECEIPT_SHA,'source_closure':a.STORED_SOURCE_CLOSURE,
            'checkpoint_sha256':'b'*64,'checkpoint_state_sha256':'c'*64,'selected_endpoint_sha256':'d'*64,
            'receipts':{name:{'sha256':'e'*64} for name in ('validation_replay.json',
                'validation_worker_exit.json','test_metrics.json','test_worker_exit.json')}} for seed in a.SEEDS]
        return {'baseline_references':baseline,'stored_channel_references':stored,
                'candidate_selection_rule':a.SELECTION_RULE}

    def test_reference_scope_keeps_three_stored_models_outside_six_evaluation_jobs(self):
        c=self.reference_scope();before=deepcopy(c)
        self.assertTrue(e.verify_reused_reference_scope(c))
        self.assertEqual(c,before)
        self.assertEqual(len(e.expected_jobs()),6)
        self.assertNotIn(a.STORED_CHANNEL,e.ARMS);self.assertNotIn(a.CNN,e.ARMS)

    def test_stored_reference_scope_rejects_retrain_runtime_capacity_and_unpinned_origin(self):
        for changes in ({'retrained':True},{'reevaluated':True},{'host':'5080'},
                        {'added_parameters':1024},{'arm':a.QUERY},{'source_closure':'a'*64},
                        {'final_receipt_sha256':'a'*64},{'original_evidence_verified':False},
                        {'receipts':{}}):
            c=self.reference_scope();c['stored_channel_references'][0].update(changes)
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Stored QKV'):
                e.verify_reused_reference_scope(c)

    def test_missing_or_duplicate_stored_reference_rejected(self):
        c=self.reference_scope();c['stored_channel_references'].pop()
        with self.assertRaises(ValueError):e.verify_reused_reference_scope(c)
        c=self.reference_scope();c['stored_channel_references'][0]=c['stored_channel_references'][1]
        with self.assertRaises(ValueError):e.verify_reused_reference_scope(c)

    def test_prewritten_candidate_selection_cannot_use_test_rank(self):
        c=self.reference_scope();c['candidate_selection_rule']='test_rmse'
        with self.assertRaisesRegex(ValueError,'selection rule'):e.verify_reused_reference_scope(c)

    def test_stored_final_receipt_mismatch_blocks_before_result_reads_or_copy(self):
        with patch.object(a,'sha',return_value='0'*64),patch.object(a,'read') as reading:
            add=Mock()
            with self.assertRaisesRegex(ValueError,'Stored Channel final receipt changed'):
                a.stored_channel_references({},'a'*64,add,{})
            reading.assert_not_called();add.assert_not_called()

    def stored_fixture(self):
        metrics={'count':5,'qty_rmse':2.,'qty_mae':1.,'time_nll':.5,
                 'tail':{'count':2,'qty_rmse':3.,'qty_mae':2.,'time_nll':.7}}
        endpoint={**deepcopy(metrics),'evaluation_scope':'validation_only',
                  'held_out_test_evaluated':False,'state_sha256':'c'*64}
        endpoint_path=self.root/'endpoint.json';a.write(endpoint_path,{'selected':endpoint})
        ref={'id':'channel42','arm':a.STORED_CHANNEL,'seed':42,'role':'stored_loop1_channel_immutable_reference',
             'selected_epoch':5,'checkpoint_sha256':'b'*64,'checkpoint_state_sha256':'c'*64,
             'source_closure':a.STORED_SOURCE_CLOSURE,'contract_sha256':a.STORED_CONTRACT_SHA,
             'final_receipt_sha256':a.STORED_RECEIPT_SHA,'original_evidence_verified':True,
             'selected_endpoint_path':'endpoint.json','selected_endpoint_sha256':a.sha(endpoint_path),
             'validation_reference':a.validation_reference(endpoint),'receipts':{}}
        for split,suffix in (('validation','validation_replay.json'),('test','test_metrics.json')):
            raw={'status':'complete','job_id':'channel42','arm':a.STORED_CHANNEL,'dataset':a.DATASETS[0],
                'seed':42,'split':split,'selected_epoch':5,'checkpoint_sha256':'b'*64,'state_sha256':'c'*64,
                'contract_sha256':a.STORED_CONTRACT_SHA,'source_closure':a.STORED_SOURCE_CLOSURE,
                'retrained':False,'reselected':False,'metrics':deepcopy(metrics),'native_runtime_checked':True,
                'target_identity_sha256':'x'*64,'truth_sha256':'y'*64,'time_sha256':'z'*64,
                'runtime':{'gpu':{'uuid':'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'},
                    'python':'3.12.13','torch':'2.11.0+cu130','cudnn':92000,'threads':4}}
            a.write(self.root/suffix,raw)
            exname=split+'_worker_exit.json'
            a.write(self.root/exname,{'actual_child_exit_observed':True,'exit_code':0,
                'job_id':ref['id'],'contract_sha256':ref['contract_sha256']})
            for name in (suffix,exname):ref['receipts'][name]={'path':name,'sha256':a.sha(self.root/name)}
        return {'stored_channel_references':[ref]}

    def test_stored_validation_uses_original_selected_endpoint_and_never_infers(self):
        c=self.stored_fixture();before=deepcopy(c)
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'evaluator',return_value=e),patch.object(a,'transport') as remote:
            rows=a.stored_channel_rows(c)
        remote.assert_not_called();self.assertEqual(c,before)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0][0]['validation_metric_origin'],'original_training_selected_endpoint')
        self.assertEqual(rows[0][1]['metrics'],c['stored_channel_references'][0]['validation_reference'])
        self.assertEqual(rows[1][0]['added_parameters'],3072)
        self.assertFalse(rows[1][0]['reevaluated'])

    def test_stored_reference_bad_source_or_wrong_checkpoint_is_rejected_even_with_new_file_hash(self):
        c=self.stored_fixture();ref=c['stored_channel_references'][0]
        path=self.root/'validation_replay.json';original=a.read(path)
        for change in ({'source_closure':'a'*64},{'checkpoint_sha256':'a'*64},
                       {'reselected':True},{'job_id':'foreign'}):
            path.write_text(json.dumps({**original,**change}))
            ref['receipts']['validation_replay.json']['sha256']=a.sha(path)
            with patch.object(a,'BUNDLE',self.root),patch.object(a,'evaluator',return_value=e),self.assertRaises(ValueError):
                a.stored_channel_rows(c)

    def test_stored_failed_worker_does_not_become_reused_success(self):
        c=self.stored_fixture();ref=c['stored_channel_references'][0]
        path=self.root/'validation_worker_exit.json';raw=a.read(path);raw['exit_code']=1
        path.write_text(json.dumps(raw));ref['receipts'][path.name]['sha256']=a.sha(path)
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'evaluator',return_value=e),self.assertRaisesRegex(ValueError,'exit identity'):
            a.stored_channel_rows(c)

    def test_twelve_row_summaries_distinguish_fresh_and_stored_counts_on_both_splits(self):
        for split in ('validation','test'):
            result=a.aggregate(self.rows(),split=split)
            self.assertEqual(result['split'],split)
            self.assertEqual((result['candidate_fit_count'],result['baseline_reference_count'],
                result['stored_channel_reference_count'],result['comparison_model_count']),(6,3,3,12))
            self.assertEqual(len(result['grouped']),4)
            self.assertEqual(len(result['seed_differences']),15)
            self.assertEqual(result['stored_channel_added_parameters'],3072)
            self.assertEqual(result['candidate_added_parameters'],1024)

    def test_fabricated_location_evaluator_routes_and_six_metrics_without_data_or_checkpoints(self):
        from paper.scripts import run_titantpp_nonlinear_location_cnn as engine
        definition=engine.fabricated_data();definition['loader']['max_seq_len']=84
        contract={'sources':{'components':{}},'datasets':{a.DATASETS[0]:{'definition':definition}},
                  'contract_sha256':'a'*64}
        with patch.object(e,'verify_contract',return_value=contract),patch.object(e,'verify_source',return_value=a.PROJECT), \
             patch.object(e,'isolate_source'),patch.object(e,'verify_imports'):
            result=e.qualify_synthetic(self.root)
        self.assertEqual(result['status'],'PASS');self.assertTrue(result['six_metrics_hand_calculation_checked'])
        self.assertEqual(set(result['arms']),{a.QUERY,a.VALUE})
        self.assertEqual(result['checkpoint_reads'],0);self.assertFalse(result['real_data_loaded'])
        self.assertFalse(result['native_gpu_replay_verified'])
        for counts in result['dataset_parameter_counts'][a.DATASETS[0]].values():
            self.assertEqual(counts['actual_active_parameters'],80451)
            self.assertEqual(counts['actual_registered_parameters'],105183)



if __name__ == '__main__':
    unittest.main()
