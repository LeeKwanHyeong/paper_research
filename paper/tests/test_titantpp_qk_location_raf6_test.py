"""Offline RAF6 orchestration contracts; no SSH and no real Test inference."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import advance_titantpp_qk_location_raf6_test as a
from paper.scripts import evaluate_titantpp_qk_location_frozen as e


class RAF6EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='qk-only-raf3-eval-')
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

    def test_gate_requires_all_three_explicit_worker_exit0(self):
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
        gate = {'schema': 'titantpp_qk_location_raf6_validation_gate_v1', 'cohort': e.COHORT,
                'status': 'PASS', 'contract_sha256': c['contract_sha256'], 'job_count': 6,
                'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': 11.}
        path = self.root / 'validation_gate.json'; a.write(path, gate)
        with patch.object(e, 'verify_validation_receipt', side_effect=lambda r, c, j: r):
            e.verify_global_validation_gate(self.root, c)
            for change in ({'job_count': 2}, {'validation_worker_exits': {}}, {'created_unix': 9.}):
                path.write_text(json.dumps({**gate, **change}))
                with self.assertRaises(ValueError):
                    e.verify_global_validation_gate(self.root, c)
            bad = self.root / 'results/0/validation_worker_exit.json'
            record = a.read(bad); record['exit_code'] = 1; bad.write_text(json.dumps(record))
            gate['validation_worker_exits']['0'] = a.sha(bad); path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, 'actual worker exit0'):
                e.verify_global_validation_gate(self.root, c)

    def test_permit_rejects_wrong_host_stage_and_expired_or_extended_budget(self):
        p = {'schema': 'titantpp_qk_location_raf6_evaluation_permit_v1', 'authorized': True, 'new_training': False,
             'frozen_selection': True, 'stage': 'validation', 'host': '5080', 'job_ids': ['job'],
             'contract_sha256': 'a' * 64, 'evaluator_sha256': e.sha(e.__file__),
             'not_before_unix': 100., 'deadline_unix': 200.}
        job = {'id': 'job', 'evaluation_host': '5080'}
        c = {'contract_sha256': 'a' * 64, 'jobs': [job], 'absolute_host_deadline_unix':225.}
        def check(value, stage='validation'):
            path = self.root / 'permit.json'; path.write_text(json.dumps(value))
            with patch.object(e, 'verify_partial_metadata', return_value={'remote_roots': {'5080': str(self.root)}}):
                return e.verify_permit(self.root, c, job, stage, 'permit.json', a.sha(path), now=150.)
        check(p)
        for change in ({'host': '5090'}, {'new_training': True}, {'job_ids': ['other']},
                       {'deadline_unix': 120.}, {'deadline_unix': 50000.}, {'deadline_unix':226.}):
            with self.assertRaises(ValueError):
                check({**p, **change})
        with self.assertRaises(ValueError):
            check(p, 'test')

    def rows(self):
        rows = []
        for d in a.DATASETS:
            for arm, shift in ((a.CNN, 0.), (a.Q_ONLY, 2.), (a.K_ONLY, 3.)):
                for seed, x in zip(a.SEEDS, (1., 2., 3.)):
                    metrics = {'qty_rmse': x + shift, 'qty_mae': (x + shift) / 2., 'time_nll': x + shift}
                    rows.append({'dataset': d, 'arm': arm, 'seed': seed, 'metrics': {**metrics, 'tail': metrics.copy()},
                                 'target_identity_sha256': 'a', 'truth_sha256': 'b', 'time_sha256': 'c'})
        return rows

    def test_aggregation_sample_sd_and_unfavorable_seed_differences(self):
        r = a.aggregate(self.rows())
        cnn = r['grouped'][0]['metrics']['whole_qty_rmse']
        self.assertEqual(cnn, {'mean': 2., 'sample_sd': 1.})
        self.assertEqual(len(r['seed_differences']), 9)
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
        self.assertEqual(e.expected_jobs(), {('raf_spare_parts', arm, seed) for arm in a.ARMS for seed in (42,52,62)})

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

    def test_exact_training3_contract_rejects_a_fresh_cnn_job(self):
        jobs=[{'id':f'{d}__{s}__{arm}'} for d in a.DATASETS for s in a.SEEDS for arm in a.ARMS]
        for changes in (jobs+[{'id':'raf_spare_parts__42__titantpp_cnn'}],jobs[:-1]):
            with patch.object(a,'TRAIN_INTEGRITY_SHA','a'*64),patch.object(a,'sha',return_value='a'*64), \
                 patch.object(a,'read',side_effect=[{'files':{},'contract_raw_sha256':'a'*64},{'jobs':changes,'science_spec_sha256':a.SCIENCE_SHA,'runner_modules':{arm:'paper.scripts.run_titantpp_qk_location_cnn' for arm in a.ARMS}}]):
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


    def test_cnn_references_remain_readonly_and_old_candidates_are_excluded(self):
        refs=[{'seed':seed,'arm':a.LEGACY_CNN,'comparison_arm':a.CNN,'dataset':a.DATASETS[0],
            'role':'historical_immutable_reference','retrained':False,'reevaluated':False,
            'receipt_sha256':'a'*64,'checkpoint_sha256':'b'*64} for seed in a.SEEDS]
        c={'baseline_references':refs,'candidate_selection_rule':a.SELECTION_RULE}
        self.assertTrue(e.verify_reused_reference_scope(c))
        for change in ({'retrained':True},{'reevaluated':True},{'receipt_sha256':'x'},
                       {'arm':a.Q_ONLY},{'seed':99}):
            bad=deepcopy(c);bad['baseline_references'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):e.verify_reused_reference_scope(bad)
        with self.assertRaises(ValueError):e.verify_reused_reference_scope({**c,'stored_channel_references':[{}]})
        with self.assertRaises(ValueError):e.verify_reused_reference_scope({**c,'candidate_selection_rule':'test_rmse'})
        self.assertEqual(e.ARMS,a.ARMS)
        self.assertNotIn(a.CNN,e.ARMS)

    def test_sha_and_hidden_width_constants_are_preserved(self):
        self.assertTrue(e.digest_valid('a'*64));self.assertFalse(e.digest_valid('a'*34))
        self.assertIn("model['hidden_dim'] == 64",Path(e.__file__).read_text())

    def test_fabricated_difference_evaluator_routes_and_six_metrics_without_data_or_checkpoints(self):
        from paper.scripts import run_titantpp_qk_location_cnn as engine
        definition=engine.fabricated_data();definition['loader']['max_seq_len']=84
        contract={'sources':{'components':{}},'datasets':{a.DATASETS[0]:{'definition':definition}},
                  'contract_sha256':'a'*64}
        with patch.object(e,'verify_contract',return_value=contract),patch.object(e,'verify_source',return_value=a.PROJECT), \
             patch.object(e,'isolate_source'),patch.object(e,'verify_imports'):
            result=e.qualify_synthetic(self.root)
        self.assertEqual(result['status'],'PASS');self.assertTrue(result['six_metrics_hand_calculation_checked'])
        self.assertEqual(set(result['arms']),set(a.ARMS))
        self.assertTrue(result['nonzero_qk_kernel_checked'])
        self.assertEqual(result['checkpoint_reads'],0);self.assertFalse(result['real_data_loaded'])
        self.assertFalse(result['native_gpu_replay_verified'])
        for counts in result['dataset_parameter_counts'][a.DATASETS[0]].values():
            self.assertEqual(counts['actual_active_parameters'],79043)
            self.assertEqual(counts['actual_registered_parameters'],103775)





def comparison_fixture():
    from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
    pop={'count':5,'target_identity_sha256':'a'*64,'truth_sha256':'b'*64,'time_sha256':'c'*64}
    refs=[]
    for model,value in (('historical_CNN',4.),('external_one',3.)):
        for split in c.SPLITS:
            for seed in c.SEEDS:
                refs.append({'model':model,'split':split,'seed':seed,'selected_epoch':5,
                    'checkpoint_sha256':str(seed)[0]*64,'metrics':{m:value for m in c.METRICS}})
    registry={'status':'verified','dataset':'raf_spare_parts','seeds':list(c.SEEDS),'rows':refs,
        'expected_population':{s:deepcopy(pop) for s in c.SPLITS},
        'historical_cnn':{s:{m:c.stats([4.,4.,4.]) for m in c.METRICS} for s in c.SPLITS},
        'external_frontier':{s:{m:{'models':['external_one'],'mean':3.,'sample_sd':{'external_one':0.}}
            for m in c.METRICS} for s in c.SPLITS}}
    rows=[{'model':arm,'split':split,'seed':seed,'dataset':'raf_spare_parts','status':'complete',
        'validation_selected_checkpoint_verified':True,'selected_epoch':8,'checkpoint_sha256':str(seed)[0]*64,
        'metrics':{m:2. for m in c.METRICS},'count':5,'population':deepcopy(pop),
        'runtime':deepcopy(c.NATIVE_RUNTIME),'local_cnn_parameter_count':192,'original_local_cnn_parameter_count':576}
        for arm in c.ARMS for split in c.SPLITS for seed in c.SEEDS]
    return registry,rows


class LocationComparisonTests(unittest.TestCase):
    def test_four_strict_checks_each_candidate_and_no_baseline_rerun(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        registry,rows=comparison_fixture();r=c.compare_historical_rows(registry,rows)
        self.assertEqual(r['user_performance_gate'],{arm:True for arm in c.ARMS})
        self.assertEqual(r['user_performance_check_count_per_candidate'],4)
        self.assertEqual(r['user_performance_check_count'],8)
        self.assertEqual(r['candidate_selection']['selected_candidate'],c.K_ONLY)
        self.assertTrue(r['candidate_selection']['test_used_for_eligibility'])
        self.assertFalse(r['candidate_selection']['test_used_for_checkpoint_selection'])
        self.assertFalse(r['historical_cnn_retrained']);self.assertFalse(r['historical_cnn_reevaluated'])
        self.assertEqual(r['fresh_same_runtime_cnn_fit_count'],0)
        self.assertFalse(r['test_independent']);self.assertFalse(r['causal_effect_established'])
        self.assertIn('576->192',r['intervention_limitation'])

    def test_each_equal_or_worse_check_only_disqualifies_affected_arm(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        for arm in c.ARMS:
            for split in c.SPLITS:
                for metric in c.METRICS:
                    for value in (3.,5.):
                        with self.subTest(arm=arm,split=split,metric=metric,value=value):
                            registry,rows=comparison_fixture()
                            for row in rows:
                                if row['model']==arm and row['split']==split:row['metrics'][metric]=value
                            result=c.compare_historical_rows(registry,rows)
                            self.assertFalse(result['user_performance_gate'][arm])
                            other=next(x for x in c.ARMS if x!=arm)
                            self.assertEqual(result['candidate_selection']['selected_candidate'],other)

    def test_selection_rmse_then_mae_then_arm_never_test_ranking(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        for rmse,mae in ((1.9,2.5),(2.,1.9)):
            registry,rows=comparison_fixture()
            for r in rows:
                if r['model']==c.Q_ONLY:
                    r['metrics']={'qty_rmse':rmse,'qty_mae':mae} if r['split']=='validation' else {'qty_rmse':2.9,'qty_mae':2.9}
            out=c.compare_historical_rows(registry,rows)
            self.assertEqual(out['candidate_selection']['selected_candidate'],c.Q_ONLY)
        for r in rows:r['metrics']={m:5. for m in c.METRICS}
        out=c.compare_historical_rows(registry,rows)
        self.assertIsNone(out['candidate_selection']['selected_candidate'])

    def test_adverse_seed_sd_and_key_minus_query_retained(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        registry,rows=comparison_fixture();rows[0]['metrics']['qty_rmse']=5.
        result=c.compare_historical_rows(registry,rows)
        row=next(r for r in result['seed_differences'] if r['candidate_model']==c.Q_ONLY and r['split']=='validation' and r['seed']==42 and r['metric']=='qty_rmse')
        delta=row['differences']['historical_CNN']
        self.assertEqual(delta['absolute_candidate_minus_reference'],1.)
        self.assertEqual(delta['relative_percent_candidate_minus_reference'],25.)
        self.assertEqual(result['summaries'][c.Q_ONLY]['validation']['qty_rmse']['candidate'],c.stats([5.,2.,2.]))
        contrast=next(r for r in result['primary_contrast_seed_differences'] if r['split']=='validation' and r['seed']==42 and r['metric']=='qty_rmse')
        self.assertEqual(contrast['absolute_candidate_minus_reference'],-3.)
        self.assertEqual(contrast['relative_percent_candidate_minus_reference'],-60.)

    def test_duplicate_missing_population_and_checkpoint_rejected(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        registry,rows=comparison_fixture();cases=[rows[:-1],rows[:-1]+[rows[0]]]
        for key,value in (('selected_epoch',9),('checkpoint_sha256','d'*64)):
            bad=deepcopy(rows);bad[-1][key]=value;cases.append(bad)
        bad=deepcopy(rows);bad[0]['population']['truth_sha256']='d'*64;cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):c.compare_historical_rows(registry,bad)

    def test_old_arms_runtime_and_parameter_mismatch_rejected(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        for change in ({'model':'titantpp_cnn'},{'model':'titantpp_cnn_qk_only_v1'},
                       {'local_cnn_parameter_count':384},{'original_local_cnn_parameter_count':192},
                       {'runtime':{**c.NATIVE_RUNTIME,'cudnn':91900}}):
            registry,rows=comparison_fixture();rows[0].update(deepcopy(change))
            with self.subTest(change=change),self.assertRaises(ValueError):c.compare_historical_rows(registry,rows)

    def test_registry_frontier_aggregate_or_selection_corruption_rejected(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        registry,rows=comparison_fixture();mutations=[]
        bad=deepcopy(registry);bad['external_frontier']['test']['qty_mae']['mean']=1.;mutations.append(bad)
        bad=deepcopy(registry);bad['external_frontier']['test']['qty_mae']['models']*=2;mutations.append(bad)
        bad=deepcopy(registry);bad['historical_cnn']['validation']['qty_rmse']['sample_sd']=1.;mutations.append(bad)
        bad=deepcopy(registry);bad['rows'][3]['selected_epoch']=7;mutations.append(bad)
        bad=deepcopy(registry);bad['rows'][-1]['checkpoint_sha256']='d'*64;mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):c.compare_historical_rows(bad,rows)

    def test_zero_reference_relative_delta_is_null(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as c
        registry,rows=comparison_fixture()
        for ref in registry['rows']:ref['metrics']={m:0. for m in c.METRICS}
        registry['historical_cnn']={s:{m:c.stats([0.,0.,0.]) for m in c.METRICS} for s in c.SPLITS}
        registry['external_frontier']={s:{m:{'models':['external_one'],'mean':0.,'sample_sd':{'external_one':0.}}
            for m in c.METRICS} for s in c.SPLITS}
        r=c.compare_historical_rows(registry,rows)
        self.assertFalse(any(r['user_performance_gate'].values()))
        self.assertTrue(all(d['relative_percent_candidate_minus_reference'] is None
            for row in r['seed_differences'] for d in row['differences'].values()))

    def test_report_keeps_all_three_contrasts_tail_time_and_relative_deltas(self):
        fixture=RAF6EvaluationTests().rows()
        for row in fixture:row['selected_epoch']=3
        result=a.aggregate(fixture)
        self.assertEqual(len(result['seed_differences']),9)
        self.assertEqual({(x['candidate'],x['baseline']) for x in result['seed_differences']},
                         {(a.Q_ONLY,a.CNN),(a.K_ONLY,a.CNN),(a.K_ONLY,a.Q_ONLY)})
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            path=Path(folder)/'report.md';a.write_summary_markdown(result,path);text=path.read_text()
        self.assertIn('큰수량 TimeNLL',text);self.assertIn('상대차(%)',text)
        self.assertIn('576→192',text);self.assertIn('인과관계',text)


class QKLocationAdditionalGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='qk-eval-guards-')
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def final_fixture(self):
        return {'status':'complete','candidate_test_count':6,'seed_count':3,'baseline_reference_count':3,
            'comparison_model_count':9,'candidate_selection':{'selection_rule':a.SELECTION_RULE},
            'validation_replay_count':6,'actual_native_worker_exit0_count':12,'actual_pipeline_exit0_count':2,
            'held_out_test_evaluated':True,'external_comparison_complete':True,'historical_cnn_retrained':False,
            'historical_cnn_reevaluated':False,'files':{},'files_sha256':a.canonical({})}

    def test_forged_complete_empty_files_never_reaches_evaluator_or_remote(self):
        a.write(self.root/'final_receipt.json',self.final_fixture())
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'REPORT',self.root/'report'), \
             patch.object(a,'verify_bootstrap',return_value={}),patch.object(a,'evaluator') as evaluate, \
             patch.object(a,'transport') as remote:
            with self.assertRaisesRegex(ValueError,'file coverage'): a.advance('a'*64)
        evaluate.assert_not_called();remote.assert_not_called()

    def test_forged_complete_only_report_files_rejected(self):
        report=self.root/'report'; report.mkdir(); r=self.final_fixture()
        for n in ('THREE_SEED_VALIDATION.json','THREE_SEED_VALIDATION.md','THREE_SEED_TEST.json',
                  'THREE_SEED_TEST.md','EXTERNAL_COMPARISON.json','EXTERNAL_COMPARISON.md'):
            p=report/n;p.write_text('{}');r['files'][str(p)]=a.sha(p)
        r['files_sha256']=a.canonical(r['files']);a.write(self.root/'final_receipt.json',r)
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'REPORT',report):
            with self.assertRaises((FileNotFoundError,ValueError)):a.verify_complete()

    def test_source141_or142_rejected_before_source_file_reads(self):
        for count in (141,143,144,146):
            source={'files':{str(i):'a'*64 for i in range(count)},'revision':'revision','root':'source'}
            source['closure']=e.canonical(source['files'])
            with self.subTest(count=count),patch.object(e,'SOURCE_CLOSURE',source['closure']), \
                 patch.object(e,'SOURCE_REVISION','revision'),patch.object(e,'sha') as hashes:
                with self.assertRaisesRegex(ValueError,'exactly145'):e.verify_source(self.root,source)
                hashes.assert_not_called()

    def test_qk_structure_no_v_dummy_and_both_local_kernels(self):
        from paper.scripts import run_titantpp_qk_location_cnn as engine
        import torch
        model,_=engine.build_model(engine.fabricated_data(),a.Q_ONLY)
        found=e.verify_qk_local_structure(model)
        self.assertEqual(found['active_local_cnn_parameters'],192)
        self.assertEqual(found['active_value_cnn_parameters'],0)
        module=model.encoder.layers[0].attn
        module.register_parameter('causal_v_kernel',torch.nn.Parameter(torch.zeros(3,64)))
        with self.assertRaisesRegex(ValueError,'physical kernel'):e.verify_qk_local_structure(model)

    def test_receipt_parameter_identity_cannot_be_inferred_from_arm_label(self):
        counts={'actual_registered_parameters':103775,'actual_active_parameters_before_evaluation_freeze':79043,
            'maximum_sequence_length':84,'active_local_cnn_parameters':192,'active_value_cnn_parameters':0,
            'original_value_kernel_absent':True,'value_operation':'unmixed_common_value_projection',
            'qk_location_arm':a.Q_ONLY,'retained_projection':'q','active_query_cnn_parameters':192,
            'active_key_cnn_parameters':0,'qk_location_parameter_delta':-384}
        self.assertTrue(e.verify_physical_counts({'arm':a.Q_ONLY,'physical_parameter_counts':counts}))
        for change in ({'actual_registered_parameters':104095},{'active_local_cnn_parameters':512},
                       {'active_value_cnn_parameters':128},{'original_value_kernel_absent':False},
                       {'value_operation':'runtime_zero_mask'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                e.verify_physical_counts({'arm':a.Q_ONLY,'physical_parameter_counts':{**counts,**change}})

    def archive_fixture(self, change=None):
        import hashlib
        c={'contract_sha256':'a'*64,'jobs':[{'id':str(i),'evaluation_host':'5090'} for i in range(6)]}
        a.write(self.root/'validation_gate.json',{'synthetic':True})
        for split in ('validation','test'):
            common={'contract_sha256':c['contract_sha256'],'host':'5090','split':split}
            records={split+'_pipeline_actual_exit.json':{**common,'actual_child_exit_observed':True,'exit_code':0},
                split+'_pipeline_complete.json':{**common,'status':'complete','completed':[str(i) for i in range(6)]},
                split+'_pipeline.claim':{**common,'pid':100},split+'_supervision.claim':{**common,'pid':99}}
            for j in c['jobs']:
                q='validation_replay' if split=='validation' else 'test_metrics'
                records['results/'+j['id']+'/'+q+'.json']={'validation_gate_sha256':a.sha(self.root/'validation_gate.json')}
                records['results/'+j['id']+'/'+q+'_claim.json']={}
                records['results/'+j['id']+'/'+split+'_worker_exit.json']={**common,'job_id':j['id'],
                    'actual_child_exit_observed':True,'exit_code':0,'pid':int(j['id'])+10,'started_unix':1.}
                records['worker_launches/'+split+'_'+j['id']+'.json']={**common,'job_id':j['id'],
                    'pid':int(j['id'])+10,'ppid':100,'started_unix':1.}
                records['logs/'+split+'_'+j['id']+'.log']={}
            if change:change(split,records)
            raw={n:json.dumps(v).encode() for n,v in records.items()}
            manifest={n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
            for name,value in raw.items():
                prefix='' if name.startswith(('results/','worker_launches/','logs/')) else 'native/5090/'
                path=self.root/(prefix+name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
            a.write(self.root/'native/5090'/ (split+'_result_manifest.json'),manifest)
            raw['result_manifest.json']=json.dumps(manifest).encode()
            with tarfile.open(self.root/('5090_'+split+'_results.tar.gz'),'w:gz') as archive:
                for name,value in raw.items():
                    item=tarfile.TarInfo(name);item.size=len(value);archive.addfile(item,io.BytesIO(value))
        return c

    def test_completed_archive_revalidates_worker_exits_and_launch_identity(self):
        mutations=(None,{'exit_code':1},{'actual_child_exit_observed':False},{'job_id':'foreign'},{'pid':999})
        for mutation in mutations:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
                self.root=Path(folder)
                def change(split,records):
                    if mutation:records['results/0/'+split+'_worker_exit.json'].update(mutation)
                c=self.archive_fixture(change)
                ev=Mock()
                with patch.object(a,'BUNDLE',self.root),patch.object(a,'evaluator',return_value=ev), \
                     patch.object(a,'check_receipt',side_effect=lambda r,*args:r):
                    if mutation:
                        with self.assertRaises(ValueError):a.verify_native_archives(c)
                    else:self.assertTrue(a.verify_native_archives(c))

    def test_completed_archive_requires_local_bytes_match_original_archive(self):
        c=self.archive_fixture()
        (self.root/'results/0/test_metrics.json').write_text('{"changed":true}')
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'evaluator',return_value=Mock()), \
             patch.object(a,'check_receipt',side_effect=lambda r,*args:r):
            with self.assertRaisesRegex(ValueError,'archive/local'):a.verify_native_archives(c)

    def test_native_normalizer_rejects_label_without_qk_physical_proof(self):
        from paper.scripts import prepare_titantpp_qk_location_raf6_test as compare
        with self.assertRaisesRegex(ValueError,'physical parameter'):
            compare.normalized_native_row({'status':'complete','native_runtime_checked':True,'arm':a.Q_ONLY},'5090')



class TwoLocationGuardTests(unittest.TestCase):
    def test_physical_route_cannot_be_relabelled_and_missing_opposite_required(self):
        from paper.scripts import run_titantpp_qk_location_cnn as engine
        import torch
        for arm in a.ARMS:
            with self.subTest(arm=arm):
                data=engine.fabricated_data();data['loader']['max_seq_len']=84
                model,_=engine.build_model(data,arm)
                counts={'actual_registered_parameters':103775,'actual_active_parameters_before_evaluation_freeze':79043,
                    'maximum_sequence_length':84,**e.verify_qk_local_structure(model)}
                self.assertTrue(e.verify_physical_counts({'arm':arm,'physical_parameter_counts':counts}))
                other=next(x for x in a.ARMS if x!=arm)
                with self.assertRaises(ValueError):e.verify_physical_counts({'arm':other,'physical_parameter_counts':counts})
                module=model.encoder.layers[0].attn
                name='causal_k_kernel' if arm==a.Q_ONLY else 'causal_q_kernel'
                module.register_parameter(name,torch.nn.Parameter(torch.zeros(3,64)))
                with self.assertRaises(ValueError):e.verify_qk_local_structure(model)

    def test_negative_quantity_metric_rejected_but_signed_time_metric_allowed(self):
        good={'count':1,'qty_rmse':1.,'qty_mae':1.,'time_nll':-.1,
              'tail':{'count':1,'qty_rmse':1.,'qty_mae':1.,'time_nll':-.2}}
        e.compare_validation_metrics(good,good)
        for group in (None,'tail'):
            for key in ('qty_rmse','qty_mae'):
                bad=deepcopy(good);(bad if group is None else bad[group])[key]=-1.
                with self.subTest(group=group,key=key),self.assertRaises(ValueError):e.compare_validation_metrics(bad,bad)

    def test_scientific_identity_rejected_before_collection_or_source_loading(self):
        for changes in ({'science_spec_sha256':'x'},{'runner_modules':{}}):
            c={'science_spec_sha256':a.SCIENCE_SHA,'runner_modules':{arm:'paper.scripts.run_titantpp_qk_location_cnn' for arm in a.ARMS},**changes}
            with patch.object(a,'TRAIN_INTEGRITY_SHA','a'*64),patch.object(a,'sha',return_value='a'*64), \
                 patch.object(a,'read',side_effect=[{'files':{},'contract_raw_sha256':'a'*64},c]),self.assertRaises(ValueError):
                a.training_contract()

    def test_six_metric_aggregate_rejects_stored_qk_only_extra_row(self):
        rows=RAF6EvaluationTests().rows()
        foreign=deepcopy(rows[0]);foreign['arm']='titantpp_cnn_qk_only_v1'
        with self.assertRaises(ValueError):a.aggregate(rows+[foreign])

    def test_stored_qk_only_reference_is_not_a_formal_comparison(self):
        refs=[{'seed':seed,'arm':a.LEGACY_CNN,'comparison_arm':a.CNN,'dataset':a.DATASETS[0],
            'role':'historical_immutable_reference','retrained':False,'reevaluated':False,
            'receipt_sha256':'a'*64,'checkpoint_sha256':'b'*64} for seed in a.SEEDS]
        with self.assertRaises(ValueError):e.verify_reused_reference_scope({'baseline_references':refs,
            'candidate_selection_rule':a.SELECTION_RULE,'stored_qk_only_references':[{}]})

    def test_native_pipeline_is_sequential_and_all_six_jobs_exit_zero(self):
        from paper.scripts import run_titantpp_qk_location_test_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root=Path(folder);jobs=[{'id':str(i),'evaluation_host':'5090'} for i in range(6)]
            c={'jobs':jobs,'contract_sha256':'a'*64,'runtimes':{'5090':{'gpu_uuid':'GPU-fixed','python':'/python'}},
               'evaluator_integrity':{'files':{'operation/pipeline.py':'hash'}}}
            ev=Mock();ev.verify_contract.return_value=c;ev.verify_partial_metadata.return_value={'remote_roots':{'5090':str(root)}}
            ev.require.side_effect=a.require;ev.sha.return_value='hash';ev.verify_permit.return_value={'deadline_unix':1e20}
            order=[];counter=[0]
            def popen(argv,**kwargs):
                counter[0]+=1;pid=counter[0]+100
                self.assertEqual(len(order),2*(counter[0]-1))
                order.append(('launch',pid))
                child=Mock(pid=pid);child.wait.side_effect=lambda:(order.append(('exit',pid)) or 0)
                return child
            with patch.object(pipeline,'load_evaluator',return_value=ev), \
                 patch.object(pipeline.subprocess,'run',side_effect=[Mock(stdout='GPU-fixed\n'),Mock(stdout='')]*6):
                self.assertEqual(pipeline.run(root,'5090','validation',popen),0)
            self.assertEqual(counter[0],6)
            self.assertEqual(a.read(root/'validation_pipeline_complete.json')['completed'],[str(i) for i in range(6)])
            for j in jobs:
                record=a.read(root/'results'/j['id']/'validation_worker_exit.json')
                self.assertIs(record['actual_child_exit_observed'],True);self.assertEqual(record['exit_code'],0)
            ev.verify_global_validation_gate.assert_not_called()

    def test_test_pipeline_gate_failure_precedes_any_child_or_claim(self):
        from paper.scripts import run_titantpp_qk_location_test_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root=Path(folder);ev=Mock();ev.require.side_effect=a.require;ev.sha.return_value='h'
            ev.verify_contract.return_value={'jobs':[],'runtimes':{'5090':{}},'evaluator_integrity':{'files':{'operation/pipeline.py':'h'}}}
            ev.verify_partial_metadata.return_value={'remote_roots':{'5090':str(root)}}
            ev.verify_global_validation_gate.side_effect=ValueError('six Validation required')
            child=Mock()
            with patch.object(pipeline,'load_evaluator',return_value=ev),self.assertRaises(ValueError):
                pipeline.run(root,'5090','test',child)
            child.assert_not_called();self.assertFalse((root/'test_pipeline.claim').exists())


if __name__ == '__main__':
    unittest.main()
