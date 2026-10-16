"""Negative fixed-scope guards; synthetic receipts, no native inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paper.scripts import evaluate_titantpp_components_taxi_raf24_frozen as e
from paper.tests.test_titantpp_components_taxi_raf24_preparation import TaxiRAF24Fixture
from paper.tests import test_titantpp_components_frozen_evaluator as fixtures

class TaxiRAF24EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(self.temp.cleanup)
        self.f=TaxiRAF24Fixture(self.temp.name);self.root=self.f.root
        self.expected,self.runtime=fixtures.ComponentEvaluatorContracts.host_runtime(self,'5080')
        self.f.runtimes={'5080':self.expected};self.c=self.f.build();self.f.write_json('contract.json',self.c)

    def seal(self,c):
        c=deepcopy(c);c.pop('contract_sha256',None);c['contract_sha256']=e.canonical(c)
        self.f.write_json('contract.json',c);return c

    def test_unknown_job_or_missing_permit_precedes_any_science_access(self):
        with self.f.pins(),patch.object(e,'verify_source',side_effect=AssertionError('Science was touched')):
            with self.assertRaisesRegex(ValueError,'Unknown'):e.worker(self.root,'foreign','test','cuda')
            with self.assertRaisesRegex(ValueError,'permit required'):e.worker(self.root,self.c['jobs'][0]['id'],'test','cuda')

    def test_foreign_schema_dataset_runtime_or_seed_rejected(self):
        for key,value in (('schema','components36'),('cohort','components_inter62_source131'),
                          ('expected_seeds',[42,52]),('evaluation_scope','test_seed62_followup')):
            self.seal({**self.c,key:value})
            with self.f.pins(),self.assertRaises(ValueError):e.verify_contract(self.root)
        c=deepcopy(self.c);c['runtimes']['5090']=self.expected;self.seal(c)
        with self.f.pins(),self.assertRaises(ValueError):e.verify_contract(self.root)

    def test_permit_stage_exact_scope_deadline_and_no_training(self):
        job=self.c['jobs'][0]
        p={'schema':'titantpp_components_taxi_raf24_evaluation_permit_v1','authorized':True,
           'new_training':False,'frozen_selection':True,'stage':'validation','contract_sha256':self.c['contract_sha256'],
           'evaluator_sha256':e.sha(e.__file__),'host':'5080','job_ids':[job['id']],
           'not_before_unix':100.,'deadline_unix':200.}
        def check(value,stage='validation',now=150.):
            self.f.write_json('permit.json',value)
            with self.f.pins():return e.verify_permit(self.root,self.c,job,stage,'permit.json',e.sha(self.root/'permit.json'),now=now)
        check(p)
        for change in ({'new_training':True},{'job_ids':['foreign']},{'host':'5090'},
                       {'deadline_unix':99.},{'deadline_unix':50000.},{'schema':'inter62permit'}):
            with self.assertRaises(ValueError):check({**p,**change})
        with self.assertRaises(ValueError):check(p,'test')

    def gate(self):
        receipts,exits={},{}
        for j in self.c['jobs']:
            p='results/'+j['id']+'/'
            receipt={'schema':e.KIND,'status':'complete','split':'validation','contract_sha256':self.c['contract_sha256'],
              **{k:j[k] for k in ('id','cohort','study','dataset','arm','seed','selected_epoch','initial_state_sha256')},
              'checkpoint_sha256':j['checkpoint']['sha256'],'state_sha256':j['checkpoint_state_sha256'],
              'source_closure':self.f.source_closure,'retrained':False,'reselected':False,'raw_predictions_written':False,
              'full_population':True,'validation_match':'PASS','native_runtime_checked':True,
              'qualification':dict.fromkeys(e.QUALIFICATION_KEYS,True),'metrics':j['validation_reference'],
              **{k:self.c['datasets'][j['dataset']]['validation_population'][k] for k in ('target_identity_sha256','truth_sha256','time_sha256')},
              'runtime':self.runtime,'runtime_after':self.runtime,'completed_unix':100.}
            receipts[j['id']]=self.f.write_json(p+'validation_replay.json',receipt)
            exits[j['id']]=self.f.write_json(p+'validation_worker_exit.json',{
              'job_id':j['id'],'contract_sha256':self.c['contract_sha256'],'exit_code':0,
              'actual_child_exit_observed':True,'completed_unix':100.})
        g={'schema':'titantpp_components_taxi_raf24_validation_gate_v1','cohort':e.COHORT,
            'status':'PASS','contract_sha256':self.c['contract_sha256'],'job_count':24,
            'validation_receipts':receipts,'validation_worker_exits':exits,'created_unix':150.}
        self.f.write_json('validation_gate.json',g);return g

    def test_exact24_validation_gate_qualifies_each_model_exit_and_population(self):
        gate=self.gate()
        with self.f.pins():self.assertEqual(e.verify_global_validation_gate(self.root,self.c)['job_count'],24)
        for change in ({'job_count':4},{'job_count':36},{'schema':'inter62gate'},
                       {'validation_worker_exits':{}},{'validation_receipts':{}}):
            self.f.write_json('validation_gate.json',{**gate,**change})
            with self.f.pins(),self.assertRaises(ValueError):e.verify_global_validation_gate(self.root,self.c)

    def test_modified_receipt_or_nonzero_actual_exit_cannot_release_test(self):
        gate=self.gate();j=self.c['jobs'][0];p='results/'+j['id']+'/'
        for key,value in (('exit_code',1),('actual_child_exit_observed',False),('completed_unix',151.)):
            receipt={'job_id':j['id'],'contract_sha256':self.c['contract_sha256'],'exit_code':0,
                     'actual_child_exit_observed':True,'completed_unix':100.,key:value}
            gate['validation_worker_exits'][j['id']]=self.f.write_json(p+'validation_worker_exit.json',receipt)
            self.f.write_json('validation_gate.json',gate)
            with self.f.pins(),self.assertRaises(ValueError):e.verify_global_validation_gate(self.root,self.c)

    def test_metric_contract_first_strict_finite_selection_and_tail_thresholds(self):
        self.assertEqual(e.strict_selected_epoch([{'epoch':1,'val_qty_rmse':3.}, {'epoch':2,'val_qty_rmse':3.}]),(1,3.))
        self.assertEqual(e.TAILS,{'yellow_trip_hourly':3449.,'raf_spare_parts':200.})
        ref=self.c['jobs'][0]['validation_reference'];e.compare_validation_metrics(ref,ref)
        for group in (None,'tail'):
            for key in e.METRICS:
                bad=deepcopy(ref);(bad if group is None else bad[group])[key]+=.01
                with self.assertRaises(ValueError):e.compare_validation_metrics(bad,ref)

if __name__=='__main__':unittest.main()
