"""Terminal exposure publication recovery: fabricated evidence, no network/GPU."""
from copy import deepcopy
import hashlib
import json
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper.scripts import observe_titantpp_retrieval_results as observer
from paper.scripts import finalize_titantpp_retrieval_results as wrapper
from paper.tests import test_titantpp_retrieval_repair_observer as original_fixtures


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frozen_raw_sha(value):
    """The immutable runtime writes exactly these bytes, not canonical JSON."""
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return hashlib.sha256(raw.encode()).hexdigest()


class TerminalExposureRecoveryTests(unittest.TestCase):
    def fixture(self, *, resumed=False, epochs=160):
        host = '5090' if resumed else '5080'
        arm = 'titantpp_retrieval_fixed' if resumed else 'titantpp_retrieval_mean'
        job = {'id':host+'__yellow_trip_hourly__42__'+arm,
            'configuration_id':'yellow_trip_hourly__42__'+arm,
            'host':host, 'dataset':'yellow_trip_hourly', 'seed':42, 'arm':arm}
        data = {'dataset_id':'yellow_trip_hourly', 'loader':{'batch_size':128},
            'inherited_data_identity':{'populations':{
                'train':{'target_count':38393}, 'validation':{'target_count':8268}}}}
        parent = {'datasets':[data]}
        c = {'parent_contract':parent, 'parent_contract_sha256':observer.PARENT_SHA,
            'jobs':[job], 'reused_jobs':[],
            'resume':{job['id']:{'last_epoch':148, 'best_epoch':120}} if resumed else {}}
        exposure = {'train':[{'epoch':i, 'split':'train', 'count':38393,
            'batches':300, 'batch_order_sha256':hashlib.sha256(('train'+str(i)).encode()).hexdigest()}
            for i in range(1, epochs+1)],
            'validation':[{'epoch':i, 'split':'validation', 'count':8268,
            'batches':65, 'batch_order_sha256':'a'*64} for i in range(1, epochs+2)]}
        prefix = deepcopy(exposure);prefix['validation'].pop()
        directory = 'runs/'+arm+'/count_only_log_regression/seed_42/'
        receipt_files = {name:hashlib.sha256(name.encode()).hexdigest() for name in
            ('last_epoch_state.pt','history.json','epoch_timing.json','train.log')}
        receipt_files['exposure.json'] = frozen_raw_sha(prefix)
        terminal_files = {directory+name:digest for name,digest in receipt_files.items()}
        terminal_files[directory+'exposure.json'] = frozen_raw_sha(exposure)
        row = {'job':job, 'started':True, 'SHA_verified':True,
            'checkpoint_receipt_SHA_verified':False, 'checkpoint_race':True,
            'actual_saved_epoch':epochs, 'actual_best_epoch':epochs-40,
            'checkpoint_receipt':{'epoch':epochs, 'contract_sha256':canonical(c),
                'parent_contract_sha256':observer.PARENT_SHA, 'files':receipt_files},
            'terminal':{'status':'complete','scientific_success':True,'job':deepcopy(job),
                'contract_sha256':canonical(c),'parent_contract_sha256':observer.PARENT_SHA,
                'evaluation_scope':'validation_only','held_out_test_evaluated':False,
                'files':terminal_files}, 'terminal_exposure':exposure}
        return c,row,directory

    def reseal_final(self,row,directory):
        row['terminal']['files'][directory+'exposure.json'] = frozen_raw_sha(row['terminal_exposure'])

    def reseal_both(self,row,directory):
        self.reseal_final(row,directory)
        prefix=deepcopy(row['terminal_exposure']);prefix['validation']=prefix['validation'][:-1]
        row['checkpoint_receipt']['files']['exposure.json'] = frozen_raw_sha(prefix)

    def test_fresh_completed_fit_accepts_exact_one_selected_validation_append(self):
        for epochs in (40,120,160,300):
            with self.subTest(epochs=epochs):
                c,row,_=self.fixture(epochs=epochs)
                self.assertIs(observer.terminal_exposure_bridge(c,row),True)

    def test_resumed_fit_accepts_same_bridge_without_dropping_prefix(self):
        c,row,_=self.fixture(resumed=True)
        self.assertIs(observer.terminal_exposure_bridge(c,row),True)
        self.assertEqual(len(row['terminal_exposure']['train']),160)
        self.assertEqual(len(row['terminal_exposure']['validation']),161)

    def test_unchanged_exposure_is_not_a_publication_transition(self):
        c,row,directory=self.fixture()
        row['checkpoint_receipt']['files']['exposure.json']=row['terminal']['files'][directory+'exposure.json']
        self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_train_mutation_cannot_be_hidden_by_resealing_terminal(self):
        c,row,directory=self.fixture()
        row['terminal_exposure']['train'][0]['batch_order_sha256']='b'*64
        self.reseal_final(row,directory)
        self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_extra_or_missing_final_validation_rejected_even_if_resealed(self):
        for change in ('extra','missing'):
            with self.subTest(change=change):
                c,row,directory=self.fixture()
                if change=='extra':
                    record=deepcopy(row['terminal_exposure']['validation'][-1]);record['epoch']+=1
                    row['terminal_exposure']['validation'].append(record)
                else:row['terminal_exposure']['validation'].pop()
                self.reseal_both(row,directory)
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_validation_order_change_or_sequence_reorder_rejected(self):
        for change in ('replay_order','prefix_order','reorder'):
            with self.subTest(change=change):
                c,row,directory=self.fixture();rows=row['terminal_exposure']['validation']
                if change=='reorder':rows[0],rows[1]=rows[1],rows[0]
                elif change=='replay_order':rows[-1]['batch_order_sha256']='b'*64
                else:rows[0]['batch_order_sha256']='b'*64
                self.reseal_both(row,directory)
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_wrong_population_batch_split_epoch_or_digest_rejected(self):
        mutations=[('train','count',38392),('validation','count',8267),
            ('train','batches',299),('validation','batches',64),
            ('validation','split','train'),('train','epoch',0),
            ('train','batch_order_sha256','z'*64),('validation','batch_order_sha256','a'*63)]
        for split,key,value in mutations:
            with self.subTest(split=split,key=key,value=value):
                c,row,directory=self.fixture();row['terminal_exposure'][split][0][key]=value
                self.reseal_both(row,directory)
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_only_exposure_file_can_change(self):
        for name in ('last_epoch_state.pt','history.json','epoch_timing.json','train.log'):
            with self.subTest(name=name):
                c,row,directory=self.fixture();row['terminal']['files'][directory+name]='f'*64
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_missing_or_ambiguous_receipt_terminal_file_rejected(self):
        for change in ('missing_receipt','missing_terminal','duplicate_suffix'):
            with self.subTest(change=change):
                c,row,directory=self.fixture()
                if change=='missing_receipt':del row['checkpoint_receipt']['files']['history.json']
                elif change=='missing_terminal':del row['terminal']['files'][directory+'history.json']
                else:row['terminal']['files']['foreign/history.json']=row['checkpoint_receipt']['files']['history.json']
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_frozen_serialization_raw_SHA_is_required(self):
        for which in ('prefix','final'):
            with self.subTest(which=which):
                c,row,directory=self.fixture()
                if which=='prefix':
                    prefix=deepcopy(row['terminal_exposure']);prefix['validation'].pop()
                    row['checkpoint_receipt']['files']['exposure.json']=canonical(prefix)
                else:row['terminal']['files'][directory+'exposure.json']=canonical(row['terminal_exposure'])
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)

    def test_scope_identity_or_unverified_terminal_rejected(self):
        for change in ('scope','heldout','terminal_job','receipt_contract','terminal_contract',
                       'parent_contract','terminal_failure','unverified_SHA'):
            with self.subTest(change=change):
                c,row,_=self.fixture()
                if change=='scope':row['terminal']['evaluation_scope']='mixed'
                elif change=='heldout':row['terminal']['held_out_test_evaluated']=True
                elif change=='terminal_job':row['terminal']['job']['seed']=52
                elif change=='receipt_contract':row['checkpoint_receipt']['contract_sha256']='0'*64
                elif change=='terminal_contract':row['terminal']['contract_sha256']='0'*64
                elif change=='parent_contract':row['terminal']['parent_contract_sha256']='0'*64
                elif change=='terminal_failure':row['terminal']['scientific_success']=False
                else:row['SHA_verified']=False
                self.assertIs(observer.terminal_exposure_bridge(c,row),False)



class RecoveryAnalysisTests(unittest.TestCase):
    def fixture(self):
        c,s=original_fixtures.RepairObserverTests().terminal()
        _,bridge,_=TerminalExposureRecoveryTests().fixture(resumed=True)
        data=c['parent_contract']['datasets'][0]
        data['loader']={'batch_size':128}
        data['inherited_data_identity']['populations']['train']={'target_count':38393}
        c['parent_contract_sha256']=observer.PARENT_SHA
        row=s['rows'][0]
        row['terminal'].update(files=bridge['terminal']['files'],contract_sha256=canonical(c))
        row['checkpoint_receipt']=deepcopy(bridge['checkpoint_receipt'])
        row['checkpoint_receipt']['contract_sha256']=canonical(c)
        row['terminal_exposure']=bridge['terminal_exposure']
        row['checkpoint_receipt_SHA_verified']=False
        row['terminal_exposure_bridge_verified']=True
        row['last_checkpoint_file_sha256']=row['checkpoint_receipt']['files']['last_epoch_state.pt']
        row['claim']['contract_sha256']=canonical(c)
        s['contract_sha256']=canonical(c)
        return c,s

    def test_verified_append_bridge_restores_terminal_without_changing_snapshot(self):
        c,s=self.fixture();before=deepcopy(s)
        result=observer.analyze(c,s)
        self.assertTrue(result['terminal_verified'])
        self.assertEqual(result['counts']['complete'],1)
        self.assertEqual(result['rows'][0]['actual_saved_epoch'],160)
        self.assertEqual(s,before)

    def test_missing_saved_metadata_cannot_be_inferred_from_endpoint_or_receipt(self):
        c,s=self.fixture();del s['rows'][0]['actual_saved_epoch']
        result=observer.analyze(c,s)
        self.assertFalse(result['terminal_verified'])
        self.assertEqual(result['counts']['uncertain'],1)
        self.assertIsNone(result['rows'][0]['actual_saved_epoch'])

    def test_absent_tampered_bridge_or_metadata_is_uncertain(self):
        for change in ('bridge_absent','bridge_false','tampered_exposure','saved_epoch',
                       'best_epoch','checkpoint_SHA','receipt_epoch'):
            with self.subTest(change=change):
                c,s=self.fixture();r=s['rows'][0]
                if change=='bridge_absent':del r['terminal_exposure_bridge_verified']
                elif change=='bridge_false':r['terminal_exposure_bridge_verified']=False
                elif change=='tampered_exposure':r['terminal_exposure']['train'][0]['count']-=1
                elif change=='saved_epoch':r['actual_saved_epoch']=159
                elif change=='best_epoch':r['actual_best_epoch']=119
                elif change=='checkpoint_SHA':r['last_checkpoint_file_sha256']='0'*64
                else:r['checkpoint_receipt']['epoch']=159
                result=observer.analyze(c,s)
                self.assertFalse(result['terminal_verified'])
                self.assertEqual(result['counts']['uncertain'],1)
                self.assertIsNone(result['rows'][0]['actual_saved_epoch'])

    def test_live_owner_gpu_pid_or_nonzero_exit_blocks_bridge(self):
        for change in ('owner','gpu','exit','missing_exit'):
            with self.subTest(change=change):
                c,s=self.fixture()
                if change=='owner':s['processes']=['8 1 /runtime/python /owned/repair/operation/repair_runtime.py --mode dispatch']
                elif change=='gpu':s['gpu_pids']='GPU-own, 8, python'
                elif change=='exit':s['supervisor_exit']['returncode']=1
                else:del s['supervisor_exit']
                result=observer.analyze(c,s)
                self.assertFalse(result['terminal_verified'])
                self.assertEqual(result['counts']['uncertain'],1)


class RecoveryProfileTests(unittest.TestCase):
    def fixture(self,b):
        def put(name,value):
            path=b/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        put('current.json',{'canonical_sha256':'c'*64})
        put('control/frozen.py',{'immutable':1})
        put('recovery.py',{'recovery':1})
        put('control/integrity_receipt.json',{'files':{'control/frozen.py':observer.sha(b/'control/frozen.py')},'project_files':{}})
        profile={'schema':'titantpp_retrieval_result_recovery_20261008_v1',
            'original_current_sha256':observer.sha(b/'current.json'),
            'original_control_integrity_sha256':observer.sha(b/'control/integrity_receipt.json'),
            'contract_sha256':'c'*64,'files':{'recovery.py':observer.sha(b/'recovery.py')},
            'project_files':{},'selection_changed':False,'new_training':False,
            'new_evaluation':False,'held_out_test_evaluated':False}
        put(observer.RECOVERY_PROFILE,profile)
        return profile,put

    def test_profile_seals_current_original_control_and_sidecar_without_rewriting(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            b=Path(tmp);profile,_=self.fixture(b)
            before={str(p.relative_to(b)):p.read_bytes() for p in b.rglob('*') if p.is_file()}
            self.assertEqual(observer.validate_recovery_profile(b),profile)
            after={str(p.relative_to(b)):p.read_bytes() for p in b.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_changed_bindings_code_contract_scope_or_unsafe_path_rejected(self):
        for change in ('current','integrity','frozen','sidecar','contract','scope','escape'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                b=Path(tmp);profile,put=self.fixture(b)
                if change=='current':put('current.json',{'canonical_sha256':'d'*64})
                elif change=='integrity':put('control/integrity_receipt.json',{'files':{},'project_files':{}})
                elif change=='frozen':put('control/frozen.py',{'immutable':2})
                elif change=='sidecar':put('recovery.py',{'recovery':2})
                else:
                    if change=='contract':profile['contract_sha256']='d'*64
                    elif change=='scope':profile['new_evaluation']=True
                    else:profile['files']={'../escape.py':'0'*64}
                    put(observer.RECOVERY_PROFILE,profile)
                with self.assertRaises(ValueError):observer.validate_recovery_profile(b)


class RecoveryFinalizerBindingTests(unittest.TestCase):
    def fixture(self):
        old=SimpleNamespace(profile_eligibility=object())
        observer_stub=SimpleNamespace(validate_recovery_profile=Mock(return_value={'profile':'sealed'}),
            preflight=Mock(),sha=Mock(return_value='a'*64),RECOVERY_PROFILE='control/profile.json')
        original_load=Mock(return_value=({'contract':'fixed'},old,{'original_integrity':'preserved'}))
        finalizer=SimpleNamespace(load_bundle=original_load,run=Mock(return_value={'status':'pending'}))
        return observer_stub,finalizer,original_load,old

    def test_wrapper_preserves_original_loader_and_adds_separate_SHA_bindings(self):
        obs,finalizer,load,old=self.fixture();b=Path('/private/tmp/fabricated-recovery')
        with patch.object(wrapper,'module',side_effect=[obs,finalizer]):
            actual,profile=wrapper.load_recovery(b)
            c,selected,integrity=actual.load_bundle(b)
        load.assert_called_once_with(b);obs.preflight.assert_called_once_with(b)
        self.assertIs(selected,obs);self.assertIs(obs.profile_eligibility,old.profile_eligibility)
        self.assertEqual(c,{'contract':'fixed'});self.assertEqual(profile,{'profile':'sealed'})
        self.assertEqual(integrity,{'original_integrity':'preserved',
            'result_recovery_profile_sha256':'a'*64,'result_recovery_observer_sha256':'a'*64,
            'result_recovery_finalizer_sha256':'a'*64})
        finalizer.run.assert_not_called()

    def test_invalid_profile_stops_before_loading_original_finalizer(self):
        obs,finalizer,_,_=self.fixture();obs.validate_recovery_profile.side_effect=ValueError('changed seal')
        with patch.object(wrapper,'module',side_effect=[obs,finalizer]) as loader:
            with self.assertRaisesRegex(ValueError,'changed seal'):wrapper.load_recovery(Path('/private/tmp/fabricated'))
        self.assertEqual(loader.call_count,1);finalizer.run.assert_not_called()

    def test_original_preflight_failure_is_not_bypassed(self):
        obs,finalizer,_,_=self.fixture();obs.preflight.side_effect=ValueError('original source changed')
        with patch.object(wrapper,'module',side_effect=[obs,finalizer]):
            actual,_=wrapper.load_recovery(Path('/private/tmp/fabricated'))
            with self.assertRaisesRegex(ValueError,'original source changed'):actual.load_bundle(Path('/private/tmp/fabricated'))
        finalizer.run.assert_not_called()


if __name__=='__main__':unittest.main()
