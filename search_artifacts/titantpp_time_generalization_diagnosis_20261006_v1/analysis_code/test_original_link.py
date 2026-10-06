"""Synthetic local receipt/terminal contracts; never reads real result inputs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT=Path(os.environ.get('ORIGINAL_LINK_SCRIPT',Path(__file__).with_name('run_context_diagnosis.py')))
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('original_link_under_test',SCRIPT)
linker=importlib.util.module_from_spec(spec);spec.loader.exec_module(linker)
EXPECTED='845a891db284f2cf88aa2ab17bee47716ea8dc11e6268646a71c772073ab7130'


class OriginalLink(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.original=self.base/'original';self.diagnostic=self.base/'diagnostic'
        self.original.mkdir();self.diagnostic.mkdir()
        self.contract={'fixture_only':True,'canonical_conditions':9,'anchors42':{},'jobs':[]}
        self.prior={'Test_read':False,'optimizer_steps':0,'GPU_used':False}
        self.patterns={'rows':[],'all_final':True,'Test_read':False,'sources':[
            {'path':str(self.original/'hourly_monitor/5090/snapshot.json'),'sha256':'d'*64,'actual_observed_utc':'2026-10-06T01:00:00+00:00'}]}
        self.summary={'scope':'Validation_only','held_out_test_evaluated':False,'rows':[],
            'grouped_3seed':[{'dataset':d,'n_seeds':3} for d in ('yellow_trip_hourly','raf_spare_parts','intermittent_frozen_5000')]}
        for index,dataset in enumerate(('yellow_trip_hourly','raf_spare_parts','intermittent_frozen_5000')):
            for seed in (42,52,62):
                q={'epoch':10,'val_qty_rmse':index+seed/10,'val_qty_mae':index+seed/20,'val_time_nll':index+seed/100,'train_time_nll':.1}
                self.patterns['rows'].append({'dataset':dataset,'seed':seed,'complete':True,'saved_epoch':50,'quantity_selected':q})
                self.summary['rows'].append({'dataset':dataset,'seed':seed,'best_epoch':10,
                    'metrics':{k:q['val_'+k] for k in ('qty_rmse','qty_mae','time_nll')}})
        for row in self.summary['rows']:
            if row['seed']!=42:continue
            dataset=row['dataset'];historical=deepcopy(row['metrics'])
            # Distinct, valid native endpoint/history receipts are not rounded
            # together or forced into a false cross-runtime identity claim.
            row['metrics']['qty_rmse']+=3.75e-7
            endpoint={'evaluation_scope':'validation_only','held_out_test_evaluated':False,'selected':deepcopy(row['metrics'])}
            rel='anchors/'+dataset+'/endpoint_replays.json';path=self.original/'source'/rel
            self.write(path,endpoint);digest=self.hash(path);row['endpoint_sha256']=digest
            self.contract['anchors42'][dataset]={'endpoint':rel,'endpoint_sha256':digest,
                'metrics':deepcopy(row['metrics']),'history_metrics':historical}
        self.caches={host:{'contract_sha256':EXPECTED,'analysis':{'counts':{'complete':count,'running':0,'waiting':0,'failed':0,'uncertain':0},'owned_fit_pids':[],'gpu_pids':[],'held_out_test_evaluated':False}}
            for host,count in (('5080',4),('5090',2))}
        self.snapshots={}
        for host in ('5080','5090'):
            jobs=[]
            for row in self.summary['rows']:
                if row['seed']==42:continue
                owner='5090' if row['dataset']=='intermittent_frozen_5000' else '5080'
                if host==owner:
                    job={'id':row['dataset']+'__'+str(row['seed']),'dataset':row['dataset'],'seed':row['seed'],'host':host}
                    jobs.append(job);self.contract['jobs'].append(job)
            snap={'contract_sha256':EXPECTED,'source_SHA_verified':True,'operation_SHA_verified':True,
                'gpu_uuid_verified':True,'processes':[],'gpu_pids':[],'supervisor_exit':{'returncode':0},'rows':[]}
            for job in jobs:
                snap['rows'].append({'job':job,'SHA_verified':True,
                    'terminal':{'status':'complete','scientific_success':True,'held_out_test_evaluated':False},
                    'endpoint':{'evaluation_scope':'validation_only','held_out_test_evaluated':False},
                    'manifest_sha256':hashlib.sha256(job['id'].encode()).hexdigest()})
            self.snapshots[host]=snap
            self.caches[host]['snapshot']=str(self.original/'hourly_monitor'/host/'snapshot.json')
        self.receipt={'status':'nine_conditions_Validation_originals_complete','contract_sha256':EXPECTED,
            'held_out_test_evaluated':False,'canonical_conditions':9,'outputs':{},'originals':{}}
        for host in ('5080','5090'):
            path=self.original/'retrieved'/host/'training_originals.tar.gz';path.parent.mkdir(parents=True);path.write_bytes(('synthetic archive '+host).encode())
            self.receipt['originals'][host]={'status':'passed','contract_sha256':EXPECTED,'archive_sha256':self.hash(path),
                'manifests':[{'id':row['job']['id'],'all_binary_small_SHA_verified':True,'terminal_manifest_sha256':row['manifest_sha256']} for row in self.snapshots[host]['rows']]}
        self.save()
        actual_canonical=linker.head.canonical
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(linker.head,'canonical',side_effect=lambda value:EXPECTED if value==self.contract else actual_canonical(value)).start()
        mock.patch.object(linker,'original_patterns',side_effect=lambda path:deepcopy(self.patterns)).start()
        mock.patch('subprocess.run',side_effect=AssertionError('Network/commands forbidden in local link')).start()

    @staticmethod
    def hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')

    def save(self):
        self.write(self.diagnostic/'context_results.json',self.prior)
        self.write(self.original/'execution_contract.json',self.contract)
        self.write(self.original/'analysis/Validation_comparison.json',self.summary)
        for name,text in [('Validation_per_seed.csv','fixture,Validation\n'),('README.md','Synthetic Validation summary\n')]:
            (self.original/'analysis'/name).write_text(text)
        self.receipt['outputs']={name:self.hash(self.original/'analysis'/name) for name in ('Validation_comparison.json','Validation_per_seed.csv','README.md')}
        for host,snap in self.snapshots.items():self.write(Path(self.caches[host]['snapshot']),snap)
        for host,cache in self.caches.items():self.write(self.original/'hourly_monitor'/('terminal_cache_'+host+'.json'),cache)
        self.write(self.original/'analysis/completion_receipt.json',self.receipt)

    def invoke(self):return linker.refresh_original_link(self.diagnostic,self.original)

    def reject(self):
        with self.assertRaises((ValueError,AssertionError,KeyError,FileNotFoundError)):self.invoke()
        self.assertFalse((self.diagnostic/'Intermittent_final_link.json').exists())

    def test_pending_does_not_require_archive_or_terminal_gate(self):
        (self.original/'analysis/completion_receipt.json').unlink()
        self.patterns['all_final']=False;self.patterns['rows'][-1]['complete']=False
        for host in ('5080','5090'):(self.original/'hourly_monitor'/('terminal_cache_'+host+'.json')).unlink()
        output=self.invoke()
        self.assertEqual(output['status'],'pending_original_terminal_and_retrieval')
        self.assertEqual(output['remote_calls'],0);self.assertFalse(output['new_inference']);self.assertFalse(output['Test_read'])
        self.assertIn('2026-10-06T01:00:00+00:00',(self.diagnostic/'INTERMITTENT_STATUS.md').read_text())

    def test_valid_nine_condition_completion_and_grouped_summary(self):
        output=self.invoke()
        self.assertEqual(output['status'],'verified_original9_final_Validation_and_binary_SHA_linked')
        self.assertEqual(output['grouped_3seed'],self.summary['grouped_3seed'])
        self.assertEqual(len(output['original_history_patterns']['rows']),9)
        self.assertEqual(output['validation_summary_sha256'],self.hash(self.original/'analysis/Validation_comparison.json'))
        self.assertEqual(output['completion_receipt_sha256'],self.hash(self.original/'analysis/completion_receipt.json'))
        self.assertEqual(output['CPU_binary_reinference_audit'],'not_performed')
        self.assertEqual(output['remote_calls'],0);self.assertFalse(output['Test_read'])

    def test_seed42_distinct_endpoint_and_history_are_preserved(self):
        output=self.invoke()
        for row in self.summary['rows']:
            if row['seed']!=42:continue
            historical=next(h for h in output['original_history_patterns']['rows'] if h['dataset']==row['dataset'] and h['seed']==42)
            self.assertNotEqual(row['metrics']['qty_rmse'],historical['quantity_selected']['val_qty_rmse'])
            anchor=self.contract['anchors42'][row['dataset']]
            self.assertEqual(row['metrics']['qty_rmse'],anchor['metrics']['qty_rmse'])
            self.assertEqual(historical['quantity_selected']['val_qty_rmse'],anchor['history_metrics']['qty_rmse'])

    def test_seed42_endpoint_sha_mismatch_rejected(self):
        anchor=next(iter(self.contract['anchors42'].values()))
        path=self.original/'source'/anchor['endpoint'];path.write_text(path.read_text()+' ');self.reject()

    def test_seed42_endpoint_foreign_scope_rejected(self):
        dataset=next(iter(self.contract['anchors42']));anchor=self.contract['anchors42'][dataset]
        path=self.original/'source'/anchor['endpoint'];endpoint=json.loads(path.read_text())
        endpoint['held_out_test_evaluated']=True;self.write(path,endpoint)
        anchor['endpoint_sha256']=self.hash(path)
        row=next(r for r in self.summary['rows'] if r['dataset']==dataset and r['seed']==42)
        row['endpoint_sha256']=anchor['endpoint_sha256'];self.save();self.reject()

    def test_seed42_history_lineage_mismatch_rejected(self):
        self.patterns['rows'][0]['quantity_selected']['val_qty_rmse']+=.01;self.reject()

    def test_archive_sha_mismatch_rejected(self):
        (self.original/'retrieved/5090/training_originals.tar.gz').write_bytes(b'changed');self.reject()

    def test_summary_file_sha_mismatch_rejected(self):
        path=self.original/'analysis/Validation_comparison.json';path.write_text(path.read_text()+' ');self.reject()

    def test_prior_diagnostic_test_scope_rejected(self):
        self.prior['Test_read']=True;self.write(self.diagnostic/'context_results.json',self.prior);self.reject()

    def test_completion_test_scope_rejected(self):
        self.receipt['held_out_test_evaluated']=True;self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_summary_test_scope_rejected_even_with_matching_sha(self):
        self.summary['held_out_test_evaluated']=True;self.save();self.reject()

    def test_missing_5090_host_rejected(self):
        del self.receipt['originals']['5090'];self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_empty_host_map_rejected(self):
        self.receipt['originals']={};self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_missing_summary_output_digest_rejected(self):
        del self.receipt['outputs']['Validation_comparison.json'];self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_missing_csv_output_digest_rejected(self):
        del self.receipt['outputs']['Validation_per_seed.csv'];self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_foreign_output_key_rejected_without_reading_it(self):
        self.receipt['outputs']['forbidden_mixed_results.json']='0'*64
        self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_wrong_completion_contract_rejected(self):
        self.receipt['contract_sha256']='0'*64;self.write(self.original/'analysis/completion_receipt.json',self.receipt);self.reject()

    def test_missing_terminal_cache_rejected(self):
        (self.original/'hourly_monitor/terminal_cache_5090.json').unlink();self.reject()

    def test_owned_fit_pid_present_rejected(self):
        self.caches['5090']['analysis']['owned_fit_pids']=[123]
        self.write(self.original/'hourly_monitor/terminal_cache_5090.json',self.caches['5090']);self.reject()

    def test_wrong_terminal_complete_count_rejected(self):
        self.caches['5090']['analysis']['counts']['complete']=1
        self.write(self.original/'hourly_monitor/terminal_cache_5090.json',self.caches['5090']);self.reject()

    def test_nonterminal_counts_and_gpu_presence_rejected(self):
        baseline=deepcopy(self.caches['5090'])
        for state in ('failed','running','waiting','uncertain'):
            self.caches['5090']=deepcopy(baseline);self.caches['5090']['analysis']['counts'][state]=1
            self.save()
            with self.subTest(state=state):self.reject()
        self.caches['5090']=deepcopy(baseline);self.caches['5090']['analysis']['gpu_pids']=[123];self.save();self.reject()

    def test_snapshot_runtime_identity_exit_and_pid_rejected(self):
        baseline=deepcopy(self.snapshots['5090'])
        mutations=[('source_SHA_verified',False),('operation_SHA_verified',False),('gpu_uuid_verified',False),
            ('processes',[{'pid':123}]),('gpu_pids',[123]),('supervisor_exit',{'returncode':1})]
        for key,value in mutations:
            self.snapshots['5090']=deepcopy(baseline);self.snapshots['5090'][key]=value;self.save()
            with self.subTest(key=key):self.reject()

    def test_missing_binary_manifest_verification_rejected(self):
        self.receipt['originals']['5090']['manifests'][0]['all_binary_small_SHA_verified']=False;self.save();self.reject()

    def test_missing_snapshot_job_rejected(self):
        self.snapshots['5090']['rows'].pop();self.save();self.reject()

    def test_snapshot_manifest_sha_mismatch_rejected(self):
        self.snapshots['5090']['rows'][0]['manifest_sha256']='0'*64;self.save();self.reject()

    def test_snapshot_endpoint_scope_and_scientific_failure_rejected(self):
        baseline=deepcopy(self.snapshots['5090'])
        self.snapshots['5090']['rows'][0]['endpoint']['held_out_test_evaluated']=True;self.save();self.reject()
        self.snapshots['5090']=deepcopy(baseline)
        self.snapshots['5090']['rows'][0]['terminal']['scientific_success']=False;self.save();self.reject()

    def test_unfinished_history_rejected_despite_completion_receipt(self):
        self.patterns['all_final']=False;self.patterns['rows'][-1]['complete']=False;self.reject()

    def test_duplicate_summary_condition_rejected(self):
        self.summary['rows'][-1]=deepcopy(self.summary['rows'][0]);self.save();self.reject()

    def test_selector_epoch_mismatch_rejected(self):
        self.summary['rows'][0]['best_epoch']=11;self.save();self.reject()

    def test_same_epoch_metric_mismatch_rejected(self):
        self.summary['rows'][0]['metrics']['time_nll']+=.01;self.save();self.reject()


if __name__=='__main__':unittest.main()
