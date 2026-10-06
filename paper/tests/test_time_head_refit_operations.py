"""Pure local contract fixtures. No GPU, SSH, source data, or evaluation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

DEFAULT = Path(__file__).resolve().parents[2]/'search_artifacts/titantpp_time_head_bias_refit_5080_20261006_v1/control'
CONTROL = Path(os.environ.get('TIME_HEAD_REFIT_CONTROL_DIR', DEFAULT))
def load(name):
    spec = importlib.util.spec_from_file_location('refit_ops_'+name, CONTROL/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
m = load('monitor_once'); f = load('finalize_once')


class Operations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = {'root':str(self.root),'source':{'files_sha256':'a'*64},
            'operation':{'files':{}},'runtime':{'gpu_uuid':'GPU-fixture','runtime_expected':{'torch':'fixture'}},
            'datasets':[{'dataset_id':d,'inherited_data_identity':{'populations':{'train':{'target_count':2},'validation':{'target_count':2}}}} for d in ('taxi','raf')],
            'fitting':{'maximum_epochs':40,'patience':10},'jobs':[]}
        for d in ('taxi','raf'):
            for seed in (42,52,62):
                jid=d+'__'+str(seed)+'__time_bias2'; source=self.root/(jid+'.source'); source.write_bytes(b'original')
                self.c['jobs'].append({'id':jid,'dataset':d,'seed':seed,'output_dir':str(self.root/'run'/jid),
                    'baseline_metrics':{'count':2,'qty_rmse':3.,'qty_mae':2.,'time_nll':1.,'tail':{'count':1,'qty_rmse':5.,'qty_mae':4.,'time_nll':1.}},
                    'checkpoint':{'path':str(source),'sha256':m.sha(source),'state_sha256':'b'*64,'epoch':7}})
        self.job=self.c['jobs'][0]
        for j in self.c['jobs']: self.make_job(j)

    def put(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))

    def make_job(self,j):
        p=Path(j['output_dir']);p.mkdir(parents=True)
        metrics=copy.deepcopy(j['baseline_metrics'])
        history=[{'epoch':i,'validation':{**metrics,'time_nll':1. if i==0 else 2.},'steps':i,'train_time_nll':None if i==0 else 2.} for i in range(11)]
        quantity={'train':'d'*64,'validation':'e'*64}
        inp={'held_out_materialized':False,'populations':self.c['datasets'][0]['inherited_data_identity']['populations'],'refit_runtime':{'gpu':{'uuid':'GPU-fixture'},'torch':'fixture'}}
        self.put(p/'input_receipt.json',inp)
        common={'evaluation_scope':'validation_only','held_out_test_evaluated':False}
        endpoint={**common,'E0':metrics,'selected':metrics,'last':history[-1]['validation'],'best_epoch':0,'completed_epochs':10,'optimizer_steps':10,
            'frozen_state_sha256':'c'*64,'quantity_predictions_exactly_equal':True,'selected_is_identity':True,'trainable_scalar_count':2}
        startup={**common,'status':'passed','job':j['id'],'contract_sha256':m.digest(self.c),'original_checkpoint':j['checkpoint'],'E0':metrics,
            'input_receipt_sha256':m.sha(p/'input_receipt.json'),'runtime':inp['refit_runtime'],'frozen_state_sha256':'c'*64,'quantity_prediction_sha256':quantity}
        result={**endpoint,'schema':'titantpp_time_head_bias_refit_v1','status':'complete','job':j['id'],'contract_sha256':m.digest(self.c),'quantity_output_unchanged':True,
            'quantity_prediction_sha256_before':quantity,'quantity_prediction_sha256_after':quantity,'frozen_state_sha256_before':'c'*64,'frozen_state_sha256_after':'c'*64,
            'trainable_tensors':['b_t','w_raw'],'original_checkpoint':j['checkpoint'],'source_files_sha256':'a'*64}
        for name,value in [('startup_gate.json',startup),('history.json',{**common,'history':history}),('endpoint_replays.json',endpoint),('result.json',result),
            ('training_permit.json',{'job':j['id'],'contract_sha256':m.digest(self.c)}),('worker_process.json',{'pid':100+self.c['jobs'].index(j),'ppid':50,'command':[]})]:self.put(p/name,value)
        (p/'selected_refit.pt').write_bytes(b'selected');(p/'last_refit.pt').write_bytes(b'last')
        terminal={**common,'status':'complete','scientific_success':True,'job':j,'contract_sha256':m.digest(self.c),'files':{x.name:m.sha(x) for x in p.iterdir()},
            'trainable_scalar_count':2,'frozen_state_sha256':'c'*64,'original_checkpoint_sha_preserved':True}
        self.put(p/'terminal_manifest.json',terminal)
        self.put(p/'supervisor_terminal_receipt.json',{**common,'status':'complete','scientific_success':True,'job':j['id'],'contract_sha256':m.digest(self.c),'files':{x.name:m.sha(x) for x in p.iterdir()}})

    def docs(self):
        return m.verify_job(self.c,self.job,self.job['output_dir'])['proof']

    def snapshot(self):
        return {'contract_sha256':m.digest(self.c),'host':'5080','held_out_test_evaluated':False,
            'actual_observed_utc':'2026-10-06T00:00:00+00:00','server_terminal_verified':True,
            'rows':[{'job':j,'state':'complete',**m.verify_job(self.c,j,j['output_dir'])} for j in self.c['jobs']],
            'counts':{'complete':6,'failed':0,'running':0,'waiting':0,'uncertain':0},'owned_processes':[],'owned_gpu_pids':[],
            'process_identity_issues':[],'failure':None,'server_status':{'status':'complete','complete':[j['id'] for j in self.c['jobs']]},
            'supervisor_exit':{'returncode':0,'contract_sha256':m.digest(self.c)}}

    def test_valid_identity_terminal(self):
        self.assertTrue(m.verify_job(self.c,self.job,self.job['output_dir'])['result']['selected_is_identity'])
        m.validate_snapshot(self.c,self.snapshot(),terminal=True)

    def test_complete_proof_mutations_rejected(self):
        mutations=[('result','job','wrong'),('result','contract_sha256','0'*64),('result','source_files_sha256','0'*64),
            ('result','frozen_state_sha256_after','0'*64),('result','trainable_tensors',['b_t','w_raw','v_t']),
            ('result','trainable_scalar_count',3),('result','quantity_prediction_sha256_after',{'train':'0'*64,'validation':'e'*64}),
            ('result','best_epoch',1),('result','selected_is_identity',False),('terminal','scientific_success',False),
            ('startup','original_checkpoint',{}),('input','populations',{})]
        for name,key,value in mutations:
            with self.subTest(name=name,key=key):
                docs=self.docs();docs[name][key]=value
                with self.assertRaises((ValueError,KeyError)):m.verify_evidence(self.c,self.job,docs)

    def test_missing_e0_and_tie_selector_rejected(self):
        docs=self.docs();docs['history']['history']=docs['history']['history'][1:]
        with self.assertRaises(ValueError):m.verify_evidence(self.c,self.job,docs)
        docs=self.docs();docs['history']['history'][1]['validation']['time_nll']=1.
        docs['result']['best_epoch']=1
        with self.assertRaises(ValueError):m.verify_evidence(self.c,self.job,docs)

    def test_premature_stop_rejected(self):
        docs=self.docs();docs['history']['history']=docs['history']['history'][:5]
        for label in ('result','endpoint'):docs[label]['completed_epochs']=4
        with self.assertRaises(ValueError):m.verify_evidence(self.c,self.job,docs)

    def test_selected_binary_tamper_rejected(self):
        (Path(self.job['output_dir'])/'selected_refit.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):m.verify_job(self.c,self.job,self.job['output_dir'])

    def test_last_binary_missing_rejected(self):
        (Path(self.job['output_dir'])/'last_refit.pt').unlink()
        with self.assertRaises(ValueError):m.verify_job(self.c,self.job,self.job['output_dir'])

    def test_original_binary_tamper_rejected(self):
        Path(self.job['checkpoint']['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):m.verify_job(self.c,self.job,self.job['output_dir'])

    def test_manifest_escape_rejected(self):
        for relative in ('../escape','/absolute'):
            with self.subTest(relative=relative),self.assertRaises(ValueError):m.safe_file(self.root,relative)

    def test_terminal_requires_actual_exit_and_absence(self):
        for key,value in [('supervisor_exit',{'returncode':1,'contract_sha256':m.digest(self.c)}),('owned_processes',[{'pid':4}]),('owned_gpu_pids',[4]),('process_identity_issues',['mismatch'])]:
            s=self.snapshot();s[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_snapshot(self.c,s,terminal=True)

    def test_duplicate_job_and_forged_counts_rejected(self):
        s=self.snapshot();s['rows'][-1]=copy.deepcopy(s['rows'][0])
        with self.assertRaises(ValueError):m.validate_snapshot(self.c,s,terminal=True)
        s=self.snapshot();s['counts']['complete']=5
        with self.assertRaises(ValueError):m.validate_snapshot(self.c,s,terminal=True)

    def process_fixture(self):
        j=self.job;controller=['python',str(self.root/'operation/controller.py'),'--contract',str(self.root/'execution_contract.json'),'--mode','dispatch']
        worker=['python',str(self.root/'operation/titantpp_time_head_refit_runtime.py'),'--contract',str(self.root/'execution_contract.json'),
            '--job',j['id'],'--training-permit',j['output_dir']+'/training_permit.json','--owner-pid','50']
        processes={50:{'pid':50,'ppid':1,'argv':controller},100:{'pid':100,'ppid':50,'argv':worker}}
        rows=[{'job':j,'state':'running','worker_process':{'pid':100,'ppid':50,'command':worker}}]
        return processes,rows

    def test_exact_process_gpu_binding(self):
        ps,rows=self.process_fixture()
        owned,issues,gpu=m.process_identity(self.c,self.root,ps,{100},{'pid':50},rows)
        self.assertEqual(issues,[]);self.assertEqual(gpu,[100]);self.assertEqual(len(owned),2)
        ps[100]['ppid']=999
        self.assertTrue(m.process_identity(self.c,self.root,ps,{100},{'pid':50},rows)[1])

    def test_missing_gpu_or_spoofed_worker_rejected(self):
        ps,rows=self.process_fixture()
        self.assertTrue(m.process_identity(self.c,self.root,ps,set(),{'pid':50},rows)[1])
        ps[100]['argv']=ps[100]['argv'][:-1]+['999']
        self.assertTrue(m.process_identity(self.c,self.root,ps,{100},{'pid':50},rows)[1])

    def test_completed_pid_still_alive_blocks_terminal(self):
        ps,rows=self.process_fixture();rows[0]['state']='complete'
        self.assertTrue(m.process_identity(self.c,self.root,ps,{100},{'pid':50},rows)[1])

    def test_history_eta_uses_same_selected_epoch(self):
        doc=self.docs()['history']
        for row in doc['history'][1:]:row['epoch_seconds']=100.;row['completed_utc']='2026-10-06T00:00:00+00:00'
        p=m.progress_history(self.c,self.job,doc)
        self.assertEqual(p['selected_epoch'],0);self.assertEqual(p['selected_validation']['time_nll'],1.)
        self.assertEqual(p['recent_epoch_seconds_median'],100.);self.assertEqual(p['seconds_to_current_patience_boundary'],0)
        self.assertEqual(p['seconds_to_max_epochs'],3000.)

    def test_retrieved_archive_manifest_and_checkpoint_integrity(self):
        import shutil
        original=self.root/'retrieved';original.mkdir()
        shutil.copytree(self.root/'run',original/'run')
        for j in self.c['jobs']:
            p=original/'source_checkpoints'/j['id'];p.mkdir(parents=True)
            shutil.copyfile(j['checkpoint']['path'],p/'source_selected.pt')
        self.put(original/'execution_contract.json',self.c)
        self.put(original/'status.json',{'status':'complete','complete':[j['id'] for j in self.c['jobs']]})
        self.put(original/'supervisor_process_exit.json',{'returncode':0,'contract_sha256':m.digest(self.c)})
        seal={'status':'sealed','contract_sha256':m.digest(self.c),'held_out_test_evaluated':False,'archive_sha256':'e'*64,
            'archive_files':{str(p.relative_to(original)):m.sha(p) for p in original.rglob('*') if p.is_file()}}
        self.assertEqual(len(f.verify_retrieved(self.c,original,seal)),6)
        (original/'run'/self.job['id']/'selected_refit.pt').write_bytes(b'tampered')
        with self.assertRaises(ValueError):f.verify_retrieved(self.c,original,seal)

    def test_remote_program_compiles_without_running(self):
        compile(m.core_code(),'observer','exec')
        compile(m.core_code()+f.SEAL_CODE,'sealer','exec')

    def test_observation_failure_is_not_training_failure(self):
        receipt=m.observation_failure(subprocess.TimeoutExpired('ssh',15))
        self.assertEqual(receipt['status'],'observation_failed');self.assertFalse(receipt['training_failure']);self.assertFalse(receipt['automatic_retry'])

    def test_cache_reuse_has_no_ssh_and_preserves_actual_time(self):
        self.put(self.root/'execution_contract.json',self.c);self.put(self.root/'current.json',{'canonical_sha256':m.digest(self.c),'root':str(self.root)})
        self.put(self.root/'hourly_monitor/terminal_cache.json',self.snapshot())
        with mock.patch.object(m,'BUNDLE',self.root),mock.patch.object(m.subprocess,'run',side_effect=AssertionError('network forbidden')),mock.patch('builtins.print') as print_mock:
            m.main()
        out=json.loads(print_mock.call_args.args[0]);self.assertEqual(out['actual_observed_utc'],'2026-10-06T00:00:00+00:00');self.assertTrue(out['terminal_cache_reused'])

    def test_finalizer_pending_and_claim_do_not_connect(self):
        self.put(self.root/'execution_contract.json',self.c);self.put(self.root/'current.json',{'canonical_sha256':m.digest(self.c),'root':str(self.root)})
        with mock.patch.object(f,'BUNDLE',self.root),mock.patch.object(f.subprocess,'run',side_effect=AssertionError('network forbidden')),mock.patch('builtins.print'):
            f.main();self.assertFalse(m.read(self.root/'analysis/pending.json')['remote_retrieval_started'])
            self.put(self.root/'analysis/retrieval.claim',{'prior':True});f.main()

    def test_archive_special_paths_links_and_duplicates_rejected(self):
        for name,kind in [('../escape',tarfile.REGTYPE),('/escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE),('device',tarfile.CHRTYPE)]:
            member=tarfile.TarInfo(name);member.type=kind
            with self.subTest(name=name),self.assertRaises(ValueError):f.validate_archive_members([member])
        member=tarfile.TarInfo('file')
        with self.assertRaises(ValueError):f.validate_archive_members([member,member])


if __name__=='__main__':unittest.main()
