"""Repair observer checks use fabricated local evidence; no network or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from paper.scripts import observe_titantpp_retrieval_repair as o

class RepairObserverTests(unittest.TestCase):
    def launch(self,c):
        spec=c['hosts']['5090'];root=spec['root']
        return {'host':'5090','root':root,'contract_sha256':o.canonical(c),'command':[spec['python'],root+'/operation/repair_runtime.py','--bundle',root,'--host','5090','--mode','dispatch']}

    def fixture(self,epoch=149):
        root='/owned/repair';host='5090';job={'host':host,'id':'fixed','dataset':'taxi','arm':'fixed','seed':42}
        data={'dataset_id':'taxi','quantity_boundaries_all_train_rows':[1,2,3,3449],'inherited_data_identity':{'populations':{'validation':{'target_count':8268}}}}
        c={'jobs':[job],'hosts':{host:{'root':root,'python':'/runtime/python','gpu_uuid':'GPU-own'}},'parent_contract':{'datasets':[data]},'resume':{'fixed':{'last_epoch':148}},'limits':{'per_condition_seconds':21600}}
        h=[{'epoch':i,'val_qty_rmse':1 if i==120 else 2,'val_qty_mae':1,'val_time_nll':.5,'train_all_finite':True} for i in range(1,epoch+1)]
        entry=root+'/operation/repair_runtime.py';argv=['/runtime/python',entry,'--bundle',root,'--host',host,'--mode','worker','--job','fixed','--claim',root+'/claims/fixed.json']
        supervisor=['/runtime/python',entry,'--bundle',root,'--host',host,'--mode','dispatch']
        snapshot={'host':host,'contract_sha256':o.canonical(c),'actual_observed_utc':'2026-10-07T06:00:00+00:00','observed_unix':100,
            'gpu_uuid_verified':True,'source_SHA_verified':True,'operation_SHA_verified':True,'parent_operation_SHA_verified':True,
            'gpu_pids':'GPU-own, 3, python','processes':['1 0 '+o.shlex.join(supervisor),'2 1 '+o.shlex.join(['timeout','--signal=TERM','--kill-after=15s','21600',*argv]),'3 2 '+o.shlex.join(argv)],
            'server_status':{'status':'running','active_job':job,'supervisor_pid':1,'worker_group_pid':2},'failure':None,
            'start_permit':{'deadline_unix':1000},'server_lease':{'job':'fixed','contract_sha256':o.canonical(c),'expires_unix':150,'supervisor_pid':1,'worker_group_pid':2},
            'rows':[{'job':job,'started':True,'claim':{'job':job,'contract_sha256':o.canonical(c),'deadline_unix':1000},'history':{'history':h},'actual_saved_epoch':epoch,'actual_best_epoch':120,
            'timing':{'epochs':[{'epoch':i,'elapsed_seconds':1 if i<=148 else 10} for i in range(1,epoch+1)]}}]}
        return c,snapshot

    def test_resume_ETA_uses_only_new_epochs(self):
        c,s=self.fixture();row=o.analyze(c,s)['rows'][0]
        self.assertEqual(row['state'],'running');self.assertEqual(row['new_saved_epochs'],1)
        self.assertEqual(row['eta']['median_completed_epoch_seconds'],10)
        self.assertEqual(row['eta']['conditional_remaining_seconds'],110)
        c,s=self.fixture(148);self.assertIsNone(o.analyze(c,s)['rows'][0]['eta'])

    def test_unowned_GPU_PID_or_wrong_worker_argv_is_uncertain(self):
        for field in ('gpu','argv','ppid','lease'):
            with self.subTest(field=field):
                c,s=self.fixture()
                if field=='gpu':s['gpu_pids']='GPU-foreign, 3, python'
                elif field=='argv':s['processes'][2]=s['processes'][2].replace('--job fixed','--job other')
                elif field=='ppid':s['processes'][2]=s['processes'][2].replace('3 2 ','3 999 ')
                else:s['server_lease']['expires_unix']=99
                self.assertEqual(o.analyze(c,s)['counts']['uncertain'],1)

    def test_native_failure_is_not_waiting_or_query_failure(self):
        c,s=self.fixture();s['failure']={'message':'Worker exited 124'};s['processes']=[]
        r=o.analyze(c,s);self.assertEqual(r['counts']['failed'],1);self.assertFalse(r['query_failed'])

    def test_checkpoint_publication_race_is_uncertain(self):
        c,s=self.fixture();s['rows'][0]['checkpoint_race']=True
        self.assertEqual(o.analyze(c,s)['counts']['uncertain'],1)

    def test_waiting_host_has_no_SSH(self):
        c,s=self.fixture();runner=Mock(side_effect=AssertionError('network forbidden'))
        with tempfile.TemporaryDirectory() as tmp:
            result=o.host_once(c,Path(tmp),'5090',runner=runner)
        runner.assert_not_called();self.assertEqual(result['counts']['waiting'],1)

    def test_one_readonly_SSH_and_snapshot_SHA_pointer(self):
        c,s=self.fixture();runner=Mock(return_value=Mock(returncode=0,stdout=json.dumps(s),stderr=''))
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);o.write(b/'deployment/launch_receipt_5090.json',self.launch(c))
            result=o.host_once(c,b,'5090',runner=runner)
            self.assertEqual(result['counts']['running'],1);self.assertEqual(runner.call_count,1)
            argv=runner.call_args.args[0];self.assertEqual(argv[0],'ssh');self.assertIn('--native',argv[-1]);self.assertNotIn('--mode',argv[-1])
            pointer=o.read(b/'hourly_monitor/latest_5090.json')
            self.assertEqual(o.sha(pointer['snapshot']),pointer['snapshot_sha256'])

    def terminal(self):
        c,s=self.fixture(160);r=s['rows'][0];job=r['job'];data=c['parent_contract']['datasets'][0]
        r['terminal']={'status':'complete','scientific_success':True,'job':job,'contract_sha256':o.canonical(c),'parent_contract_sha256':o.PARENT_SHA,'evaluation_scope':'validation_only','held_out_test_evaluated':False}
        def ep(epoch):
            h=r['history']['history'][epoch-1]
            return {'evaluation_scope':'validation_only','held_out_test_evaluated':False,'count':8268,'quantity_boundaries':data['quantity_boundaries_all_train_rows'],'quantity_cells':[{}]*5,'tail':{'count':79,'qty_rmse':2,'qty_mae':1,'time_nll':0},'qty_rmse':h['val_qty_rmse'],'qty_mae':h['val_qty_mae'],'time_nll':h['val_time_nll']}
        r['endpoint']={'job':job,'evaluation_scope':'validation_only','held_out_test_evaluated':False,'best_epoch':120,'completed_epochs':160,'selected':ep(120),'last':ep(160)};r['SHA_verified']=True
        s['processes']=[];s['gpu_pids']='';s['server_status']={'status':'complete'};s['supervisor_exit']={'returncode':0}
        return c,s

    def test_terminal_requires_full_Validation_same_selector_epoch_and_exit(self):
        c,s=self.terminal();self.assertTrue(o.analyze(c,s)['terminal_verified'])
        for change in ('metric','population','SHA','exit'):
            with self.subTest(change=change):
                altered=copy.deepcopy(s)
                if change=='metric':altered['rows'][0]['endpoint']['selected']['qty_rmse']=.9
                elif change=='population':altered['rows'][0]['endpoint']['selected']['count']=8000
                elif change=='SHA':altered['rows'][0]['SHA_verified']=False
                else:altered['supervisor_exit']['returncode']=1
                self.assertFalse(o.analyze(c,altered)['terminal_verified'])

    def test_terminal_cache_revalidation_preserves_time_no_SSH(self):
        c,s=self.terminal();runner=Mock(return_value=Mock(returncode=0,stdout=json.dumps(s),stderr=''))
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);o.write(b/'deployment/launch_receipt_5090.json',self.launch(c))
            first=o.host_once(c,b,'5090',runner=runner)
            forbidden=Mock(side_effect=AssertionError('extra SSH'))
            second=o.host_once(c,b,'5090',runner=forbidden)
            forbidden.assert_not_called();self.assertEqual(first['actual_observed_utc'],second['actual_observed_utc'])
            p=o.read(b/'hourly_monitor/terminal_cache_5090.json');Path(p['snapshot']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'bytes changed'):o.host_once(c,b,'5090',runner=forbidden)

if __name__=='__main__':unittest.main()
