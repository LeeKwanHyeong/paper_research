import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import titantpp_gru_cap_parallel3_v2_runtime as r
from paper.scripts import advance_titantpp_gru_cap_parallel3_v2 as a


class ParallelTests(unittest.TestCase):
    def test_handoff_accepts_actual_gpu_pid_set_and_preserves_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[dict(id=str(i),host='5080') for i in range(3)]
            c={'jobs':jobs,'hosts':{'5080':{}},'limits':{'per_condition_seconds':100}}
            r.write(root/'status.json',{'status':'running','completed':{},'supervisor_pid':11,
                'active_jobs':{'0':{'worker_group_pid':22}}},True)
            launch={'pid':11,'ppid':10,'argv':['python','original']}
            r.write(root/'phases/train_launch.json',launch,True)
            r.write(root/'start_permit.json',{'deadline_unix':1000},True)
            before=(root/'status.json').read_bytes()
            module=SimpleNamespace(common=SimpleNamespace(gpu_pids=lambda _: {33}),
                authorization=lambda *args:{'deadline_unix':1000},verify_training_permit=lambda *args:None)
            with patch.object(r,'proc',side_effect=lambda pid:launch if pid==11 else {'pid':33,'ppid':22}),\
                 patch.object(r.time,'time',return_value=100):
                h=r.initial_handoff(module,root,c,'5080')
            self.assertEqual(h['original_active_gpu_pid'],33)
            self.assertEqual(h['retained_job_ids'],['0'])
            self.assertEqual((root/'status.json').read_bytes(),before)
            self.assertFalse((root/'claims/0.json').exists())
            self.assertTrue((root/'claims/1.json').exists())
            self.assertFalse((root/'claims/2.json').exists())

    def test_partition_retains_completed_and_active(self):
        jobs=[dict(id=str(i),host='5080') for i in range(7)]
        done,active,todo=r.partition({'jobs':jobs},'5080',{'status':'running','completed':{'1':{},'0':{}},'active_jobs':{'2':{}}})
        self.assertEqual((done,active,[j['id'] for j in todo]),(['0','1'],'2',['3','4','5','6']))

    def test_partition_rejects_skip_duplicate_or_no_live(self):
        c={'jobs':[dict(id=str(i),host='5080') for i in range(4)]}
        for state in [{'status':'running','completed':{'1':{}},'active_jobs':{'2':{}}},
            {'status':'running','completed':{},'active_jobs':{'0':{},'1':{}}},
            {'status':'failed','completed':{},'active_jobs':{'0':{}}}]:
            with self.assertRaises(ValueError):r.partition(c,'5080',state)

    def test_only_lease_and_progress_routes_change(self):
        root=Path('/owned');reads=[];writes=[]
        f=SimpleNamespace(read=lambda p:reads.append(p), common=SimpleNamespace(write_json=lambda p,v,**kw:writes.append(p)))
        r.route_frozen(f,root,{'id':'a'})
        f.read(root/'server_lease.json');f.read(root/'run/a/history.json')
        f.common.write_json(root/'progress.json',{});f.common.write_json(root/'run/a/history.json',{})
        self.assertEqual(reads,[root/'leases/a.json',root/'run/a/history.json'])
        self.assertEqual(writes,[root/'parallel3_v2/progress/a.json',root/'run/a/history.json'])

    def test_unexpected_parent_training_failure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r.write(root/'failure.json',{'type':'ValueError','message':'NaN loss','cleanup':None,
                'active_job':{'id':'next'},'completed':{},'contract_sha256':r.canonical({})},True)
            h={'retained_job_ids':['old'],'remaining_jobs':[{'id':'next'}]}
            with self.assertRaisesRegex(ValueError,'Unexpected original failure'):r.parent_terminal(root,{},h,'5080')

    def test_science_source_unchanged_and_operational_patch_unique(self):
        bundle=a.BUNDLE
        self.assertEqual(r.sha(bundle/'execution_contract.json'),r.CONTRACT_RAW)
        self.assertEqual(r.sha(bundle/'operation/campaign.py'),r.CAMPAIGN_RAW)
        source=(bundle/'operation/campaign.py').read_text()
        self.assertEqual(source.count('frozen.PulseBudget = frozen.PulseBudget'),1)

    def test_transport_compiles_and_uses_exact_manual2_root(self):
        compile(a.INSTALL,'install','exec');compile(a.EXTRA_PROBES,'probes','exec')
        self.assertIn("spec['tmux']+'_parallel3_v2'",a.INSTALL)
        self.assertNotIn('kill',a.INSTALL)
        text=a.status_script(a.launch.launcher_for())
        compile(text,'whole-status','exec')
        self.assertIn('parallel_process_observations(state,files,pids,c,host)',text)

    def test_dispatch_admits_two_alongside_parent_then_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[{'id':str(i),'seed':42+i,'dataset':'x'} for i in range(4)]
            c={'hosts':{'5080':{'python':'python'}},'limits':{'per_condition_seconds':100}}
            first={'deadline_unix':1000};r.write(root/'claims/0.json',first,True)
            h={'remaining_jobs':jobs,'original_active_guard':1,'first_claim_raw_sha256':r.sha(root/'claims/0.json')}
            created=[];iterations=[0];maxima=[]
            class Child:
                def __init__(self):self.pid=100+len(created);self.birth=iterations[0]
                def poll(self):return 0 if iterations[0]>self.birth else None
            def spawn(*args,**kwargs):
                child=Child();created.append(child);return child
            module=SimpleNamespace(common=SimpleNamespace(gpu_pids=lambda *_:[]),stop_owned=lambda _:None)
            actual_write=r.write
            def write(path,value,exclusive=False):
                if Path(path).name=='status.json':maxima.append(len(value.get('active_jobs',{})))
                return actual_write(path,value,exclusive)
            with patch.object(r,'authorization',return_value={'deadline_unix':1000}),\
                 patch.object(r,'parent_terminal',side_effect=lambda *x:iterations[0]>0),\
                 patch.object(r,'capacity'),patch.object(r,'worker_exit',return_value={}),\
                 patch.object(r.subprocess,'Popen',side_effect=spawn),\
                 patch.object(r.time,'time',return_value=100),\
                 patch.object(r.time,'sleep',side_effect=lambda _:iterations.__setitem__(0,iterations[0]+1)),\
                 patch.object(r,'write',side_effect=write):
                r.dispatch(module,root,c,'5080',h)
            self.assertEqual(len(created),4);self.assertEqual(maxima[0],2);self.assertLessEqual(max(maxima),3)


if __name__=='__main__':unittest.main()
