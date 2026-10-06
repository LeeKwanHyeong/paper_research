"""Fake-runner observation tests: no SSH, CUDA, activation or scientific files."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/observe_titantpp_contextual_retrieval_hourly.py'
spec=importlib.util.spec_from_file_location('_pilot_observer_test',SCRIPT)
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fixture():
    return {'schema':o.NAME,'jobs':[{'id':host+'__Taxi__42__'+arm,'host':host,'dataset':'yellow_trip_hourly',
        'seed':42,'arm':'titantpp_retrieval_'+arm} for host in o.HOSTS for arm in ('mean','weighted','fixed','adaptive','generic')]}


def analysis(c,host):
    rows=[{'job':j,'id':j['id'],'dataset':j['dataset'],'arm':j['arm'],'seed':42,'state':'waiting',
        'actual_saved_epoch':None,'best_epoch':None,'eta':None} for j in c['jobs'] if j['host']==host]
    rows[0].update(state='running',actual_saved_epoch=7,best_epoch=5,eta={'conditional_remaining_seconds':200.,'max300_remaining_seconds':2000.})
    return {'host':host,'actual_observed_utc':'2026-10-07T00:01:02+00:00','rows':rows,
        'counts':{'complete':0,'failed':0,'running':1,'waiting':4,'uncertain':0}}


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.bundle=(Path(self.temp.name)/o.NAME).resolve();self.bundle.mkdir();self.reports=(Path(self.temp.name)/'reports').resolve()
        self.c=fixture();self.calls=[];self.sealed=False
        def seal(bundle):self.assertEqual(bundle,self.bundle);self.sealed=True;return self.c
        self.monitor=types.SimpleNamespace(load_bundle=seal,digest=digest)
    def launch(self,host):o.write(self.bundle/'deployment'/('launch_receipt_'+host+'.json'),{'host':host})
    def run_report(self,runner,hosts=o.HOSTS):
        return o.run(self.bundle,hosts,runner=runner,monitor_loader=lambda b:self.monitor,report_root=self.reports)
    def forbidden(self,*args,**kwargs):raise AssertionError('Unexpected monitor/SSH invocation')
    def test_pending_hosts_seal_before_reads_no_calls(self):
        report,path=self.run_report(self.forbidden)
        self.assertTrue(self.sealed);self.assertEqual(report['execution_counts']['waiting'],10)
        self.assertFalse(report['activation_performed']);self.assertFalse(report['finalizer_invoked'])
        self.assertIsNone(report['hosts']['5090']['actual_observed_kst']);self.assertTrue(path.exists())
        self.assertIn('5개',path.read_text());self.assertEqual(o.read(self.reports/'latest.json')['markdown'],str(path))
    def test_launched_hosts_exact_one_monitor_call_each(self):
        for h in o.HOSTS:self.launch(h)
        def runner(cmd,**kwargs):
            self.calls.append(cmd);host=cmd[-1]
            return subprocess.CompletedProcess(cmd,0,json.dumps(analysis(self.c,host)),'')
        report,_=self.run_report(runner)
        self.assertEqual(len(self.calls),2)
        for h in o.HOSTS:
            self.assertEqual(sum(cmd[-1]==h for cmd in self.calls),1)
            self.assertIn('--bundle',next(cmd for cmd in self.calls if cmd[-1]==h))
        self.assertEqual(report['hosts']['5090']['actual_observed_kst'],'2026-10-07 09:01:02 KST')
        self.assertEqual(report['execution_counts']['running'],2)
    def test_query_failure_is_uncertain_other_host_continues(self):
        for h in o.HOSTS:self.launch(h)
        def runner(cmd,**kwargs):
            self.calls.append(cmd)
            return subprocess.CompletedProcess(cmd,0,json.dumps({'query_failed':True,'error':'transport'}) if cmd[-1]=='5090' else json.dumps(analysis(self.c,'5080')),'')
        report,_=self.run_report(runner)
        self.assertEqual(len(self.calls),2);self.assertEqual(report['execution_counts']['uncertain'],5)
        self.assertEqual(report['execution_counts']['failed'],0);self.assertFalse(report['hosts']['5090']['training_failure'])
        self.assertEqual(report['hosts']['5080']['counts']['running'],1)
    def test_runner_exception_and_bad_coverage_are_query_failures(self):
        for h in o.HOSTS:self.launch(h)
        def runner(cmd,**kwargs):
            if cmd[-1]=='5090':raise subprocess.TimeoutExpired(cmd,240)
            bad=analysis(self.c,'5080');bad['rows'].pop();return subprocess.CompletedProcess(cmd,0,json.dumps(bad),'')
        report,_=self.run_report(runner)
        self.assertEqual(report['execution_counts']['uncertain'],10);self.assertEqual(report['execution_counts']['failed'],0)
    def test_terminal_cache_actual_time_and_zero_additional_ssh(self):
        self.launch('5090')
        def runner(cmd,**kwargs):
            self.calls.append(cmd);item=analysis(self.c,'5090')
            item.update(terminal_cache_reused=True,observation_time_preserved=True)
            for r in item['rows']:r['state']='complete'
            item['counts']={'complete':5,'failed':0,'running':0,'waiting':0,'uncertain':0}
            return subprocess.CompletedProcess(cmd,0,json.dumps(item),'')
        report,path=self.run_report(runner)
        self.assertEqual(len(self.calls),1);self.assertTrue(report['hosts']['5090']['terminal_cache_reused'])
        self.assertEqual(report['hosts']['5090']['actual_observed_utc'],'2026-10-07T00:01:02+00:00')
        self.assertIn('추가 SSH 조회는 없습니다',path.read_text())
    def test_single_host_does_not_query_unselected_block(self):
        for h in o.HOSTS:self.launch(h)
        def runner(cmd,**kwargs):self.calls.append(cmd);return subprocess.CompletedProcess(cmd,0,json.dumps(analysis(self.c,cmd[-1])),'')
        report,_=self.run_report(runner,('5090',))
        self.assertEqual(len(self.calls),1);self.assertEqual(report['hosts']['5080']['counts']['uncertain'],5)
        self.assertFalse(report['hosts']['5080']['query_failed'])
    def test_capacity_pending_and_cost_pass_are_separate(self):
        o.write(self.bundle/'capacity/pending_5080.json',{'status':'waiting_original54'})
        o.write(self.bundle/'qualification/5090/receipt.json',{'status':'passed','technical_qualification':'passed',
            'cost_candidate_eligibility':{'titantpp_retrieval_adaptive':{'eligible':False}}})
        report,path=self.run_report(self.forbidden)
        self.assertIn('기존54',report['hosts']['5080']['reason'])
        costs=report['hosts']['5090']['qualification'];self.assertEqual(costs['technical_status'],'passed')
        self.assertEqual(costs['cost_ineligible_arms'],['titantpp_retrieval_adaptive'])
        self.assertEqual(report['execution_counts']['waiting'],10);self.assertIn('학습 시작 허가와 분리',path.read_text())
    def test_invalid_qualification_cannot_report_native_pass(self):
        o.write(self.bundle/'qualification/5090/receipt.json',{'status':'passed','host':'5080'})
        def verify(host,receipt):
            if receipt.get('host')!=host:raise ValueError('foreign native receipt')
        self.monitor.campaign=types.SimpleNamespace(verify_qualification=lambda c,h,q:verify(h,q))
        report,_=self.run_report(self.forbidden)
        self.assertEqual(report['hosts']['5090']['qualification']['technical_status'],'unverified')
        self.assertEqual(report['execution_counts']['waiting'],10)
    def test_corrupt_qualification_does_not_abort_other_host(self):
        path=self.bundle/'qualification/5090/receipt.json';path.parent.mkdir(parents=True);path.write_text('invalid')
        report,_=self.run_report(self.forbidden)
        self.assertEqual(report['hosts']['5090']['qualification']['technical_status'],'unverified')
        self.assertEqual(report['hosts']['5080']['counts']['waiting'],5)
    def test_source_seal_failure_aborts_before_monitor_and_reporting(self):
        def invalid(bundle):raise ValueError('source/op changed')
        monitor=types.SimpleNamespace(load_bundle=invalid,digest=digest)
        with self.assertRaises(ValueError):o.run(self.bundle,runner=self.forbidden,monitor_loader=lambda b:monitor,report_root=self.reports)
        self.assertFalse(self.reports.exists())
    def sealed_fixture(self):
        c=fixture();c['hosts']={h:{'root':'/owned/'+h} for h in o.HOSTS}
        source=self.bundle/'source/science.py';source.parent.mkdir();source.write_text('# frozen scientific fixture\n')
        files={}
        for name in ('campaign','monitor','qualify'):
            path=self.bundle/'operation'/(name+'.py');path.parent.mkdir(exist_ok=True);path.write_text('# immutable operation fixture\n')
            files['operation/'+name+'.py']=o.raw_sha(path)
        c['source']={'files':{'science.py':o.raw_sha(source)},'files_sha256':o.canonical({'science.py':o.raw_sha(source)})}
        c['operation']={'files':files,'files_sha256':o.canonical(files)}
        controls={}
        for name in ('finalize_once','test_finalize_offline'):
            path=self.bundle/'control'/(name+'.py');path.parent.mkdir(exist_ok=True);path.write_text('# immutable control fixture\n')
            controls['control/'+name+'.py']=o.raw_sha(path)
        integrity={'schema':'titantpp_contextual_retrieval_control_integrity_v1','contract_sha256':o.canonical(c),
            'files':controls,'files_sha256':o.canonical(controls),'automatic_retry':False}
        o.write(self.bundle/'execution_contract.json',c);o.write(self.bundle/'control/integrity_receipt.json',integrity)
        current={'schema':o.NAME,'canonical_sha256':o.canonical(c),'hosts':{h:s['root'] for h,s in c['hosts'].items()},
            'source_closure_sha256':c['source']['files_sha256'],'operation_closure_sha256':c['operation']['files_sha256'],
            'control_integrity_receipt_sha256':o.raw_sha(self.bundle/'control/integrity_receipt.json'),'control_closure_sha256':o.canonical(controls)}
        o.write(self.bundle/'current.json',current)
        return c
    def test_modified_operation_rejected_before_exec_module(self):
        c=self.sealed_fixture();self.assertEqual(o.preflight(self.bundle),c)
        (self.bundle/'operation/monitor.py').write_text("raise AssertionError('unverified code executed')\n")
        with mock.patch.object(o.importlib.util,'spec_from_file_location') as importer:
            with self.assertRaisesRegex(ValueError,'operation file changed before import'):o.load_monitor(self.bundle)
            importer.assert_not_called()
    def test_source_and_control_bytes_rejected_before_import(self):
        self.sealed_fixture()
        for relative in ('source/science.py','control/finalize_once.py'):
            path=self.bundle/relative;original=path.read_bytes();path.write_bytes(original+b'# altered\n')
            with mock.patch.object(o.importlib.util,'spec_from_file_location') as importer:
                with self.assertRaises(ValueError):o.load_monitor(self.bundle)
                importer.assert_not_called()
            path.write_bytes(original)
    def test_current_closure_pointer_rejected_before_import(self):
        self.sealed_fixture();path=self.bundle/'current.json';current=o.read(path)
        for key in ('canonical_sha256','source_closure_sha256','operation_closure_sha256','control_integrity_receipt_sha256','control_closure_sha256'):
            altered=dict(current);altered[key]='0'*64;path.write_text(json.dumps(altered))
            with mock.patch.object(o.importlib.util,'spec_from_file_location') as importer:
                with self.assertRaises(ValueError):o.load_monitor(self.bundle)
                importer.assert_not_called()
        path.write_text(json.dumps(current))
    def test_latest_binds_written_report_and_markdown_raw_bytes(self):
        _,path=self.run_report(self.forbidden);latest=o.read(self.reports/'latest.json')
        self.assertEqual(latest['markdown_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(latest['report_sha256'],hashlib.sha256(Path(latest['report']).read_bytes()).hexdigest())
        self.assertNotEqual(latest['report_sha256'],digest(o.read(latest['report'])))
    def test_reports_append_without_replacing_old_record(self):
        first,path1=self.run_report(self.forbidden);raw=path1.read_bytes()
        second,path2=self.run_report(self.forbidden)
        self.assertNotEqual(path1,path2);self.assertEqual(path1.read_bytes(),raw)
        self.assertEqual(len(list(self.reports.glob('*.md'))),2)
        self.assertEqual(o.read(self.reports/'latest.json')['markdown'],str(path2))


if __name__=='__main__':unittest.main()
