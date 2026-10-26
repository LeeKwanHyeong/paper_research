"""Consume verified new local collections; no encoder inference or fitting."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def rec(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    a=argparse.ArgumentParser();a.add_argument('--host',choices=['5080','5090'],required=True);args=a.parse_args()
    run=HERE/'feature_processes'/args.host;run.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',args.host]
    (run/'INTENT.json').write_text(json.dumps(dict(command=cmd,assembler=rec(Path(__file__)),driver=rec(HERE/'data_control.py')),indent=2)+'\n')
    start=time.monotonic()
    with (run/'stdout.log').open('xb') as so,(run/'stderr.log').open('xb') as se:r=subprocess.run(cmd,stdout=so,stderr=se)
    d=dict(actual_exit=r.returncode,elapsed_seconds=time.monotonic()-start,stdout=rec(run/'stdout.log'),stderr=rec(run/'stderr.log'))
    (run/'ACTUAL_EXIT.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True);raise SystemExit(r.returncode)
def worker(host):
    p=HERE/'data_control.py';assert rec(p)['sha256']=='43f2091127941efc3bcfc155db17b7d4843964b8c37a01ace115002907cc3546'
    spec=importlib.util.spec_from_file_location('feature_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s=m.contract();colpath=HERE/'execution_v1'/host/'retrieval/full/COLLECTED.json';c=json.loads(colpath.read_text())
    assert c['status']=='COLLECTED_AND_VERIFIED' and c['all_child_exits_zero'] and c['actual_controller_exit']==0
    for r in [c['archive'],c['pack_record'],c['members'],c['spec']]:m.verify(r)
    for co in sorted(c['cohorts'],key=lambda co:(co['split']!='train',co['checkpoint'])):
        for r in co['source_refs'].values():m.verify(r)
        m.write_cohort(co['seed'],co['checkpoint'],co['split'],co['chunks'],{**co['source_refs'],'collection':rec(colpath)},s)
        m.audit_math(co['seed'],co['checkpoint'],co['split'],s)
if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
