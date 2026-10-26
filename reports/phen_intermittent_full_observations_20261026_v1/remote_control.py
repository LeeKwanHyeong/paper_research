"""Isolated observation job control. No training, existing-process signals or retries."""
import argparse,datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,tarfile,time,traceback
P=pathlib.Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rec(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2)
def proc(pid):
    p=pathlib.Path('/proc')/str(pid)
    s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,exists=True,state=s[0],pgid=int(s[2]),start_ticks=int(s[19]),cwd=os.readlink(p/'cwd'),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def query(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=15)
    if r.returncode:raise RuntimeError(str(args)+': '+r.stderr)
    return r.stdout
def verify_files():
    m=json.loads((P/'DEPLOY_MANIFEST.json').read_text())
    assert m['files']
    for rel in m.get('empty_directories',[]):
        d=(P/rel).resolve();assert d.is_relative_to(P) and d.is_dir() and not any(d.iterdir())
    for rel,r in m['files'].items():
        p=(P/rel).resolve();assert p.is_relative_to(P) and p.is_file()
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],rel
    return sha(P/'DEPLOY_MANIFEST.json')
def verify_claim(pre,spec):
    claim=json.loads(pre.with_suffix('.claim.json').read_text())
    assert verify_files()==claim['manifest_sha256'],'Manifest differs from launch claim'
    assert sha(spec)==claim['spec_sha256'],'Spec differs from launch claim'
def verify_output(out,spec):
    s=json.loads(spec.read_text());complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['spec_sha256']==sha(spec) and complete['collector_sha256']==s['runner_sha256']
    assert complete['actual_child_exits_all_zero'] and len(complete['receipts'])==len(s['jobs'])
    expected={out/'COMPLETE.json'}
    def check(r):
        p=pathlib.Path(r['path']);assert p.is_relative_to(out) and p.is_file()
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];expected.add(p)
    for r in complete['receipts']:check(r)
    for j in s['jobs']:
        d=out/j['id'];receipt=json.loads((out/(j['id']+'.ACTUAL_EXIT.json')).read_text())
        assert receipt['actual_child_exit']==0 and receipt['spec_sha256']==sha(spec) and receipt['collector_sha256']==s['runner_sha256']
        check(receipt['stdout']);check(receipt['stderr'])
        result=json.loads((d/'COMPLETE.json').read_text());expected|={d/'INTENT.json',d/'COMPLETE.json'}
        assert result['status']==('complete' if s['mode']=='full' else 'qualification_complete')
        assert result['checkpoint_sha256']==j['checkpoint_sha256'] and set(result['splits'])==set(j['splits'])
        for split,v in result['splits'].items():
            path=d/(split+'_manifest.json');assert json.loads(path.read_text())==v;expected.add(path)
            assert v['rows']==(s['population_counts'][split] if s['mode']=='full' else 128*s['qualification_batches'])
            assert sum(c['rows'] for c in v['chunks'])==v['rows'];check(v['progress'])
            for c in v['chunks']:
                check(c['observations'])
                if v['head_hidden_saved']:check(c['head_hidden'])
    assert expected=={p for p in out.rglob('*') if p.is_file()},'Missing or extra output member'
def guard(g):
    assert str(P)==g['remote_root']
    assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==g['boot_id'],'Boot changed'
    gpu=query(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']).strip().split(',')
    assert len(gpu)==2 and gpu[0].strip()==g['gpu_uuid'],'GPU changed'
    assert float(gpu[1])>=g['minimum_free_mib'],'Insufficient available GPU memory'
    protected=[proc(x['pid']) for x in g['protected']]
    assert protected==g['protected'],'Protected process changed'
    apps=query(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    pids=[int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()]
    allowed={x['pid'] for x in protected if x['state']=='T'}
    assert set(pids)<=allowed,'Another GPU compute process is present'
    return dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_free_mib=float(gpu[1]),compute_pids=pids,protected=protected,boot_id=g['boot_id'],gpu_uuid=g['gpu_uuid'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','run','status','pack']);ap.add_argument('--host',choices=['5080','5090'],required=True);ap.add_argument('--mode',choices=['qualification','full'],required=True);a=ap.parse_args()
    g=json.loads((P/(a.host+'_GUARD.json')).read_text());control=P/'control';control.mkdir(exist_ok=True)
    pre=control/a.mode;out=P/('output_'+a.mode);spec=P/(a.host+'_'+a.mode.upper()+'_SPEC_v2.json')
    if a.action=='launch':
        binding=verify_files();before=guard(g)
        assert not out.exists() and not pre.with_suffix('.claim.json').exists(),'Existing attempt/partial'
        if a.mode=='full':
            qr=json.loads((control/'qualification.ACTUAL_EXIT.json').read_text());assert qr['actual_collector_exit']==0 and qr['post_guard_passed']
            assert json.loads((P/'output_qualification'/'COMPLETE.json').read_text())['status']=='qualification_complete'
        assert pathlib.Path(g['python']).is_file() and pathlib.Path(g['tmux']).is_file()
        name='phen-full-obs-'+a.host+'-'+a.mode+'-20261026'
        save(pre.with_suffix('.claim.json'),dict(host=a.host,mode=a.mode,guard=before,manifest_sha256=binding,spec_sha256=sha(spec),session=name))
        cmd=shlex.join([g['python'],str(P/'remote_control.py'),'run','--host',a.host,'--mode',a.mode])
        query([g['tmux'],'new-session','-d','-s',name,'env PYTHONDONTWRITEBYTECODE=1 '+cmd])
        print(json.dumps(dict(status='launched',session=name,claim=rec(pre.with_suffix('.claim.json')))),flush=True);return
    if a.action=='run':
        code=None;post=False;started=time.monotonic()
        try:
            verify_claim(pre,spec);save(pre.with_suffix('.before_guard.json'),guard(g));save(pre.with_suffix('.supervisor.json'),proc(os.getpid()))
            cmd=[g['python'],str(P/'collector_v2.py'),'--spec',str(spec),'--spec-sha',sha(spec),'--output',str(out)]
            with pre.with_suffix('.stdout').open('x') as stdout,pre.with_suffix('.stderr').open('x') as stderr:
                child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':''})
                try:save(pre.with_suffix('.collector_identity.json'),proc(child.pid))
                finally:code=child.wait()
            if code!=0:raise RuntimeError('Collector actual exit '+str(code))
            verify_claim(pre,spec);save(pre.with_suffix('.after_guard.json'),guard(g));post=True
        except BaseException as e:
            save(pre.with_suffix('.FAILURE.json'),dict(error=str(e),traceback=traceback.format_exc(),actual_collector_exit=code,retry_attempted=False))
        finally:
            save(pre.with_suffix('.ACTUAL_EXIT.json'),dict(actual_collector_exit=code,post_guard_passed=post,elapsed_seconds=time.monotonic()-started,spec=rec(spec),stdout=rec(pre.with_suffix('.stdout')) if pre.with_suffix('.stdout').exists() else None,stderr=rec(pre.with_suffix('.stderr')) if pre.with_suffix('.stderr').exists() else None,supervisor_own_os_exit_directly_observed=False))
        if not post:raise SystemExit(1)
        return
    if a.action=='status':
        result=dict(mode=a.mode,claimed=pre.with_suffix('.claim.json').exists(),complete=out.joinpath('COMPLETE.json').exists())
        for ext in ['ACTUAL_EXIT.json','FAILURE.json','collector_identity.json']:
            path=pre.with_suffix('.'+ext)
            if path.exists():result[ext]=json.loads(path.read_text())
        result['jobs']=[]
        if out.exists():
            for job in sorted(out.iterdir()):
                if job.is_dir():
                    progress={}
                    for p in sorted(job.glob('*_progress.jsonl')):
                        text=p.read_text();lines=text.split('\n')[:-1]
                        progress[p.stem]=[json.loads(x) for x in lines[-1:]]
                    result['jobs'].append(dict(id=job.name,complete=(job/'COMPLETE.json').exists(),progress=progress,failure=json.loads((job/'FAILURE.json').read_text()) if (job/'FAILURE.json').exists() else None))
        print(json.dumps(result),flush=True);return
    if a.action=='pack':
        receipt=json.loads(pre.with_suffix('.ACTUAL_EXIT.json').read_text());assert receipt['actual_collector_exit']==0 and receipt['post_guard_passed']
        verify_claim(pre,spec);verify_output(out,spec);before=guard(g);archive=P/(a.mode+'_results.tar.gz');manifest=P/(a.mode+'_RESULT_MEMBERS.json')
        assert not archive.exists() and not manifest.exists(),'Existing collection/partial'
        files=[p for p in out.rglob('*') if p.is_file()]+[p for p in control.glob(a.mode+'.*') if p.is_file()]
        save(manifest,dict(guard=before,files={str(p.relative_to(P)):rec(p) for p in files}))
        with tarfile.open(archive,'x:gz') as tar:
            for p in [*files,manifest]:tar.add(p,arcname=str(p.relative_to(P)),recursive=False)
        print(json.dumps(dict(archive=rec(archive),manifest=rec(manifest))),flush=True)
if __name__=='__main__':main()
