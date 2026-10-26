"""Bounded transports for this isolated, inference-only artifact directory."""
import argparse,hashlib,json,pathlib,shlex,subprocess,tarfile,time,datetime,sys
P=pathlib.Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rec(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2)
def transport(host,label,command,input_data=None,timeout=45):
    prefix=P/(host+'_'+label)
    save(prefix.with_suffix('.intent.json'),dict(argv=command,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_bytes=len(input_data) if input_data else 0,input_sha256=hashlib.sha256(input_data).hexdigest() if input_data else None))
    start=time.monotonic();exitcode=None;failure=None;exception_message=None
    try:r=subprocess.run(command,input=input_data,capture_output=True,timeout=timeout);out,err,exitcode=r.stdout,r.stderr,r.returncode
    except subprocess.TimeoutExpired as e:out,err=e.stdout or b'',e.stderr or b'';failure='transport_timeout'
    except OSError as e:out,err=b'',b'';failure=type(e).__name__;exception_message=str(e)
    prefix.with_suffix('.stdout').write_bytes(out);prefix.with_suffix('.stderr').write_bytes(err)
    receipt=dict(actual_exit=exitcode,failure=failure,exception_message=exception_message,elapsed_seconds=time.monotonic()-start,stdout=rec(prefix.with_suffix('.stdout')),stderr=rec(prefix.with_suffix('.stderr')))
    save(prefix.with_suffix('.receipt.json'),receipt)
    if exitcode!=0:raise RuntimeError('Transport failed; original preserved: '+str(prefix))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','deploy','launch','status','pack','collect']);ap.add_argument('--host',choices=['5080','5090'],required=True);ap.add_argument('--mode',choices=['qualification','full'],default='qualification');ap.add_argument('--round',default='01');a=ap.parse_args()
    g=json.loads((P/(a.host+'_GUARD.json')).read_text());remote=g['remote_root'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',a.host]
    if a.action=='build':
        files=[p for p in (P/'bundle').rglob('*') if p.is_file()]
        files += [P/'USER_LOG_MSE_GATE_APPROVAL.json',P/'LOG_MSE_GATE_AUTHORIZATION_V8.json',P/'USER_NUMERIC_GATE_APPROVAL.json',P/'NUMERIC_GATE_AUTHORIZATION_V6.json',P/'collector_v5.py',P/'remote_control.py',P/(a.host+'_GUARD.json'),P/(a.host+'_QUALIFICATION_SPEC_v8.json'),P/(a.host+'_FULL_SPEC_v8.json')]
        directories=['bundle/source42/sample_data','bundle/source52/sample_data']
        manifest=P/(a.host+'_DEPLOY_MANIFEST.json');save(manifest,dict(empty_directories=directories,files={str(p.relative_to(P)):{k:v for k,v in rec(p).items() if k!='path'} for p in files}))
        archive=P/(a.host+'_deployment.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in files:tar.add(p,arcname=str(p.relative_to(P)),recursive=False)
            for d in directories:tar.add(P/d,arcname=d,recursive=False)
            tar.add(manifest,arcname='DEPLOY_MANIFEST.json',recursive=False)
        save(P/(a.host+'_DEPLOYMENT.json'),dict(archive=rec(archive),manifest=rec(manifest)))
        print(json.dumps(dict(archive=rec(archive),files=len(files))));return
    if a.action=='deploy':
        d=json.loads((P/(a.host+'_DEPLOYMENT.json')).read_text());archive=pathlib.Path(d['archive']['path']);assert rec(archive)==d['archive']
        # The new directory and incoming archive are exclusive; never overwrite an attempt.
        code='import pathlib,sys,hashlib,tarfile,json; p=pathlib.Path('+repr(remote)+'); p.mkdir(); b=sys.stdin.buffer.read(); assert hashlib.sha256(b).hexdigest()=='+repr(sha(archive))+'; f=p/"incoming.tar.gz"; f.write_bytes(b); t=tarfile.open(f); members=t.getmembers(); assert len(members)==len({m.name for m in members}); assert all((m.isfile() or (m.isdir() and m.name in ["bundle/source42/sample_data","bundle/source52/sample_data"])) and not pathlib.PurePosixPath(m.name).is_absolute() and ".." not in pathlib.PurePosixPath(m.name).parts for m in members); t.extractall(p); d=json.loads((p/"DEPLOY_MANIFEST.json").read_text()); assert d["files"]; assert all((p/k).stat().st_size==v["bytes"] and hashlib.sha256((p/k).read_bytes()).hexdigest()==v["sha256"] for k,v in d["files"].items()); print(json.dumps({"status":"verified","files":len(d["files"]),"manifest_sha256":hashlib.sha256((p/"DEPLOY_MANIFEST.json").read_bytes()).hexdigest()}))'
        out=transport(a.host,'DEPLOY_'+a.round,ssh+[shlex.join(['python3','-c',code])],archive.read_bytes(),180);print(out.decode());return
    base=[g['python'],remote+'/remote_control.py']
    if a.action in ['launch','status','pack']:
        cmd=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1',*base,a.action,'--host',a.host,'--mode',a.mode])
        out=transport(a.host,a.mode.upper()+'_'+a.action.upper()+'_'+a.round,ssh+[cmd],timeout=180 if a.action=='pack' else 45)
        print(out.decode());return
    if a.action=='collect':
        packed=json.loads((P/(a.host+'_'+a.mode.upper()+'_PACK_'+a.round+'.stdout')).read_text());target=P/'retrieval'/a.host/a.mode;target.mkdir(parents=True,exist_ok=False)
        archive=target/'results.tar.gz'
        # Stream archive bytes to a dedicated file so raw binary output is not duplicated in logs.
        cmd=ssh+[shlex.join(['cat',packed['archive']['path']])];pre=target/'TRANSFER'
        save(pre.with_suffix('.intent.json'),dict(argv=cmd,remote_archive=packed['archive']))
        start=time.monotonic()
        code=None;failure=None;exception_message=None
        with archive.open('xb') as output,pre.with_suffix('.stderr').open('x') as err:
            try:r=subprocess.run(cmd,stdout=output,stderr=err,timeout=180);code=r.returncode
            except (subprocess.TimeoutExpired,OSError) as e:failure=type(e).__name__;exception_message=str(e)
        save(pre.with_suffix('.receipt.json'),dict(actual_exit=code,failure=failure,exception_message=exception_message,elapsed_seconds=time.monotonic()-start,archive=rec(archive),stderr=rec(pre.with_suffix('.stderr'))))
        assert code==0 and archive.stat().st_size==packed['archive']['bytes'] and sha(archive)==packed['archive']['sha256']
        originals=target/'originals';originals.mkdir()
        with tarfile.open(archive) as tar:
            ms=tar.getmembers();assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in ms)
            assert len(ms)==len({m.name for m in ms}),'Duplicate archive members'
            tar.extractall(originals)
        manifest=originals/(a.mode+'_RESULT_MEMBERS.json');assert sha(manifest)==packed['manifest']['sha256']
        members=json.loads(manifest.read_text())['files'];assert members
        assert {str(p.relative_to(originals)) for p in originals.rglob('*') if p.is_file()}==set(members)|{manifest.name},'Archive inventory differs'
        for rel,ref in members.items():
            p=originals/rel;assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],rel
        save(target/'COLLECTED.json',dict(archive=rec(archive),manifest=rec(manifest),verified_members=len(members),actual_transport_exit=0))
        print(json.dumps(dict(status='collected',host=a.host,mode=a.mode,files=len(members),bytes=archive.stat().st_size)))
if __name__=='__main__':main()
