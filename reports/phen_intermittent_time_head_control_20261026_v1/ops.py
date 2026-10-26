"""Bounded transports for this approved isolated extraction, with raw receipts."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,subprocess,sys,time
HERE=Path(__file__).resolve().parent
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
def rec(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def run(host,action,cmd,input_data=None,timeout=45):
    d=HERE/'transports';d.mkdir(exist_ok=True)
    prefix=d/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+host+'_'+action)
    save(prefix.with_suffix('.intent.json'),dict(command=cmd,timeout=timeout,input_bytes=len(input_data or b''),input_sha256=hashlib.sha256(input_data or b'').hexdigest()))
    start=time.monotonic()
    try:
        r=subprocess.run(cmd,input=input_data,capture_output=True,timeout=timeout)
        output,error,code=r.stdout,r.stderr,r.returncode;timed_out=False
    except subprocess.TimeoutExpired as e:
        output,error,code=e.stdout or b'',e.stderr or b'',None;timed_out=True
    prefix.with_suffix('.stdout').write_bytes(output);prefix.with_suffix('.stderr').write_bytes(error)
    receipt=dict(actual_exit=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,stdout=rec(prefix.with_suffix('.stdout')),stderr=rec(prefix.with_suffix('.stderr')))
    save(prefix.with_suffix('.receipt.json'),receipt)
    print(json.dumps(dict(host=host,action=action,**receipt)),flush=True)
    if code!=0 or timed_out:raise SystemExit(1)
    if action!='fetch':
        try: print(json.dumps(json.loads(output)),flush=True)
        except (ValueError,UnicodeDecodeError):print(output.decode(errors='replace')[:1000],flush=True)
    return output
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','launch','status','pack','fetch','upload-fullspec']);p.add_argument('--host',choices=['5080','5090'],required=True);p.add_argument('--mode',choices=['qualification','full'],default='qualification');p.add_argument('--destination',type=Path);a=p.parse_args()
    e=HERE/'execution_v1'/a.host;config=json.loads((e/'PREPARED.json').read_text());remote=config['remote_root'];python=config['python']
    if a.action=='deploy':
        archive=e/'deploy.tar.gz';assert rec(archive)['sha256']==config['archive']['sha256']
        source="""import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()==sys.argv[2]
assert root.is_absolute() and root.name.startswith('phen_intermittent_time_head_control_20261026_v1_') and not root.exists()
root.mkdir();(root/'DEPLOY_UPLOAD.tar.gz').write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
 for m in t.getmembers():
  assert not m.issym() and not m.islnk() and not m.isdev() and (root/m.name).resolve().is_relative_to(root)
 t.extractall(root,filter='data')
sys.path.insert(0,str(root));import remote_control as c
print(json.dumps({'status':'deployed','manifest_sha256':c.verify_files(),'files_verified':True}))
"""
        return run(a.host,a.action,SSH+[a.host,shlex.join([python,'-c',source,remote,config['archive']['sha256']])],archive.read_bytes(),timeout=120)
    if a.action=='upload-fullspec':
        path=e/'stage'/(a.host+'_FULL_SPEC.json');blob=path.read_bytes()
        source="import pathlib,hashlib,sys,json;p=pathlib.Path(sys.argv[1]);b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==sys.argv[2];f=p.open('xb');f.write(b);f.close();print(json.dumps({'path':str(p),'bytes':len(b),'sha256':sys.argv[2]}))"
        return run(a.host,a.action,SSH+[a.host,shlex.join([python,'-c',source,remote+'/'+path.name,rec(path)['sha256']])],blob)
    if a.action=='fetch':
        assert a.destination and not a.destination.exists()
        source="import pathlib,sys;p=pathlib.Path(sys.argv[1]);sys.stdout.buffer.write(p.read_bytes())"
        blob=run(a.host,a.action,SSH+[a.host,shlex.join([python,'-c',source,remote+'/'+a.mode+'_results.tar.gz'])],timeout=120)
        a.destination.parent.mkdir(parents=True,exist_ok=True)
        with a.destination.open('xb') as f:f.write(blob)
        save(a.destination.with_suffix('.receipt.json'),rec(a.destination));return
    spec=e/'stage'/(a.host+'_'+a.mode.upper()+'_SPEC.json')
    command=[python,remote+'/remote_control.py',a.action,'--host',a.host,'--mode',a.mode,'--spec-sha',rec(spec)['sha256']]
    run(a.host,a.action+'_'+a.mode,SSH+[a.host,shlex.join(command)],timeout=45 if a.action!='pack' else 120)
if __name__=='__main__':main()
