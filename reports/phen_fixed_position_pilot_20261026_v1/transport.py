"""Bounded, recorded transport of the approved two-fit 5080 pilot."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
REMOTE='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/phen_fixed_position_pilot_20261026_v1_5080'
PYTHON='/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','5080']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture(name, argv, *, data=None, timeout=45):
    paths={k:HERE/(name+'.'+k) for k in ('stdout','stderr','receipt.json')}
    assert not any(p.exists() for p in paths.values()), 'Existing transport attempt'
    started=time.time()
    try:
        r=subprocess.run(argv,input=data,capture_output=True,timeout=timeout)
        code,so,se,timed=r.returncode,r.stdout,r.stderr,False
    except subprocess.TimeoutExpired as e:
        code,so,se,timed=None,e.stdout or b'',e.stderr or b'',True
    paths['stdout'].write_bytes(so);paths['stderr'].write_bytes(se)
    receipt={'actual_exit':code,'timeout':timed,'started_unix':started,'elapsed_seconds':time.time()-started,
             **{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in [('stdout',so),('stderr',se)]}}
    paths['receipt.json'].write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));print(so.decode(errors='replace')[:15000]);print(se.decode(errors='replace')[:2000])
    assert code==0 and not timed, 'Transport failed; no automatic retry'
    return so
def remote(code): return SSH+[shlex.join(['env','PYTHONDONTWRITEBYTECODE=1',PYTHON,'-c','REMOTE='+repr(REMOTE)+'\n'+code])]
def main(mode):
    if mode=='package':
        with tarfile.open(HERE/'pilot_bundle.tar.gz','x:gz') as t:
            for p in sorted((HERE/'bundle').rglob('*')):
                if p.is_file():
                    assert not p.is_symlink()
                    t.add(p,arcname=str(p.relative_to(HERE/'bundle')),recursive=False)
        print(json.dumps({'bytes':(HERE/'pilot_bundle.tar.gz').stat().st_size,'sha256':sha(HERE/'pilot_bundle.tar.gz')}))
    elif mode=='deploy':
        code='ARCHIVE_SHA='+repr(sha(HERE/'pilot_bundle.tar.gz'))+'\n'+'''import sys,io,tarfile,hashlib,json
from pathlib import Path
r=Path(REMOTE);blob=sys.stdin.buffer.read();assert hashlib.sha256(blob).hexdigest()==ARCHIVE_SHA
r.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  p=r/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
for name in ['models','utils','sample_data']:(r/'source'/name).mkdir(exist_ok=True)
c=json.loads((r/'execution_contract.json').read_text())
for n,s in {**c['source']['files'],**c['input_files']}.items():assert hashlib.sha256((r/'source'/n).read_bytes()).hexdigest()==s,n
print(json.dumps({'deployed':True,'files_verified':len(c['source']['files'])+len(c['input_files']),'archive_sha256':ARCHIVE_SHA}))
'''
        capture('DEPLOY',remote(code),data=(HERE/'pilot_bundle.tar.gz').read_bytes(),timeout=120)
    elif mode=='launch':
        code='PYTHON='+repr(PYTHON)+'\n'+'''import sys,json,subprocess,shlex,os
from pathlib import Path
r=Path(REMOTE);assert not (r/'parent_attempt.json').exists()
sys.path.insert(0,str(r/'source'))
from paper.scripts.run_phen_fixed_position_pilot import verify_files,gpu_guard
c=json.loads((r/'execution_contract.json').read_text());verify_files(c);guard=gpu_guard(c)
command=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONHASHSEED=42',PYTHON,str(r/'source/pilot_assets/remote_parent.py')])+' > '+shlex.quote(str(r/'parent.stdout'))+' 2> '+shlex.quote(str(r/'parent.stderr'))
p=subprocess.run(['/usr/bin/tmux','new-session','-d','-s','phen-fixed-pos-20261026-v1',command],capture_output=True,text=True)
assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
print(json.dumps({'tmux_launch_actual_exit':p.returncode,'gpu_guard':guard,'source_verified':True}))
'''
        capture('LAUNCH',remote(code))
    elif mode=='status':
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        code='''import json,subprocess
from pathlib import Path
r=Path(REMOTE);out={'jobs':{}}
for name in ['parent_attempt.json','COMPLETE.json','FAILURE.json','dispatch_receipt.json','dispatcher.exit.json']:
 p=r/name
 if p.is_file():out[name]=json.loads(p.read_text())
for name in ['parent.stderr','dispatcher.stderr']:
 p=r/name
 if p.is_file():out[name]=p.read_text()[-3000:]
for d in ['raf_spare_parts','yellow_trip_hourly']:
 name=d+'__fixed_sinusoidal__seed42';p=r/'jobs'/name;v={}
 files=sorted((p/'curve_epochs').glob('epoch_*.json'))
 if files:v['epoch_count']=len(files);v['latest_epoch']=json.loads(files[-1].read_text())
 for suffix in ['launch.json','exit.json']:
  q=r/(name+'.'+suffix)
  if q.is_file():v[suffix]=json.loads(q.read_text())
 q=p/'receipt.json'
 if q.is_file():
  receipt=json.loads(q.read_text());v['receipt']={k:receipt[k] for k in ['status','selected_epoch','completed_epochs','individual_predictions_match']}
 q=r/(name+'.stderr.log')
 if q.is_file():v['stderr_tail']=q.read_text()[-2500:]
 out['jobs'][d]=v
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True,timeout=15).strip()
print(json.dumps(out))
'''
        capture('STATUS_'+stamp,remote(code))
    elif mode=='collect':
        code='''import json,hashlib,tarfile
from pathlib import Path
r=Path(REMOTE);done=json.loads((r/'COMPLETE.json').read_text());manifest=r/'RESULT_MANIFEST.json'
assert done['all_children_actual_exit_zero'] and hashlib.sha256(manifest.read_bytes()).hexdigest()==done['result_manifest_sha256']
files=json.loads(manifest.read_text())['files']
for n,rec in files.items():
 p=r/n;assert p.stat().st_size==rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],n
with tarfile.open(r/'results.tar.gz','x:gz') as t:
 for n in list(files)+['RESULT_MANIFEST.json','COMPLETE.json','parent.stdout','parent.stderr']:t.add(r/n,arcname=n,recursive=False)
p=r/'results.tar.gz';print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
'''
        rec=json.loads(capture('PACK_RESULTS',remote(code),timeout=120))
        target=HERE/'results.tar.gz';assert not target.exists()
        capture('COLLECT',['scp','-o','BatchMode=yes','-o','ConnectTimeout=10','5080:'+REMOTE+'/results.tar.gz',str(target)],timeout=120)
        assert sha(target)==rec['sha256'] and target.stat().st_size==rec['bytes']
        dest=HERE/'retrieval';dest.mkdir(exist_ok=False)
        with tarfile.open(target,'r:gz') as t:
            for m in t.getmembers():
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                p=dest/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
        for n,v in json.loads((dest/'RESULT_MANIFEST.json').read_text())['files'].items():
            assert sha(dest/n)==v['sha256'] and (dest/n).stat().st_size==v['bytes']
        print('All collected bytes and SHA256 verified')
    else: raise ValueError(mode)
if __name__=='__main__':main(sys.argv[1])
