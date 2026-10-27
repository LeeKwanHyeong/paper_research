"""Preserve stdout, stderr and parent-observed exits for local qualification."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('label');p.add_argument('--timeout',type=int,default=300)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command
if cmd and cmd[0]=='--':cmd=cmd[1:]
assert cmd
dest=Path(__file__).resolve().parent/'qualification'/a.label
dest.mkdir(parents=True,exist_ok=False)
start=time.time();timed=False
try:
 r=subprocess.run(cmd,capture_output=True,timeout=a.timeout)
 out,err,code=r.stdout,r.stderr,r.returncode
except subprocess.TimeoutExpired as e:
 out,err,code,timed=e.stdout or b'',e.stderr or b'',124,True
for name,data in [('stdout.log',out),('stderr.log',err)]:
 with (dest/name).open('xb') as f:f.write(data)
rec={'argv':cmd,'actual_child_exit_code':None if timed else code,'wrapper_exit_code':code,
     'timeout':timed,'elapsed_seconds':time.time()-start,
     'streams':{n:{'bytes':(dest/n).stat().st_size,'sha256':hashlib.sha256((dest/n).read_bytes()).hexdigest()} for n in ['stdout.log','stderr.log']}}
with (dest/'ACTUAL_EXIT.json').open('x') as f:json.dump(rec,f,indent=2);f.write('\n')
print(json.dumps(rec));print(out.decode(errors='replace')[-7000:]);print(err.decode(errors='replace')[-7000:],file=sys.stderr)
raise SystemExit(code)
