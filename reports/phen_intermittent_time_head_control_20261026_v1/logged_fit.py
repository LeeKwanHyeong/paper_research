"""Observe each local head-fit process exit and preserve complete byte logs."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time,os
HERE=Path(__file__).resolve().parent
def rec(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[52,42],required=True);p.add_argument('--kind',choices=['best','last'],required=True);a=p.parse_args()
    out=HERE/'fit_processes'/f'seed{a.seed}_{a.kind}';out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(HERE/'data_control.py'),'fit','--seed',str(a.seed),'--kind',a.kind]
    (out/'INTENT.json').write_text(json.dumps(dict(command=cmd,driver=rec(HERE/'data_control.py'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
    start=time.monotonic()
    with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:
        result=subprocess.run(cmd,stdout=so,stderr=se,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    receipt=dict(actual_exit=result.returncode,elapsed_seconds=time.monotonic()-start,stdout=rec(out/'stdout.log'),stderr=rec(out/'stderr.log'))
    (out/'ACTUAL_EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
