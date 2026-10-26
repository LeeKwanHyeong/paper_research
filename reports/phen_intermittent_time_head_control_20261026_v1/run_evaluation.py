"""Single selected-head evaluation with externally observed local process exit."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def rec(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    out=HERE/'evaluation_process';out.mkdir(exist_ok=False)
    go=HERE/'EVALUATION_GO.json';gate=json.loads(go.read_text());assert gate['approved']
    source=HERE/'analysis/evaluate_selected.py';assert rec(source)['sha256']==gate['analysis_code_sha256']
    cmd=[sys.executable,str(source),'--output',str(HERE/'results_v1'),'--go-receipt',str(go)]
    (out/'INTENT.json').write_text(json.dumps(dict(command=cmd,source=rec(source),go=rec(go)),indent=2)+'\n')
    start=time.monotonic()
    with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:
        result=subprocess.run(cmd,stdout=so,stderr=se,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(HERE/'plot_cache'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    r=dict(actual_exit=result.returncode,elapsed_seconds=time.monotonic()-start,stdout=rec(out/'stdout.log'),stderr=rec(out/'stderr.log'))
    (out/'ACTUAL_EXIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);raise SystemExit(result.returncode)
if __name__=='__main__':main()
