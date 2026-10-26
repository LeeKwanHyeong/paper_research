import datetime,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
def rec(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
spec=P/'CPU_QUALIFICATION_SPEC_v2.json';prefix=P/'CPU_QUALIFICATION_02'
cmd=['/usr/local/bin/python3',str(P/'collector_v2.py'),'--spec',str(spec),'--spec-sha',rec(spec)['sha256'],'--output',str(P/'cpu_qualification_v2')]
with prefix.with_suffix('.intent.json').open('x') as f:json.dump(dict(argv=cmd,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2)
start=time.monotonic()
with prefix.with_suffix('.stdout').open('x') as out,prefix.with_suffix('.stderr').open('x') as err:
    r=subprocess.run(cmd,stdout=out,stderr=err,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
d=dict(actual_exit=r.returncode,elapsed_seconds=time.monotonic()-start,stdout=rec(prefix.with_suffix('.stdout')),stderr=rec(prefix.with_suffix('.stderr')),spec=rec(spec))
with prefix.with_suffix('.receipt.json').open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d));raise SystemExit(r.returncode)
