"""Deploy a sealed selected-checkpoint evaluation and run each phase once."""
from pathlib import Path
import argparse,base64,hashlib,io,json,shlex,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'search_artifacts/titantpp_retrieval_test_20261008_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def verify():
 c=read(B/'execution_contract.json');m=read(B/'deployment_manifest.json')
 require(sha(B/'execution_contract.json')==m['contract_sha256'],'Contract bytes changed')
 for rel,d in m['files'].items():require(sha(B/rel)==d,'Sealed file changed: '+rel)
 return c,m
DEPLOY='''import sys,hashlib,io,tarfile,pathlib,json
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];buf=sys.stdin.buffer.read();assert hashlib.sha256(buf).hexdigest()==expected
assert not root.exists(),"Preserve existing evaluation root; no overwrite"
with tarfile.open(fileobj=io.BytesIO(buf),mode="r:gz") as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not m.name.startswith("/") and ".." not in pathlib.PurePosixPath(m.name).parts for m in members)
 root.mkdir(parents=True,exist_ok=False)
 tar.extractall(root,filter="data")
m=json.loads((root/"deployment_manifest.json").read_text())
for rel,d in m["files"].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==d,rel
print(json.dumps({"status":"deployed","root":str(root),"contract_sha256":m["contract_sha256"],"package_sha256":expected,"files":len(m["files"])}))
'''
RUN='''import pathlib,subprocess,sys,json,base64,hashlib,os,time
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];split=sys.argv[3];expected=sys.argv[4];py=sys.executable
assert hashlib.sha256((root/"execution_contract.json").read_bytes()).hexdigest()==expected
c=json.loads((root/"execution_contract.json").read_text());request=json.loads(sys.stdin.buffer.read())
assert str(root)==c["hosts"][host]["root"] and py==c["hosts"][host]["python"]
assert not (root/"claims"/(split+"_"+host+".json")).exists(),"Preserve phase claim; no retry"
if split=="test":
 for rel,encoded in request["validation_files"].items():
  assert rel in ("results/validation_5080.json","results/validation_5090.json","validation_gate.json")
  raw=base64.b64decode(encoded);p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw,"Preserve conflicting validation gate/receipt"
  else:
   with p.open("xb") as f:f.write(raw)
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True).strip(),"GPU became busy; no inference started"
args=[py,str(root/"operation/evaluate.py"),"--contract",str(root/"execution_contract.json"),"--host",host,"--split",split]
started=time.time();result=subprocess.run(args,capture_output=True,text=True,timeout=950)
log=root/"logs"/(split+"_"+host+".json");log.parent.mkdir(parents=True,exist_ok=True)
with log.open("x") as f:json.dump({"argv":args,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr,"elapsed_seconds":time.time()-started},f)
p=root/"results"/(split+"_"+host+".json")
raw=p.read_bytes() if p.exists() else b""
print(json.dumps({"status":"complete" if result.returncode==0 else "failed","returncode":result.returncode,"sha256":hashlib.sha256(raw).hexdigest(),"result_base64":base64.b64encode(raw).decode(),"elapsed_seconds":time.time()-started,"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}))
sys.exit(result.returncode)
'''
def remote(host,code,args,input_bytes,c,timeout):
 h=c['hosts'][host];env=h['environment']
 cmd=shlex.join(['env']+[key+'='+value for key,value in sorted(env.items())]+[h['python'],'-c',code]+args)
 return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,cmd],input=input_bytes,capture_output=True,timeout=timeout)
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',choices=['5080','5090'],required=True);p.add_argument('--mode',choices=['deploy','validation','test'],required=True);a=p.parse_args();c,m=verify();h=c['hosts'][a.host]
 if a.mode=='deploy':
  local=B/'deployment'/('receipt_'+a.host+'.json');require(not local.exists(),'Existing deployment preserved')
  stream=io.BytesIO()
  with tarfile.open(fileobj=stream,mode='w:gz') as tar:
   for rel in sorted(list(m['files'])+['deployment_manifest.json']):tar.add(B/rel,arcname=rel,recursive=False)
  buf=stream.getvalue();r=remote(a.host,DEPLOY,[h['root'],hashlib.sha256(buf).hexdigest()],buf,c,180)
  require(r.returncode==0,'Deployment failed: '+r.stderr.decode()[-1600:]);receipt=json.loads(r.stdout);require(receipt['contract_sha256']==m['contract_sha256'],'Deployment receipt differs');write(local,receipt);print(json.dumps({'host':a.host,'mode':a.mode,'status':'deployed'}));return
 local=B/'results'/(a.mode+'_'+a.host+'.json');require(not local.exists(),'Existing result preserved; no inference retry')
 require((B/'deployment'/('receipt_'+a.host+'.json')).is_file(),'Deployment not verified')
 request={}
 if a.mode=='test':
  gate=read(B/'validation_gate.json');require(gate['passed'] and gate['conditions']==10 and gate['contract_sha256']==sha(B/'execution_contract.json'),'Global Validation gate missing')
  request={'validation_files':{rel:base64.b64encode((B/rel).read_bytes()).decode() for rel in ['validation_gate.json','results/validation_5080.json','results/validation_5090.json']}}
 r=remote(a.host,RUN,[h['root'],a.host,a.mode,m['contract_sha256']],json.dumps(request).encode(),c,1000)
 if not r.stdout.strip():
  write(B/'deployment'/('failure_'+a.mode+'_'+a.host+'.json'),{'returncode':r.returncode,'stderr':r.stderr.decode()[-5000:]});raise RuntimeError('Native evaluation did not return a result: '+r.stderr.decode()[-1200:])
 returned=json.loads(r.stdout);raw=base64.b64decode(returned['result_base64']);require(hashlib.sha256(raw).hexdigest()==returned['sha256'],'Returned raw result SHA differs')
 if raw:
  local.parent.mkdir(parents=True,exist_ok=True)
  with local.open('xb') as f:f.write(raw)
 write(B/'deployment'/('execution_'+a.mode+'_'+a.host+'.json'),{k:v for k,v in returned.items() if k!='result_base64'})
 require(r.returncode==0 and returned['status']=='complete','Evaluation failed; original failure preserved')
 print(json.dumps({'host':a.host,'mode':a.mode,'status':'complete','sha256':returned['sha256'],'elapsed_seconds':returned['elapsed_seconds']}))
if __name__=='__main__':main()
