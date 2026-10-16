import pathlib,json,hashlib,tarfile,subprocess,time,sys
root=pathlib.Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_instacart_test_20261012_v1_5080")
archive=pathlib.Path("/tmp/titantpp_frozen_test_22db16310ed0b1af96b3fda32980f46095e387bc5c36e51531b2b964d14d3888.tar.gz")
expected_archive="2258906b00b409963e87f359abf347788058702c0d725c0ca1996a06498e7c5a"
expected_seal="22db16310ed0b1af96b3fda32980f46095e387bc5c36e51531b2b964d14d3888"
expected_contract="f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c"
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
require(not root.exists(),"Isolated root already exists; deployment repeat prohibited")
require(sha(archive)==expected_archive,"Transport archive SHA mismatch")
require(not pathlib.Path('/var/run/reboot-required').exists(),"Reboot pending")
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
require(len(gpu.splitlines())==1 and gpu.split(',')[0].strip()=='GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6','GPU identity mismatch')
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
require(not compute,'GPU busy; no deployment')
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();require(len(members)==499,'Wrong member count')
 for m in members:
  p=pathlib.PurePosixPath(m.name)
  require(m.isfile() and not p.is_absolute() and '..' not in p.parts,'Unsafe archive member')
 require(len({m.name for m in members})==len(members),'Duplicate archive paths')
 root.mkdir(parents=True,exist_ok=False)
 t.extractall(root,filter='data')
require(sha(root/'operation_integrity.json')==expected_seal,'Operation seal SHA mismatch')
seal=json.loads((root/'operation_integrity.json').read_text());inputs=json.loads((root/'input_integrity.json').read_text())
require(seal['contract_sha256']==inputs['contract_sha256']==expected_contract,'Contract mismatch')
for rel,expected in {**inputs['files'],**seal['files']}.items():
 target=(root/rel).resolve();require(target.is_relative_to(root),'Manifest path escapesroot')
 require(sha(target)==expected,'Extracted file SHA mismatch: '+rel)
receipt={'status':'verified','contract_sha256':expected_contract,'operation_integrity_sha256':expected_seal,'archive_sha256':expected_archive,'remote_root':str(root),'verified_member_count':499,'observed_unix':time.time(),'gpu':gpu,'gpu_idle_observed':True,'python_executable':sys.executable,'reboot_required':False,'training_started':False,'inference_started':False,'automatic_retry':False}
with (root/'control/deployment_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
