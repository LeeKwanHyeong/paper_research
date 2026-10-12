"""Explicit host-local lifecycle with the corrected frozen-interface v5 contract.

``check`` uses only Python's standard library and makes no remote call. Every
other mode requires a fresh user authorization and a digest-bound hold release.
No mode retries, resumes, polls, changes shared Runtime, or removes evidence.
Every invocation targets exactly one host. Qualification completion is fetched
only by an explicit ``status`` invocation; another host's approval or observation
is never requested. The immutable v1/v2/v3/v4 launchers, bundles and administrative
bridges remain unchanged. Health uses read-only inode/device lock observation
and a pinned shutdown-waiter identity; real or unknown updaters still block.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "titantpp_cnn_gru_components_20261012_v5"
PARENT_SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
REVISION = "57da14a2a7a5687e1bdef7d3164f3e8c0051da59"
FROZEN_DESIGN_SHA256 = "d3a579750f60c037e45b881ee03ad27a211bdbcc61f2677a4fa6ca128b0336a5"
HOSTS = ("5080", "5090")
ARMS = ("titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_cnn_only",
        "titantpp_cnn_gru54_gru_only", "titantpp_cnn_gru54")
FULL, CNN, GRU = ARMS[3], ARMS[1], ARMS[2]
ROOT_PARENT = Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts")
GPU_UUIDS = {"5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6",
             "5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"}
PYTHONS = {"5080": "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12",
           "5090": str(ROOT_PARENT / "titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12")}
MAX_JSON_BYTES = 64 * 1024 ** 2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    require(Path(path).stat().st_size <= MAX_JSON_BYTES, "Oversized JSON")
    return json.loads(Path(path).read_text())


def safe(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts), "Unsafe relative path")
    resolved = (Path(root) / path).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), "Owned root escape")
    return resolved


def write_exclusive(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    require(len(raw) <= MAX_JSON_BYTES, "Oversized receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def expected_members(c, contract_raw):
    result = {"execution_contract.json": contract_raw, **c["operation"]["files"]}
    result.update({"source/" + name: digest for name, digest in {
        **c["source"]["files"], **c["input_files"]}.items()})
    return result


def inspect_archive(path, expected):
    """Reject links, traversal, duplicates, extras and altered bytes before writes."""
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in seen,
                "Archive member is unsafe, duplicate or unsealed")
            safe(Path("/sealed_archive"), member.name)
            require(0 <= member.size <= 8 * 1024 ** 3, "Oversized archive member")
            stream = archive.extractfile(member)
            require(stream is not None, "Unreadable archive member")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 ** 2), b""):
                digest.update(block)
            require(digest.hexdigest() == expected[member.name], "Archive member SHA mismatch")
            seen.add(member.name)
    require(seen == set(expected), "Archive members incomplete")


def validate_contract(c):
    require(c["schema"] == SCHEMA and c["stage"] == "components", "Foreign study")
    require(c.get("design_sha256") == FROZEN_DESIGN_SHA256, "Frozen time-interface design binding missing or changed")
    require(c["arms"] == list(ARMS) and c["seeds"] == [42, 52, 62], "Arm or seed drift")
    assignments = {"5080": [[d, s] for d in ("yellow_trip_hourly", "raf_spare_parts") for s in (42, 52, 62)],
        "5090": [["intermittent_frozen_5000", s] for s in (42, 52, 62)]}
    require(c["assignments"] == assignments and set(c["hosts"]) == set(HOSTS), "Host assignment drift")
    expected = [{"id": f"{d}__{s}__{arm}", "host": host, "dataset": d, "seed": s, "arm": arm}
        for host, groups in assignments.items() for d, s in groups for arm in ARMS]
    require(c["jobs"] == expected and len(expected) == 36, "36-job identity drift")
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
        "Train/Validation boundary drift")
    require(c["training"] == {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40,
        "patience": 40, "monitor": "validation_raw_quantity_rmse",
        "tie": "strict_earliest_finite_minimum", "warm_start": False}, "Training/selector drift")
    require(c["parallelism"] == {"maximum_workers": 3,
        "grouping": "same_dataset_seed_four_arms_waves_3_plus_1",
        "per_process_cuda_fraction": .30, "aggregate_gpu_fraction": .8}, "Concurrency drift")
    source = c["source"]
    old = source["frozen_science_files"]
    require(len(old) == 129 and canonical(old) == source["frozen_science_files_sha256"] == PARENT_SOURCE,
        "Frozen science129 closure drift")
    require(len(source["files"]) == 131 and all(source["files"].get(k) == v for k, v in old.items())
        and set(source["files"]) - set(old) == {"models/TPPs/CountAwareTitanCNNGRUComponents.py",
            "paper/scripts/run_titantpp_cnn_gru_components.py"}, "Source131 extension drift")
    require(source["git_revision"] == source["frozen_science_git_revision"] == REVISION
        and source["extension_provenance"] == "uncommitted component extension pinned by SHA256",
        "Extension provenance drift")
    require(canonical(source["files"]) == source["files_sha256"], "Source closure drift")
    require(len(c["input_files"]) == 7 and not set(source["files"]) & set(c["input_files"]), "Input seal drift")
    require(set(c["operation"]["files"]) == {"operation/campaign.py", "operation/frozen_runtime.py",
        "operation/qualification.py", "operation/pipeline.py"}
        and canonical(c["operation"]["files"]) == c["operation"]["files_sha256"], "Operation4 seal drift")
    for data in c["datasets"]:
        require(canonical(data) == c["dataset_sha256"][data["dataset_id"]], "Dataset definition drift")
    for host, spec in c["hosts"].items():
        root = ROOT_PARENT / (SCHEMA + "_" + host)
        require(spec["root"] == str(root) and spec["source_root"] == str(root / "source")
            and spec["operation_root"] == str(root / "operation") and spec["output_dir"] == str(root / "run"),
            "Remote owned root mismatch")
        require(spec["python"] == PYTHONS[host] and spec["gpu_uuid"] == GPU_UUIDS[host], "GPU/Runtime owner drift")
        require(spec["runtime_expected"] == {"cuda": "13.0", "cudnn": 91900 if host == "5080" else 92000,
            "numpy": "2.4.4", "polars": "1.39.3", "torch": "2.11.0+cu130"}, "Pinned Runtime drift")
        require(spec["environment"]["PYTHONPATH"] == spec["source_root"]
            and spec["environment"]["SOURCE_REVISION"] == REVISION, "Isolated environment drift")
    require(c["limits"]["automatic_retry"] is False and c["limits"]["workers_per_host"] == 3
        and c["limits"]["total_wall_seconds"] == 168 * 3600
        and c["limits"]["per_condition_seconds"] == 36 * 3600, "Lease/retry drift")
    require(c["launch_policy"] == {"host_independent_start": True,
        "all_hosts_qualification_before_any_fit": False, "fresh_user_start_instruction_required": True,
        "automatic_reassignment": False, "continuous_remote_polling": False}, "Independent host start gate drift")


def check_bundle(bundle):
    bundle = Path(bundle).resolve()
    c = read(bundle / "execution_contract.json")
    validate_contract(c)
    digest = canonical(c)
    current, receipt, hold = [read(bundle / name) for name in ("current.json", "preparation_receipt.json", "launch_hold.json")]
    require(current["schema"] == SCHEMA and current["contract_sha256"] == digest
        and current["bundle"] == str(bundle), "Current pointer mismatch")
    require(receipt["contract_sha256"] == digest and receipt["contract_raw_sha256"] == sha(bundle / "execution_contract.json"),
        "Preparation contract receipt mismatch")
    require(hold["contract_sha256"] == digest and hold["hold"] is True, "Original immutable hold changed")
    for name, expected in {**c["source"]["files"], **c["input_files"]}.items():
        require(sha(safe(bundle / "source", name)) == expected, "Source/input mismatch: " + name)
    for name, expected in c["operation"]["files"].items():
        require(sha(safe(bundle, name)) == expected, "Operation mismatch: " + name)
    tools_path = bundle / "local_tools_integrity.json"
    require(sha(tools_path) == receipt["local_tools_integrity_raw_sha256"], "Local tool seal rawSHA mismatch")
    seal = read(tools_path)
    require(bool(seal["files"]) and seal["files_sha256"] == canonical(seal["files"]), "Local tool seal closure mismatch")
    require("paper/scripts/launch_titantpp_cnn_gru_components_v5.py" in seal["files"], "Launcher not sealed")
    for name, expected in seal["files"].items():
        require(sha(safe(PROJECT, name)) == expected, "Local tool/source mismatch: " + name)
    require(sha(bundle / "deployment.tar.gz") == receipt["payload_sha256"], "Payload rawSHA mismatch")
    inspect_archive(bundle / "deployment.tar.gz", expected_members(c, receipt["contract_raw_sha256"]))
    return c, {"status": "prepared_on_hold", "contract_sha256": digest, "source_files": 131,
        "input_files": 7, "operation_files": 4, "remote_calls": 0,
        "hold_raw_sha256": sha(bundle / "launch_hold.json"), "local_tools_integrity_raw_sha256": sha(tools_path)}


def authorization(bundle, c, host):
    require(host in HOSTS, "Unknown requested host; SSH0")
    auth_path = Path(bundle) / ("launch_authorization_" + host + ".json")
    release_path = Path(bundle) / ("launch_release_" + host + ".json")
    require(auth_path.exists() and release_path.exists(), "Fresh user start authorization and hold release required; SSH0")
    approval, release = read(auth_path), read(release_path)
    digest = canonical(c)
    require(approval.get("approved") is True and approval.get("explicit_new_user_instruction") is True
        and isinstance(approval.get("user_instruction"), str) and bool(approval["user_instruction"].strip())
        and approval.get("contract_sha256") == digest and approval.get("hosts") == [host],
        "Explicit new single-host user start approval missing; SSH0")
    require(release.get("released") is True and release.get("contract_sha256") == digest
        and release.get("original_hold_raw_sha256") == sha(Path(bundle) / "launch_hold.json")
        and release.get("authorization_raw_sha256") == sha(auth_path), "Immutable hold release binding mismatch; SSH0")
    return approval


# Source sent to the pinned host Python. It imports no project code, writes only
# the exact newly owned root, and every extraction/member is verified first.
REMOTE_SCRIPT = r'''
import base64,fcntl,hashlib,json,math,os,pathlib,shlex,subprocess,sys,tarfile,time
mode,payload=sys.argv[1],json.load(sys.stdin); c=payload['contract']; host=payload['host']; spec=c['hosts'][host]
root=pathlib.Path(spec['root']); digest=payload['contract_sha256']
def need(ok,msg):
 if not ok: raise ValueError(msg)
def canon(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def safe(name):
 p=pathlib.Path(name); need(not p.is_absolute() and '..' not in p.parts and bool(p.parts),'unsafe path')
 result=(root/p).resolve(); need(result.is_relative_to(root.resolve()),'root escape'); return result
def read(p):
 need(pathlib.Path(p).stat().st_size<=67108864,'oversized receipt'); return json.loads(pathlib.Path(p).read_text())
def write(p,v):
 p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def capture(argv): return subprocess.run(argv,capture_output=True,text=True,check=True,timeout=20).stdout
def start_lease():
 now=time.time(); previous=c.get('operational_replacement',{}).get('host_leases',{}).get(host)
 if previous is not None:
  started,deadline=previous['started_at_unix'],previous['deadline_unix']
  need(type(started) in (int,float) and type(deadline) in (int,float) and math.isfinite(started) and math.isfinite(deadline),'nonfinite inherited lease')
  need(deadline-started==c['limits']['total_wall_seconds'] and started<=now<deadline,'expired or extended inherited lease')
 else: started,deadline=now,now+c['limits']['total_wall_seconds']
 result={'host':host,'contract_sha256':digest,'approval_sha256':canon(approval),'started_at_unix':started,'deadline_unix':deadline}
 if previous is not None: result['inherited_host_lease']=previous
 return result
need(canon(c)==digest,'contract mismatch')
need(root==pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts')/(c['schema']+'_'+host),'foreign root')
need(pathlib.Path(sys.executable).resolve()==pathlib.Path(spec['python']).resolve(),'foreign Python')
if mode=='health':
 rows=[x.strip().split(',') for x in capture(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader,nounits']).splitlines() if x.strip()]
 selected=[r for r in rows if r[0].strip()==spec['gpu_uuid']]; need(len(selected)==1,'GPU UUID missing')
 need(spec['gpu_name_contains'] in selected[0][1],'GPU name mismatch')
 pids=[]
 for row in capture(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines():
  fields=[x.strip() for x in row.split(',')]; need(len(fields)==2,'unresolved GPU process inventory')
  if fields[0]==spec['gpu_uuid']: pids.append(int(fields[1]))
 updates=[]; idle_shutdown_observers=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit(): continue
  try:
   comm=(p/'comm').read_text().strip()
   if comm in ('apt','apt-get','dpkg','unattended-upgr','packagekitd'):
    observer=False
    if comm=='unattended-upgr':
     try:
      before=(p/'stat').read_text().rsplit(')',1)[1].split()
      argv=(p/'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
      status=(p/'status').read_text()
      after=(p/'stat').read_text().rsplit(')',1)[1].split()
      expected=['/usr/bin/python3','/usr/share/unattended-upgrades/unattended-upgrade-shutdown','--wait-for-signal']
      if argv==expected and before[19]==after[19] and after[0]=='S' and int(after[1])==1:
       uid=next(line.split()[1:] for line in status.splitlines() if line.startswith('Uid:'))
       helper_sha=sha(expected[1])
       shutdown=capture(['busctl','get-property','org.freedesktop.login1','/org/freedesktop/login1','org.freedesktop.login1.Manager','PreparingForShutdown']).strip()
       observer=uid==['0','0','0','0'] and helper_sha=='8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0' and shutdown=='b false'
       if observer: idle_shutdown_observers.append({'pid':int(p.name),'argv':argv,'state':after[0],'ppid':1,'start_ticks':after[19],'helper_source_sha256':helper_sha,'PreparingForShutdown':False})
     except (OSError,ValueError,StopIteration,subprocess.SubprocessError): observer=False
    if not observer: updates.append({'pid':int(p.name),'comm':comm})
  except (FileNotFoundError,PermissionError,ProcessLookupError): pass
 locks=[]
 import re
 lock_text=pathlib.Path('/proc/locks').read_text()
 locked_objects=set()
 for row in lock_text.splitlines():
  for token in row.split():
   match=re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9]+)',token)
   if match:
    locked_objects.add((int(match.group(1),16),int(match.group(2),16),int(match.group(3))))
 for name in ('/var/lib/dpkg/lock','/var/lib/dpkg/lock-frontend','/var/lib/apt/lists/lock','/var/cache/apt/archives/lock'):
  path=pathlib.Path(name)
  if not path.exists(): continue
  stat=path.stat()
  if (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino) in locked_objects: locks.append(name)
 probe_code="import json,sys,torch,numpy,polars; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,'polars':polars.__version__}))"
 probe=subprocess.Popen([spec['python'],'-c',probe_code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 probe_out,probe_err=probe.communicate(timeout=60); need(probe.returncode==0,'native Runtime probe failed '+probe_err)
 runtime=json.loads(probe_out); runtime['gpu']={'uuid':spec['gpu_uuid'],'driver':selected[0][2].strip()}
 pids=[]
 for row in capture(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines():
  fields=[x.strip() for x in row.split(',')]; need(len(fields)==2,'post-probe GPU inventory unresolved')
  if fields[0]==spec['gpu_uuid']: pids.append(int(fields[1]))
 need(runtime['python']=='3.12.13' and all(runtime[k]==v for k,v in spec['runtime_expected'].items()),'Runtime mismatch')
 blocked=(bool(pids) or bool(updates) or bool(locks) or pathlib.Path('/var/run/reboot-required').exists())
 if payload.get('require_root_absent'): blocked=blocked or root.exists()
 result={'status':'waiting' if blocked else 'ready','host':host,'contract_sha256':digest,'root':str(root),'root_exists':root.exists(),'gpu_uuid':spec['gpu_uuid'],'gpu_pids':pids,'reboot_required':pathlib.Path('/var/run/reboot-required').exists(),'update_processes':updates,'idle_shutdown_observers':idle_shutdown_observers,'package_locks':locks,'runtime':runtime,'native_probe':{'pid':probe.pid,'actual_child_exit_observed':True,'returncode':probe.returncode,'gpu_inventory_after_probe_exit':True},'observed_unix':time.time()}
elif mode=='reserve':
 need(not root.exists(),'existing root preserved')
 root.mkdir(); (root/'incoming').mkdir()
 write(root/'deployment_intent.json',{'host':host,'contract_sha256':digest,'payload_sha256':payload['payload_sha256'],'started_unix':time.time()})
 result={'status':'reserved','host':host,'root':str(root),'contract_sha256':digest}
else:
 if mode=='deploy':
  intent=read(root/'deployment_intent.json'); need(intent['contract_sha256']==digest and intent['payload_sha256']==payload['payload_sha256'],'deployment intent mismatch')
  archive_path=safe('incoming/deployment_'+payload['payload_sha256']+'.tar.gz'); need(sha(archive_path)==payload['payload_sha256'],'payload mismatch')
  expected=payload['expected_members']; seen=set()
  with tarfile.open(archive_path,'r:gz') as archive:
   members=archive.getmembers()
   for member in members:
    need(member.isfile() and member.name in expected and member.name not in seen,'unsafe archive member'); safe(member.name)
    stream=archive.extractfile(member); h=hashlib.sha256()
    for b in iter(lambda:stream.read(1048576),b''): h.update(b)
    need(h.hexdigest()==expected[member.name],'member SHA mismatch'); seen.add(member.name)
   need(seen==set(expected),'incomplete archive')
   for member in members:
    target=safe(member.name); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as output,archive.extractfile(member) as source:
     for b in iter(lambda:source.read(1048576),b''): output.write(b)
  approval=payload['approval']; need(approval['approved'] is True and approval['contract_sha256']==digest,'approval mismatch')
  start=start_lease(); write(root/'approval.json',approval)
  write(root/'start_permit.json',start)
  write(root/'deployment_receipt.json',{'host':host,'root':str(root),'contract_sha256':digest,'payload_sha256':payload['payload_sha256'],'source_files_sha256':c['source']['files_sha256'],'status':'deployed','files':expected,'completed_unix':time.time()})
  result={'status':'deployed','host':host,'root':str(root),'contract_sha256':digest,'start_permit':start}
 else:
  need(read(root/'execution_contract.json')==c,'remote contract drift')
  for name,value in {**c['source']['files'],**c['input_files']}.items(): need(sha(safe('source/'+name))==value,'source/input drift '+name)
  for name,value in c['operation']['files'].items(): need(sha(safe(name))==value,'operation drift '+name)
  approval=read(root/'approval.json'); need(approval==payload['approval'],'remote approval drift')
  if mode=='phase':
   phase=payload['phase']; need(phase in ('qualify','train'),'foreign phase')
   for name in ('failure.json','qualification/failure.json','phases/'+phase+'_failure.json','phases/'+phase+'_intent.json','control/'+phase+'_launch_intent.json'):
    need(not safe(name).exists(),'existing intent/failure preserved '+name)
   session=spec['tmux']+'_'+phase
   present=subprocess.run([spec['tmux_binary'],'has-session','-t',session],capture_output=True).returncode
   need(present==1,'tmux session already exists or lookup failed')
   if phase=='train':
    permit=payload['training_permit']; need(permit['contract_sha256']==digest and permit['start_permit_sha256']==canon(read(root/'start_permit.json')),'training permit mismatch')
    for name,raw in payload['qualification_raw'].items():
     target=safe('qualification/host_'+name+'_receipt.json')
     content=base64.b64decode(raw,validate=True); need(hashlib.sha256(content).hexdigest()==permit['qualifications'][name]['raw_sha256'],'bridge SHA mismatch')
     with target.open('xb') as f: f.write(content)
    write(root/'training_permit.json',permit)
   write(root/'control'/(phase+'_launch_intent.json'),{'host':host,'phase':phase,'contract_sha256':digest,'requested_unix':time.time()})
   command=[spec['python'],str(root/'operation/pipeline.py'),'--contract',str(root/'execution_contract.json'),'--host',host,'--phase',phase]
   for key in ('MPLCONFIGDIR','XDG_CACHE_HOME'): pathlib.Path(spec['environment'][key]).mkdir(parents=True,exist_ok=True)
   shell_command='cd '+shlex.quote(spec['source_root'])+' && exec env '+ ' '.join(shlex.quote(k+'='+v) for k,v in spec['environment'].items())+' '+ ' '.join(shlex.quote(v) for v in command)
   subprocess.run([spec['tmux_binary'],'new-session','-d','-s',session,shell_command],check=True,capture_output=True,timeout=20)
   result={'status':'started','host':host,'phase':phase,'root':str(root),'contract_sha256':digest,'argv':command,'phase_os_exit_code':'not_yet_observed','started_unix':time.time()}
  elif mode=='status':
   files={}
   names=['approval.json','start_permit.json','deployment_receipt.json','status.json','failure.json','qualification/failure.json','supervisor.claim']
   names += ['phases/'+phase+'_'+tail+'.json' for phase in ('qualify','train') for tail in ('intent','launch','exit','complete','failure')]
   if (root/'qualification/receipt.json').exists():
    q=read(root/'qualification/receipt.json'); names += ['qualification/receipt.json']+list(q['qualification_files'])
   state=read(root/'status.json') if (root/'status.json').exists() else {}
   checkpoints={}; terminal_hashes={}
   for job in (j for j in c['jobs'] if j['host']==host):
    base='run/'+job['id']; run=base+'/runs/'+job['arm']+'/count_only_log_regression/seed_'+str(job['seed'])
    names += [base+'/status.json',base+'/terminal_manifest.json','claims/'+job['id']+'.json','leases/'+job['id']+'.json','worker_exits/'+job['id']+'.json']
    names += [run+'/'+name for name in ('server_checkpoint_receipt.json','history.json','epoch_timing.json')]
    if safe(run+'/server_checkpoint_receipt.json').exists():
     saved=read(safe(run+'/server_checkpoint_receipt.json')); verified={}
     for name,value in saved['files'].items():
      target=safe(run+'/'+name); verified[name]=target.exists() and sha(target)==value
     checkpoints[job['id']]={'saved_epoch':saved['epoch'],'files_verified':verified,'snapshot_complete':all(verified.values()),'best_selected_file_present':safe(run+'/best_val_qty_rmse_model.pt').exists()}
    if safe(base+'/terminal_manifest.json').exists():
     terminal=read(safe(base+'/terminal_manifest.json')); verified={}
     for name,value in terminal['files'].items():
      target=safe(base+'/'+name); verified[name]=target.exists() and sha(target)==value
     terminal_hashes[job['id']]={'all_files_verified':all(verified.values()),'files':verified}
   for name in dict.fromkeys(names):
    path=safe(name)
    if path.exists():
     need(path.stat().st_size<=67108864,'snapshot file too large'); raw=path.read_bytes()
     files[name]={'raw_sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
   rows=capture(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']); pids=[]
   for row in rows.splitlines():
    fields=[x.strip() for x in row.split(',')]; need(len(fields)==2,'GPU process inventory unresolved')
    if fields[0]==spec['gpu_uuid']: pids.append(int(fields[1]))
   processes={}; process_ids=set(pids)
   if state.get('supervisor_pid'): process_ids.add(state['supervisor_pid'])
   process_ids.update(item['worker_group_pid'] for item in state.get('active_jobs',{}).values())
   for pid in process_ids:
    try:
     proc=pathlib.Path('/proc')/str(pid); stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
     processes[str(pid)]={'pid':pid,'ppid':int(stat[1]),'pgrp':int(stat[2]),'argv':(proc/'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')}
    except (FileNotFoundError,ProcessLookupError): processes[str(pid)]={'pid':pid,'disappeared_during_observation':True}
   result={'host':host,'root':str(root),'contract_sha256':digest,'source_files_sha256':c['source']['files_sha256'],'gpu_uuid':spec['gpu_uuid'],'gpu_pids':pids,'files':files,'processes':processes,'checkpoints':checkpoints,'terminal_hashes':terminal_hashes,'observed_unix':time.time(),'pipeline_os_exit_directly_observed':False,'whole_queue_eta':'unconfirmed'}
  else: raise ValueError('unknown action')
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
'''


class SSHRemote:
    """Bounded transport. No method is called before the complete local gate."""
    def call(self, host, action, payload):
        require(host in HOSTS and payload["host"] == host, "Foreign host")
        spec = payload["contract"]["hosts"][host]
        environment = spec["environment"]
        require(environment.get("PYTHONDONTWRITEBYTECODE", "1") == "1", "Shared Runtime bytecode writes forbidden")
        values = ["env", "PYTHONDONTWRITEBYTECODE=1", *(key + "=" + value for key, value in sorted(environment.items())),
            spec["python"], "-c", REMOTE_SCRIPT, action]
        command = " ".join(shlex.quote(value) for value in values)
        completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
            input=json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            capture_output=True, text=True, timeout=120, check=True)
        require(len(completed.stdout.encode()) <= MAX_JSON_BYTES, "Remote output exceeded receipt bound")
        result = json.loads(completed.stdout)
        result["transport_os_exit_code"] = completed.returncode
        return result

    def upload(self, host, local, remote_path):
        require(host in HOSTS and Path(remote_path).is_relative_to(ROOT_PARENT / (SCHEMA + "_" + host)),
            "Upload outside exact owned root")
        return subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", str(local),
            host + ":" + shlex.quote(str(remote_path))], capture_output=True, text=True,
            timeout=600, check=True).returncode


def intent(bundle, host, action):
    folder = Path(bundle) / "control" / "intents"
    target = folder / (action + "_" + host + ".json")
    exit_path = folder / (action + "_" + host + "_exit.json")
    require(not exit_path.exists(), "Existing action exit preserved; no retry")
    write_exclusive(target, {"host": host, "action": action, "pid": os.getpid(), "ppid": os.getppid(),
        "argv": [sys.executable, *sys.argv], "started_unix": time.time(), "automatic_retry": False})
    return exit_path


def one_call(bundle, remote, host, action, payload, *, label=None):
    exit_path = intent(bundle, host, label or action)
    try:
        result = remote.call(host, action, payload)
        require(result.get("transport_os_exit_code") == 0, "Remote helper OS exit was not observed0")
        write_exclusive(exit_path, {"host": host, "action": action, "transport_os_exit_code": 0,
            "result": result, "completed_unix": time.time(), "scientific_phase_exit_inferred": False})
        return result
    except BaseException as exc:
        write_exclusive(exit_path, {"host": host, "action": action, "status": "failed_or_unobserved",
            "error": type(exc).__name__, "message": str(exc), "completed_unix": time.time(), "automatic_retry": False})
        raise


def validate_health(result, c, host, *, require_root_absent):
    spec = c["hosts"][host]
    require(result["host"] == host and result["root"] == spec["root"]
        and result["contract_sha256"] == canonical(c) and result["gpu_uuid"] == spec["gpu_uuid"], "Health identity mismatch")
    runtime = result["runtime"]
    require(runtime["python"] == "3.12.13" and runtime["gpu"]["uuid"] == spec["gpu_uuid"]
        and all(runtime[k] == v for k, v in spec["runtime_expected"].items()), "Health Runtime drift")
    probe = result["native_probe"]
    require(probe["actual_child_exit_observed"] is True and probe["returncode"] == 0
        and probe["gpu_inventory_after_probe_exit"] is True, "Native Runtime probe exit/GPU reobservation missing")
    return (result["status"] == "ready" and not result["gpu_pids"] and not result["reboot_required"]
        and not result["update_processes"] and not result["package_locks"]
        and (not result["root_exists"] if require_root_absent else result["root_exists"]))


def decode_snapshot(snapshot, c, host):
    require(snapshot["host"] == host and snapshot["root"] == c["hosts"][host]["root"]
        and snapshot["contract_sha256"] == canonical(c) and snapshot["gpu_uuid"] == c["hosts"][host]["gpu_uuid"]
        and snapshot["source_files_sha256"] == c["source"]["files_sha256"], "Snapshot identity mismatch")
    files = {}
    for name, item in snapshot["files"].items():
        safe(Path("/snapshot"), name)
        raw = base64.b64decode(item["base64"], validate=True)
        require(len(raw) <= MAX_JSON_BYTES and hashlib.sha256(raw).hexdigest() == item["raw_sha256"], "Snapshot rawSHA mismatch")
        files[name] = raw
    return files


def qualification_binding(snapshot, c, host):
    files = decode_snapshot(snapshot, c, host)
    require(not snapshot["gpu_pids"], "Native GPU still occupied; qualification is pending")
    require(not any(name.endswith("failure.json") for name in files), "Preserved failure; no automatic retry")
    def json_file(name):
        require(name in files, "Native qualification pending: " + name)
        return json.loads(files[name])
    q, phase_exit, complete = [json_file(name) for name in (
        "qualification/receipt.json", "phases/qualify_exit.json", "phases/qualify_complete.json")]
    require(q["status"] == "passed" and q["host"] == host and q["contract_sha256"] == canonical(c)
        and q["source_files_sha256"] == c["source"]["files_sha256"] and q["held_out_test_evaluated"] is False,
        "Native qualification identity drift")
    require(phase_exit["phase"] == "qualify" and phase_exit["contract_sha256"] == canonical(c)
        and phase_exit["os_exit_code"] == 0 and type(phase_exit["pid"]) is int
        and complete["host"] == host and complete["phase"] == "qualify"
        and complete["contract_sha256"] == canonical(c) and complete["subprocess_os_exit_code"] == 0
        and not complete["gpu_compute_pids"], "Native actual phase exit/completion incomplete")
    launch = json_file("phases/qualify_launch.json")
    require(launch["pid"] == phase_exit["pid"] and launch["ppid"] == phase_exit["ppid"]
        and launch["argv"] == phase_exit["argv"], "Native launch/exit process identity drift")
    runtime = q["runtime"]
    require(runtime["python"] == "3.12.13" and runtime["gpu"]["uuid"] == c["hosts"][host]["gpu_uuid"]
        and all(runtime[k] == v for k, v in c["hosts"][host]["runtime_expected"].items()), "Native qualification Runtime drift")
    native = q["native_qualification_exit"]
    require(native["actual_child_exit_observed"] is True and native["returncode"] == 0,
        "Native qualification actual child OS exit incomplete")
    admission = q["concurrency_admission"]
    require(admission["status"] == "passed" and admission["workers"] == 3
        and admission["arms"] == [FULL, CNN, GRU] and admission["shape"] == [128, 256]
        and admission["contract_sha256"] == canonical(c) and admission["simultaneous_ready_verified"] is True
        and admission["scientific_optimizer_updates"] == 0
        and admission["timing_is_scientific_efficiency_evidence"] is False
        and len(admission["GPU_witness"]["gpu_pids"]) == 3
        and not admission["final_GPU_observation"]["gpu_pids"], "Native concurrency admission incomplete")
    exits = admission["actual_child_exits"]
    require(set(exits) == {FULL, CNN, GRU} and all(item["returncode"] == 0
        and item["actual_child_exit_observed"] is True for item in exits.values())
        and len({item["worker_group_pid"] for item in exits.values()}) == 3, "Three native worker actual exits incomplete")
    for name, expected in q["qualification_files"].items():
        require(name in files and hashlib.sha256(files[name]).hexdigest() == expected, "Native qualification evidence drift")
    return {"receipt": q, "raw_sha256": hashlib.sha256(files["qualification/receipt.json"]).hexdigest(), "os_exit_code": 0}


def host_training_gate(bundle, c, host):
    require(host in HOSTS, "Foreign training host; SSH0")
    bindings, raw = {}, {}
    for host in (host,):
        path = Path(bundle) / "control" / "qualification_cache" / host / "snapshot.json"
        require(path.exists(), "Target host native qualification required before any real fit; SSH0")
        snapshot = read(path)
        bindings[host] = qualification_binding(snapshot, c, host)
        raw[host] = snapshot["files"]["qualification/receipt.json"]["base64"]
    return bindings, raw


def training_status(snapshot, c, host):
    """Report saved evidence and live ownership without inferring fit failures."""
    files = decode_snapshot(snapshot, c, host)
    digest = canonical(c)
    def record(name):
        return json.loads(files[name]) if name in files else None
    failure = next((name for name in files if name == "failure.json" or name == "phases/train_failure.json"), None)
    if failure:
        return {"status": "training_failed", "failure": record(failure), "evidence": failure}
    state = record("status.json") or {}
    active = state.get("active_jobs", {})
    owned = {job["id"]: job for job in c["jobs"] if job["host"] == host}
    jobs = []
    for name, job in owned.items():
        terminal = record("run/" + name + "/terminal_manifest.json")
        item = {"job": job, "status": "waiting", "saved_epoch": None, "best_epoch": None}
        run = "run/" + name + "/runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"])
        if terminal:
            require(terminal["job"] == job and terminal["contract_sha256"] == digest
                and terminal["status"] == "complete" and terminal["scientific_success"] is True
                and terminal["evaluation_scope"] == "validation_only" and terminal["held_out_test_evaluated"] is False,
                "Foreign training terminal")
            item["status"] = "complete" if snapshot.get("terminal_hashes", {}).get(name, {}).get("all_files_verified") is True else "terminal_integrity_unconfirmed"
        elif name in active:
            require(active[name]["job"] == job, "Foreign active job")
            item["status"] = "running_before_first_checkpoint"
        saved = snapshot.get("checkpoints", {}).get(name)
        if saved:
            if saved["snapshot_complete"] is True:
                history = record(run + "/history.json")
                require(history is not None and len(history["history"]) == saved["saved_epoch"], "Saved history/epoch mismatch")
                rows = history["history"]
                finite = [(row["val_qty_rmse"], row["epoch"]) for row in rows
                    if type(row.get("val_qty_rmse")) in (int, float) and math.isfinite(row["val_qty_rmse"])]
                require(bool(finite), "No finite saved full Validation selector history")
                item.update(saved_epoch=saved["saved_epoch"], best_epoch=min(finite, key=lambda row: (row[0], row[1]))[1],
                    best_epoch_source="saved full Validation history; original strict earliest selector",
                    selected_checkpoint_file_present=saved["best_selected_file_present"])
                if item["status"] == "running_before_first_checkpoint":
                    item["status"] = "running"
            else:
                item["status"] = "checkpoint_snapshot_unconfirmed"
        jobs.append(item)
    completed = sum(item["status"] == "complete" for item in jobs)
    report = {"jobs": jobs, "completed": completed, "total": len(owned),
        "gpu_pids": snapshot["gpu_pids"], "observed_unix": snapshot.get("observed_unix"),
        "whole_queue_eta": "unconfirmed", "pipeline_os_exit_directly_observed": False}
    complete, exit_record = record("phases/train_complete.json"), record("phases/train_exit.json")
    if complete:
        launch, supervisor_claim = record("phases/train_launch.json"), record("supervisor.claim")
        expected_phase_argv = [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
            "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
        require(complete["phase"] == "train" and complete["host"] == host and complete["contract_sha256"] == digest
            and complete["subprocess_os_exit_code"] == 0 and not complete["gpu_compute_pids"]
            and exit_record is not None and exit_record["phase"] == "train"
            and exit_record["os_exit_code"] == 0 and exit_record["contract_sha256"] == digest,
            "Training phase completion/actual child exit mismatch")
        require(launch is not None and launch["phase"] == "train" and launch["contract_sha256"] == digest
            and launch["pid"] == exit_record["pid"] and launch["ppid"] == exit_record["ppid"]
            and launch["argv"] == exit_record["argv"] == expected_phase_argv,
            "Training phase launch/exit PID/PPID/argv mismatch")
        require(supervisor_claim is not None and supervisor_claim["pid"] == exit_record["pid"]
            and supervisor_claim["ppid"] == exit_record["ppid"] and supervisor_claim["argv"] == expected_phase_argv
            and supervisor_claim["contract_sha256"] == digest, "Training dispatcher ownership claim mismatch")
        require(state.get("status") == "complete" and set(state.get("completed", {})) == set(owned)
            and state.get("completed_conditions") == len(owned) and state.get("held_out_test_evaluated") is False,
            "Training dispatcher completion identities incomplete")
        for name, job in owned.items():
            worker_name, terminal_name = "worker_exits/" + name + ".json", "run/" + name + "/terminal_manifest.json"
            require(worker_name in files and terminal_name in files, "Worker actual exit/terminal raw evidence missing")
            binding = state["completed"][name]
            require(binding["worker_exit_sha256"] == hashlib.sha256(files[worker_name]).hexdigest()
                and binding["terminal_manifest_sha256"] == hashlib.sha256(files[terminal_name]).hexdigest(),
                "Dispatcher worker-exit/terminal rawSHA mismatch")
            worker, claim = record(worker_name), record("claims/" + name + ".json")
            require(worker["actual_child_exit_observed"] is True and worker["returncode"] == 0
                and worker["job"] == job and worker["contract_sha256"] == digest
                and worker["supervisor_pid"] == exit_record["pid"] and type(worker["worker_group_pid"]) is int
                and worker["worker_group_pid"] > 0, "Worker actual OS exit/job/parent mismatch")
            argv = worker["argv"]
            require(claim is not None and claim["job"] == job and claim["contract_sha256"] == digest
                and len(argv) == 16 and Path(argv[0]).name == "timeout"
                and argv[1:3] == ["--signal=TERM", "--kill-after=15s"]
                and argv[4:14] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
                    "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host,
                    "--mode", "fit", "--job", name] and argv[14] == "--deadline"
                and float(argv[15]) == claim["deadline_unix"], "Worker timeout argv/claim identity mismatch")
        require(not snapshot["gpu_pids"] and completed == len(owned), "Terminal completion or GPU idle unconfirmed")
        return {**report, "status": "training_complete_remote_verified", "immutable_local_binary_retrieval": "not_performed"}
    if state.get("status") == "running":
        try:
            processes = snapshot["processes"]
            parent = processes[str(state["supervisor_pid"])]
            require(parent["argv"] == state["supervisor_argv"], "Supervisor PID/argv mismatch")
            require(state["supervisor_argv"] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
                "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"],
                "Supervisor argv outside approved study")
            require(len(active) <= 3 and time.time() < state["deadline_unix"], "Worker ceiling or lease expired")
            for name, worker in active.items():
                proc = processes[str(worker["worker_group_pid"])]
                require(proc["argv"] == worker["worker_guard_argv"] and proc["ppid"] == state["supervisor_pid"]
                    and proc["pgrp"] == worker["worker_group_pid"], "Worker timeout PID/PPID/argv mismatch")
                lease = record("leases/" + name + ".json")
                require(lease and lease["job"] == name and lease["contract_sha256"] == digest
                    and lease["worker_group_pid"] == worker["worker_group_pid"] and lease["supervisor_pid"] == state["supervisor_pid"]
                    and snapshot["observed_unix"] < lease["expires_unix"] <= worker["condition_deadline_unix"], "Active job lease mismatch")
            for pid in snapshot["gpu_pids"]:
                proc = processes[str(pid)]
                owners = [worker for worker in active.values() if proc["ppid"] == worker["worker_group_pid"]]
                require(len(owners) == 1 and proc["pgrp"] == owners[0]["worker_group_pid"]
                    and proc["argv"] == owners[0]["worker_guard_argv"][4:], "Foreign or unresolved GPU process ownership")
        except (KeyError, ValueError) as exc:
            return {**report, "status": "training_identity_unconfirmed", "reason": str(exc), "user_action_required": True}
        return {**report, "status": "training_running", "active_jobs": len(active), "identity_verified": True}
    return {**report, "status": "training_phase_pending_or_identity_unconfirmed",
        "reason": "A launch intent does not prove a live scientific fit"}


def run_mode(bundle, mode, host, *, remote=None):
    require(host in HOSTS, "Unknown requested host; SSH0")
    targets = (host,)
    bundle = Path(bundle).resolve()
    c, checked = check_bundle(bundle)
    if mode == "check":
        return {**checked, "requested_host": host}
    require(mode in ("deploy", "qualify", "train", "status"), "Unknown launcher mode")
    approval = authorization(bundle, c, host)  # This runs before constructing any remote action.
    remote = remote or SSHRemote()
    digest = canonical(c)
    payload = lambda host, **extra: {"contract": c, "contract_sha256": digest, "host": host,
        "approval": approval, **extra}
    if mode == "status":
        report = {}
        for host in targets:
            cache = bundle / "control" / "qualification_cache" / host / "snapshot.json"
            if cache.exists() and not (bundle / "control" / "intents" / ("train_" + host + ".json")).exists():
                binding = qualification_binding(read(cache), c, host)
                report[host] = {"status": "qualification_passed_cached", "raw_sha256": binding["raw_sha256"], "remote_calls": 0}
                continue
            token = uuid.uuid4().hex
            try:
                snapshot = one_call(bundle, remote, host, "status", payload(host), label="status_" + token)
                decode_snapshot(snapshot, c, host)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                report[host] = {"status": "query_failed_or_identity_unconfirmed", "reason": str(exc),
                    "training_failure_inferred": False, "remote_calls": 1}
                continue
            write_exclusive(bundle / "control" / "status" / host / (token + ".json"), snapshot)
            if (bundle / "control" / "intents" / ("train_" + host + ".json")).exists():
                try:
                    report[host] = {**training_status(snapshot, c, host), "remote_calls": 1}
                except ValueError as exc:
                    report[host] = {"status": "training_identity_unconfirmed", "reason": str(exc),
                        "training_failure_inferred": False, "remote_calls": 1}
                continue
            try:
                binding = qualification_binding(snapshot, c, host)
            except ValueError as exc:
                report[host] = {"status": "pending_or_failed", "reason": str(exc), "remote_calls": 1}
            else:
                if not cache.exists():
                    write_exclusive(cache, snapshot)
                report[host] = {"status": "qualification_passed", "raw_sha256": binding["raw_sha256"], "remote_calls": 1}
        return {"mode": mode, "hosts": report, "continuous_polling": False}
    # A fresh health observation after a busy result is allowed. Actual action
    # intents are irreversible one-attempt records and block this host before
    # any additional health/SSH operation once either mutation was attempted.
    labels = ("deploy_reserve", "deploy_upload", "deploy") if mode == "deploy" else (mode,)
    require(not any((bundle / "control" / "intents" / (label + "_" + host + ".json")).exists()
        for label in labels for host in targets), "Existing mutation intent preserved; no automatic retry; SSH0")
    bindings, raw = host_training_gate(bundle, c, host) if mode == "train" else ({}, {})
    deployments = {}
    if mode in ("qualify", "train"):
        for host in targets:
            path = bundle / "control" / "deployment" / (host + ".json")
            require(path.exists(), "Target host deployment receipt required; SSH0")
            deployment = read(path)
            require(deployment["host"] == host and deployment["root"] == c["hosts"][host]["root"]
                and deployment["contract_sha256"] == digest and deployment["status"] == "deployed",
                "Deployment identity drift; SSH0")
            start = deployment["start_permit"]
            require(start["host"] == host and start["contract_sha256"] == digest
                and start["approval_sha256"] == canonical(approval)
                and start["deadline_unix"] - start["started_at_unix"] == c["limits"]["total_wall_seconds"]
                and start["started_at_unix"] <= time.time() < start["deadline_unix"], "Start lease expired or drifted; SSH0")
            deployments[host] = deployment
    # Read only the target host. Another host can remain occupied and unapproved.
    # Any busy result preserves the observation and starts no target mutation.
    health_token = uuid.uuid4().hex
    health = {host: one_call(bundle, remote, host, "health", payload(host,
        require_root_absent=mode == "deploy"), label=mode + "_health_" + health_token) for host in targets}
    ready = {host: validate_health(health[host], c, host, require_root_absent=mode == "deploy") for host in targets}
    if not all(ready.values()):
        return {"mode": mode, "status": "waiting", "hosts": health, "deploys": 0, "phases_started": 0}
    result = {}
    if mode == "deploy":
        receipt = read(bundle / "preparation_receipt.json")
        members = expected_members(c, receipt["contract_raw_sha256"])
        for host in targets:
            one_call(bundle, remote, host, "reserve", payload(host, payload_sha256=receipt["payload_sha256"]), label="deploy_reserve")
            destination = Path(c["hosts"][host]["root"]) / "incoming" / ("deployment_" + receipt["payload_sha256"] + ".tar.gz")
            upload_exit = intent(bundle, host, "deploy_upload")
            try:
                code = remote.upload(host, bundle / "deployment.tar.gz", destination)
                require(code == 0, "Payload transfer did not exit0")
                write_exclusive(upload_exit, {"host": host, "scp_os_exit_code": code, "destination": str(destination)})
            except BaseException as exc:
                write_exclusive(upload_exit, {"host": host, "status": "failed_or_unobserved", "message": str(exc), "automatic_retry": False})
                raise
            result[host] = one_call(bundle, remote, host, "deploy", payload(host,
                payload_sha256=receipt["payload_sha256"], expected_members=members))
            write_exclusive(bundle / "control" / "deployment" / (host + ".json"), result[host])
    else:
        permits = {}
        if mode == "train":
            # Bind only this host's actual qualification and start lease.
            for host in targets:
                deployment = deployments[host]
                start = deployment["start_permit"]
                require(start["host"] == host and start["contract_sha256"] == digest
                    and start["approval_sha256"] == canonical(approval)
                    and start["deadline_unix"] - start["started_at_unix"] == c["limits"]["total_wall_seconds"]
                    and start["started_at_unix"] <= time.time() < start["deadline_unix"], "Start lease expired or drifted")
                permits[host] = {"contract_sha256": digest, "start_permit_sha256": canonical(start),
                    "qualification_sha256": bindings[host]["raw_sha256"], "qualification": bindings[host]["receipt"],
                    "qualifications": bindings, "issued_unix": time.time(), "fresh_fits": True, "automatic_retry": False}
            for host in targets:
                write_exclusive(bundle / "control" / "training_permits" / (host + ".json"), permits[host])
        for host in targets:
            require((bundle / "control" / "deployment" / (host + ".json")).exists(), "Host deployment receipt missing")
            extra = {"phase": mode}
            if mode == "train":
                extra.update(training_permit=permits[host], qualification_raw=raw)
            result[host] = one_call(bundle, remote, host, "phase", payload(host, **extra), label=mode)
    return {"mode": mode, "hosts": result, "continuous_polling": False, "automatic_retry": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "deploy", "qualify", "train", "status"), required=True)
    parser.add_argument("--host", choices=HOSTS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_mode(args.bundle, args.mode, args.host), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
