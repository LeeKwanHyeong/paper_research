"""Read-only component status transport with lossless bounded diagnostics.

No project code or remote action runs at import. The caller supplies the sealed
launcher namespace and completes its integrity gate. Only status is permitted;
all old source, launchers, remote files and scientific receipts remain unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

MAX_BYTES = 64 * 1024 ** 2

# Shared verbatim with the remote status collector; synthetic tests execute the
# same functions against fabricated /proc data. This code writes no files.
PROCESS_HELPERS = r'''
def status_need(ok, message):
 if not ok: raise ValueError(message)
def status_pid(value, source):
 status_need(type(value) is int and 0 < value <= 2147483647, 'malformed PID in '+source)
 return value
def status_argv(value, source):
 status_need(isinstance(value,list) and bool(value) and all(isinstance(x,str) and bool(x) for x in value), 'malformed argv in '+source)
 return value
def status_record(files, name):
 item=files[name]
 if isinstance(item,bytes): raw=item
 else:
  raw=base64.b64decode(item['base64'],validate=True)
  status_need(hashlib.sha256(raw).hexdigest()==item['raw_sha256'],'snapshot bytes changed '+name)
 value=json.loads(raw.decode('utf-8'))
 status_need(isinstance(value,dict),'malformed ownership receipt '+name)
 return value
def status_proc_read(pid, proc_root):
 proc=pathlib.Path(proc_root)/str(pid)
 before=(proc/'stat').read_text()
 try:
  command=(proc/'cmdline').read_bytes().decode('utf-8').rstrip('\x00').split('\x00')
  after=(proc/'stat').read_text()
 except (FileNotFoundError,ProcessLookupError):
  # Only the initial stat lookup can prove an explicitly absent PID. Once its
  # identity was observed, disappearing mid-query is unresolved evidence.
  raise OSError('process disappeared after initial stat observation') from None
 def fields(raw):
  prefix,tail=raw.rsplit(')',1)
  status_need(int(prefix.split('(',1)[0].strip())==pid,'/proc PID differs')
  values=tail.split(); status_need(len(values)>19,'malformed /proc stat')
  ppid,pgrp,ticks=int(values[1]),int(values[2]),int(values[19])
  status_need(ppid>=0 and pgrp>0 and ticks>=0,'malformed /proc identity')
  return ppid,pgrp,ticks
 pre,post=fields(before),fields(after)
 return {'pid':pid,'ppid':post[0],'pgrp':post[1],'argv':command,'start_ticks':post[2],
         'identity_stable_during_query':pre==post}
def collect_process_observations(state, files, gpu_pids, contract, host, proc_root='/proc', reader=None):
 status_need(isinstance(state,dict),'malformed campaign state')
 active=state.get('active_jobs',{})
 if isinstance(active,list):
  status_need(state.get('status')=='failed' and active==[], 'active_jobs list is not an exact failed-empty boundary')
  active={}
 status_need(isinstance(active,dict),'active_jobs must be a dict or exact failed-empty list')
 spec=contract['hosts'][host]
 digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 dispatcher=[spec['python'],spec['root']+'/operation/campaign.py','--contract',spec['root']+'/execution_contract.json','--host',host,'--mode','dispatch']
 wrapper=[spec['python'],spec['root']+'/operation/pipeline.py','--contract',spec['root']+'/execution_contract.json','--host',host,'--phase','train']
 candidates={}; owner_pids=set(); owner_records=[]
 def add(value, source, key, expected=None, owner=True):
  pid=status_pid(value,source+'/'+key)
  entry=candidates.setdefault(pid,{'evidence_sources':[], 'evidence_pid_fields':{}, 'expected_identities':[]})
  if source not in entry['evidence_sources']: entry['evidence_sources'].append(source)
  fields=entry['evidence_pid_fields'].setdefault(source,[])
  if key not in fields: fields.append(key)
  if expected is not None: entry['expected_identities'].append({'source':source,'pid_field':key,**expected})
  if owner: owner_pids.add(pid)
 for pid in gpu_pids: add(pid,'gpu_inventory','pid',owner=False)
 if 'supervisor_pid' in state:
  argv=status_argv(state.get('supervisor_argv'),'status.json')
  status_need(argv==dispatcher,'state dispatcher argv differs')
  add(state['supervisor_pid'],'status.json','supervisor_pid',{'argv':argv,'pgrp':state['supervisor_pid']})
 for name,item in active.items():
  status_need(isinstance(name,str) and isinstance(item,dict),'malformed active worker entry')
  argv=status_argv(item.get('worker_guard_argv'),'status.json:'+name)
  add(item.get('worker_group_pid'),'status.json','worker_group_pid',{'argv':argv,'pgrp':item.get('worker_group_pid'),
      **({'ppid':state['supervisor_pid']} if 'supervisor_pid' in state else {})})
 for name in ('supervisor.claim','phases/train_launch.json','phases/train_exit.json'):
  if name not in files: continue
  record=status_record(files,name)
  status_need(record.get('contract_sha256')==digest,'owner contract mismatch '+name)
  pid,ppid=status_pid(record.get('pid'),name+'/pid'),status_pid(record.get('ppid'),name+'/ppid')
  argv=status_argv(record.get('argv'),name)
  status_need(argv==dispatcher,'owner dispatcher argv differs '+name)
  owner_records.append((pid,ppid,argv))
  add(pid,name,'pid',{'argv':argv,'ppid':ppid,'pgrp':pid})
  add(ppid,name,'ppid',{'argv':wrapper})
 status_need(not owner_records or all(row==owner_records[0] for row in owner_records),'owner launch/claim/exit identity mismatch')
 for name in files:
  if not name.startswith('worker_exits/') or not name.endswith('.json'): continue
  record=status_record(files,name)
  status_need(record.get('contract_sha256')==digest,'worker exit contract mismatch '+name)
  pid=status_pid(record.get('worker_group_pid'),name+'/worker_group_pid')
  parent=status_pid(record.get('supervisor_pid'),name+'/supervisor_pid')
  argv=status_argv(record.get('argv'),name)
  add(pid,name,'worker_group_pid',{'argv':argv,'ppid':parent,'pgrp':pid})
  add(parent,name,'supervisor_pid',{'argv':dispatcher,'pgrp':parent})
  for key in ('pid','ppid'):
   if key in record: add(record[key],name,key)
 observations={}; processes={}; read_proc=status_proc_read if reader is None else reader
 for pid, evidence in sorted(candidates.items()):
  base={'pid':pid,**evidence}
  try:
   actual=read_proc(pid,proc_root)
   status_need(actual.get('pid')==pid,'queried process PID mismatch')
   matched=actual.get('identity_stable_during_query') is True and all(
       all(actual.get(key)==value for key,value in identity.items() if key not in ('source','pid_field'))
       for identity in evidence['expected_identities'])
   label='live' if matched else 'reused_pid'
   observations[str(pid)]={**base,'state':label,'actual':actual}
   processes[str(pid)]={**actual,'observation_state':label}
  except (FileNotFoundError,ProcessLookupError):
   observations[str(pid)]={**base,'state':'absent','absence_observed':True}
  except (OSError,ValueError,UnicodeError) as error:
   observations[str(pid)]={**base,'state':'query_error','absence_observed':False,
                         'error_type':type(error).__name__,'error_message':str(error)[:400]}
   processes[str(pid)]={'pid':pid,'observation_state':'query_error','query_error':True,
                        'error_type':type(error).__name__}
 return observations,processes,sorted(owner_pids)
'''

# Standard-library aliases used by the exact remote helper text.
import pathlib
exec(compile(PROCESS_HELPERS, __file__ + ':process_helpers', 'exec'), globals())

OLD_PROCESS_BLOCK = """   processes={}; process_ids=set(pids)
   if state.get('supervisor_pid'): process_ids.add(state['supervisor_pid'])
   process_ids.update(item['worker_group_pid'] for item in state.get('active_jobs',{}).values())
   for pid in process_ids:
    try:
     proc=pathlib.Path('/proc')/str(pid); stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
     processes[str(pid)]={'pid':pid,'ppid':int(stat[1]),'pgrp':int(stat[2]),'argv':(proc/'cmdline').read_bytes().decode().rstrip('\\x00').split('\\x00')}
    except (FileNotFoundError,ProcessLookupError): processes[str(pid)]={'pid':pid,'disappeared_during_observation':True}
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def write_bytes_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path, value):
    write_bytes_exclusive(path, json_bytes(value) + b'\n')


def patched_remote_script(launcher_namespace):
    """Patch only this new transport; never change the frozen namespace."""
    source = launcher_namespace.REMOTE_SCRIPT
    marker = "need(canon(c)==digest,'contract mismatch')"
    require(source.count(marker) == 1 and source.count(OLD_PROCESS_BLOCK) == 1,
            'Frozen remote-script patch anchors changed; SSH0')
    script = source.replace(marker, "need(mode=='status','read-only status transport rejects action')\n" + marker, 1)
    script = script.replace(OLD_PROCESS_BLOCK,
        "   inventory=capture(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'])\n"
        "   gpu_inventory_uuids=[row.strip() for row in inventory.splitlines() if row.strip()]\n"
        "   need(gpu_inventory_uuids.count(spec['gpu_uuid'])==1,'fixed GPU UUID inventory missing or ambiguous')\n"
        "   process_observations,processes,observed_owner_pids=collect_process_observations(state,files,pids,c,host)\n", 1)
    marker = 'print(json.dumps(result,ensure_ascii=False,allow_nan=False))'
    require(script.count(marker) == 1, 'Frozen result patch anchor changed; SSH0')
    script = script.replace(marker,
        "result.update(process_observations=process_observations,observed_owner_pids=observed_owner_pids,gpu_inventory_uuids=gpu_inventory_uuids)\n" + marker, 1)
    # Helpers are definitions only; guard remains before every remote action.
    script = PROCESS_HELPERS + '\n' + script
    compile(script, 'component-status-read-only-v2', 'exec')
    return script


class StatusEvidenceError(RuntimeError):
    def __init__(self, message, receipt_path, metadata):
        super().__init__(message)
        self.evidence_receipt_path = str(receipt_path)
        self.evidence_metadata = metadata


def preserve_stream(folder, name, value):
    raw = b'' if value is None else value.encode('utf-8') if isinstance(value, str) else bytes(value)
    preserved = raw[:MAX_BYTES]
    path = folder / (name + '.bin')
    write_bytes_exclusive(path, preserved)
    return {'path': str(path), 'full_sha256': sha_bytes(raw), 'byte_count': len(raw),
            'preserved_sha256': sha_bytes(preserved), 'preserved_byte_count': len(preserved),
            'truncated': len(preserved) != len(raw)}


class StatusOnlyRemote:
    """Same pinned SSH environment/Python command; no upload or other action."""
    def __init__(self, launcher_namespace, evidence_folder):
        self.launcher = launcher_namespace
        self.folder = Path(evidence_folder)
        self.remote_script = patched_remote_script(launcher_namespace)
        self.evidence_receipt_path = self.folder / 'transport_evidence.json'

    def upload(self, *args, **kwargs):
        raise ValueError('Status transport forbids upload; remote0')

    def call(self, host, action, payload):
        require(action == 'status', 'Only read-only status is allowed; remote0')
        require(host in ('5080', '5090') and payload['host'] == host, 'Foreign status host; remote0')
        spec = payload['contract']['hosts'][host]
        environment = spec['environment']
        require(environment.get('PYTHONDONTWRITEBYTECODE', '1') == '1', 'Runtime bytecode writes forbidden; remote0')
        values = ['env', 'PYTHONDONTWRITEBYTECODE=1', *(key + '=' + value for key, value in sorted(environment.items())),
                  spec['python'], '-c', self.remote_script, 'status']
        command = ' '.join(shlex.quote(value) for value in values)
        metadata = {'schema': 'titantpp_components_status_transport_evidence_v2', 'host': host,
                    'action': 'status', 'remote_script_sha256': sha_bytes(self.remote_script.encode()),
                    'started_unix': time.time(), 'actual_transport_exit_observed': False,
                    'transport_os_exit_code': None, 'automatic_retry': False, 'remote_calls': 1}
        stdout = stderr = b''
        outcome = 'unobserved_error'
        try:
            completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                input=json_bytes(payload), capture_output=True, timeout=120, check=True)
            stdout, stderr = completed.stdout, completed.stderr
            metadata.update(actual_transport_exit_observed=True, transport_os_exit_code=completed.returncode)
            require(completed.returncode == 0, 'Status SSH returned nonzero exit')
            outcome = 'received'
        except subprocess.CalledProcessError as error:
            stdout, stderr = error.stdout, error.stderr
            metadata.update(actual_transport_exit_observed=True, transport_os_exit_code=error.returncode)
            outcome = 'nonzero_exit'
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout, error.stderr
            outcome = 'timeout_exit_unobserved'
        except (OSError, ValueError) as error:
            outcome = 'local_transport_error'
            metadata['exception_type'] = type(error).__name__
        metadata['stdout'] = preserve_stream(self.folder, 'stdout', stdout)
        metadata['stderr'] = preserve_stream(self.folder, 'stderr', stderr)
        result = None
        if outcome == 'received':
            try:
                require(not metadata['stdout']['truncated'] and not metadata['stderr']['truncated'], 'Status output exceeded64MiB')
                result = json.loads(bytes(stdout).decode('utf-8'))
                require(isinstance(result, dict), 'Status response is not a JSON object')
                result['transport_os_exit_code'] = metadata['transport_os_exit_code']
                outcome = 'complete'
            except UnicodeError:
                outcome = 'decode_error'
            except (ValueError, TypeError):
                outcome = 'invalid_or_oversized_json'
        metadata.update(status=outcome, completed_unix=time.time(), pipeline_OS_exit_directly_observed=False)
        write_json_exclusive(self.evidence_receipt_path, metadata)
        if outcome != 'complete':
            raise StatusEvidenceError('Read-only status transport ' + outcome + '; evidence preserved',
                                      self.evidence_receipt_path, metadata) from None
        result['transport_evidence_path'] = str(self.evidence_receipt_path)
        result['transport_evidence_sha256'] = sha_bytes(self.evidence_receipt_path.read_bytes())
        return result


def read_status_once(bundle, launcher_namespace, c, host, diagnostics_root):
    """One authorized status SSH, immutable raw evidence, no retry/other action."""
    bundle = Path(bundle)
    approval = launcher_namespace.authorization(bundle, c, host)
    token = uuid.uuid4().hex
    folder = Path(diagnostics_root) / token
    folder.mkdir(parents=True, exist_ok=False)
    remote = StatusOnlyRemote(launcher_namespace, folder)
    exit_path = launcher_namespace.intent(bundle, host, 'status_repair_v2_' + token)
    snapshot_path = bundle / 'control' / 'status' / host / (token + '.json')
    try:
        snapshot = remote.call(host, 'status', {'contract': c, 'contract_sha256': launcher_namespace.canonical(c),
                                              'host': host, 'approval': approval})
        # Preserve the actual raw response before/after local snapshot serialization.
        write_json_exclusive(snapshot_path, snapshot)
        launcher_namespace.write_exclusive(exit_path, {'host': host, 'action': 'status',
            'transport_os_exit_code': 0, 'actual_transport_exit_observed': True,
            'snapshot_path': str(snapshot_path), 'snapshot_raw_sha256': sha_bytes(snapshot_path.read_bytes()),
            'transport_evidence_path': snapshot['transport_evidence_path'],
            'transport_evidence_sha256': snapshot['transport_evidence_sha256'],
            'completed_unix': time.time(), 'scientific_phase_exit_inferred': False, 'automatic_retry': False})
        return snapshot_path, snapshot
    except BaseException as error:
        metadata = getattr(error, 'evidence_metadata', {})
        launcher_namespace.write_exclusive(exit_path, {'host': host, 'action': 'status',
            'status': 'failed_or_unobserved', 'error': type(error).__name__,
            'message': str(error)[:500], 'actual_transport_exit_observed': metadata.get('actual_transport_exit_observed', False),
            'transport_os_exit_code': metadata.get('transport_os_exit_code'),
            'evidence_receipt_path': getattr(error, 'evidence_receipt_path', None),
            'completed_unix': time.time(), 'automatic_retry': False})
        raise
