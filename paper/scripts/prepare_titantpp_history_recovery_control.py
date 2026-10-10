"""Prepare isolated, local recovery controls; never allocate or contact a Pod.

The scientific bundle is prepared separately. These controls preserve its exact
contract and the deleted parent's originals, account for both Pods under USD50,
and authorize one measured-runtime launch. Use --self-test before preparation.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'titantpp_contribution_history_recovery_20261010_v1'
PARENT = ROOT / 'search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1'
BUNDLE = ROOT / 'search_artifacts' / SCHEMA
PARENT_BILLED = 30.124985997681506


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def substitute(text, before, after):
    require(before in text, 'Missing control transform: ' + before[:80])
    return text.replace(before, after)


def original(name):
    text = (PARENT / 'control' / name).read_text()
    return text.replace(PARENT.name, SCHEMA)


COMMON_RECOVERY = r'''
def verify_local_parent():
    binding = read(B/'control/parent_preservation.json')
    assert sha(binding['archive']) == binding['archive_sha256'], 'Parent archive drift'
    assert sha(binding['manifest']) == binding['manifest_sha256'], 'Parent manifest drift'
    manifest = read(binding['manifest'])
    assert len(manifest['files']) == 393, 'Parent original membership changed'
    originals = Path(binding['originals']).resolve()
    for name, entry in manifest['files'].items():
        p = originals/name
        assert p.resolve().is_relative_to(originals) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size == entry['size'] and sha(p) == entry['sha256'], name
    c = read(B/'execution_contract.json')
    assert binding['archive_sha256'] == c['recovery']['parent_archive_sha256']
    assert binding['manifest_sha256'] == c['recovery']['parent_manifest_sha256']
    assert sha(B/'recovery_parent/parent_archive.tar.gz') == binding['archive_sha256']
    assert sha(B/'recovery_parent/parent_manifest.json') == binding['manifest_sha256']
    for name, digest in c['recovery']['parent_files'].items():
        p = B/'recovery_parent'/name
        assert p.resolve().is_relative_to((B/'recovery_parent').resolve()) and sha(p) == digest, name
    return binding

def verify_local_tools():
    sealed = read(B/'control/local_tools_integrity.json')
    assert sealed['contract_canonical'] == canonical(read(B/'execution_contract.json'))
    assert sha(B/'control/integrity_receipt.json') == sealed['integrity_raw_sha256']
    for name, digest in sealed['files'].items():
        assert sha(B/name) == digest, name
    return sealed

def combined_cost(a, now):
    return a['parent_billed_usd'] + prior_storage_cost(a) + max(0., now-a['created_unix'])/3600*a['running_rate_usd_hour']

def prior_storage_cost(a):
    if not a.get('network_volume_id'):return 0.
    if 'prior_network_storage_usd' in a:
        amount=a['prior_network_storage_usd']
        assert type(amount) in (int,float) and amount>=0 and a['prior_network_storage_basis']
        return amount
    assert a['network_volume_created_unix']<=a['created_unix'] and a['network_volume_usd_hour']>=0
    return (a['created_unix']-a['network_volume_created_unix'])/3600*a['network_volume_usd_hour']

def verify_current_originals(receipt):
    a=read_allocation();assert receipt['pod_id']==a['pod_id'] and receipt['status'] in ('verified','already_verified')
    assert receipt['mode'] in ('complete','partial')
    base=B/'originals'/receipt['mode'];manifest_path=base/'verified/manifest.json';originals=base/'verified/originals'
    assert sha(base/'originals.tar.gz')==receipt['archive_sha256'] and sha(manifest_path)==receipt['manifest_sha256']
    manifest=read(manifest_path)
    assert manifest['contract_sha256']==canonical(read(B/'execution_contract.json')) and manifest['pod_id']==a['pod_id']
    for name,entry in manifest['files'].items():
        p=originals/name
        assert p.resolve().is_relative_to(originals.resolve()) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size==entry['size'] and sha(p)==entry['sha256'], name
    return manifest

def owned_volume(a=None):
    a=a or read_allocation();binding=a['network_volume_identity']
    assert binding['id']==a['network_volume_id'], 'Wrong dedicated volume binding'
    v=api('network-volume','get',a['network_volume_id'])
    assert all(v.get(k)==value for k,value in binding.items()), 'Dedicated volume readback changed'
    return v

def delete_preserved_volume(a, receipt):
    a=a or read_allocation()
    if not a.get('network_volume_id'):return {'network_volume_id':None,'deleted_confirmed':True,'remote_calls':0}
    # Caller has already verified both archives/all original SHAs and confirmed
    # this exact Pod absent. Never infer permission from an arbitrary volume ID.
    verify_local_parent();verify_local_tools();verify_current_originals(receipt)
    inventory=api('pod','list');pods=inventory if isinstance(inventory,list) else inventory.get('pods')
    assert isinstance(pods,list) and not any(p.get('id')==a['pod_id'] for p in pods), 'Pod absence required before volume deletion'
    owned_volume(a)
    target=B/'control/volume_deletion_request.json'
    write(target,{'network_volume_id':a['network_volume_id'],'time':time.time(),'reason':'Both immutable original archives verified and dedicated Pod absent','automatic_retry':False},True)
    api('network-volume','delete',a['network_volume_id'])
    inventory=api('network-volume','list');volumes=inventory if isinstance(inventory,list) else inventory.get('networkVolumes',inventory.get('network_volumes'))
    assert isinstance(volumes,list) and not any(v.get('id')==a['network_volume_id'] for v in volumes), 'Volume deletion requested; absence unconfirmed'
    result={'network_volume_id':a['network_volume_id'],'deleted_confirmed':True,'time':time.time(),'automatic_retry':False}
    write(B/'control/volume_cleanup_receipt.json',result,True)
    return result
'''


SEALER = r'''"""Seal the prepared recovery contract without changing it or contacting a Pod."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys,tarfile,time
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PYTHON = '/opt/titantpp-history27-venv/bin/python'
KEYS = ('torch','cuda','cudnn','numpy','polars')
def require(ok,message):
    if not ok: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def safe(root,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe package name')
    target=root/p
    require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root.resolve()),'Unsafe/missing package file '+name)
    return target
def finite(v,name,positive=False):
    require(type(v) in (int,float) and math.isfinite(v) and (v>0 if positive else v>=0),'Invalid '+name)
    return v
def build(bundle=B,now=None):
    bundle=Path(bundle).resolve();now=time.time() if now is None else now
    c=read(bundle/'execution_contract.json');a=read(bundle/'control/allocation.json');r=read(bundle/'deployment/runtime.json')
    approval=read(bundle/'approval.json');current=read(bundle/'current.json');digest=canonical(c)
    require(c['schema']==bundle.name and current['contract_sha256']==digest,'Prepared contract identity drift')
    require(approval['approved'] is True and approval['contract_sha256']==digest and approval['maximum_combined_usd']==50
        and approval['held_out_test_authorized'] is False and approval['user_instruction'],'Recovery approval drift')
    h=c['hosts']['a100'];budget=c['budget'];recovery=c['recovery']
    require(a['bundle']==bundle.name and a['remote_root']==h['root'] and a['pod_id']==recovery['destination_pod_id']
        and a['pod_id']!='qq2uk2dpfi2rum','New physical allocation identity drift')
    require(a['total_cap_usd']==50 and a['automatic_recharge'] is False and a['autopay'] is False,'Budget/recharge drift')
    prior=finite(a['parent_billed_usd'],'parent billing')
    require(prior==budget['parent_billed_usd'] and prior>=30.124985997681506,'Unaccounted parent billing')
    require(r['pod_id']==a['pod_id'] and r['python']==PYTHON and h['python']==PYTHON,'Measured owner/Python changed')
    require(-10<=now-finite(r['observed_unix'],'runtime observation',True)<=1800,'Measured runtime stale/future')
    require('A100-SXM4-80GB' in r['gpu']['name'] and r['gpu']['uuid']==h['gpu_uuid'],'Measured GPU identity changed')
    require({k:r['runtime'][k] for k in KEYS}==h['runtime_expected'],'Frozen native Runtime changed')
    gpu=finite(a['gpu_rate_usd_hour'],'GPU hourly',True);storage=finite(a['storage_rate_usd_hour'],'storage hourly')
    rate=finite(a['running_rate_usd_hour'],'all-in hourly',True)
    require(math.isclose(gpu+storage,rate,abs_tol=1e-10,rel_tol=0) and rate==budget['confirmed_running_hourly_usd']
        and gpu==budget['confirmed_gpu_hourly_usd'],'Rate binding drift')
    started=finite(a['created_unix'],'billing start',True);deadline=finite(a['deadline_unix'],'work deadline',True)
    reserve=finite(a['retrieval_cleanup_reserve_usd'],'cleanup reserve')
    require(started==budget['billing_started_at_unix'] and deadline==budget['hard_deadline_unix']
        and a['stop_deadline_unix']==budget['absolute_stop_deadline_unix'],'Monetary lease drift')
    require(started<=r['observed_unix'] and started<=now<deadline and reserve==budget['recovery_reserve_usd']==5,'Expired/invalid recovery lease')
    require(deadline-started==c['limits']['total_wall_seconds'],'Contract wall lease drift')
    prior_storage=0.
    if a.get('network_volume_id'):
        volume_rate=finite(a['network_volume_usd_hour'],'volume hourly')
        if 'prior_network_storage_usd' in a:
            prior_storage=finite(a['prior_network_storage_usd'],'pre-Pod storage upper bound')
            require(a['prior_network_storage_basis'] and prior_storage==budget['prior_network_storage_usd'],'Pre-Pod storage upper bound changed')
        else:
            volume_started=finite(a['network_volume_created_unix'],'volume billing start',True)
            require(volume_started<=started,'Unknown volume billing start')
            prior_storage=(started-volume_started)/3600*volume_rate
        require(a['network_volume_identity']['id']==a['network_volume_id'],'Dedicated volume identity binding drift')
    require(prior+prior_storage+(deadline-started)/3600*rate+reserve<=50+1e-9,'Combined work/reserve exceeds USD50')
    require(prior+prior_storage+(a['stop_deadline_unix']-started)/3600*rate<=50+1e-9,'Absolute cap deadline exceeds USD50')
    require(len(recovery['reused_jobs'])==10 and len(recovery['resumed_jobs'])==2 and len(recovery['fresh_jobs'])==15
        and len(c['jobs'])==27 and recovery['automatic_retry'] is False,'Recovery execution counts drift')
    all_ids=recovery['reused_jobs']+list(recovery['resumed_jobs'])+recovery['fresh_jobs']
    require(len(set(all_ids))==27 and set(all_ids)=={j['id'] for j in c['jobs']},'Recovery execution ownership drift')
    for name,digest2 in {**c['source']['files'],**c['input_files']}.items():require(sha(safe(bundle,'source/'+name))==digest2,'Frozen source/input drift '+name)
    for name,digest2 in c['operation']['files'].items():require(sha(safe(bundle,name))==digest2,'Operation drift '+name)
    for name,digest2 in recovery['parent_files'].items():require(sha(safe(bundle,'recovery_parent/'+name))==digest2,'Preserved parent run drift '+name)
    require(sha(bundle/'recovery_parent/parent_archive.tar.gz')==recovery['parent_archive_sha256'],'Parent raw archive drift')
    require(sha(bundle/'recovery_parent/parent_manifest.json')==recovery['parent_manifest_sha256'],'Parent raw manifest drift')
    spec=importlib.util.spec_from_file_location('recovery_sealer_operation',bundle/'operation/campaign.py');op=importlib.util.module_from_spec(spec)
    bytecode=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(op)
    finally:sys.dont_write_bytecode=bytecode
    previous=os.environ.get('SOURCE_REVISION');os.environ['SOURCE_REVISION']=c['source']['git_revision']
    try:op.validate(c,source_root=bundle/'source',operation_root=bundle/'operation');op.validate_recovery(c,bundle)
    finally:
        if previous is None:os.environ.pop('SOURCE_REVISION',None)
        else:os.environ['SOURCE_REVISION']=previous
    permit={'host':'a100','contract_sha256':digest,'approval_sha256':canonical(approval),'started_at_unix':started,
        'deadline_unix':deadline,'issued_unix':now,'lease_billing_start_is_not_training_start':True}
    remote_names=['control/pipeline.py','control/concurrency_admission.py',*c['operation']['files']]
    integrity={'contract_sha256':digest,'files':{n:sha(safe(bundle,n)) for n in remote_names},'created_unix':now,
        'prepared_contract_preserved':True,'local_sealer_raw_sha256':sha(Path(__file__))}
    documents={'start_permit.json':permit,'control/integrity_receipt.json':integrity,
        'deployment/sealed_runtime.json':r}
    existing={'source/'+p:v for p,v in {**c['source']['files'],**c['input_files']}.items()}
    existing.update(integrity['files']);existing.update({'recovery_parent/'+p:v for p,v in recovery['parent_files'].items()})
    existing.update({'recovery_parent/parent_archive.tar.gz':recovery['parent_archive_sha256'],
        'recovery_parent/parent_manifest.json':recovery['parent_manifest_sha256'],
        'recovery_parent/parent_contract.json':sha(bundle/'recovery_parent/parent_contract.json')})
    existing.update({n:sha(safe(bundle,n)) for n in ('execution_contract.json','approval.json','current.json')})
    files={**existing,**{n:hashlib.sha256(encoded(v)).hexdigest() for n,v in documents.items()}}
    manifest={'schema':c['schema']+'_package','contract_sha256':digest,'host':'a100','pod_id':a['pod_id'],
        'remote_root':h['root'],'files':files,'created_unix':now,'only_allowlisted_regular_files':True,
        'provider_credentials_included':False,'completed_reused':10,'new_terminal_required':17}
    return {'documents':documents,'manifest':manifest}
def seal(bundle=B,now=None):
    bundle=Path(bundle).resolve();prepared=build(bundle,now);docs=prepared['documents'];manifest=prepared['manifest']
    names=[*docs,'package_manifest.json','deployment/launch_package.tar.gz','deployment/seal_receipt.json','deployment/seal_intent.json','control/local_tools_integrity.json','control/observer_binding.json']
    require(not any((bundle/n).exists() for n in names),'Sealing already attempted; preserve the attempt')
    def write_new(n,v):
        p=bundle/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(encoded(v))
    write_new('deployment/seal_intent.json',{'created_unix':time.time(),'contract_sha256':manifest['contract_sha256'],'automatic_retry':False})
    for name,value in {**docs,'package_manifest.json':manifest}.items():write_new(name,value)
    archive=bundle/'deployment/launch_package.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for n in sorted([*manifest['files'],'package_manifest.json']):
            p=safe(bundle,n)
            if n in manifest['files']:require(sha(p)==manifest['files'][n],'Bytes changed during package '+n)
            tar.add(p,arcname=n,recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();require(len(members)==len(manifest['files'])+1 and all(m.isfile() for m in members),'Package membership drift')
        for m in members:require(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(safe(bundle,m.name)),'Archived bytes changed '+m.name)
    receipt={'status':'sealed_not_launched','remote_calls':0,'contract_sha256':manifest['contract_sha256'],
        'integrity_raw_sha256':sha(bundle/'control/integrity_receipt.json'),'archive':str(archive),'archive_raw_sha256':sha(archive),
        'manifest_raw_sha256':sha(bundle/'package_manifest.json'),'files':len(manifest['files']),'created_unix':time.time(),'automatic_retry':False}
    write_new('deployment/seal_receipt.json',receipt)
    write_new('control/observer_binding.json',{'remote_snapshot_sha256':sha(bundle/'control/remote_snapshot.py'),'contract_sha256':manifest['contract_sha256']})
    tool_files={str(p.relative_to(bundle)):sha(p) for p in sorted((bundle/'control').glob('*.py'))}
    tool_files.update({n:sha(bundle/n) for n in ('control/allocation.json','control/parent_preservation.json','control/observer_binding.json','deployment/runtime.json')})
    write_new('control/local_tools_integrity.json',{'files':tool_files,'integrity_raw_sha256':receipt['integrity_raw_sha256'],
        'contract_canonical':manifest['contract_sha256'],'created_unix':time.time()})
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=B);p.add_argument('--seal',action='store_true');a=p.parse_args()
    result=seal(a.bundle) if a.seal else {'status':'validated_not_sealed','remote_calls':0,'writes':0,'files':len(build(a.bundle)['manifest']['files'])}
    print(json.dumps(result,ensure_ascii=False))
'''


MANAGER = r'''"""Combined parent+recovery cost guard; no recharging or automatic relaunch."""
import fcntl,time
from host import B,read_allocation,write,stop,combined_cost
def main():
    with (B/'control/manager.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a=read_allocation()
        while True:
            now=time.time()
            if (B/'control/cleanup_receipt.json').exists() or (B/'control/stop_receipt.json').exists():return
            write(B/'control/manager_heartbeat.json',{'observed_unix':now,'pod_id':a['pod_id'],
                'work_deadline_unix':a['deadline_unix'],'compute_stop_unix':a['stop_deadline_unix'],
                'parent_billed_usd':a['parent_billed_usd'],'new_Pod_estimated_usd':combined_cost(a,now)-a['parent_billed_usd'],
                'estimated_cost_usd':combined_cost(a,now),'maximum_combined_usd':50})
            due=min(a['setup_deadline_unix'],a['deadline_unix']) if not (B/'deployment/launch_receipt_a100.json').exists() else a['stop_deadline_unix']
            # Stop compute before, never after, the absolute combined monetary cap.
            if now>=due-60:
                try:stop('Setup deadline' if not (B/'deployment/launch_receipt_a100.json').exists() else 'Combined budget stop; retain originals');return
                except Exception as exc:write(B/'control/guard_stop_error.json',{'time':now,'type':type(exc).__name__,'message':str(exc),'user_action_required':True})
            time.sleep(min(30,max(.1,due-60-now)))
if __name__=='__main__':main()
'''


def controls():
    texts = {name: original(name) for name in ('host.py','pipeline.py','concurrency_admission.py','deploy.py','monitor.py','backup.py','remote_snapshot.py','retrieve.py','terminal_service.py')}
    texts['host.py'] = substitute(texts['host.py'], "assert a['total_cap_usd']==50 and a['automatic_recharge'] is False", "assert a['total_cap_usd']==50 and a['automatic_recharge'] is False and a['autopay'] is False\n    assert a['parent_billed_usd']>=30.124985997681506\n    assert a['parent_billed_usd']+prior_storage_cost(a)+(a['deadline_unix']-a['created_unix'])/3600*a['running_rate_usd_hour']+5<=50+1e-9\n    assert a['parent_billed_usd']+prior_storage_cost(a)+(a['stop_deadline_unix']-a['created_unix'])/3600*a['running_rate_usd_hour']<=50+1e-9")
    texts['host.py'] = substitute(texts['host.py'], "a = read(B/'control/allocation.json')", "a = read(B/'control/allocation.json')\n    if (B/'control/local_tools_integrity.json').exists():\n        assert sha(B/'control/allocation.json')==read(B/'control/local_tools_integrity.json')['files']['control/allocation.json'], 'Allocation binding changed after seal'")
    texts['host.py'] = substitute(texts['host.py'], "write(B/'control/stop_receipt.json'", "now=time.time();remaining=max(0.,50-combined_cost(a,now))\n    retained_rate=a.get('stopped_storage_usd_hour',0.)+a.get('network_volume_usd_hour',0.)\n    assert retained_rate>0, 'Stopped storage rate must be bound before retaining originals'\n    write(B/'control/storage_preservation_budget.json',{'pod_id':a['pod_id'],'network_volume_id':a.get('network_volume_id'),\n        'stopped_unix':now,'estimated_combined_cost_usd':combined_cost(a,now),'remaining_combined_budget_usd':remaining,\n        'retained_storage_hourly_usd':retained_rate,'retained_storage_budget_deadline_unix':now+remaining/retained_rate*3600,\n        'user_action_required':True,'reason':'Compute stopped; immutable retrieval/cleanup pending; stored originals remain billable','actual_final_invoice_verified':False})\n    write(B/'control/stop_receipt.json'")
    texts['host.py'] += COMMON_RECOVERY
    texts['manager.py'] = MANAGER
    texts['seal_launch.py'] = SEALER
    texts['deploy.py'] = substitute(texts['deploy.py'], 'remote,owned', 'remote,owned,verify_local_tools,verify_local_parent')
    texts['deploy.py'] = substitute(texts['deploy.py'], " assert not (B/'deployment/launch_intent_a100.json').exists()", " verify_local_tools();verify_local_parent()\n assert not (B/'deployment/launch_intent_a100.json').exists()")
    texts['monitor.py'] = substitute(texts['monitor.py'], 'canonical,remote', 'canonical,remote,verify_local_tools,verify_local_parent,combined_cost')
    texts['monitor.py'] = substitute(texts['monitor.py'], 'def observe():\n', 'def observe():\n verify_local_tools();verify_local_parent()\n')
    texts['monitor.py'] = substitute(texts['monitor.py'], "max(0,time.time()-read_allocation()['created_unix'])/3600*read_allocation()['running_rate_usd_hour']", 'combined_cost(read_allocation(),time.time())')
    texts['backup.py'] = substitute(texts['backup.py'], "'--exclude=cache/'", "'--exclude=cache/','--exclude=recovery_parent/'")
    snapshot = texts['remote_snapshot.py']
    snapshot = substitute(snapshot,"for j in c['jobs']:\n folder=R/'run'/j['id'];row={'job':j,'state':'waiting'}", "for n,v in c['recovery']['parent_files'].items():assert sha(R/'recovery_parent'/n)==v,n\nassert sha(R/'recovery_parent/parent_archive.tar.gz')==c['recovery']['parent_archive_sha256']\nassert sha(R/'recovery_parent/parent_manifest.json')==c['recovery']['parent_manifest_sha256']\nfor j in c['jobs']:\n reused=j['id'] in c['recovery']['reused_jobs']\n folder=R/('recovery_parent/run' if reused else 'run')/j['id'];row={'job':j,'state':'waiting','completed_reused':reused}\n expected_contract=c['recovery']['parent_contract_sha256'] if reused else digest")
    snapshot = substitute(snapshot,"m['contract_sha256']==digest", "m['contract_sha256']==expected_contract")
    snapshot = substitute(snapshot,"assert r['contract_sha256']==digest", "entry=c['recovery']['resumed_jobs'].get(j['id'])\n    if entry and r['contract_sha256']==c['recovery']['parent_contract_sha256']:\n     assert r['epoch']==entry['epoch'] and sha(receipt_path)==entry['receipt_sha256']\n     row.update(checkpoint_snapshot='before_first_new_save',parent_saved_epoch=entry['epoch'],parent_best_epoch=entry['best_epoch'])\n    else:assert r['contract_sha256']==digest")
    snapshot = substitute(snapshot, "read(run/'epoch_timing.json')['epochs'][-10:]", "[t for t in read(run/'epoch_timing.json')['epochs'] if t['epoch']>(entry['epoch'] if entry else 0)][-10:]")
    snapshot = substitute(snapshot,"row.update(state='training',saved_epoch=e,best_epoch=best,selected_history=hist[best-1],checkpoint_receipt=r)","if row.get('checkpoint_snapshot')!='before_first_new_save':row.update(state='training',saved_epoch=e,best_epoch=best,selected_history=hist[best-1],checkpoint_receipt=r)\n     else:row.update(parent_checkpoint_receipt=r,new_completed_epochs=0)")
    snapshot = substitute(snapshot,"elif row.get('checkpoint_snapshot')=='before_first_save':", "elif row.get('checkpoint_snapshot') in ('before_first_save','before_first_new_save'):")
    # Parent terminal observations retain the old actual observation; copied bytes are not a fresh GPU observation.
    snapshot = substitute(snapshot,"row['state']='complete';row['terminal_manifest_sha256']", "row['state']='complete';row['original_gpu_uuid']=c['recovery']['source_gpu_uuid'] if reused else c['hosts']['a100']['gpu_uuid'];row['terminal_manifest_sha256']")
    texts['remote_snapshot.py'] = snapshot
    retrieval = texts['retrieve.py']
    retrieval = substitute(retrieval,"p=root/'run'/job['id']/'terminal_manifest.json'", "reused=job['id'] in c['recovery']['reused_jobs']\n p=root/('recovery_parent/run' if reused else 'run')/job['id']/'terminal_manifest.json'\n expected_contract=c['recovery']['parent_contract_sha256'] if reused else request['contract_sha256']")
    retrieval = substitute(retrieval,"manifest['contract_sha256']==request['contract_sha256']", "manifest['contract_sha256']==expected_contract")
    retrieval = substitute(retrieval,"'worker_exits','concurrency_admission')", "'worker_exits','concurrency_admission','recovery_parent')")
    retrieval = substitute(retrieval,"terminals={}\n", "require(sha(root/'recovery_parent/parent_archive.tar.gz')==c['recovery']['parent_archive_sha256'],'Parent archive drift')\nrequire(sha(root/'recovery_parent/parent_manifest.json')==c['recovery']['parent_manifest_sha256'],'Parent manifest drift')\nfor name,digest in c['recovery']['parent_files'].items():require(sha(root/'recovery_parent'/name)==digest,'Parent run drift '+name)\nterminals={}\n")
    retrieval = substitute(retrieval,"'verified_scientific_terminal_count':len(terminals)", "'completed_reused':len(c['recovery']['reused_jobs']),'new_scientific_terminal_count':len([j for j in terminals if j not in c['recovery']['reused_jobs']]),'verified_scientific_terminal_count':len(terminals)")
    retrieval = substitute(retrieval,"prefix='run/'+job['id']+'/'", "reused=job['id'] in contract['recovery']['reused_jobs']\n                prefix=('recovery_parent/run/' if reused else 'run/')+job['id']+'/'\n                expected_contract=contract['recovery']['parent_contract_sha256'] if reused else canonical(contract)")
    retrieval = substitute(retrieval,"terminal['contract_sha256']==canonical(contract)", "terminal['contract_sha256']==expected_contract")
    retrieval = substitute(retrieval,"if metadata['mode']=='complete':\n", "require(manifest['files'].get('recovery_parent/parent_archive.tar.gz',{}).get('sha256')==contract['recovery']['parent_archive_sha256'],'Archived parent archive drift')\n        require(manifest['files'].get('recovery_parent/parent_manifest.json',{}).get('sha256')==contract['recovery']['parent_manifest_sha256'],'Archived parent manifest drift')\n        for name,digest in contract['recovery']['parent_files'].items():require(manifest['files'].get('recovery_parent/'+name,{}).get('sha256')==digest,'Archived parent run drift '+name)\n        if metadata['mode']=='complete':\n            require(manifest['completed_reused']==10 and manifest['new_scientific_terminal_count']==17,'Recovery terminal counts drift')\n")
    texts['retrieve.py'] = retrieval
    service = texts['terminal_service.py']
    service = substitute(service, 'owned,api,stop', 'owned,api,stop,verify_local_tools,verify_local_parent,verify_current_originals,combined_cost,delete_preserved_volume')
    service = substitute(service, "endpoints=list((original/'run'/job['id']).rglob('endpoint_replays.json'));assert len(endpoints)==1", "reused=job['id'] in c['recovery']['reused_jobs'];folder=original/('recovery_parent/run' if reused else 'run')/job['id']\n  endpoints=list(folder.rglob('endpoint_replays.json'));assert len(endpoints)==1")
    service = substitute(service,"'elapsed_seconds':e['elapsed_seconds']", "'elapsed_seconds':e['elapsed_seconds'],'execution_provenance':'reused_parent' if reused else ('resumed_on_new_GPU' if job['id'] in c['recovery']['resumed_jobs'] else 'fresh_recovery'),'physical_gpu_uuid':c['recovery']['source_gpu_uuid'] if reused else c['hosts']['a100']['gpu_uuid']")
    service = substitute(service,"'time_cost':'shared A100 three concurrent fits; isolated structural-efficiency claim prohibited'", "'time_cost':'A100 concurrent fit segments; times are not pooled across physical GPUs and do not establish isolated structural efficiency','recovery_limitations':c['recovery']['limitations'],'same_physical_GPU_for_all_controls':False,'complete_reused':10,'resumed':2,'fresh':15")
    service = substitute(service, "'동시 실행 fit 시간은 처리량 관측이며 구조 자체의 독립 실행 효율과 혼합하지 않는다.'", "'동시 실행 fit 시간은 처리량 관측이며 구조 자체의 독립 실행 효율과 혼합하지 않는다.','기존 완료10개 재사용, RAF seed42 두 조건은 checkpoint 재개, 나머지15개 신규. 물리 A100이 교체되었으며 RAF seed42의 disabled는 원래 GPU 결과이므로 같은 GPU 대조로 주장하지 않는다.'")
    before=" parent=read(B/'deployment/parent_preservation.json');parentdir=Path(parent['parent'])/'preserved_attempt1'\n for n,s in parent['files'].items():assert sha(parentdir/n)==s,n"
    service = substitute(service,before," parent=verify_local_parent();verify_local_tools();verify_current_originals(receipt)")
    service = substitute(service," write(B/'control/cleanup_receipt.json'", " volume=delete_preserved_volume(a,receipt)\n write(B/'control/cleanup_receipt.json'")
    service = substitute(service,"'deleted_confirmed':True,'mode':mode", "'deleted_confirmed':True,'network_volume_cleanup':volume,'mode':mode")
    service = substitute(service,"'maximum_total_usd':50}", "'maximum_total_usd':50,'parent_billed_usd':a['parent_billed_usd'],'combined_estimated_usd':combined_cost(a,time.time()),'parent_original_file_count':393}")
    service = substitute(service,"'time':time.time()},True)\n\ndef main():", "'time':time.time(),'completed_reused':10,'new_terminal_required':17,'physical_GPU_migration':True,'Q1_acceptance_separate_from_scientific_completion':True},True)\n\ndef main():")
    texts['terminal_service.py'] = service
    for name,text in texts.items():ast.parse(text, filename=name)
    return texts


def prepare(bundle=BUNDLE):
    bundle=Path(bundle).resolve();c=read(bundle/'execution_contract.json')
    require(bundle.name==SCHEMA and c['schema']==SCHEMA,'Wrong recovery bundle')
    require(c['budget']['parent_billed_usd']>=PARENT_BILLED,'Unaccounted parent bill')
    require(not (bundle/'control').exists(),'Controls already prepared; no overwrite')
    parent=read(PARENT/'originals/partial/verified/manifest.json')
    require(len(parent['files'])==393,'Parent original count drift')
    for name,entry in parent['files'].items():require(sha(PARENT/'originals/partial/verified/originals'/name)==entry['sha256'],'Parent original drift '+name)
    require(sha(PARENT/'originals/partial/originals.tar.gz')==c['recovery']['parent_archive_sha256'],'Parent archive binding drift')
    require(sha(PARENT/'originals/partial/verified/manifest.json')==c['recovery']['parent_manifest_sha256'],'Parent manifest binding drift')
    texts=controls();(bundle/'control').mkdir()
    for name,text in texts.items():
        with (bundle/'control'/name).open('x') as out:out.write(text)
    preservation={'archive':str(PARENT/'originals/partial/originals.tar.gz'),'archive_sha256':c['recovery']['parent_archive_sha256'],
        'manifest':str(PARENT/'originals/partial/verified/manifest.json'),'manifest_sha256':c['recovery']['parent_manifest_sha256'],
        'originals':str(PARENT/'originals/partial/verified/originals'),'original_count':393,'parent_deleted_confirmed':True}
    with (bundle/'control/parent_preservation.json').open('xb') as out:out.write(encoded(preservation))
    with (bundle/'control/README.md').open('x') as out:out.write(CONTROL_README)
    result={'status':'controls_prepared_not_sealed','bundle':str(bundle),'files':len(texts),'remote_calls':0,'writes_allowlisted':True}
    print(json.dumps(result,ensure_ascii=False));return result


CONTROL_README = '''# A100 이력 보정 복구 운영 절차

이 디렉터리는 새 복구 bundle 전용이다. 부모 Pod와 원래 control·실패 기록을
수정하지 않는다. 과학 조건 27개를 유지하며 완료 10개 원본을 재사용하고,
RAF seed42의 두 partial checkpoint를 검증 후 재개하며 15개는 처음 학습한다.
첫 복구 batch는 이미 완료한 disabled를 제외해 두 개가 실행된다. 이후 batch는
같은 dataset/seed의 세 조건을 동시에 실행한다.

## 준비와 시작

1. 부모 archive·manifest 및 393개 원본의 SHA를 먼저 검증한다. 신규 control
   생성기는 로컬 파일만 작성하며 Pod 생성·SSH·학습은 수행하지 않는다.
2. 전용 새 Pod/volume의 실제 identity, SSH, 측정 Runtime와 GPU UUID를
   `allocation.json` 및 `deployment/runtime.json`에 기록한다. 기존 Runtime·소스·
   optimizer·Validation 선택 규칙을 유지한다.
3. `manager.py`를 먼저 시작하고 최신 heartbeat를 확인한다. 부모의 실제 지출과
   새 Pod의 GPU·container·volume 비용을 더한 총 상한은 50달러이다. 회수·정리
   예비비 5달러를 남기며, Pod보다 먼저 생성한 volume 비용은 별도 상한으로
   기록한다. `autopay: false`는 이 작업에서 자동 충전을 허용하지 않는 계약값이다.
   provider가 공개하지 않은 계정의 live autopay 설정을 확인했다는 뜻은 아니다.
4. `seal_launch.py`의 기본 모드는 읽기 전용 검증이다. `--seal`은 prepared
   contract를 그대로 두고 start permit·integrity·allowlist archive를 한 번만
   생성한다. 발행 시각과 비용 lease의 시작 시각은 구분한다.
5. `deploy.py`는 한 번만 실행한다. qualification 실제 exit0와 자기 CPU/GPU
   PID 부재, 세 CUDA worker의 동시 admission·실제 exit0를 확인한 후 dispatcher를
   시작한다. 실패 시 claim·failure를 보존하며 자동 재시도하지 않는다.
6. `terminal_service.py`는 10분 간격 관측과 미검증 live backup을 수행한다.
   `monitor.py`는 호출당 읽기 전용 SSH 한 번이며 새 학습을 시작하지 않는다.

## 관측과 완료

- 재사용한 원본은 원래 GPU 결과로 표시한다. 재개 checkpoint의 원래 epoch를
  새 진척으로 표시하지 않으며 ETA 표본은 재개 이후 실제 완료 epoch만 사용한다.
- 과학 완료는 부모 완료 10개와 새 terminal 17개, supervisor 실제 exit0 및 idle,
  새 immutable archive의 모든 SHA, 부모 archive/manifest와 393개 원본 SHA,
  Validation 보고가 모두 확인된 상태이다. Q1의 개발 지지 여부는 별도 판정한다.
- 정확히 이 Pod가 삭제되어 부재가 확인된 후에만 정확히 이 전용 volume을
  삭제한다. volume 삭제도 부재가 확인되어야 cleanup 완료 receipt를 남긴다.
  원본 검증 실패나 삭제 미확정 상태는 failure로 보존하고 사용자 조치가 필요하다.
- 비용 guard가 compute를 중지하면 volume 원본을 자동 삭제하지 않는다.
  `storage_preservation_budget.json`에 남은 저장 비용 한도와 정리 필요 시각을
  기록한다. 이후 재접속·재시작은 자동으로 하지 않는다.
- 물리 A100 GPU가 교체되어 RAF seed42 disabled와 두 재개 조건이 같은 UUID의
  대조가 아니다. 시간은 물리 GPU별 구간으로 기록하며 혼합해 구조 효율을 주장하지
  않는다. Test·CPU binary 재추론·새 독립 평가는 이 작업에 포함하지 않는다.

## 로컬 검증

`/usr/local/bin/python3 paper/scripts/prepare_titantpp_history_recovery_control.py --self-test`
는 생성 control 11개의 문법, 누적 비용, lease/Runtime/원본 SHA 차단, 재개 ETA,
전용 volume 삭제 순서를 네트워크 없이 검증한다. 명령 예시는 실행 승인 자체가 아니다.
'''


def self_test():
    texts=controls()
    # Smoke-load the stdlib-only generated sealer and monetary helper without a Pod.
    namespace={'__file__':str(BUNDLE/'control/seal_launch.py'),'__name__':'controls_test'}
    exec(compile(texts['seal_launch.py'],'seal_launch.py','exec'),namespace)
    for value in (float('nan'),float('inf'),-1,True):
        try:namespace['finite'](value,'test')
        except ValueError:pass
        else:raise AssertionError('Invalid monetary value accepted')
    ns={'read':read,'sha':sha,'B':BUNDLE,'Path':Path,'canonical':lambda x:''}
    exec(COMMON_RECOVERY,ns)
    a={'parent_billed_usd':PARENT_BILLED,'created_unix':1000,'running_rate_usd_hour':1.6}
    require(ns['combined_cost'](a,4600)==PARENT_BILLED+1.6,'Parent spend omitted')
    require(ns['combined_cost'](a,0)==PARENT_BILLED,'Negative elapsed billing')
    storage={**a,'network_volume_id':'new-volume','prior_network_storage_usd':.0009722222222222222,
        'prior_network_storage_basis':'Conservative one-hour pre-Pod upper bound'}
    require(ns['combined_cost'](storage,4600)==PARENT_BILLED+.0009722222222222222+1.6,'Pre-Pod volume cost omitted')
    require("t['epoch']>(entry['epoch'] if entry else 0)" in texts['remote_snapshot.py'],'Resume ETA includes old GPU epochs')
    require("recovery_parent/run/" in texts['retrieve.py'] and "new_scientific_terminal_count" in texts['retrieve.py'],'Parent terminals conflated')
    require("verify_local_parent()" in texts['terminal_service.py'],'Delete before parent preservation verification')
    checks=0
    # Exercise the sealer's actual identity, cap and runtime gates in an isolated
    # fixture. Only the already-tested operation validator is stubbed; no GPU,
    # network, real contract, original checkpoint, or actual seal is touched.
    with tempfile.TemporaryDirectory(prefix='titantpp-recovery-control-test-') as directory:
        b=Path(directory)/SCHEMA;(b/'control').mkdir(parents=True);(b/'deployment').mkdir();(b/'recovery_parent').mkdir();(b/'operation').mkdir()
        for name,value in texts.items():(b/'control'/name).write_text(value)
        namespace['__file__']=str(b/'control/seal_launch.py')
        for n,data in [('parent_archive.tar.gz',b'parent archive'),('parent_manifest.json',b'parent manifest'),('parent_contract.json',b'{}')]:
            (b/'recovery_parent'/n).write_bytes(data)
        runtime={'torch':'2.11.0+cu130','cuda':'13.0','cudnn':91900,'numpy':'2.4.4','polars':'1.39.3'}
        ids=['job'+str(i) for i in range(27)];start=100000.;deadline=start+3600;rate=1.6
        c={'schema':SCHEMA,'hosts':{'a100':{'root':'/workspace/'+SCHEMA+'_a100','python':namespace['PYTHON'],'gpu_uuid':'GPU-destination','runtime_expected':runtime}},
           'budget':{'parent_billed_usd':PARENT_BILLED,'confirmed_running_hourly_usd':rate,'confirmed_gpu_hourly_usd':1.59,
                     'billing_started_at_unix':start,'hard_deadline_unix':deadline,'absolute_stop_deadline_unix':start+4*3600,'recovery_reserve_usd':5},
           'limits':{'total_wall_seconds':3600},'jobs':[{'id':i} for i in ids],'source':{'files':{},'git_revision':'frozen'},'input_files':{},'operation':{'files':{}},
           'recovery':{'destination_pod_id':'new-pod','reused_jobs':ids[:10],'resumed_jobs':{i:{} for i in ids[10:12]},'fresh_jobs':ids[12:],
                       'automatic_retry':False,'parent_files':{},'parent_archive_sha256':sha(b/'recovery_parent/parent_archive.tar.gz'),
                       'parent_manifest_sha256':sha(b/'recovery_parent/parent_manifest.json')}}
        a={'bundle':SCHEMA,'remote_root':c['hosts']['a100']['root'],'pod_id':'new-pod','total_cap_usd':50,'automatic_recharge':False,'autopay':False,
           'parent_billed_usd':PARENT_BILLED,'gpu_rate_usd_hour':1.59,'storage_rate_usd_hour':.01,'running_rate_usd_hour':rate,
           'created_unix':start,'deadline_unix':deadline,'stop_deadline_unix':start+4*3600,'retrieval_cleanup_reserve_usd':5}
        r={'pod_id':'new-pod','python':namespace['PYTHON'],'observed_unix':start+30,'gpu':{'name':'NVIDIA A100-SXM4-80GB','uuid':'GPU-destination'},'runtime':runtime}
        digest=namespace['canonical'](c)
        approval={'approved':True,'contract_sha256':digest,'maximum_combined_usd':50,'held_out_test_authorized':False,'user_instruction':'Approved recovery'}
        current={'contract_sha256':digest}
        def fixture_write():
            for n,v in [('execution_contract.json',c),('approval.json',approval),('current.json',current),('control/allocation.json',a),('deployment/runtime.json',r)]:
                (b/n).write_bytes(encoded(v))
        fixture_write()
        fake_op=SimpleNamespace(validate=lambda *args,**kwargs:None,validate_recovery=lambda *args,**kwargs:None)
        namespace['importlib']=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *args:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None)),module_from_spec=lambda spec:fake_op))
        prepared=namespace['build'](b,now=start+60)
        require(prepared['documents']['start_permit.json']['started_at_unix']==start,'Billing lease changed during setup')
        require(prepared['documents']['start_permit.json']['issued_unix']==start+60,'Issue time confused with billing lease')
        checks+=1
        for obj,key,bad in [(a,'parent_billed_usd',0),(a,'automatic_recharge',True),(a,'stop_deadline_unix',start+100*3600),
                            (r['gpu'],'uuid','GPU-foreign'),(r,'observed_unix',start-4000),(r,'python','/other/python')]:
            old=obj[key];obj[key]=bad;fixture_write()
            try:namespace['build'](b,now=start+60)
            except ValueError:checks+=1
            else:raise AssertionError('Sealer accepted invalid '+key)
            finally:obj[key]=old;fixture_write()
        (b/'recovery_parent/parent_archive.tar.gz').write_bytes(b'changed')
        try:namespace['build'](b,now=start+60)
        except ValueError:checks+=1
        else:raise AssertionError('Sealer accepted changed immutable parent archive')
    volume_checks=0
    allocation={'pod_id':'new-pod','network_volume_id':'new-volume','network_volume_identity':{'id':'new-volume','name':'dedicated-new','size':10,'dataCenterId':'US-MD-1'}}
    trace=[];state={'pod_present':False,'volume_present_after_delete':False,'foreign_volume':False}
    def mock_api(*args):
        trace.append(('api',args))
        if args==('pod','list'):return [{'id':'new-pod'}] if state['pod_present'] else []
        if args==('network-volume','get','new-volume'):
            return {**allocation['network_volume_identity'],'id':'foreign' if state['foreign_volume'] else 'new-volume'}
        if args==('network-volume','delete','new-volume'):return {}
        if args==('network-volume','list'):return [{'id':'new-volume'}] if state['volume_present_after_delete'] else []
        raise AssertionError('Unexpected provider action '+repr(args))
    ns.update({'read_allocation':lambda:allocation,'api':mock_api,'time':SimpleNamespace(time=lambda:1000.),
               'verify_local_parent':lambda:trace.append(('verify','parent')),'verify_local_tools':lambda:trace.append(('verify','controls')),
               'verify_current_originals':lambda receipt:trace.append(('verify','new_archive')),
               'write':lambda p,v,exclusive=False:trace.append(('write',p.name))})
    result=ns['delete_preserved_volume'](allocation,{'mode':'complete'})
    require(result['deleted_confirmed'] and trace.index(('verify','new_archive'))<trace.index(('api',('network-volume','delete','new-volume'))), 'Volume deleted before new archive verification')
    require(trace.index(('api',('pod','list')))<trace.index(('api',('network-volume','delete','new-volume'))), 'Volume deleted before Pod absence readback')
    require(trace.index(('api',('network-volume','list')))<trace.index(('write','volume_cleanup_receipt.json')), 'Cleanup recorded before actual volume absence')
    volume_checks+=1
    for flag in ('pod_present','foreign_volume','volume_present_after_delete'):
        trace.clear();state[flag]=True
        try:ns['delete_preserved_volume'](allocation,{'mode':'complete'})
        except AssertionError:volume_checks+=1
        else:raise AssertionError('Unsafe volume cleanup accepted '+flag)
        require(('write','volume_cleanup_receipt.json') not in trace,'Unconfirmed cleanup recorded as complete')
        if flag!='volume_present_after_delete':require(('api',('network-volume','delete','new-volume')) not in trace,'Unsafe volume deleted')
        state[flag]=False
    print(json.dumps({'status':'local_control_checks_passed','compiled_controls':len(texts),'sealer_scenarios':checks,
        'volume_cleanup_scenarios':volume_checks,'remote_calls':0}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=BUNDLE);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    self_test() if a.self_test else prepare(a.bundle)


if __name__=='__main__':main()
