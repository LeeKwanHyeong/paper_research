"""Archive only a fully exited Intermittent Train/Validation18 in six joint workers evaluation in its authorized root."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import time

root = Path(sys.argv[1]).resolve()
expected_seal, expected_contract = sys.argv[2:4]
assert root == Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_intermittent_time_cross_20261012_v1_5080')

def read(path):
    return json.loads(path.read_bytes())

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

require(not (root/'retrieval_claim.json').exists() and not (root/'retrieval').exists(), 'Prior retrieval prohibits retry')
require(sha(root/'operation_integrity.json') == expected_seal, 'Operation rawSHA mismatch')
seal, inputs, c = (read(root/name) for name in ('operation_integrity.json','input_integrity.json','contract.json'))
require(seal['contract_sha256'] == inputs['contract_sha256'] == c['contract_sha256'] == expected_contract,
        'Contract binding mismatch')
complete, pipeline_exit = (read(root/'control'/name) for name in ('pipeline_complete.json','pipeline_exit.json'))
require(complete['status'] == 'complete' and complete['contract_sha256'] == expected_contract
        and complete['joint_worker_count']==6 and complete['model_split_passes']==18
        and complete['actual_all_joint_worker_exit0'] is True and complete['new_fits'] == 0, 'Train pipeline not complete')
require(pipeline_exit['exit_code'] == 0 and pipeline_exit['actual_child_exit_observed'] is True,
        'Actual pipeline exit0 not established')
for name in ('pipeline_failure.json','cleanup_failure.json'):
    require(not (root/'control'/name).exists(), 'Preserved pipeline failure present')
require(read(root/'control/pipeline_latest.json')['phase'] == 'complete', 'Last phase not complete')
qualification = read(root/'control/native_qualification.json')
routes = {job['study']+'__'+job['arm']:(job['study'],job['arm']) for job in c['jobs']}
require(qualification['status'] == 'PASS' and qualification['contract_sha256'] == expected_contract
        and set(qualification['routes']) == set(routes), 'Native synthetic3 incomplete')
owned = [read(root/'control/pipeline_claim.json')['pid']]
for name,(study,arm) in routes.items():
    folder = root/'control/native_probes'/name
    probe, child_exit = read(folder/'receipt.json'), read(folder/'worker_exit.json')
    require(sha(folder/'receipt.json') == qualification['routes'][name], 'Native receipt SHA changed')
    require(probe['status'] == 'PASS' and probe['contract_sha256'] == expected_contract
            and probe['study'] == study and probe['arm'] == arm and probe['device'] == 'cuda'
            and probe['source_closure'] == c['sources'][study]['closure'], 'Native probe identity differs')
    require(probe['native_pinned_contract_checked'] is True and probe['optimizer_updates'] == 0
            and probe['checkpoint_reads'] == 0 and probe['real_data_loaded'] is False
            and probe['held_out_test_evaluated'] is False and all(v is True for v in probe['qualification'].values())
            and probe['hooked_and_unhooked_all_original_outputs_bitwise_equal'] is True
            and probe['telemetry_finite'] is True, 'Native qualification flags differ')
    require(child_exit['exit_code'] == 0 and child_exit['actual_child_exit_observed'] is True
            and child_exit['contract_sha256'] == expected_contract, 'Actual native exit0 missing')
    launch=read(folder/'launch.json')
    expected_argv=[c['runtime']['python'],str(root/'control/evaluate.py'),'--root',str(root),'--native-probe','--study',study,'--arm',arm,'--device','cuda']
    require(child_exit['argv']==launch['argv']==expected_argv and child_exit['pid']==launch['pid']
        and child_exit['ppid']==launch['ppid']==owned[0], 'Native launch/actual exit bridge differs')
    owned.append(child_exit['pid'])
module_spec = importlib.util.spec_from_file_location('train_fit_terminal_validator', root/'control/evaluate.py')
adapter = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(adapter)
require(adapter.verify_contract(root) == c, 'Canonical RAF contract differs')
identities = {}
for seed in (42,52,62):
    for split in ('train','validation'):
        dest=root/'results'/('seed'+str(seed))/(split+'_joint')
        joint=adapter.verify_joint_receipt(root,c,seed,split)
        key='seed'+str(seed)+'__'+split
        launch,x=read(dest/'launch.json'),read(dest/'worker_exit.json')
        require(sha(dest/'receipt.json')==complete['joint_receipts'][key]
                and sha(dest/'worker_exit.json')==complete['joint_worker_exits'][key], 'Joint pipeline binding differs')
        argv=[c['runtime']['python'],str(root/'control/evaluate.py'),'--root',str(root),
              '--seed',str(seed),'--split',split,'--device','cuda']
        require(x['argv']==launch['argv']==argv and joint['argv']==argv[1:]
                and x['pid']==launch['pid']==joint['pid'] and x['ppid']==launch['ppid']==joint['ppid']==owned[0]
                and x['exit_code']==0 and x['actual_child_exit_observed'] is True
                and x['timed_out'] is False and x['supervisor_aborted'] is False
                and x['contract_sha256']==expected_contract and x['split']==split and x['seed']==seed,
                'Joint actual identity/exit differs')
        require(joint['started_unix']>=qualification['completed_unix'], 'Joint inference preceded qualification')
        for job in (j for j in c['jobs'] if j['seed']==seed):
            path=root/'results'/job['id']/(split+'_metrics.json');output=read(path)
            require(sha(path)==complete['model_receipts'][job['id']][split], 'Model pass receipt differs')
            require(output['pid']==joint['pid'] and output['ppid']==joint['ppid'] and output['argv']==joint['argv'], 'Model shared process bridge differs')
            runtime=output['runtime']
            require(runtime['python']=='3.12.13' and runtime['gpu']['uuid']==c['runtime']['gpu_uuid'],'Native Python/GPU differs')
            for name in ('torch','cuda','cudnn','numpy','polars'):
                require(runtime[name]==c['runtime'][name],'Native Runtime differs: '+name)
            identity=(output['target_identity_sha256'],output['truth_sha256'])
            require(identities.setdefault(split,identity)==identity, 'Model target population differs')
        owned.append(x['pid'])
for pid in owned:
    cmdline = Path('/proc')/str(pid)/'cmdline'
    require(not cmdline.exists() or str(root).encode() not in cmdline.read_bytes(), 'Own process still alive')
smi = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],
                     capture_output=True,text=True,check=True).stdout.strip()
for row in smi.splitlines():
    values = [v.strip() for v in row.split(',')]
    require(len(values)<2 or int(values[1]) not in owned, 'Own GPU PID remains')
for rel,expected in {**inputs['files'],**seal['files']}.items():
    target = (root/rel).resolve()
    require(target.is_relative_to(root) and not target.is_symlink() and sha(target) == expected,
            'Immutable binary/input/operation changed: '+rel)
write(root/'retrieval_claim.json', {'contract_sha256':expected_contract,'observed_unix':time.time(),
                                  'exclusive_retrieval':True,'automatic_retry':False})
files = set(inputs['files'])|set(seal['files'])|{'operation_integrity.json','retrieval_claim.json'}
for folder in ('results','control'):
    for path in (root/folder).rglob('*'):
        rel = path.relative_to(root)
        if path.is_file() and not any('cache' in part or part=='__pycache__' for part in rel.parts):
            files.add(rel.as_posix())
for rel in files:
    path = root/rel
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe archive source')
manifest = {rel:sha(root/rel) for rel in sorted(files)}
dest = root/'retrieval'
dest.mkdir(exist_ok=False)
write(dest/'manifest.json', {'schema':'immutable_intermittent_time_cross_originals_v1','contract_sha256':expected_contract,'files':manifest})
archive = dest/'originals.tar.gz'
with tarfile.open(archive,'x:gz') as t:
    for rel in sorted(files):
        t.add(root/rel,arcname=rel,recursive=False)
with tarfile.open(archive,'r:gz') as t:
    members = t.getmembers()
    require(len(members) == len(manifest) and {m.name for m in members} == set(manifest), 'Archive member set differs')
    for member in members:
        require(member.isfile() and member.size == (root/member.name).stat().st_size, 'Archive member type/size differs')
        h = hashlib.sha256()
        with t.extractfile(member) as stream:
            for data in iter(lambda:stream.read(1048576),b''):
                h.update(data)
        require(h.hexdigest() == manifest[member.name], 'Archived rawSHA differs: '+member.name)
for rel,expected in manifest.items():
    require(sha(root/rel) == expected, 'Source changed during archive: '+rel)
receipt={'status':'verified_terminal_archived','contract_sha256':expected_contract,
    'operation_integrity_sha256':expected_seal,'remote_root':str(root),
    'manifest_sha256':sha(dest/'manifest.json'),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
    'file_count':len(manifest),'actual_pipeline_exit0':True,'actual_worker_exit0_count':6,'actual_joint_worker_exit0_count':6,'model_split_passes':18,
    'native_qualification_pass':True,'train_count':9,'validation_count':9,'new_fits':0,'test_inference':0,
    'own_processes_absent':True,'own_gpu_pids_absent':True,'gpu_processes_observed':smi,
    'completed_unix':time.time(),'automatic_retry':False}
write(dest/'receipt.json',receipt)
print(json.dumps(receipt),flush=True)
