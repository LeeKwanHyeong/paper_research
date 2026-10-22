"""One bounded read-only preflight per explicitly authorized host."""
from pathlib import Path
import hashlib
import json
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from paper.scripts.advance_shared_component_ablation import transport

parent = json.loads((ROOT / 'search_artifacts/shared_component_ablation_20261022_v1/execution_contract.json').read_text())
for name, digest in parent['source']['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
folder = Path(__file__).resolve().parent
for host in ('5080', '5090'):
    spec = parent['hosts'][host]
    program = '''import json, os, subprocess, sys
from pathlib import Path
spec = json.loads(SPEC)
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':s[0],'start_ticks':int(s[19]),'pgid':int(s[2]),'argv':(p/'cmdline').read_bytes().decode().rstrip('\\0').split('\\0'),'cwd':os.readlink(p/'cwd')}
    except FileNotFoundError:return None
paused=[]
for expected in spec.get('paused_processes',[]):
    live=proc(expected['pid'])
    assert live and live['state']=='T' and all(live[k]==expected[k] for k in ('start_ticks','pgid','argv','cwd')), 'Paused original changed'
    paused.append(live)
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
assert spec['gpu_uuid'] in gpu,'GPU UUID differs'
pids=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout
active={int(x) for x in pids.splitlines() if x.strip()}
assert active <= {p['pid'] for p in paused},'Unknown active GPU owner'
import torch
new_root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/shared_history_mlp_control_20261022_v1_'+HOST)
assert not new_root.exists(),'New root already exists; preserve'
print(json.dumps({'status':'preflight_passed','host':HOST,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'python_executable':sys.executable,'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_raw':gpu,'compute_pids':sorted(active),'paused_preserved':paused,'new_root_exists':False,'network_or_runtime_changes':False}))
'''.replace('SPEC', repr(json.dumps(spec))).replace('HOST', repr(host))
    child_env = {**spec['environment'], 'PYTHONDONTWRITEBYTECODE': '1'}
    remote_command = ['env', *[key+'='+str(value) for key,value in child_env.items()], spec['python'], '-']
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=1', host, shlex.join(remote_command)]
    prior = folder / (host + '.json')
    if prior.exists():
        value = json.loads(prior.read_text())
        assert value['command'] == command, 'Prior preflight command differs'
        for stream in ('stdout', 'stderr'):
            raw = (folder / (host + '.' + stream)).read_bytes()
            assert len(raw) == value[stream+'_bytes'] and hashlib.sha256(raw).hexdigest() == value[stream+'_sha256']
    else:
        value, _ = transport(command, folder, host, timeout=45, input_bytes=program.encode())
    assert value['actual_os_returncode'] == 0 and not value['timeout'] and value['transport_error'] is None, value
    result = json.loads((folder / (host + '.stdout')).read_text())
    for key in ('python', 'torch', 'cuda', 'cudnn'):
        assert result[key] == spec['runtime_expected'][key], (host,key,result[key])
    print(json.dumps({'host':host,'status':result['status'],'actual_os_returncode':value['actual_os_returncode'],'elapsed_seconds':value['elapsed_seconds'],'paused_preserved':len(result['paused_preserved'])}))
