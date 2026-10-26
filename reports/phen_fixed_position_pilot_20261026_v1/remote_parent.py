"""One owned tmux parent: native qualification, two sequential fits, curves."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
CONTRACT = ROOT / 'execution_contract.json'
def child_environment(contract, root=ROOT):
    env = {**os.environ, **contract['hosts']['5080']['environment']}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='42',
               PYTHONPATH=str(root/'source'), MPLCONFIGDIR=str(root/'cache/matplotlib'),
               XDG_CACHE_HOME=str(root/'cache'), TORCHINDUCTOR_CACHE_DIR=str(root/'cache/torchinductor'))
    return env
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
def run(label, command):
    started = time.time()
    out, err = ROOT / (label + '.stdout'), ROOT / (label + '.stderr')
    with out.open('xb') as so, err.open('xb') as se:
        env = child_environment(json.loads(CONTRACT.read_text()))
        child = subprocess.Popen(command, cwd=SOURCE, stdout=so, stderr=se, env=env)
        write(label + '.launch.json', {'pid':child.pid, 'command':command, 'started_unix':started})
        rc = child.wait()
    result = {'actual_os_returncode':rc, 'pid':child.pid, 'elapsed_seconds':time.time()-started,
              'stdout':{'path':out.name,'bytes':out.stat().st_size,'sha256':sha(out)},
              'stderr':{'path':err.name,'bytes':err.stat().st_size,'sha256':sha(err)}}
    write(label + '.exit.json', result)
    if rc != 0:
        write('FAILURE.json', {'phase':label, 'actual_os_returncode':rc, 'scientific_retry':False})
        raise SystemExit(rc)
    return result
def main():
    write('parent_attempt.json', {'pid':os.getpid(),'started_unix':time.time(),'contract_raw_sha256':sha(CONTRACT)})
    c = json.loads(CONTRACT.read_text())
    for relative in ('cache/matplotlib','cache/torchinductor','cache/torch/kernels'):
        (ROOT/relative).mkdir(parents=True,exist_ok=True)
    write('effective_environment.json', {k:child_environment(c)[k] for k in c['hosts']['5080']['environment']})
    command = [sys.executable,'-u','-m','paper.scripts.run_phen_fixed_position_pilot',
               '--contract',str(CONTRACT),'--contract-sha',sha(CONTRACT),'--host','5080','--device','cuda:0']
    run('source_check', command + ['--mode','check'])
    for dataset in ('raf_spare_parts','yellow_trip_hourly'):
        for mode in ('synthetic','capacity'):
            run(dataset + '.' + mode, command + ['--mode',mode,'--dataset',dataset])
    dispatch = run('dispatcher', command + ['--mode','dispatch'])
    result = json.loads((ROOT / 'dispatch_receipt.json').read_text())
    assert result['status'] == 'complete' and all(x['actual_os_returncode']==0 for x in result['exits'])
    plots = run('plots', [sys.executable,str(SOURCE / 'pilot_assets/plot_curves.py'),
                         '--root',str(ROOT),'--output',str(ROOT / 'curves')])
    files = {}
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.is_relative_to(SOURCE) or p.is_relative_to(ROOT/'cache'):
            continue
        if p.name in ('parent.stdout','parent.stderr'):
            continue
        files[str(p.relative_to(ROOT))] = {'bytes':p.stat().st_size,'sha256':sha(p)}
    write('RESULT_MANIFEST.json', {'files':files, 'held_out_test_evaluated':False})
    write('COMPLETE.json', {'status':'complete','all_children_actual_exit_zero':True,
                           'dispatcher_actual_exit':dispatch['actual_os_returncode'],
                           'plots_actual_exit':plots['actual_os_returncode'],
                           'result_manifest_sha256':sha(ROOT/'RESULT_MANIFEST.json'),
                           'supervisor_own_exit_directly_observed':False,'completed_unix':time.time()})
if __name__ == '__main__': main()
