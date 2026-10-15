from pathlib import Path
import tempfile, json, shutil, hashlib, subprocess, os, time
root=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
parent=root/'search_artifacts/titantpp_backbone_location_raf6_20261015_v1'
c=json.loads((parent/'execution_contract.json').read_text())
with tempfile.TemporaryDirectory(prefix='v_diff_frozen_cpu_') as d:
    temporary=Path(d);source=temporary/'source';source.mkdir();(source/'sample_data').mkdir()
    for n,h in c['source']['files'].items():
        p=parent/'source'/n
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h
        target=source/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for n in ['models/TPPs/CountAwareTitanValueDifferenceCNN.py','paper/scripts/run_titantpp_value_difference_cnn.py','paper/scripts/titantpp_value_difference_raf3_runtime.py']:
        target=source/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/n,target)
    program=temporary/'smoke.py';program.write_text('''import json, torch, polars as pl, sys
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from paper.scripts import titantpp_value_difference_raf3_runtime as rt
from paper.scripts import run_titantpp_value_difference_cnn as runner
source=Path(sys.argv[1]).resolve();rt.runner=runner
c={'datasets':[runner.fabricated_data()]};c['datasets'][0]['loader']['max_seq_len']=84
original_fork=torch.random.fork_rng
torch.set_num_threads(4)
with ExitStack() as patches:
    patches.enter_context(patch.object(pl,'read_parquet',side_effect=AssertionError('Real data forbidden')))
    patches.enter_context(patch.object(pl,'scan_parquet',side_effect=AssertionError('Real data forbidden')))
    patches.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self))
    patches.enter_context(patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self))
    patches.enter_context(patch.object(torch.random,'fork_rng',lambda **k:original_fork(devices=[])))
    patches.enter_context(patch.object(rt,'start_ticks',return_value=123))
    for name in ('set_per_process_memory_fraction','empty_cache','reset_peak_memory_stats','synchronize'):
        patches.enter_context(patch.object(torch.cuda,name,return_value=None))
    for name in ('max_memory_reserved','max_memory_allocated'):
        patches.enter_context(patch.object(torch.cuda,name,return_value=0))
    result=rt.synthetic_arm(c,source,rt.ARMS[0])
for name,module in list(sys.modules.items()):
    if name.startswith(('models.','paper.','simple_lab_test.')) and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(source)
assert result['status']=='passed' and result['active_local_cnn_parameters']==512
assert result['active_value_difference_parameters']==128 and result['target_blocked']
assert result['shape']==[128,84] and result['updates']==4
result.update(mode='CPU fabricated integration with CUDA resource/process calls mocked; NOT native qualification',research_data_read=False,remote_calls=0,scientific_fits_started=0,inherited_source_files=139)
print(json.dumps(result,sort_keys=True))
''')
    t=time.time();p=subprocess.run(['/usr/local/bin/python3',str(program),str(source)],cwd=source,env={**os.environ,'PYTHONPATH':str(source),'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(temporary/'mpl'),'XDG_CACHE_HOME':str(temporary/'cache')},capture_output=True)
    out=Path('/private/tmp/value_difference_runtime_smoke_v3_stdout.txt');out.write_bytes(p.stdout)
    err=Path('/private/tmp/value_difference_runtime_smoke_v3_stderr.txt');err.write_bytes(p.stderr)
    receipt={'actual_exit':p.returncode,'passed':int(p.returncode==0),'failed':int(p.returncode!=0),'elapsed_seconds':time.time()-t,'stdout':str(out),'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr':str(err),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest(),'actual_native_qualification':False,'remote_calls':0,'scientific_fits_started':0}
    Path('/private/tmp/value_difference_runtime_smoke_v3_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));print(p.stdout.decode());print(p.stderr.decode());raise SystemExit(p.returncode)
