"""Run the predeclared validation gates, then the approved test queue."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def write(value):
    p=HERE/'pipeline_status.json'
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(p)

def main():
    assert not (HERE/'pipeline_status.json').exists(), 'Preserve the existing pipeline attempt'
    os.chdir(ROOT)
    os.environ.update({'CUBLAS_WORKSPACE_CONFIG':':4096:8','CUDA_VISIBLE_DEVICES':'0',
       'CUDA_DEVICE_ORDER':'PCI_BUS_ID','NVIDIA_TF32_OVERRIDE':'0','OMP_NUM_THREADS':'4',
       'MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONHASHSEED':'42',
       'LD_LIBRARY_PATH':'/home/leekwanhyeong/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib'})
    steps=[('qualification_cpu',['dispatch.py','--phase','qualification_cpu']),
           ('qualification_cuda',['dispatch.py','--phase','qualification_cuda']),
           ('validation_full',['dispatch.py','--phase','validation_full']),
           ('qualification_gate',['qualify.py']),
           ('test',['dispatch.py','--phase','test'])]
    started=time.time()
    for name,args in steps:
        state={'status':'running','phase':name,'pid':os.getpid(),'started_unix':started,
               'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(state)
        result=subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]])
        if result.returncode:
            write({**state,'status':'failed','exit_code':result.returncode,
                   'elapsed_seconds':time.time()-started,'test_started':name=='test'})
            return result.returncode
    write({'status':'complete','phase':'test','pid':os.getpid(),'started_unix':started,
           'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'elapsed_seconds':time.time()-started,'postprocessing_pending':True})
    return 0

if __name__=='__main__':
    sys.exit(main())
