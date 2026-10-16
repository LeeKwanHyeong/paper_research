"""Read the new Pod's package versions, target GPU and idle state."""
import json
import subprocess
import sys
import time
import torch
import numpy
import polars

def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=20).stdout.strip()

gpu_row = output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader']).split(', ', 1)
assert gpu_row == ['GPU-17e020b0-7bbe-d835-af19-881e511f9a2a', 'NVIDIA A100-SXM4-80GB'], gpu_row
processes = output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
assert not processes, 'No training/admission may start before measured idle'
runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               numpy=numpy.__version__, polars=polars.__version__)
assert runtime == dict(torch='2.11.0+cu130', cuda='13.0', cudnn=91900, numpy='2.4.4', polars='1.39.3'), runtime
assert sys.version_info[:2] == (3, 12)
print(json.dumps({'pod_id': 'z278euk8d74nqt', 'python': sys.executable, 'observed_unix': time.time(),
                  'gpu': dict(uuid=gpu_row[0], name=gpu_row[1]), 'runtime': runtime,
                  'gpu_processes': [], 'idle': True, 'native_qualification': 'pending',
                  'python_version': sys.version}), flush=True)
