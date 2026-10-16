"""One read-only health and prior-root identity observation before frozen Train fit-state evaluation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import shutil
import sys
import time

PARENT = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v2_5080')
DEST = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080')

def read(path):
    return json.loads(path.read_bytes()) if path.exists() else None

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return {'exit_code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}

result = {'observed_unix': time.time(), 'parent_root': str(PARENT), 'new_root_exists': DEST.exists(),
          'python_executable': sys.executable, 'reboot_required': Path('/var/run/reboot-required').exists(),
          'gpu': command(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']),
          'gpu_processes': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader']),
          'tmux_path': shutil.which('tmux'),
          'tmux': command(['tmux', '-V']) if shutil.which('tmux') else {'unavailable': True},
          'parent_contract_raw_sha256': sha(PARENT/'contract.json'),
          'parent_operation_raw_sha256': sha(PARENT/'operation_integrity.json'),
          'parent_pipeline_complete': read(PARENT/'control/pipeline_complete.json'),
          'parent_pipeline_exit': read(PARENT/'control/pipeline_exit.json'),
          'parent_failure_exists': (PARENT/'control/pipeline_failure.json').exists(),
          'package_manager_processes': command(['pgrep', '-af', r'(^|/)(apt|apt-get|dpkg|unattended-upgrade|do-release-upgrade)( |$)']),
          'training_started': False, 'model_inference_started': False, 'mutations': 0}
print(json.dumps(result, ensure_ascii=False))
