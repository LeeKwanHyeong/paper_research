"""Record the real pipeline child exit; a single invocation and no retry."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

r = Path(sys.argv[1]).resolve()
contract_bytes = (r / 'contract.json').read_bytes()
contract_sha = hashlib.sha256(contract_bytes).hexdigest()
c = json.loads(contract_bytes)
assert str(r) == c['authorized_remote_root']
assert hashlib.sha256((r / 'operation/pipeline.py').read_bytes()).hexdigest() == c['pipeline_sha256']
assert not (r / 'pipeline_actual_exit.json').exists()
started = time.time()
lib=Path(c['process_environment']['LD_LIBRARY_PATH'])/'libnvrtc-builtins.so.13.0'
assert hashlib.sha256(lib.read_bytes()).hexdigest()==c['nvrtc_builtins_sha256']
child = subprocess.Popen([c['runtime']['python'], str(r / 'operation/pipeline.py'), str(r)], cwd=r, env={**os.environ,**c['process_environment']})
code = child.wait()
unchanged = hashlib.sha256((r/'contract.json').read_bytes()).hexdigest() == contract_sha
wrapper_code = code if unchanged else 1
with (r / 'pipeline_actual_exit.json').open('x') as f:
    json.dump({'actual_child_exit_observed': True, 'exit_code': code, 'pipeline_pid': child.pid,
        'supervisor_pid': os.getpid(), 'started_unix': started, 'completed_unix': time.time(),
        'automatic_retry': False, 'contract_sha256': contract_sha, 'contract_unchanged': unchanged,
        'supervisor_exit_code': wrapper_code}, f)
sys.exit(wrapper_code)
