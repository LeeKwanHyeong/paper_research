"""Local collection/test command receipts. Never opens a remote connection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
mode = sys.argv[1]
assert mode in ('test', 'collect')
cmd = [sys.executable, '-m', 'unittest', '-v', 'test_collect_results'] if mode == 'test' else [sys.executable, str(ROOT/'collect_results.py'), *sys.argv[2:]]
logs = ROOT/'receipts'; logs.mkdir(exist_ok=True)
prefix = logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+mode)
receipt = {'argv': cmd, 'started_utc': datetime.now(timezone.utc).isoformat(), 'cwd': str(ROOT), 'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
prefix.with_suffix('.intent.json').write_text(json.dumps(receipt,indent=2)+'\n')
start = time.monotonic()
with prefix.with_suffix('.stdout').open('wb') as out, prefix.with_suffix('.stderr').open('wb') as err:
    child = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, timeout=180, check=False)
receipt.update(actual_child_exit=child.returncode, elapsed_seconds=time.monotonic()-start, finished_utc=datetime.now(timezone.utc).isoformat())
for key in ('stdout','stderr'):
    p=prefix.with_suffix('.'+key); receipt[key]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2)); sys.exit(child.returncode)
