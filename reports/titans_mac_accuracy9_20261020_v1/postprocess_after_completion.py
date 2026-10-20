"""Finite local-only continuation after the already-running native finisher."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
BUNDLE=ROOT/'search_artifacts/titans_mac_accuracy9_20261020_v1'
INTEGRITY='1dd974eccccce4c9b338bfdcd6b3f3755ee96a1172ee46de6b63c6f19db0cd04'
COMPARATOR_SHA='a3e5e18627ab19b1984ff6b8d9a54407225db1defcef7c9a3809299a876098b5'

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def run():
    write(REPORT/'POSTPROCESS_STARTED.json',{'started_at':datetime.now(timezone.utc).isoformat(),
        'native_work_started':False,'remote_calls':0,'automatic_retry':False})
    contract=json.loads((BUNDLE/'execution_contract.json').read_text())
    deadline=contract['hosts']['5090']['absolute_host_deadline_unix']+1800
    while not (REPORT/'QUEUE_COMPLETE.json').exists():
        if (REPORT/'QUEUE_FAILURE.json').exists():
            write(REPORT/'POSTPROCESS_RECEIPT.json',{'status':'native_failed_no_comparison',
                'remote_calls':0,'automatic_retry':False});return 1
        if time.time()>=deadline:
            write(REPORT/'POSTPROCESS_RECEIPT.json',{'status':'local_wait_timeout_no_comparison',
                'remote_calls':0,'automatic_retry':False});return 1
        time.sleep(min(15.,max(.01,deadline-time.time())))
    source=ROOT/'paper/scripts/compare_titans_mac_accuracy9.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=COMPARATOR_SHA:
        raise RuntimeError('Reviewed local comparator changed; no execution')
    command=[sys.executable,'-B','-m','paper.scripts.compare_titans_mac_accuracy9',
             '--integrity-sha',INTEGRITY]
    with (REPORT/'comparison.stdout.log').open('xb') as out,(REPORT/'comparison.stderr.log').open('xb') as err:
        child=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=1800)
    write(REPORT/'POSTPROCESS_RECEIPT.json',{'status':'complete' if child.returncode==0 else 'comparison_failed',
        'actual_local_comparison_exit':child.returncode,'automatic_retry':False,'remote_calls':0,
        'model_training_or_inference':False,'completed_at':datetime.now(timezone.utc).isoformat()})
    return child.returncode

if __name__=='__main__':
    sys.exit(run())
