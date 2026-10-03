"""One scheduler check and at most one authorized launch; never polls or retries."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from paper.scripts import local_detail_instacart_replication_contract as contract
from paper.scripts import run_local_detail_benchmark as runner
from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, write_json


def check_once(root, expected, *, start=False):
    root=Path(root)
    for name,digest in expected.items():
        require(sha_file(root/name) == digest, 'Queue identity changed: '+name)
    c=runner.read(root/'frozen_execution/execution_contract.json')
    approval,permit=runner.read(root/'approval.json'),runner.read(root/'start_permit.json')
    runner.verify_authorization(c,approval,permit,'5090')
    runner.source_and_host(c,'5090')
    runner.verify_process_environment(c,'5090')
    spec=c['hosts']['5090']; require(Path(spec['root']) == root, 'Queue root changed')
    attempted=any((root/n).exists() for n in ('queue_claim.json','dispatch_claim.json','qualification','run'))
    if attempted:
        return {'status':'already_requested','dispatch':runner.read(root/'dispatch_status.json') if (root/'dispatch_status.json').exists() else None,
                'training':runner.read(root/'run/status.json') if (root/'run/status.json').exists() else None,
                'automatic_retry':False}
    state=contract.dependency_status(c,'5090')
    if state['status'] != 'ready_for_audit':return state
    # A terminal status can be written just before its owner exits. Wait for that
    # normal exit rather than treating the few-second handoff as a failed run.
    old=c['replication']['dependency']
    if subprocess.run([spec['tmux_binary'],'has-session','-t',old['tmux']],capture_output=True).returncode == 0:
        return {'status':'waiting_for_predecessor_exit'}
    audit=contract.verify_predecessors(c,'5090')
    try:runner.verify_start_memory(c,'5090')
    except ValueError as error:
        if 'Insufficient free GPU memory' in str(error):return {'status':'waiting_for_free_gpu_memory'}
        raise
    runner.shared.check_storage(root,runner.storage_limits(c))
    require(subprocess.run([spec['tmux_binary'],'has-session','-t',spec['tmux']],capture_output=True).returncode != 0,
            'New run tmux already exists without a claim')
    if not start:return {'status':'ready_not_started','audit':audit}
    # Atomic claim survives lost SSH responses and prevents a second launch.
    write_json(root/'queue_claim.json',{'contract_sha256':sha_json(c),'claimed_at_unix':time.time(),
                                      'automatic_retry':False},exclusive=True)
    write_json(root/'predecessor_completion_audit.json',audit,exclusive=True)
    cmd='cd '+shlex.quote(spec['source_root'])+' && '+shlex.join([spec['python'],str(root/'dispatch.py'),'--host','5090'])
    cmd+=' > '+shlex.quote(str(root/'dispatch.log'))+' 2>&1'
    try:
        subprocess.run([spec['tmux_binary'],'new-session','-d','-s',spec['tmux'],cmd],check=True)
    except BaseException as error:
        write_json(root/'queue_launch_failure.json',{'error':str(error),'automatic_retry':False},exclusive=True)
        raise
    result={'status':'dispatch_started','host':'5090','tmux':spec['tmux'],'contract_sha256':sha_json(c),
            'deadline_unix':permit['deadline_unix'],'requested_at_unix':time.time()}
    write_json(root/'queue_launch_receipt.json',result,exclusive=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-json',required=True);p.add_argument('--start-if-ready',action='store_true');a=p.parse_args()
    print(json.dumps(check_once(a.root,json.loads(a.expected_json),start=a.start_if_ready),ensure_ascii=False))


if __name__ == '__main__':main()
