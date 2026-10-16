"""Explicitly approved local-only finalization of already recovered CAP9 files.

Never acquires, extracts, normalizes, retries or edits original artifacts. The old
collector, summary, integrity seals and failure record remain byte-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from paper.scripts import summarize_titantpp_gru_cap_validation_v2 as summary

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2'
REPORT = PROJECT / 'reports/titantpp_gru_residual_cap_validation_20261014_v1'
RETRIEVAL = BUNDLE / 'retrieval'
REPAIR = REPORT / 'summary_repair_v2'
CONTROL_SHA = '64f143793b2134e0db2fc9dd2e9ca80c782f00b5304c32989d5305e6c4c32ef7'
COLLECTION_SHA = '56740fb9797c6ad076ed5ba156c393043bd194ae196cbb45dbb057a7681b0251'
FAILURE_SHA = '1279c1134e59ab6c10de4c54aea00ee583ac6ee9dd21281e6f20dcad0e1f8345'
INTENT_SHA = 'a1eef36cc97d8f16ca3a3c7a402cd3267493841f4a2918ca22dc70b47881985f'
ROUND_SHA = 'c2bc7b4ee4863a420fab8824bd99d3bb4730c3cb30159e42f2bb556fe26d2104'
HOSTS = ('5080', '5090')
USER_APPROVAL = 'ㅇㅇ 승인할게. 기존 원본·실패 기록과 선택 checkpoint는 그대로 두고 집계 도구 오류만 수정·검증해. 통과하면 CAP9 Test까지 이어서 실행하고 기존 CNN·FULL과 데이터셋별 3seed RMSE·MAE, seed별 차이를 정리해줘.'
need, sha, read, canonical = summary.require, summary.sha, summary.read, summary.canonical


def inventory(root):
    paths = [p for p in root.rglob('*') if p.is_file()]
    need(all(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) for p in paths),
         'Symlink or foreign path in recovered files')
    return {str(p): sha(p) for p in sorted(paths)}


def verify(approval_sha):
    approval_path = REPAIR / 'approval.json'
    need(sha(approval_path) == approval_sha, 'Summary repair approval changed')
    approval = read(approval_path)
    need(approval['user_instruction'] == USER_APPROVAL and approval['remote_calls'] == 0,
         'Summary repair authorization differs')
    for name, value in approval['repair_files'].items():
        need(sha(PROJECT / name) == value, 'Summary repair code changed')
    need(sha(RETRIEVAL / 'failure.json') == FAILURE_SHA
         and sha(RETRIEVAL / 'intent.json') == INTENT_SHA, 'Original failure/intent changed')
    from paper.scripts import advance_titantpp_gru_cap_parallel3_v3 as monitor
    native, seal, observer, collector = monitor.gate(CONTROL_SHA)
    observer, collector = monitor.helpers(native, observer, collector)
    # Check the full existing collection closure without changing the frozen helper.
    collection_path = BUNDLE / 'parallel3_v3_control/collection_integrity.json'
    need(sha(collection_path) == COLLECTION_SHA, 'Original collection seal changed')
    collection = read(collection_path)
    need(canonical(collection['files']) == collection['files_sha256'], 'Collection closure changed')
    for name, value in collection['files'].items():
        need(sha(PROJECT / name) == value, 'Original collection dependency changed: ' + name)
    intent = read(RETRIEVAL / 'intent.json')
    round_path = Path(intent['round_path'])
    need(round_path.resolve().is_relative_to(monitor.OUTPUT / 'rounds')
         and sha(round_path) == intent['round_raw_sha256'] == ROUND_SHA, 'Original terminal round changed')
    observation = read(round_path)
    need(observation['contract_sha256'] == canonical(native)
         and observation['monitor_integrity_raw_sha256'] == seal['monitor_integrity_raw_sha256']
         and observation['all_training_terminal_verified'] is True, 'Training terminal round invalid')
    counts = observer.unique_counts(native, observation['hosts'])
    need(counts == observation['counts'] == dict(total=9, completed=9, running=0,
         waiting=0, unconfirmed=0, all_training_terminal_verified=True), 'Not all nine fits terminal')
    launcher = monitor.launch.launcher_for(BUNDLE)
    expected, caches = {}, {}
    for host in HOSTS:
        cached = observer.load_cache(BUNDLE, monitor.OUTPUT, native, host, launcher)
        need(cached is not None, 'Verified terminal cache absent')
        path, snapshot, report = cached
        source = observation['hosts'][host]
        need(source['snapshot_path'] == str(path) and source['snapshot_raw_sha256'] == sha(path)
             and source['observed_unix'] == snapshot['observed_unix']
             and source['status'] == observer.TERMINAL and source['owner_process_absence_verified'] is True,
             'Round and cache differ')
        need({r['job']['id'] for r in report['jobs']} == {r['job']['id'] for r in source['jobs']}
             and all(r['status'] == 'complete' for r in report['jobs']), 'Terminal job identity differs')
        host_files = collector.terminal_files(native, host, snapshot, launcher)
        need(not set(expected).intersection(host_files), 'Duplicate cross-host originals')
        expected.update(host_files)
        archive = RETRIEVAL / 'archives' / host / 'originals.tar.gz'
        transport = read(archive.parent / 'transport_receipt.json')
        need(transport['host'] == host and transport['actual_transport_exit'] == 0
             and transport['timeout'] is False and transport['truncation'] is False
             and transport['remote_calls'] == 1 and transport['remote_mutations'] == 0
             and transport['archive_path'] == str(archive)
             and sha(archive) == transport['archive_raw_sha256']
             and archive.stat().st_size == transport['stdout_bytes']
             and sha(archive.parent / 'stderr.raw') == transport['stderr_raw_sha256'],
             'Original retrieval transport evidence differs')
        collector.verify_archive(archive, host_files)
        for name, value in host_files.items():
            need(sha(collector.safe(RETRIEVAL / 'originals', name)) == value, 'Recovered original changed')
        cache = observer.cache_path(monitor.OUTPUT, host)
        caches[host] = dict(path=str(cache), raw_sha256=sha(cache), snapshot_path=str(path),
                            snapshot_raw_sha256=sha(path), original_observed_unix=snapshot['observed_unix'])
    result = summary.summarize(BUNDLE / 'execution_contract.json', RETRIEVAL / 'originals',
                               RETRIEVAL / 'retrieval_index.json')
    need(result['status'] == 'complete_validation_only' and result['candidate_fit_count'] == 9
         and result['baseline_reference_count'] == 18 and result['held_out_test_evaluated'] is False,
         'Repaired summary incomplete')
    return dict(contract=native, expected=expected, terminal_cache_evidence=caches,
                round_path=str(round_path), round_raw_sha256=ROUND_SHA, result=result)


def finalize(approval_sha):
    json_path, md_path = REPORT / 'THREE_SEED_VALIDATION.json', REPORT / 'THREE_SEED_VALIDATION.md'
    final_path = RETRIEVAL / 'final_receipt.json'
    need(not any(p.exists() for p in (json_path, md_path, final_path)), 'Existing final outputs preserved')
    before = inventory(RETRIEVAL)
    gate = verify(approval_sha)
    summary.write_exclusive(json_path, json.dumps(gate['result'], ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    summary.write_exclusive(md_path, summary.markdown(gate['result']))
    need(before == inventory(RETRIEVAL), 'Originals changed during local finalization')
    need(read(json_path) == gate['result'] and md_path.read_text() == summary.markdown(gate['result']),
         'Summary write verification failed')
    local_files = {**before, str(json_path): sha(json_path), str(md_path): sha(md_path)}
    receipt = dict(schema='titantpp_gru_cap_collection_final_v1', status='complete',
        candidate_fit_count=9, seed_count=3, baseline_reference_count=18,
        contract_raw_sha256=sha(BUNDLE / 'execution_contract.json'), contract_sha256=canonical(gate['contract']),
        collection_integrity_raw_sha256=COLLECTION_SHA, round_path=gate['round_path'], round_raw_sha256=ROUND_SHA,
        terminal_cache_evidence=gate['terminal_cache_evidence'], held_out_test_evaluated=False,
        remote_mutations=0, automatic_retry=False, original_files_sha256=canonical(gate['expected']),
        original_file_count=len(gate['expected']), local_files=local_files, local_files_sha256=canonical(local_files),
        summary_json_path=str(json_path), summary_json_raw_sha256=sha(json_path),
        summary_markdown_path=str(md_path), summary_markdown_raw_sha256=sha(md_path), completed_unix=time.time(),
        summary_repair=dict(approval_path=str(REPAIR / 'approval.json'), approval_raw_sha256=approval_sha,
            summary_code_raw_sha256=sha(summary.__file__), finalizer_code_raw_sha256=sha(__file__),
            original_failure_path=str(RETRIEVAL / 'failure.json'), original_failure_raw_sha256=FAILURE_SHA,
            local_finalization_only=True, remote_calls_this_repair=0, original_retrieval_repeated=False))
    summary.write_exclusive(final_path, json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    need(read(final_path) == receipt and all(sha(p) == v for p, v in local_files.items()), 'Final receipt verification failed')
    return dict(status='complete', final_receipt_path=str(final_path), final_receipt_raw_sha256=sha(final_path),
                summary_json_raw_sha256=sha(json_path), summary_markdown_raw_sha256=sha(md_path),
                original_file_count=len(gate['expected']), preserved_file_count=len(before), remote_calls=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('verify', 'finalize'), required=True)
    p.add_argument('--approval-sha', required=True)
    args = p.parse_args()
    if args.mode == 'finalize':
        result = finalize(args.approval_sha)
    else:
        gate = verify(args.approval_sha)
        result = dict(status='verified', candidate_fit_count=9, baseline_reference_count=18,
                      original_file_count=len(gate['expected']), remote_calls=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
