"""Finalize existing validation evidence. No training, predictions, network, or raw data."""
from pathlib import Path
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics as st

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CORE = ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1'
AUDIT = CORE / 'final_audit_20260930_v1'
TERMS = {'5080': CORE / 'monitor/20260928T195732Z/terminal_5080', '5090': AUDIT / 'terminal_5090'}
LEDGER = CORE / 'aggregation/20260929T232132255352Z/ledger.json'
FIRST = ROOT / 'reports/titantpp_first_analysis_20260930_v1/analysis.json'
DATASETS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart'}
LABELS = {'titantpp': 'B（이력 보완 없음）', 'titantpp_local_detail': 'Full', 'titantpp_history_mlp': 'TitanTPP（이력 MLP）', 'titantpp_level_only': '수준만', 'titantpp_change_only': '변화만', 'titantpp_no_static_lmm': '정적 검색 제거（Full 기반）', 'rmtpp': 'RMTPP', 'thp': 'THP', 'nhp': 'NHP', 'sahp': 'SAHP'}
METRICS = ['qty_mae', 'qty_rmse', 'time_nll']
KEYS = ['dataset', 'seed', 'arm']
SOURCES = {}

def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def rel(p): return str(p.relative_to(ROOT))
def read(p):
    SOURCES[rel(p)] = file_sha(p)
    return json.loads(p.read_text())
def canonical(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
def put(name, value): (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def close(a, b): assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)
def stats(values): return dict(n=len(values), mean=st.mean(values), sample_sd=st.stdev(values) if len(values) > 1 else None)
def key(r): return tuple(r[k] for k in KEYS)
def csv_out(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
def fmt(x): return f"{x['mean']:.4f} ± {x['sample_sd']:.4f}"

def reconcile(ep):
    assert ep['evaluation_scope'] == 'validation_only' and ep['held_out_test_evaluated'] is False
    assert ep['count'] > 0 and all(math.isfinite(ep[m]) for m in METRICS)
    close(ep['qty_rmse'] ** 2 * ep['count'], ep['qty_sse'])
    # Frozen definitions: body=q<=boundary[2], tail=q>boundary[3].
    # Middle bin3 lies between them, so body+tail alone is not a partition.
    for cells in [[ep['body'], ep['quantity_cells'][3], ep['tail']]] + [ep[g] for g in ['quantity_cells', 'history_cells', 'additional_history_cells']]:
        assert sum(c['count'] for c in cells) == ep['count']
        close(sum(c['qty_sse'] for c in cells), ep['qty_sse'])
        for m in ['qty_mae', 'time_nll']:
            close(sum(c[m] * c['count'] for c in cells if c['count']) / ep['count'], ep[m])
        for c in cells:
            if c['count']: close(c['qty_rmse'] ** 2 * c['count'], c['qty_sse'])

def main():
    spec = importlib.util.spec_from_file_location('saved_evidence_audit', CORE / 'collect_progress.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    c = read(CORE / 'frozen_execution/execution_contract.json')
    assert canonical(c) == audit.CONTRACT_SHA and canonical(c['source']['files']) == audit.SOURCE_SHA
    ledger, prior = read(LEDGER), read(FIRST)
    assert ledger['counts']['new_audited'] == 36 and ledger['counts']['reuse_audited'] == 18
    checked_bytes = {}
    for host, folder in TERMS.items():
        receipt, terminal = read(folder / 'collection_manifest.json'), read(folder / 'terminal_audit.json')
        assert terminal['status'] == 'passed' and terminal['fits'] == (24 if host == '5080' else 12)
        assert terminal['endpoint_records'] == terminal['fits'] * 2 and terminal['owned_processes_absent']
        assert file_sha(folder / 'collection.tar') == terminal['archive_sha256']
        for name, meta in receipt['files'].items():
            p = folder / name
            assert p.stat().st_size == meta['bytes'] and file_sha(p) == meta['sha256'], p
        checked_bytes[host] = len(receipt['files'])
        assert receipt['source_files'] == c['source']['files'] and receipt['source_files_verified'] == 97
    for name, digest in c['source']['files'].items():
        assert file_sha(TERMS['5090'] / 'source' / name) == digest
    baseline_audit = read(CORE / 'baseline_native_completion.json')
    assert baseline_audit['status'] == 'passed' and baseline_audit['conditions'] == 18
    assert baseline_audit['native_initialization_rebuild']['status'] == 'passed'
    preserved = read(CORE / 'recovery_5090_20260929_v1/preservation.json')
    assert preserved == read(TERMS['5090'] / 'recovery_5090_v1/preservation.json')
    for name, digest in preserved['files'].items():
        assert file_sha(CORE / 'recovery_5090_20260929_v1/pre_reboot' / name) == digest
    handoff = read(TERMS['5090'] / 'concurrency_v2/original_completed.json')
    old_receipt = read(TERMS['5090'] / 'run/arm_receipts' / (handoff['job'] + '.json'))
    assert canonical(old_receipt) == handoff['original_fit_receipt_sha256']
    jid = handoff['transferred_job']
    worker = read(TERMS['5090'] / 'run/workers' / (jid + '.json'))
    claim = read(TERMS['5090'] / 'concurrency_v2/job_claims' / (jid + '.json'))
    assert worker['authority']['owner_pid'] == claim['controller_pid']
    assert worker['amendment_sha256'] == claim['amendment_sha256'] == '946eb5e102633d9c9965c1727f07dd041d46e735364ca6c0e864fdd7ebbcea9d'
    assert worker['job'] == claim['job'] == jid
    read(AUDIT / 'collector_correction.json')
    read(CORE.parent / 'titantpp_concurrency_5090_20260928_v1/timing_annotation.json')
    rows, exposure_by_key, stability, cost_rows = [], {}, [], []
    datasets = {d['dataset_id']: d for d in c['datasets']}
    for old in ledger['conditions']:
        r = copy.deepcopy(old)
        if r['reused']:
            folder = ROOT / r['source']
            reuse = next(x for x in c['reuse'] if key(x) == key(r))
            for name, digest in reuse['file_sha256'].items(): assert file_sha(folder / name) == digest
        else:
            folder = TERMS[r['host']] / f"run/{r['dataset']}/seed_{r['seed']}/runs/{r['arm']}/count_only_log_regression/seed_{r['seed']}"
            audited = read(TERMS[r['host']] / 'terminal_audit.json')['checkpoint_checks']
            for label in ['selected', 'last']:
                check = next(x for x in audited if x['job'] == r['id'] and x['endpoint'] == label)
                assert check['state_sha256'] == r[label]['state_sha256']
        values, exposure, replay = audit.audit_condition(folder, datasets[r['dataset']], r['initial_state_sha256'], c['training'])
        assert all(values[k] == r[k] for k in values), key(r)
        history = read(folder / 'history.json')['history']
        assert history[min(range(len(history)), key=lambda i: history[i]['val_qty_rmse'])]['epoch'] == r['best_epoch']
        for window, hh in [('first40', history[:40]), ('last30', history[-30:])]:
            ss = stats([h['val_qty_rmse'] for h in hh])
            for field, original in [('mean', 'mean'), ('sample_sd', 'sd')]: close(ss[field], replay[window][original])
            stability.append({**{k: r[k] for k in KEYS}, 'window': window, 'epochs': len(hh), 'rmse_mean': ss['mean'], 'rmse_sample_sd': ss['sample_sd']})
        exposure_by_key[key(r)] = exposure
        r['original_snapshot_source'] = r['source']; r['source'] = rel(folder); r['campaign'] = 'core'
        r['checkpoint_audit_status'] = 'passed'; r['first40'] = stats([h['val_qty_rmse'] for h in history[:40]])
        for endpoint in ['selected', 'last']: reconcile(r[endpoint])
        timing_path = folder / 'combined_epoch_timing.json'
        if not timing_path.exists(): timing_path = folder / 'epoch_timing.json'
        epoch_seconds = None
        if timing_path.exists():
            timings = read(timing_path)['epochs']
            assert [x['epoch'] for x in timings] == list(range(1, r['completed_epochs'] + 1))
            epoch_seconds = sum(x['elapsed_seconds'] for x in timings)
        recovery = read(folder / 'recovery_cost.json') if (folder / 'recovery_cost.json').exists() else None
        if recovery:
            close(epoch_seconds, recovery['complete_epoch_seconds'])
            close(r['fit_elapsed_seconds'], recovery['post_resume_trainer_seconds'])
            prefix_dir = CORE / 'recovery_5090_20260929_v1/pre_reboot' / folder.relative_to(TERMS['5090'])
            restoration = read(TERMS['5090'] / 'recovery_5090_v1/jobs' / r['id'] / 'restoration.json')
            saved_n = restoration['saved_epoch']
            assert file_sha(prefix_dir / 'last_epoch_state.pt') == restoration['checkpoint_sha256']
            assert restoration['next_epoch'] == saved_n + 1
            assert history[:saved_n] == read(prefix_dir / 'history.json')['history']
            prefix_exposure = read(prefix_dir / 'exposure.json')
            assert exposure['train'][:saved_n] == prefix_exposure['train']
            assert exposure['validation'][:saved_n] == prefix_exposure['validation']
            assert timings[:saved_n] == read(prefix_dir / 'epoch_timing.json')['epochs']
        cost_rows.append({**{k:r[k] for k in KEYS}, 'host':r['host'], 'reused':r['reused'], 'parameters':r['parameters'], 'completed_epochs':r['completed_epochs'], 'optimizer_steps':r['optimizer_steps'], 'recorded_train_one_seconds':r['fit_elapsed_seconds'], 'recorded_interval_scope':'post_resume_only' if recovery else 'original_train_one', 'completed_epoch_seconds_sum':epoch_seconds, 'preserved_pre_resume_epoch_seconds':recovery['preserved_completed_epoch_seconds'] if recovery else None, 'peak_allocated_mib':r['peak_allocated_bytes']/2**20, 'standalone_eligible':not r['timing_interference_recorded'] and not recovery, 'timing_interference':r['timing_interference_recorded'], 'recovery':bool(recovery), 'amendment_sha256':r.get('concurrency_amendment_sha256'), 'diagnostic_epochs':','.join(map(str,r.get('affected_epochs',[]))), 'source':rel(folder)})
        r['recovery_cost'] = recovery; rows.append(r)
    # Cross-model train prefix and complete validation batch order, including restored epochs.
    for r in rows:
        if r['reused']: continue
        e = exposure_by_key[key(r)]
        for arm in ['titantpp', 'titantpp_local_detail']:
            other = exposure_by_key[(r['dataset'], r['seed'], arm)]
            n = min(len(e['train']), len(other['train']))
            assert e['train'][:n] == other['train'][:n]
            assert len({v['batch_order_sha256'] for ex in [e, other] for v in ex['validation']}) == 1
    assert len(rows) == len({key(r) for r in rows}) == 54
    external = read(ROOT / 'reports/local_detail_3seed_final_20260927_v1/condition_results.json')
    idx = {key(r):r for r in rows}; overlap = 0
    for x in external:
        e = dict(x, arm=x['model'], campaign='external_controlled')
        for endpoint in ['selected','last']: reconcile(e[endpoint])
        if e['arm'] in ['titantpp','titantpp_local_detail']:
            old = idx[key(e)]
            for endpoint in ['selected','last']: assert all(old[endpoint][m] == e[endpoint][m] for m in METRICS + ['state_sha256'])
            overlap += 1
        else: rows.append(e)
    assert overlap == 18 and len(rows) == len({key(r) for r in rows}) == 90
    idx = {key(r):r for r in rows}
    aggregates, aggregate_csv, strata = [], [], []
    for ds in DATASETS:
        for arm in LABELS:
            rr = [idx[ds,s,arm] for s in [42,52,62]]
            a = dict(dataset=ds,model=arm,seeds=[42,52,62])
            for endpoint in ['selected','last']:
                a[endpoint] = {m:stats([r[endpoint][m] for r in rr]) for m in METRICS}
                aggregate_csv.append({'dataset':ds,'model':arm,'endpoint':endpoint,'seed_count':3,**{f'{m}_{f}':a[endpoint][m][f] for m in METRICS for f in ['mean','sample_sd']}})
                for group in ['body','tail','quantity_cells','history_cells','additional_history_cells']:
                    cells = [r[endpoint][group] for r in rr]
                    for i in range(len(cells[0]) if isinstance(cells[0],list) else 1):
                        cc = [v[i] if isinstance(v,list) else v for v in cells]
                        n = cc[0]['count']; assert all(v['count'] == n for v in cc)
                        strata.append({'dataset':ds,'model':arm,'endpoint':endpoint,'group':group,'bin':i if isinstance(cells[0],list) else None,'count_per_seed':n,**{f'{m}_{f}':stats([v[m] for v in cc])[f] if n else None for m in METRICS for f in ['mean','sample_sd']}})
            aggregates.append(a)
    amap = {(a['dataset'],a['model']):a for a in aggregates}
    for old in prior['aggregates']:
        for endpoint in ['selected','last']:
            for m in METRICS:
                for field in ['mean','sample_sd']: close(old[endpoint][m][field], amap[old['dataset'],old['model']][endpoint][m][field])
    # Reapply the originally frozen Full-vs-MLP gate; other comparisons are exploratory.
    gate_rows = []
    ac = c['comparison']['acceptance']
    for ds in DATASETS:
        for arm in list(LABELS)[:6]:
            if arm == 'titantpp_local_detail': continue
            for seed in [42,52,62]:
                full, ref = idx[ds,seed,'titantpp_local_detail'],idx[ds,seed,arm]
                x,y = full['selected'],ref['selected']
                checks = {'overall_rmse':x['qty_rmse']<y['qty_rmse'],'overall_mae':x['qty_mae']<=ac['overall_mae_ratio_max']*y['qty_mae'],'body_mae':x['body']['qty_mae']<=ac['body_mae_ratio_max']*y['body']['qty_mae'],'tail_mae':x['tail']['qty_mae']<=ac['tail_mae_ratio_max']*y['tail']['qty_mae'],'tail_rmse':x['tail']['qty_rmse']<y['tail']['qty_rmse'],'time_nll':x['time_nll']<=y['time_nll']+ac['recorded_time_nll_increase_max'],'last30_mean':full['last30']['mean']<=ac['last30_rmse_mean_ratio_max']*ref['last30']['mean'],'last30_sd':full['last30']['sd']<=ac['last30_rmse_sample_sd_ratio_max']*ref['last30']['sd']}
                for i in [1,2,3]:
                    if x['quantity_cells'][i]['count']:
                        for m in ['qty_mae','qty_rmse']: checks[f'quantity_bin{i}_{m}'] = x['quantity_cells'][i][m] <= ac['each_nonempty_middle_bin_mae_and_rmse_ratio_max'] * y['quantity_cells'][i][m]
                gate_rows.append({'dataset':ds,'seed':seed,'candidate':'titantpp_local_detail','reference':arm,'prespecified_primary':arm=='titantpp_history_mlp','passed_all':all(checks.values()),'failed_checks':','.join(k for k,v in checks.items() if not v),**checks})
    paired = []
    for ds in DATASETS:
        for arm in LABELS:
            if arm == 'titantpp_history_mlp':continue
            for seed in [42,52,62]:
                x,y = idx[ds,seed,'titantpp_history_mlp']['selected'],idx[ds,seed,arm]['selected']
                paired.append({'dataset':ds,'seed':seed,'candidate':'titantpp_history_mlp','reference':arm,**{f'{m}_difference':x[m]-y[m] for m in METRICS},**{f'{m}_reduction_pct':100*(1-x[m]/y[m]) for m in ['qty_mae','qty_rmse']}})
    # Referenced, already completed reports are immutable evidence for the separate panels.
    read(ROOT / 'reports/titantpp_efficiency_5080_20260930_v1/verification.json')
    read(ROOT / 'reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json')
    sources = {**SOURCES, **audit.SOURCES}
    verification = dict(status='passed',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='validation_only',core_conditions=54,core_endpoint_roles=108,new_conditions=36,reused_conditions=18,external_additional_conditions=36,core_external_unique_conditions=90,deduplicated_overlap=18,new_checkpoint_roles=72,reused_checkpoint_roles=36,final_core_binary_source_audit_complete=True,final_report_complete=True,terminal_5090_retrieved_files=checked_bytes['5090'],terminal_5080_existing_files_reverified=checked_bytes['5080'],source_files=97,pre_reboot_files_reverified=len(preserved['files']),core_byte_integrity_failures=0,core_record_audit_failures=0,all_terminal_metrics_reconcile_with_first_analysis=True,selected_and_last_aggregates=60,stratum_rows=len(strata),strict_selector_stop_exposure_prefix_audit=True,full_vs_mlp_primary_gate_passes=sum(g['passed_all'] for g in gate_rows if g['prespecified_primary']),full_vs_mlp_primary_gate_pairs=9,checkpoint_audit_basis='5090 new CPU strict/tensor/optimizer/RNG state audit; 5080 and reused CPU audits reused after byte SHA revalidation',new_training=False,new_gpu_evaluation=False,held_out_or_raw_data_read=False,scheduler_created=False,limitations=['Existing native validation endpoint replays reused; no new predictions.', 'Recorded native initialization qualifications reused; no new initialization trial.', 'Validation model choice is exploratory; independent held-out and significance unverified.', 'External36 metrics reuse the prior final campaign evidence; this operation does not newly audit external36 binaries.', 'Historical process absence is recorded at original terminal collection times, not an ongoing observation.'])
    result = dict(status='final_validation_evidence_audit_complete',evaluation_scope='validation_only',representative_model='titantpp_history_mlp',representative_choice='post-validation user decision; fixed across datasets, not a preregistered superiority hypothesis',selection=c['comparison']['selector'],seed_statistic='equal weight seeds42/52/62; sample SD ddof1, not confidence interval',contract_sha256=canonical(c),source_closure_sha256=c['source']['files_sha256'],verification=verification,aggregates=aggregates,core_conditions=[r for r in rows if r['campaign']=='core'],external_conditions=[r for r in rows if r['campaign']!='core'],strata=strata,stability=stability,full_comparison_gate=gate_rows,mlp_paired_comparisons=paired,costs=cost_rows,acceptance=ac,sources_sha256=sources)
    put('final_results.json',result); put('verification.json',verification)
    csv_out('final_three_seed_metrics.csv',aggregate_csv); csv_out('final_strata.csv',strata); csv_out('final_stability.csv',stability); csv_out('final_costs.csv',cost_rows); csv_out('final_gate_checks.csv',gate_rows); csv_out('final_mlp_paired.csv',paired)
    condition_csv = []
    for r in rows:
        for endpoint in ['selected','last']:
            condition_csv.append({**{k:r[k] for k in KEYS},'campaign':r['campaign'],'endpoint':endpoint,'epoch':r['best_epoch'] if endpoint=='selected' else r['completed_epochs'],'count':r[endpoint]['count'],**{m:r[endpoint][m] for m in METRICS},'state_sha256':r[endpoint]['state_sha256'],'source':r.get('source',r.get('source_path','reports/local_detail_3seed_final_20260927_v1/condition_results.json'))})
    csv_out('final_conditions.csv',condition_csv)
    write_report(result,amap)
    outputs=['final_report.md','final_results.json','verification.json','final_three_seed_metrics.csv','final_conditions.csv','final_strata.csv','final_stability.csv','final_costs.csv','final_gate_checks.csv','final_mlp_paired.csv','finalize.py']
    put('final_output_manifest.json',{'at_utc':verification['at_utc'],'files':{n:file_sha(OUT/n) for n in outputs},'sources_sha256':sources})
    print(json.dumps(verification,ensure_ascii=False))

def write_report(result,amap):
    v=result['verification'];rows=result['core_conditions']+result['external_conditions'];idx={key(r):r for r in rows}
    lines=['# TitanTPP 핵심 구조 비교 — 최종 validation 결과 및 원본 감사','',f"최종화: {v['at_utc']} · 대표 구조: `titantpp_history_mlp` · 평가: validation only",'',
      '## 결론','',
      '**54조건·108개 selected/last 평가 역할의 원본 감사가 완료됐다.** 신규36조건의72개 checkpoint와 재사용18조건의36개 checkpoint를 감사 체인으로 연결했다. 이번5090 회수376파일·24checkpoint·원본97소스와 기존5080/재사용 파일의 SHA 대조에서 무결성 오류가 없었다. 학습·재평가 완료와 원본 감사 완료가 모두 충족됐다.',
      '', '**논문 대표 구조는 세 데이터 공통 이력 MLP이며, 논문에서는 TitanTPP로 표기한다.** Taxi·Intermittent의 수량 MAE/RMSE 경쟁력은 유지된다. Instacart에서 외부 최저 오차를 넘지 못했고, 수량 개선이 시간 NLL이나 모든 구간의 개선을 뜻하지 않는다. Full의 사전 성능 보호기준은 MLP 대비0/9 통과다. 실행·감사 성공을 성능 기준 통과로 바꾸어 해석하지 않는다.',
      '', '## 범위와 검증 방법','',
      '- Core: 6구조×3데이터×seed42/52/62 =54조건. 신규36 + 재사용 B/Full18. 같은 가중치가 두 역할에 쓰여도 selected/last 역할은 각각 보존한다.',
      '- 외부: RMTPP·THP·NHP·SAHP36조건은 기존 검증 결과를 재사용한다. 외부 캠페인의 B/Full18은 endpoint 수치와 state SHA로 중복 대조 후 제외했다. core+외부 고유90조건이며180역할이다. 이번5090 binary 감사 범위를 외부36조건까지 확대했다고 보고하지 않는다.',
      '- Gate: Full A/B2조건 + MLP4조건은 별도seed42 탐색이며 이3seed 표에 섞지 않는다. Full Taxi C/D는 미시작 보류다.',
      '- 모든 지표는 최초 strict validation raw 수량 RMSE 최소 checkpoint에서 함께 읽었다. MAE 최소 epoch로 재선택하지 않았다. ±는3seed 표본표준편차(ddof1)이며 신뢰구간·통계적 유의성·동등성 증명이 아니다.',
      '- batch128/max300/min40/patience40·전체train/validation·공통head/loss/optimizer·초기화·data/split identity·배치prefix·조기 종료를 기존 계약과 대조했다. 시간 지표는 계약의 `recorded_positive_integer_time_nll`로, v0.7 과거 clamped 점수와 섞지 않았다.',
      '- 새 학습·예측·GPU replay·raw data·held-out 열람 없이 CPU와 저장 기록으로 감사했다. 원래 native 초기화 검증과 validation replay를 재사용한다. 대표 모델 결정은 validation을 본 뒤의 선택이며 독립 검증은 아직 남았다.','',
      '## 대표 TitanTPP의 결과','']
    lines+=table(['데이터','MAE ↓','RMSE ↓','시간 NLL ↓'],[[name]+[fmt(amap[ds,'titantpp_history_mlp']['selected'][m]) for m in METRICS] for ds,name in DATASETS.items()])
    lines+=['','외부4모델 중 지표별 최저 평균과 비교한다. 감소율은 (비교군 평균−TitanTPP 평균)/비교군 평균이다.','']
    effects=[]
    for ds in DATASETS:
        row=[DATASETS[ds]]
        for m in ['qty_mae','qty_rmse']:
            best=min(['rmtpp','thp','nhp','sahp'],key=lambda a:amap[ds,a]['selected'][m]['mean']);a=amap[ds,'titantpp_history_mlp']['selected'][m]['mean'];b=amap[ds,best]['selected'][m]['mean'];wins=sum(idx[ds,s,'titantpp_history_mlp']['selected'][m]<idx[ds,s,best]['selected'][m] for s in [42,52,62]);row+=[f'{LABELS[best]} 대비 {100*(1-a/b):+.2f}% ({wins}/3 seed)']
        effects.append(row)
    lines+=table(['데이터','MAE 감소','RMSE 감소'],effects)
    lines+=['','Taxi·Intermittent에서는 같은seed끼리 비교해 두 수량 지표 모두 외부4모델 전체보다3/3 낮다. 이는 공통head·고정학습 조건의 비교 구현에 대한 결과이며, 원논문의 native head·최적 튜닝 모델 전체의 순위를 의미하지 않는다.','']
    for ds,name in DATASETS.items():
        lines += [f'## {name}: 전체 구조·외부 비교','']
        lines += table(['모델','선택 MAE ↓','선택 RMSE ↓','선택 시간 NLL ↓'],[[LABELS[a]]+[fmt(amap[ds,a]['selected'][m]) for m in METRICS] for a in LABELS])
        lines += ['']
    lines += ['## 내부 비교의 반례와 사전 기준','',
      '- Taxi에서는 Full의 평균 RMSE가 MLP보다1.17% 낮다. Full의 MAE 승리는1/3seed, RMSE 승리는2/3seed다. MLP를 모든 내부 구조보다 우월하다고 쓰지 않는다.',
      '- Intermittent에서는 MLP가 Full 대비 RMSE3/3seed, MAE2/3seed에서 낮다. MLP의 시간 NLL0.4971은 B0.3846·RMTPP0.2748보다 높다.',
      '- Instacart에서는 변화만의 core 평균 MAE/RMSE가 가장 낮지만 외부 지표별 최저를 넘지 못한다. MLP의 MAE는 RMTPP보다약0.16%, RMSE는THP보다약0.42% 높다. 작아 보이는 차이를 동등성으로 선언하지 않는다.',
      '- 수준·변화 모듈은 추가4,096파라미터, Full/MLP는6,144개다. Full 기반 정적 검색 제거 결과를 MLP에서 해당 검색의 필수성이나 불필요성 증명으로 사용하지 않는다.','']
    gate=result['full_comparison_gate']
    lines+=table(['데이터','Full의 비교 기준','전체 보호기준 통과','비교 성격'],[[DATASETS[ds],LABELS[a],f"{sum(g['passed_all'] for g in gate if g['dataset']==ds and g['reference']==a)}/3",'사전 주 비교' if a=='titantpp_history_mlp' else '탐색적 동일기준 적용'] for ds in DATASETS for a in list(LABELS)[:6] if a!='titantpp_local_detail'])
    lines+=['','기준은 전체RMSE 개선, 전체MAE≤1.01배, body/tail MAE≤1.02배, tailRMSE 개선, 중간bin1–3 MAE/RMSE≤1.02배, 시간NLL 증가≤0.01, last30 평균·표본SD≤1.05배다. 각seed 실패 항목은 `final_gate_checks.csv`에 모두 남겼다. MLP 우월성 기준으로 사후 뒤집어 쓰지 않았다.','', '## 구간별 반례와 표본 수','']
    strata=result['strata']
    find=lambda ds,a,g:next(s for s in strata if s['dataset']==ds and s['model']==a and s['group']==g and s['endpoint']=='selected')
    lines+=table(['데이터','모델','body MAE','tail MAE','tail RMSE','tail 표본/seed'],[[DATASETS[ds],LABELS[a],f"{find(ds,a,'body')['qty_mae_mean']:.4f}",f"{find(ds,a,'tail')['qty_mae_mean']:.4f}",f"{find(ds,a,'tail')['qty_rmse_mean']:.4f}",find(ds,a,'tail')['count_per_seed']] for ds in DATASETS for a in ['titantpp','titantpp_local_detail','titantpp_history_mlp','titantpp_no_static_lmm']])
    lines+=['','TitanTPP의 Instacart tail RMSE24.6419는 B24.4246보다 높다. Taxi tail 표본은seed당79개로 적고 동일validation 표본을3seed 반복한 것이므로237개 독립 표본이 아니다. Intermittent에서는 MLP의 body MAE0.5324가 정적검색 제거0.5238보다 높다. 전체 이득과 세부 구간의 반례를 함께 보존한다.',
      '', '모든 수량·이력 구간 경계는 train에서 고정했다. 기본 이력>64의 Instacart 빈 구간은 결측으로 표시하고0점으로 채우지 않았다. `final_strata.csv`에는30모델/데이터 조합의 selected/last840행, 각 구간의 count·MAE/RMSE/시간NLL 평균·표본SD를 남겼다. 경계와원값은 `final_results.json`에 있다.','', '## 선택·마지막 checkpoint와 학습 안정성','']
    lines+=table(['데이터','모델','last MAE','last RMSE','last 시간 NLL'],[[DATASETS[ds],LABELS[a]]+[fmt(amap[ds,a]['last'][m]) for m in METRICS] for ds in DATASETS for a in list(LABELS)[:6]])
    lines+=['','모델별 조기 종료 epoch가 다르다. `final_stability.csv`의108행에는core54조건마다 고정 첫40epoch와 마지막30epoch RMSE 평균·epoch 간 표본SD를 따로 기록했다. 이는seed 간 표본SD와 다르다. 마지막30epoch는 서로 다른 학습 시점이므로 고정40epoch 비교와 함께 해석한다.','', '## 실행 비용과 측정 범위','']
    costs=result['costs']; cost_table=[]
    for ds in DATASETS:
        for arm in list(LABELS)[:6]:
            rr=[r for r in costs if r['dataset']==ds and r['arm']==arm]
            cost_table.append([DATASETS[ds],LABELS[arm],rr[0]['parameters'],f"{st.mean(r['recorded_train_one_seconds'] for r in rr)/60:.2f}",f"{st.mean(r['completed_epochs'] for r in rr):.1f}",f"{st.mean(r['peak_allocated_mib'] for r in rr):.2f}",sum(r['standalone_eligible'] for r in rr)])
    lines+=table(['데이터','모델','파라미터','기록fit 평균분','종료epoch 평균','peak allocated 평균MiB','단독비교 가능seed수'],cost_table)
    lines+=['','fit시간은 train_one 준비/학습/validation/보고 구간이며별도endpoint replay를 제외한다. 복구 조건은 복구 이후 구간만 포함한원값으로 표시했다. `final_costs.csv`에 복구 전 완료epoch합·전체 완료epoch합·진단겹침epoch62/63/64·병렬amendment를 구분했다. 병렬fit합은GPU장치시간이 아니며, 소실부분epoch·중단·겹침을 추정 보정하지 않았다. 장치/Runtime가 다른데이터 간속도비교도 하지 않는다.','']
    lines+=table(['Instacart seed62 복구조건','복구 전 완료epoch초','복구 후 train_one초','전체 완료epoch 기록합초'],[[LABELS[r['arm']],f"{r['preserved_pre_resume_epoch_seconds']:.3f}",f"{r['recorded_train_one_seconds']:.3f}",f"{r['completed_epoch_seconds_sum']:.3f}"] for r in costs if r['recovery']])
    lines+=['','학습peak는 프로세스allocator 값이며 장치전체메모리나추론메모리가 아니다. 개인GPU의추가cloud임대료는0달러이며전기료/에너지는미측정이다. 총전기비용을0으로주장하지 않는다.',
      '', '별도의 승인된5080 MLP–Titans-MAC 동일입력짧은측정은 [효율보고서](../titantpp_efficiency_5080_20260930_v1/report.md)에 분리되어 있다. 학습step5.74배/5.58배, 평가batch8.49배/8.21배 가속과학습peak약75.5%감소를보였으나, 평가peak는약3.98배증가했다. 이는전체epoch·순수추론·동일정확도비용의증거가아니다. 이번최종화에서측정을추가하지않았다.','', '## 실패·인계·복구 기록과 감사 범위','',
      '- 5090 v1 인계는새fit전 실패한과거기록이다. v2 원관리자의정확한FileExistsError는원fit/두replay완료·이전조건worker/claim소유권·canonical receipt SHA·종료증거와연결했다. 오류원본은그대로보존했다.',
      '- 재부팅전103파일과MLP17/수준11epoch checkpoint를보존SHA와재대조했다. 승인된커널7.0.0-28→7.0.0-34만변경했고, 최대2worker복구후나머지조건을완료했다. 합성66updates는과학적fit에포함하지않는다.',
      '- 5090 최종회수는원격읽기전용으로source97·라이브러리·계약·병렬/복구계약·소유process/tmux종료를확인했다. 로컬CPU에서24checkpoint의바이트/텐서SHA·엄격한로딩·epoch/seed/route·초기화식별·optimizerstep/모멘트·RNG/shuffle상태·history/exposure/beststate를검증했다. 기존5080 48checkpoint와재사용36checkpoint는기존CPU감사및현재byteSHA재대조를사용했다.',
      '- 과거회수의빈archive는수집기가canonical JSON SHA를파일byte SHA와비교한오류였다. 생산코드와원본JSON으로원인을확인하고수집기만수정했다. 이전실패폴더·수정전수집기·교정증거를보존했다. 과학적소스/결과수정이나학습실패가아니다.',
      '- 새prediction pass는실행하지않았다. 이미실행된native validation replay를history/summary/SSE·구간가중합·checkpoint state와연결했으며, native초기화qualification을재사용했다. 완료된학습에새qualification/재학습을추가하지않았다.',
      '', '## Gate 탐색은 별도 보고','',
      '[Gate 최종보고서](../titantpp_mlp_gate_execution_20260929_v1/final_report.md)의기존감사를재사용한다. 조건부MLP Gate는두데이터seed42에서plain MLP의MAE·RMSE를개선하지못했다. Taxi상수만수량MAE1.42%·RMSE2.76%개선했고시간NLL0.842735→1.304540으로악화했다. Full A/B완료,Full Taxi C/D보류,Full B epoch73 ACK실패및74epoch부터복구이력도유지한다. 이탐색을3seed우월성으로일반화하지않는다.','', '## 현재 완료와 다음 작업','',
      '**원본 감사·최종 결과표 — 완료**: core54조건108역할, 외부중복제거, MAE/RMSE/시간NLL·selected/last·3seed변동·구간반례·비용·실패이력보고완료. 수치집계는1차분석과모두일치하며이번변경은원본감사완료와최종근거연결이다.',
      '', '**차별성과 비교공정성 확인 — 다음 작업**: 원문·코드로방법차별성과기존TPP 비교조건을정리하고, 필요한추가비교의최소범위를결정한다. 새실험을자동시작하지않는다.',
      '', '**원고 통합 — 다음 작업**: 확정MLP 방법·이최종표·기존효율측정을연결한다. 방법서술과문헌검토는병행가능하며공통주장/결과표는현재세션에서일치시킨다.',
      '', '**필요 보완·독립held-out — 승인 필요**: 비교/모델/선택규칙을동결한뒤별도승인범위로진행한다. 기존스케줄러재생성·commit/Push·외부제출은이번작업에포함하지않는다.','', '## 원본과 재현','',
      f'- 5090 최종감사: `{rel(TERMS["5090"] / "terminal_audit.json")}`',
      f'- 5080 기존감사: `{rel(TERMS["5080"] / "terminal_audit.json")}`',
      f'- 재사용감사: `{rel(CORE / "baseline_native_completion.json")}`',
      '- `final_results.json`: 전체조건·selected/last·구간·seed별대조·cost·sourceSHA.',
      '- `final_conditions.csv`:90고유조건×2역할의180행. `final_three_seed_metrics.csv`:30모델/데이터×2역할의60행.',
      '- `final_strata.csv`, `final_stability.csv`, `final_gate_checks.csv`, `final_mlp_paired.csv`, `final_costs.csv`: 반례·안정성·기준실패·비용원값.',
      '- `verification.json`, `final_output_manifest.json`, `finalize.py`: 완료판정·출처/결과SHA·로컬재현코드. 기존1차분석과원본파일은보존했다.',
      f'- 원본계약 `{result["contract_sha256"]}`,97소스closure `{result["source_closure_sha256"]}`.','']
    # Keep the narrative readable separately from dense machine-readable field names.
    polished = {
      'fit시간은': '기록된 fit 시간은 train_one의 준비·학습·validation·보고 구간이며 별도의 endpoint replay를 제외한다. 복구 조건은 복구 이후 시간만 포함한 원값이다. `final_costs.csv`에는 복구 전 완료 epoch 합, 전체 완료 epoch 합, 진단과 겹친 epoch 62·63·64, 병렬 실행 계약을 구분했다. 병렬 fit 시간의 합을 GPU 장치 사용시간으로 해석하지 않으며, 소실된 부분 epoch·중단·겹침을 추정 보정하지 않았다. 서로 다른 GPU·Runtime에서 측정한 데이터 간 속도 비교도 하지 않는다.',
      '학습peak는': '학습 peak 메모리는 프로세스 allocator 값이며 장치 전체 메모리나 추론 메모리가 아니다. 개인 GPU의 추가 cloud 임대료는 0달러이며, 전기료와 에너지는 미측정이다. 총비용이 0이라는 의미는 아니다.',
      '별도의 승인된5080': '별도로 승인된 5080 MLP–Titans-MAC 동일 입력 측정은 [효율 보고서](../titantpp_efficiency_5080_20260930_v1/report.md)에 있다. Taxi·Intermittent에서 학습 step은 각각 5.74배·5.58배, 평가 batch는 8.49배·8.21배 빨랐고 학습 peak 메모리는 약 75.5% 감소했다. 반면 평가 peak 메모리는 약 3.98배 증가했다. 전체 epoch·순수 추론·동일 정확도 비용을 입증한 측정은 아니며, 이번 감사에서 새 측정을 추가하지 않았다.',
      '- 5090 v1 인계는': '- 5090 v1 인계는 새 fit 전에 실패한 과거 기록이다. v2 원래 관리자의 정확한 FileExistsError는 기존 fit·두 replay 완료, 이전된 조건의 worker/claim 소유권, canonical receipt SHA, 종료 증거와 연결했다. 오류 원본은 그대로 보존했다.',
      '- 재부팅전103': '- 재부팅 전 103개 파일과 MLP 17·수준만 11epoch checkpoint를 보존 SHA와 다시 대조했다. 승인된 커널 변경(7.0.0-28→7.0.0-34) 후 최대 두 worker로 복구와 나머지 조건을 완료했다. 합성 66updates는 과학적 fit에 포함하지 않는다.',
      '- 5090 최종회수는': '- 5090 최종 회수에서는 원격 읽기 전용으로 소스 97개·라이브러리·원본 및 병렬/복구 계약·소유 프로세스/tmux 종료를 확인했다. 로컬 CPU에서 24개 checkpoint의 바이트/텐서 SHA, 엄격한 모델 로딩, epoch·seed·모델 식별, 초기화 기록, optimizer step·모멘트, RNG·shuffle 상태, 학습 이력·노출·best state를 검증했다. 5080의 48개와 재사용 36개 checkpoint는 기존 CPU 감사와 현재 byte SHA 대조를 연결했다.',
      '- 과거회수의빈archive': '- 과거 회수의 빈 archive는 수집기가 canonical JSON SHA를 파일 byte SHA와 비교한 오류였다. 생산 코드와 원본 JSON으로 원인을 확인하고 수집기만 수정했다. 이전 실패 폴더·수정 전 수집기·교정 증거를 보존했다. 과학적 소스나 결과를 수정한 것이 아니며 학습 실패도 아니다.',
      '- 새prediction pass': '- 새로운 prediction pass는 실행하지 않았다. 이미 수행한 native validation replay를 history·summary·SSE·구간 가중합·checkpoint state와 연결했고, 기존 native 초기화 qualification을 재사용했다.',
      '[Gate 최종보고서]': '[Gate 최종 보고서](../titantpp_mlp_gate_execution_20260929_v1/final_report.md)의 기존 감사를 재사용한다. 조건부 MLP Gate는 두 데이터의 seed42에서 plain MLP의 MAE·RMSE를 개선하지 못했다. Taxi 상수만 수량 MAE 1.42%·RMSE 2.76%를 개선했고 시간 NLL은 0.842735→1.304540으로 악화했다. Full A/B 완료, Full Taxi C/D 보류, Full B의 epoch73 ACK 실패와 epoch74부터 복구한 이력을 유지한다. 이 탐색을 3seed 우월성으로 일반화하지 않는다.',
      '**원본 감사·최종 결과표 — 완료**:': '**원본 감사·최종 결과표 — 완료**: core 54조건·108역할의 원본 검증과 외부 중복 제거를 마쳤다. MAE·RMSE·시간 NLL, selected/last, 3seed 변동, 구간 반례, 비용과 실패 이력을 최종 표에 연결했다. 수치는 1차 분석과 모두 일치하며 원본 감사가 최종 완료됐다.',
      '**차별성과 비교공정성': '**차별성과 비교 공정성 확인 — 다음 작업**: 원문·코드로 방법의 차별성과 기존 TPP 비교 조건을 정리하고 필요한 추가 비교의 최소 범위를 결정한다.',
      '**원고 통합 — 다음 작업**:': '**원고 통합 — 다음 작업**: 확정한 MLP 방법, 이 최종 결과표, 기존 효율 측정을 원고로 연결한다. 방법 서술과 문헌 검토는 병행 가능하며, 공통 주장과 표는 현재 세션에서 일치시킨다.',
      '**필요 보완·독립held-out': '**필요 보완·독립 held-out — 승인 필요**: 비교군·모델·선택 규칙을 동결한 뒤 별도로 승인된 범위에서 진행한다. 스케줄러 재생성, commit/Push, 외부 제출은 이번 작업에 포함하지 않는다.',
      '모델별 조기 종료 epoch가': '모델별 조기 종료 epoch가 다르다. `final_stability.csv`의 108행에는 core 54조건마다 고정 첫 40epoch와 마지막 30epoch의 RMSE 평균·epoch 간 표본표준편차를 따로 기록했다. 이는 seed 간 표본표준편차와 다르다. 마지막 30epoch는 서로 다른 학습 시점이므로 고정 40epoch 비교와 함께 해석한다.',
      '모든 수량·이력 구간 경계는': '수량·이력 구간 경계는 train에서 고정했다. body는 세 번째 수량 경계 이하, tail은 네 번째 경계 초과다. 두 집합 사이의 bin3도 별도로 보고한다. Instacart의 이력 >64 구간은 비어 있어 결측으로 표시했다. `final_strata.csv`에는 selected/last 840행과 각 구간의 표본 수·MAE·RMSE·시간 NLL 평균 및 표본표준편차를 담았다. 모든 경계와 원값은 `final_results.json`에 있다.'
    }
    for i, line in enumerate(lines):
        for prefix, replacement in polished.items():
            if line.startswith(prefix):
                lines[i] = replacement
                break
    (OUT/'final_report.md').write_text('\n'.join(lines))

if __name__ == '__main__': main()
