"""Read frozen local validation snapshots; never evaluate data or change training."""
import hashlib
import json
import math
import statistics as st
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'search_artifacts/titantpp_core_ablation_20260928_v1'
OUT = Path(__file__).parent / 'interim_20260928_0820'
STAMP = '20260927T232003Z'
MODELS = ['titantpp', 'titantpp_local_detail', 'titantpp_history_mlp',
          'titantpp_level_only', 'titantpp_change_only', 'titantpp_no_static_lmm']
LABELS = dict(zip(MODELS, ['B', 'Full', '이력 MLP', '수준만', '변화만', '정적 검색 제거']))
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(PROJECT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


snaps = {h: read(ROOT / 'monitor' / STAMP / h / 'snapshot.json') for h in ['5080', '5090']}
contract = read(ROOT / 'frozen_execution/execution_contract.json')
canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
assert canonical == 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
assert read(ROOT / 'baseline_audit.json')['status'] == 'passed'
assert read(ROOT / 'baseline_native_completion.json')['status'] == 'passed'
rows = []
exposures = {}
for host, dataset, seeds, models in [
    ('5080', 'yellow_trip_hourly', [42, 52, 62], MODELS),
    ('5080', 'intermittent_frozen_5000', [42], MODELS[:3]),
    ('5090', 'insta_market_basket', [42], MODELS[:3]),
]:
    files = snaps[host]['files']
    qualification = files['qualification/receipt.json']
    assert qualification['status'] == 'passed'
    assert qualification['contract_sha256'] == canonical
    ds_contract = next(d for d in contract['datasets'] if d['dataset_id'] == dataset)
    for seed in seeds:
        inp = files[f'run/{dataset}/seed_{seed}/input_receipt.json']
        assert not inp['held_out_materialized']
        assert inp['populations'] == ds_contract['inherited_data_identity']['populations']
        for model in models:
            reused = model in MODELS[:2]
            if reused:
                folder = ROOT / f'baselines/{host}/{dataset}/seed_{seed}/{model}'
            else:
                receipt = files[f'run/arm_receipts/{dataset}__{seed}__{model}.json']
                assert receipt['status'] == 'complete'
                assert receipt['contract_sha256'] == canonical
                folder = ROOT / 'monitor' / STAMP / host / f'run/{dataset}/seed_{seed}/runs/{model}/count_only_log_regression/seed_{seed}'
            summary = read(folder / 'summary.json')
            replay = read(folder / 'endpoint_replays.json')
            history = read(folder / 'history.json')['history']
            exposure = read(folder / 'exposure.json')
            exposures[dataset, seed, model] = exposure
            assert summary['status'] == 'success'
            assert summary['evaluation_scope'] == 'validation_only'
            assert summary['held_out_test_evaluated'] is False
            assert summary['completed_epochs'] == len(history) == replay['completed_epochs']
            assert [h['epoch'] for h in history] == list(range(1, len(history) + 1))
            assert all(h['train_all_finite'] and math.isfinite(h['val_qty_rmse']) for h in history)
            best = min(history, key=lambda h: h['val_qty_rmse'])
            assert best['epoch'] == summary['best_epoch'] == replay['best_epoch']
            assert summary['checkpoint_monitor'] == 'validation_raw_quantity_rmse'
            assert summary['stopped_early'] == replay['stopped_early']
            expected_stop = max(40, best['epoch'] + 40)
            assert len(history) == (expected_stop if summary['stopped_early'] else 300)
            assert replay['global_steps'] == sum(h['train_batch_count'] for h in history)
            assert replay['initial_state_sha256'] == summary['initial_state_sha256']
            assert summary['initial_state_sha256'] == qualification['initialization'][dataset][str(seed)][model]
            assert len(exposure['train']) == len(history)
            for split in ['train', 'validation']:
                assert all(x['count'] == inp['populations'][split]['target_count'] for x in exposure[split])
            for endpoint, epoch_row in [('selected', best), ('last', history[-1])]:
                ep = replay[endpoint]
                assert ep['evaluation_scope'] == 'validation_only' and not ep['held_out_test_evaluated']
                assert ep['count'] == inp['populations']['validation']['target_count']
                close(ep['qty_rmse'], math.sqrt(ep['qty_sse'] / ep['count']))
                for metric in ['qty_rmse', 'qty_mae', 'time_nll']:
                    close(ep[metric], epoch_row['val_' + metric])
                    if endpoint == 'selected':
                        close(ep[metric], summary['best_val_' + metric])
                for cells in ['quantity_cells', 'history_cells']:
                    assert sum(c['count'] for c in ep[cells]) == ep['count']
                    close(sum(c['qty_sse'] for c in ep[cells]), ep['qty_sse'])
            assert replay['selected']['state_sha256'] == summary['checkpoint_state_sha256']
            for key in ['data_sha256', 'split_manifest_sha256']:
                assert summary['resume_identity']['arguments'][key] == inp['input_identity'][key]
            if not reused:
                assert receipt['result'] == replay
            rows.append(dict(dataset=dataset, host=host, seed=seed, model=model, reused=reused,
                             source=str(folder.relative_to(PROJECT)), best_epoch=best['epoch'],
                             completed_epochs=len(history), steps=replay['global_steps'],
                             parameters=summary['parameter_count'], elapsed_seconds=summary['elapsed_seconds'],
                             peak_allocated_bytes=summary['cuda_peak_memory_allocated_bytes'],
                             selected=replay['selected'], last=replay['last'], last30=replay['last30']))

prefix_checks = []
for row in rows:
    for base in MODELS[:2]:
        a = exposures[row['dataset'], row['seed'], row['model']]
        b = exposures[row['dataset'], row['seed'], base]
        lengths = {}
        for split in ['train', 'validation']:
            n = min(len(a[split]), len(b[split]))
            assert a[split][:n] == b[split][:n]
            lengths[split] = n
        prefix_checks.append(dict(dataset=row['dataset'], seed=row['seed'], model=row['model'], baseline=base, matched_epochs=lengths))

taxi = {m: [r for r in rows if r['dataset'] == 'yellow_trip_hourly' and r['model'] == m] for m in MODELS}
aggregate = {}
for m, rr in taxi.items():
    assert len(rr) == 3
    aggregate[m] = {metric: dict(mean=st.mean(r['selected'][metric] for r in rr), sample_sd=st.stdev(r['selected'][metric] for r in rr)) for metric in ['qty_rmse', 'qty_mae', 'time_nll']}
    aggregate[m].update(parameters=rr[0]['parameters'], mean_fit_minutes=st.mean(r['elapsed_seconds'] / 60 for r in rr), mean_seconds_per_epoch=st.mean(r['elapsed_seconds'] / r['completed_epochs'] for r in rr), mean_peak_mib=st.mean(r['peak_allocated_bytes'] / 1024**2 for r in rr))

OUT.mkdir(exist_ok=True)
payload = dict(as_of_kst='2026-09-28 08:20', evaluation_scope='validation_only', contract_sha256=canonical,
               checked_conditions=len(rows), checked_endpoint_records=2*len(rows), all_checks_passed=True,
               limitation='Recorded local evidence audited; new checkpoint binaries not downloaded or independently re-evaluated in this analysis.',
               source_sha256=SOURCES, prefix_checks=prefix_checks, taxi_aggregate=aggregate, conditions=rows)
(OUT / 'comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
md = ['# 진행 중 실험의 validation 비교 — 2026-09-28 08:20 KST', '',
      'Taxi 6종 × seed42·52·62의 완료된 18조건과, Intermittent/Instacart seed42의 완료된 B·Full·MLP 6조건을 비교한다. 신규 14조건과 재사용 10조건이며, 전체 캠페인 완료 보고가 아니다. 학습 중인 조건의 점수는 제외했다.', '',
      '모든 지표는 각 학습에서 validation raw-RMSE가 가장 낮았던 최초 checkpoint의 replay이다. ±는 seed 간 표본표준편차(ddof=1)이며 신뢰구간이나 통계적 유의성 검정이 아니다. test는 열람하지 않았다.', '',
      '## Taxi: 완료된 3개 seed', '', '| 모델 | RMSE ↓ | MAE ↓ | 기록된 시간 NLL ↓ | 파라미터 | 평균 fit 분 | 평균 초/epoch |', '|---|---:|---:|---:|---:|---:|---:|']
for m in MODELS:
    a=aggregate[m]
    md.append('| ' + LABELS[m] + ' | ' + ' | '.join(f"{a[k]['mean']:.4f} ± {a[k]['sample_sd']:.4f}" for k in ['qty_rmse','qty_mae','time_nll']) + f" | {a['parameters']:,} | {a['mean_fit_minutes']:.2f} | {a['mean_seconds_per_epoch']:.2f} |")
md += ['', 'fit 시간은 해당 원본 summary의 elapsed_seconds다. 조기 종료 시점이 다르고 재사용 학습과 신규 학습은 별도 실행이므로 순수 구조 속도 벤치마크나 배포 추론 비용으로 해석하지 않는다.', '', '### Seed별 원값과 실제 노출', '', '| 모델 | seed | RMSE | MAE | 시간 NLL | 선택/종료 epoch | optimizer steps |', '|---|---:|---:|---:|---:|---:|---:|']
for m in MODELS:
    for r in taxi[m]:
        e=r['selected'];md.append(f"| {LABELS[m]} | {r['seed']} | {e['qty_rmse']:.5f} | {e['qty_mae']:.5f} | {e['time_nll']:.5f} | {r['best_epoch']}/{r['completed_epochs']} | {r['steps']:,} |")
md += ['', '### 구간별 반례 (각 셀은 3-seed 평균)', '', '| 모델 | 최상위 수량 bin4 RMSE (79개) | bin4 MAE | 이력≤64 MAE (1,241개) | 이력65–128 MAE (1,475개) | 이력>128 MAE (5,552개) |', '|---|---:|---:|---:|---:|---:|']
for m, rr in taxi.items():
    vals=[st.mean(r['selected']['quantity_cells'][4][k] for r in rr) for k in ['qty_rmse','qty_mae']]
    vals += [st.mean(r['selected']['history_cells'][i]['qty_mae'] for r in rr) for i in range(3)]
    md.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
md += ['', '수량 구간은 원본 quantity_boundaries=[7,686,1562,3449]와 bin index를 그대로 사용했다. 최상위 bin4를 전체 큰 수량 구간과 혼동하지 않는다.', '', '## 다른 데이터셋: seed42만 완료된 부분 비교', '', '| 데이터 | 모델 | RMSE ↓ | MAE ↓ | 시간 NLL ↓ |', '|---|---|---:|---:|---:|']
for r in rows:
    if r['dataset']=='yellow_trip_hourly': continue
    e=r['selected'];md.append(f"| {r['dataset']} | {LABELS[r['model']]} | {e['qty_rmse']:.6f} | {e['qty_mae']:.6f} | {e['time_nll']:.6f} |")
md += ['', '이 두 데이터셋은 신규 MLP의 seed52·62와 나머지 구조가 미완료다. seed42를 전체 데이터셋의 결론으로 일반화하지 않는다.', '',
       '## 현재 해석', '',
       '- Taxi에서 Full은 B보다 평균 RMSE 12.96%, MAE 11.80% 낮다(평균끼리의 상대 차이). 세 seed 모두 수량 RMSE/MAE 개선 방향이다. 그러나 같은 RMSE 선택 checkpoint의 시간 NLL은 모든 seed에서 악화되어 기존 +0.01 시간 보호 기준을 충족하지 못한다. 수량과 시간을 모두 개선했다고 주장할 수 없다.',
       '- 주된 구조 비교인 Full 대 같은 파라미터 수의 MLP에서는 Full의 평균 RMSE 이득이 1.17%다. seed42·62에서는 Full, seed52에서는 MLP가 좋다. Full의 seed 간 RMSE 표준편차가 작지만 n=3으로 통계적 우월성이나 보편적 안정성을 확정하지 않는다.',
       '- Full이 수준만·변화만보다 Taxi 전체 RMSE에서 각 seed 모두 좋다. 다만 두 제거 모델은 각각 2,048개 파라미터도 적어, 차이를 정보 구성의 효과만으로 설명할 수 없다.',
       '- 정적 검색 제거는 평균 RMSE가 Full보다 약간 높지만 평균 MAE·최상위 수량 bin4 오차·시간 NLL이 낮고, 측정된 peak allocated GPU memory도 낮다. encoder persistent memory는 남아 있으므로 모든 메모리가 불필요하다는 뜻은 아니다. 마지막 정적 검색의 비용 대비 필요성은 미확정이다.',
       '- Full은 B에 비해 이력 65–128 구간의 평균 MAE와 RMSE가 오히려 높다. 전체 수량 개선을 모든 이력 길이에서의 개선으로 확장하지 않는다.',
       '- Intermittent seed42는 MLP가 Full보다 수량 RMSE 약11.94% 낮다. Instacart seed42는 B·Full·MLP 수량 RMSE가 근접하며 B가 가장 낮다. 추가 seed와 구조가 끝나기 전의 반례 후보로 보존한다.',
       '- 현재 지지되는 범위는 “Taxi에서 이력 정보를 추가하는 것이 B 대비 수량 예측에 도움이 된다”이다. 수준·변화를 나누는 Full 구조가 단순 MLP보다 꼭 필요하다는 핵심 주장은 아직 충분히 입증되지 않았다. 학습 조건이나 범위를 중간 결과에 맞춰 변경하지 않는다.', '',
       '## 근거 확인 — 완료', '',
       '- 로컬에 회수된 24조건·48 endpoint 기록에 대해 완료 상태, 최초 strict RMSE 선택, 실제 조기 종료, steps/학습 표본수, validation 표본수, 초기화 해시, replay와 history/summary의 지표 및 선택 상태 해시 일치를 확인했다.',
       '- 같은 dataset/seed의 B·Full과 실제 train/validation 배치 prefix를 겹치는 전체 epoch에서 비교했다. Taxi validation 표본은 모든 조건에서 8,268개다.',
       '- 새 조건은 arm 완료 receipt와 replay가 동일하다. 데이터/분할 해시와 train/validation 모집단은 동결 계약과 일치한다.',
       '- 이는 저장된 증적의 대조다. 이번 분석에서 신규 checkpoint binary 다운로드·독립 GPU 재평가를 수행한 것은 아니다. source/Runtime/학습 설정 변경이나 재시작은 없다.',
       '- 원본 경로·SHA256, 24조건 원값과 prefix 확인은 [comparison.json](comparison.json)에 보존했다. 해당 분석은 [analyze_interim_0820.py](../analyze_interim_0820.py)로 재생성한다.', '',
       '## 남은 작업', '', '**나머지 승인 학습과 replay — 진행 중**', '- 대상: 5080·5090. 신규 36조건 중 14조건 완료, 2조건 진행, 20조건 미시작. 현재 계약을 유지한다.', '', '**전체 결과 취합과 주장 판단 — 다음 작업**', '- 대상: paper_research. 3개 데이터셋 × 3개 seed × 6모델의 결과·비용·반례를 모두 모아 최종 감사하고, 공통 head의 validation 비교라는 범위에서 주장을 판단한다.', '']
(OUT / 'report.md').write_text('\n'.join(md))
print(json.dumps({'report':str(OUT/'report.md'), 'checks_passed':True,'conditions':len(rows),'taxi':aggregate},ensure_ascii=False,indent=2))
