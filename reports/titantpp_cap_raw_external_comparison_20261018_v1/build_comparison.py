"""Compare compatible stored raw results; keep capped targets separate."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics as st

OUT = Path(__file__).resolve().parent
DS = ['yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000']
NAMES = dict(zip(DS, ['Taxi', 'RAF', 'Intermittent']))
LABELS = {'titantpp_history_mlp': 'MLP4', 'titantpp_history_mlp_width16': 'MLP16',
          'titantpp_all_available_history_mlp': 'AllAvailable',
          'titantpp_current_only_param_matched': 'CurrentOnly'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(name):
    return list(csv.DictReader((OUT / name).open()))


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); w.writeheader(); w.writerows(rows)


def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in data])


def label(model):
    return LABELS.get(model, model.replace('_matched_head', '').upper())


def value(r, metric):
    return f"{float(r[metric+'_mean']):.6f} ± {float(r[metric+'_sample_sd']):.6f}"


def main():
    sources = {'cap_mlp': 'CAP_ROWS.csv', 'raw_mlp': 'RAW_MLP_ROWS.csv', 'raw_external': 'EXTERNAL_ROWS.csv'}
    data = []
    for group, file in sources.items():
        rs = read_csv(file)
        for r in rs:
            if r.get('row_type', 'summary') != 'summary':
                continue
            assert int(r['seed_count']) == 3
            item = dict(group=group, task='cap' if group == 'cap_mlp' else 'raw',
                        dataset=r['dataset'], model=r['model'], model_label=label(r['model']),
                        split=r['split'], seed_count=3, count_per_seed=int(r['count_per_seed']))
            for metric in ['qty_rmse', 'qty_mae']:
                for suffix in ['_mean', '_sample_sd', '_seed42', '_seed52', '_seed62']:
                    item[metric + suffix] = float(r[metric + suffix])
                vals = [item[metric + '_seed' + str(s)] for s in [42,52,62]]
                assert math.isclose(st.mean(vals), item[metric + '_mean'], rel_tol=1e-10, abs_tol=1e-10)
                assert math.isclose(st.stdev(vals), item[metric + '_sample_sd'], rel_tol=1e-10, abs_tol=1e-10)
            data.append(item)
    assert len(data) == 84
    lookup = {(r['group'],r['dataset'],r['model'],r['split']):r for r in data}
    assert len(lookup) == len(data)
    populations = {}
    for r in data:
        key = r['dataset'], r['split']
        if key in populations:
            assert populations[key] == r['count_per_seed']
        populations[key] = r['count_per_seed']
    # Same row counts do not imply equal targets: Cap quantity digests differ.
    frontier = []
    selected = []
    for group in sources:
        for ds in DS:
            group_rows = [r for r in data if r['group'] == group and r['dataset'] == ds]
            assert len(group_rows) == (12 if group == 'raw_external' else 8)
            candidates = [r for r in group_rows if r['split'] == 'validation']
            winner = min(candidates, key=lambda r:(r['qty_rmse_mean'],r['qty_mae_mean'],r['model']))
            for split in ['validation','test']:
                row = dict(lookup[group,ds,winner['model'],split])
                row['selection_rule'] = 'minimum_3seed_mean_fullValidation_RMSE_then_MAE_then_model'
                row['prospective_independent_selection'] = False
                selected.append(row)
                split_rows = [r for r in group_rows if r['split'] == split]
                for metric in ['qty_rmse','qty_mae']:
                    best = min(split_rows, key=lambda r:(r[metric+'_mean'],r['model']))
                    frontier.append(dict(group=group,task=best['task'],dataset=ds,split=split,metric=metric,
                                         model=best['model'],model_label=best['model_label'],
                                         mean=best[metric+'_mean'],sample_sd=best[metric+'_sample_sd'],
                                         count_per_seed=best['count_per_seed'],seed_count=3,
                                         criterion='posthoc_minimum_within_fixed_model_set',
                                         test_based_architecture_selection=split=='test'))
    fidx = {(r['group'],r['dataset'],r['split'],r['metric']):r for r in frontier}
    contrasts, seed_deltas = [], []
    for ds in DS:
        for split in ['validation','test']:
            for metric in ['qty_rmse','qty_mae']:
                a, b = [fidx[g,ds,split,metric] for g in ['raw_mlp','raw_external']]
                assert a['count_per_seed'] == b['count_per_seed']
                ar, br = lookup['raw_mlp',ds,a['model'],split], lookup['raw_external',ds,b['model'],split]
                diffs = []
                for seed in [42,52,62]:
                    av, bv = ar[metric+'_seed'+str(seed)], br[metric+'_seed'+str(seed)]
                    diffs.append(av-bv)
                    seed_deltas.append(dict(dataset=ds,split=split,metric=metric,seed=seed,
                                            candidate=a['model'],reference=b['model'],candidate_value=av,
                                            reference_value=bv,difference=av-bv,
                                            relative_percent=100*(av-bv)/bv,selection='posthoc_metric_minimum'))
                d = a['mean']-b['mean']
                assert math.isclose(st.mean(diffs),d,rel_tol=1e-10,abs_tol=1e-10)
                contrasts.append(dict(dataset=ds,split=split,metric=metric,
                                      candidate=a['model'],reference=b['model'],candidate_mean=a['mean'],
                                      reference_mean=b['mean'],difference=d,relative_percent=100*d/b['mean'],
                                      delta_sample_sd=st.stdev(diffs),improved_seed_count=sum(x<0 for x in diffs),
                                      worsened_seed_count=sum(x>0 for x in diffs),selection='posthoc_metric_minimum'))
    write_csv('ALL_SUMMARIES.csv',data)
    write_csv('VALIDATION_SELECTED_MODELS.csv',selected)
    write_csv('POSTHOC_METRIC_FRONTIERS.csv',frontier)
    write_csv('RAW_MLP_VS_EXTERNAL_DIFFERENCES.csv',contrasts)
    write_csv('RAW_MLP_VS_EXTERNAL_SEED_DIFFERENCES.csv',seed_deltas)
    # Keep non-matched-head and deterministic references visible, without
    # silently changing the fixed six-model matched-head comparison set.
    parent = OUT.parents[1] / 'reports/titantpp_three_dataset_final_comparison_20261005_v1'
    parent_receipt = json.loads((parent/'comparison_receipt.json').read_text())
    assert sha(parent/'metrics_aggregated.csv') == parent_receipt['outputs']['metrics_aggregated.csv']
    with (parent/'metrics_aggregated.csv').open() as f:
        aux = [dict(r) for r in csv.DictReader(f) if r['view']=='overall' and r['model'] in
               ['deep_renewal_event_native_nb','last_observed_quantity','mean_quantity_in_same_observed_window']]
    assert len(aux) == 18
    for r in aux:
        r['comparison_role'] = ('different_native_head_research_reference_excluded_from_manuscript' if
                                r['model']=='deep_renewal_event_native_nb' else 'deterministic_quantity_only_reference')
        assert int(r['count_per_seed']) == populations[r['dataset'],r['split']]
    write_csv('AUXILIARY_RAW_REFERENCES.csv',aux)
    report = ['# Q3 Cap·원본 MLP·외부 비교군의 저장 결과 비교',
              '\n상태: **저장 결과 비교 완료 / 새 학습·추론·서버 작업0 / 기존 CNN·checkpoint·원본 불변**.',
              '\n**비교 대상과 한계 — 확인 완료**\n\n'
              'Cap과 원본은 MLP4·MLP16·AllAvailable·CurrentOnly의 동일한 네 구조로 범위를 맞췄다. '
              '아래 “원본 최저”는 이 네 구조 안의 값이며 과거 모든 CNN·GRU·폭8/12·후속 후보를 포함한 전역 최저가 아니다. '
              '외부 비교군은 기존 matched-head6개 모델이다. 각 결과는 seed42·52·62의 평균±표본SD(ddof1)다. '
              'RMSE·MAE 모두 Validation 수량RMSE로 선택한 같은 checkpoint의 값을 사용한다. '
              '모델은 재평가하거나 재선택하지 않았다. 이번에는 저장된 결과의 모델 간 최소값을 따로 표시한다.\n\n'
              'Cap의 Train Q3 상한은 Taxi65·RAF10·Intermittent20이다. 모든 split의 관측 이력과 정답에 같은 상한을 적용했고 '
              'log1p는 유지했다. 예측 출력을 사후 clipping한 결과가 아니다. '
              '**Cap과 원본은 입력과 정답이 다르므로 점수 차이·개선율·우월성은 계산하지 않는다.** '
              '외부 모델의 같은 Cap 학습·평가 결과는 확인한 고정 등록부와 보고 범위에 없어 N/A다. '
              '원본 외부 결과를 Cap 기준선으로 대체하지 않는다.',
              '\n**베스트의 의미 — 고정**\n\n'
              '첫 표는 각 조건에서 Validation 평균RMSE가 가장 낮은 한 구조를 정해 그 구조의 Validation/Test 두 지표를 함께 보여준다. '
              '이는 이미 알려진 결과의 기술적 선택이며 새로운 독립 선택·검증이 아니다. '
              '그 뒤 지표별 최저는 고정된 비교 모델들 중 해당 split 평균이 가장 낮은 값이다. '
              '특히 Test 최저는 **사후 최저**이므로 검증된 단일 채택 모델로 부르지 않는다. '
              'RMSE 최저 모델의 값과 다른 MAE 최저 모델의 값을 하나의 모델 결과로 합치지 않는다.',
              '\n**Validation 평균RMSE 기준으로 한 구조를 고른 결과 — 완료**\n']
    vals = []
    group_names={'cap_mlp':'Cap MLP','raw_mlp':'원본 MLP','raw_external':'원본 외부'}
    for ds in DS:
        for group in sources:
            a = next(r for r in selected if r['dataset']==ds and r['group']==group and r['split']=='validation')
            b = next(r for r in selected if r['dataset']==ds and r['group']==group and r['split']=='test')
            vals.append([NAMES[ds],group_names[group],a['model_label'],
                         f"{a['qty_rmse_mean']:.6f} / {a['qty_mae_mean']:.6f}",
                         f"{b['qty_rmse_mean']:.6f} / {b['qty_mae_mean']:.6f}"])
    report.append(table(['데이터','조건','Validation으로 고른 구조','Validation RMSE / MAE','같은 모델 Test RMSE / MAE'],vals))
    report += ['\n위 표의 원본 외부 행은 Validation RMSE로 고른 모델이다. Test에서 가장 낮은 외부 모델은 아래와 다를 수 있다. '
               '따라서 원본 모델의 경쟁력을 해석할 때 아래의 지표별 외부 최저도 함께 확인한다.',
               '\n**Cap의 지표별 최저 — 완료 / 외부 Cap 비교 N/A**\n']
    vals=[]
    for ds in DS:
        for split in ['validation','test']:
            r,m=[fidx['cap_mlp',ds,split,k] for k in ['qty_rmse','qty_mae']]
            vals.append([NAMES[ds],split,f"{r['model_label']}: {r['mean']:.6f} ± {r['sample_sd']:.6f}",
                         f"{m['model_label']}: {m['mean']:.6f} ± {m['sample_sd']:.6f}",'N/A'])
    report.append(table(['데이터','split','최저 RMSE 모델·값','최저 MAE 모델·값','동일 Cap 외부'],vals))
    report += ['\n**원본 MLP 최저와 원본 외부 최저 — 동일 목표에서 비교 완료**\n\n'
               '후보−외부이며 음수는 후보의 점수가 낮다는 뜻이다. 각 metric별로 모델 이름을 보존한다. '
               '특히 Test 행은 사후 비교이며 통계적 유의성이나 새 독립 일반화 증거가 아니다.']
    for metric in ['qty_rmse','qty_mae']:
        vals=[]
        for ds in DS:
            for split in ['validation','test']:
                a,b=[fidx[g,ds,split,metric] for g in ['raw_mlp','raw_external']]
                c=next(r for r in contrasts if r['dataset']==ds and r['split']==split and r['metric']==metric)
                vals.append([NAMES[ds],split,f"{a['model_label']}: {a['mean']:.6f} ± {a['sample_sd']:.6f}",
                             f"{b['model_label']}: {b['mean']:.6f} ± {b['sample_sd']:.6f}",
                             f"{c['difference']:+.6f} ({c['relative_percent']:+.3f}%)",f"{c['improved_seed_count']}/3"])
        report += ['\n' + metric.replace('qty_','').upper() + '\n',table(['데이터','split','원본4MLP 내 최저','원본 외부6 내 최저','평균 차이 (상대%)','낮은 seed 수'],vals)]
    aux_names={'deep_renewal_event_native_nb':'Deep Renewal native NB',
               'last_observed_quantity':'마지막 관측 수량',
               'mean_quantity_in_same_observed_window':'관측 이력 평균'}
    report += ['\n**다른 출력부·단순 참조의 결과 — 별도 보존**\n\n'
               '주 비교의 외부6종은 공통 출력부를 맞춘 집합이다. 부모 결과에 있는 Deep Renewal native NB는 '
               '다른 출력부의 연구 기록이며 기존 원고 제외 상태를 유지한다. 마지막값·이력평균은 결정적 수량 기준선으로 '
               '학습 seed의 SD와 TimeNLL이N/A다. 이를 외부6종의 평균·SD 집합에 섞지 않는다. '
               '다만 불리한 참조를 숨기지 않도록 아래에 그대로 보존한다.']
    vals=[]
    for ds in DS:
        for split in ['validation','test']:
            for name in aux_names:
                r=next(r for r in aux if r['dataset']==ds and r['split']==split and r['model']==name)
                vals.append([NAMES[ds],split,aux_names[name],f"{float(r['qty_rmse_mean']):.6f}",
                             f"{float(r['qty_mae_mean']):.6f}",'3seed' if int(r['seed_count']) else '결정적1회'])
    report += ['\n'+table(['데이터','split','보조 참조','RMSE','MAE','집계'],vals),
               '\n중요한 예외: Intermittent Validation에서 Deep Renewal MAE0.582465와 마지막값 MAE0.564733은 '
               '네 MLP 구조의 최저 MAE0.697524보다 낮다. RAF Test의 이력평균 RMSE39.357906도 네 MLP 구조의 최저39.464394보다 낮지만 '
               'MAE11.301169는 더 높다. 따라서 외부6종에 대한 일부 수량 우위를 모든 기준선·모든 지표의 우위로 확대하지 않는다.',
               '\n**결과 해석 — 완료**\n\n'
               'Cap에서 수량 오류 숫자가 작아진 것은 예측 대상의 상한이 바뀐 결과를 포함하므로 원본 문제의 해결로 해석하지 않는다. '
               'Cap에서도 세 데이터셋·Validation/Test·RMSE/MAE를 모두 개선한 공통 구조는 확인되지 않았다. '
               'RAF 폭16의 Test 개선과 Intermittent AllAvailable의 평균 이득은 보존하되 seed 예외와 '
               'Validation의 다른 순위를 생략하지 않는다. 원본 외부 대비 개선 여부는 위의 raw 조건 안에서만 판단한다. '
               '외부 비교군의 Cap 결과가 없으므로 Cap 모델의 외부 우월성·열등성은 판정하지 않는다.',
               '\n**표본·출처·계산 검증 — 완료**\n\n'
               'Cap36개의 Validation/Test checkpoint epoch·binary/state 동일성과24그룹144개 평균/SD를 확인했다. '
               '원본 MLP는기존3dataset 비교표와 원본checkpoint 결속을 재사용하고 평균/SD를 재계산했다. '
               '외부는 같은 원본 비교표 및 고정 등록부로 연결했다. 세 자료는 같은 dataset/split의 대상 개수가 같지만 '
               '동일 개수만으로 같은 truth를 증명하지 않는다. Cap의quantity digest는 별도이며 각 부모 receipt의 '
               'target/loader/input lineage와 누락된 digest 범위를 RAW_MLP_RECEIPT.json과 EXTERNAL_RECEIPT.json에 남겼다.\n\n'
               '원본 표는2026-10-05 고정 비교 자료를 일관되게 사용했다. 일부 후속 CPU/native Validation 재현과 '
               '원래 GPU endpoint에는 작은 수치 차이가 있어 이전2026-10-18 endpoint 분석과 자릿수 끝이 다를 수 있다. '
               '서로 다른 관측 원본을 임의로 교체하거나 기존 보고서를 수정하지 않았다. '
               '과거 환경·서버·평가 관측시점 차이를 완전히 통제한 새 실험이 아니며, Cap Intermittent42는5080/cuDNN91900, '
               '52·62는5090/cuDNN92000이다. 같은 dataset·seed의 Cap 네 모델은 같은 서버에서 비교했다.\n\n'
               '모든 평균은 세 개별 seed의 산술평균이다. 사건을 seed 간 합쳐RMSE를 계산하거나 ensemble한 값이 아니다. '
               '서로 다른 데이터셋의 raw 오류는 평균내지 않는다. p값·유의성 검정은0이며, 방법 오류11종을 '
               '검토해 선택편향·다중탐색·집단구성·범위외일반화를 한계로 남겼다. '
               '기존 Test를 참고한 탐색이며 한 구조의 평균 이득을 모든 seed·데이터셋의 우월성으로 확대하지 않는다.',
               '\n**남은 작업 — 완료 및 미확인 구분**\n\n'
               '이번 요청의 저장 결과 비교는 완료했다. 기존 CNN·source·checkpoint·결과는 변경하지 않았다. '
               '같은 Cap 조건의 외부 비교군 결과는 미확인이므로 해당 비교는N/A로 남긴다. '
               '새 학습·재추론·서버 실행·자동화 변경은 수행하지 않았다.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    result=dict(status='complete_stored_comparison_cap_external_not_available',
                compared_tasks=['capped_target','original_uncapped_target'],summary_rows=len(data),
                selected_model_rows=len(selected),posthoc_metric_frontiers=frontier,
                raw_best_vs_external_best=contrasts,cap_vs_raw_gain_computed=False,
                capped_external_result_available=False,capped_external_superiority='N/A',
                raw_scope='same four MLP arms as Cap, not every historical architecture',
                new_training=False,new_inference=False,remote_server_calls=0,cnn_modified=False,
                statistical_tests=0,statistical_method_review='11/11 checked',
                input_files={f:sha(OUT/f) for f in [*sources.values(),'CAP_RECEIPT.json','RAW_MLP_RECEIPT.json','EXTERNAL_RECEIPT.json','ANALYSIS_PLAN.json']})
    result['auxiliary_references']={'row_count':18,'source_sha256':sha(parent/'metrics_aggregated.csv'),
                                    'source_receipt_sha256':sha(parent/'comparison_receipt.json'),
                                    'different_head_and_deterministic_references_not_merged_with_external6':True}
    (OUT/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary_rows':len(data),'frontiers':len(frontier),'contrasts':len(contrasts),'seed_deltas':len(seed_deltas)}))


if __name__ == '__main__':
    main()
