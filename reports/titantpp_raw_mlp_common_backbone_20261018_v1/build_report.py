"""Render the completed descriptive analysis; no model/data inference."""
from pathlib import Path
import csv
import json
import hashlib

OUT = Path(__file__).resolve().parent
DS = ['yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000']
NAMES = dict(zip(DS, ['Taxi', 'RAF', 'Intermittent']))
ARMS = [('titantpp_history_mlp', 'MLP4'), ('titantpp_history_mlp_width16', 'MLP16'),
        ('titantpp_all_available_history_mlp', 'AllAvailable'),
        ('titantpp_current_only_param_matched', 'CurrentOnly')]


def rows(name):
    return list(csv.DictReader((OUT / name).open()))


def table(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, v)) + ' |' for v in values])


def pick(data, **kw):
    found = [r for r in data if all(r[k] == str(v) for k, v in kw.items())]
    assert len(found) == 1, (kw, len(found))
    return found[0]


def f(value):
    return f'{float(value):.6f}'


def main():
    endpoints = rows('ENDPOINT_SUMMARY.csv')
    metrics = rows('REFERENCE_AND_REGIME_METRICS.csv')
    matches = rows('STANDARDIZED_REGIME_COMPARISONS.csv')
    pairs = rows('PAIRED_MODEL_DELTAS.csv')
    text = ['# 원본 MLP 네 구조·세 데이터셋의 공통 오류와 다음 가설',
            '\n작성: 2026-10-18. **저장 자료 분석 완료 / 다음 모델 효과 미검증 / 새 학습 보류**.',
            '\n**판단 — 완료**\n\n세 데이터셋을 함께 설명하는 단일 백본 결함은 확인하지 못했다. '
            '단순 폭 확대와 분기 가용성 확대의 효과는 데이터셋·seed·지표에 따라 갈린다. '
            '낮은 정답 수량에서 관측 하락 조건의 과대예측은 공통으로 보였지만, '
            '개체·정답 수량·활성 분기 수를 맞춘 공통 범위에서는 Taxi와 RAF의 오류 차이가 반전했다. '
            '따라서 이 현상을 현재 관측 정보 소실의 증거로 삼아 새 학습에 진입하지 않는다. '
            '기능 중복을 찾지 못한 가설 하나는 **현재 관측 임베딩의 후기 재주입**이며, '
            '이번 결과에서는 설계 후보로만 보존한다.',
            '\n**대조군의 실제 의미 — 확인 완료**\n\n이 네 모델은 독립적인 단순 MLP가 아니라 '
            '두 Titan attention block 사이에 MLP 보정을 넣은 구조다. MLP4/16의8개 분기는 '
            '모두 첫 block의 현재·직전 관측 hidden 쌍을 읽는다. 관측 개수가1·2·4·8·16·32·64·128을 넘을 때 '
            '각 분기를 켜며 항상8로 나눈다. AllAvailable은 같은 쌍·폭4를 유지하고 직전 관측이 있으면 '
            '8개를 모두 켠다. 더 먼 이력을 새로 읽거나 전체 이력을 평균하는 모델이 아니다. '
            'CurrentOnly는 추가 MLP에서 첫 block 이후의 현재 hidden만 쓰고 폭6으로 MLP4와6144개 '
            '추가 가중치를 맞춘다. 공통 attention의 이력 처리는 유지한다. MLP16은24576개다.',
            '\n**저장 자료의 범위 — 확인 완료**\n\n원본 uncapped 자료만 수치 계산에 사용했다. '
            '전체 Validation 집계는4구조×3데이터셋×3seed=36개다. 개별 예측은 MLP4 seed42 전체 Validation과 '
            'MLP16 세 seed 전체 Validation, MLP4·AllAvailable seed42의 가중 층화 Train/Validation 표본이다. '
            'MLP16·CurrentOnly Train 및 MLP4·대조군 seed52/62의 개별 행은 찾은 범위에 없다. '
            '같은 native 환경의 MLP4·AllAvailable·CurrentOnly 첫512개 Validation 행은 보조 분석으로만 보존했다. '
            '첫4batch 표본이므로 전체 성능이나 대표성을 주장하지 않는다. 누락 자료를 생성하거나 추론하지 않았다.\n\n'
            '입력140파일 SHA, 33개 행 cohort,18개 짝 비교를 확인했다. 전체 표본은 겹치므로 cohort 수를 '
            '독립 반복 수로 해석하지 않는다. MLP4 전체 Validation과 일부 폭16은 native CUDA이고, '
            'Train 표본과 Taxi/RAF 폭16 seed42 행은 CPU torch2.14.0이다. Runtime 차이를 통제한 실험이 아니다. '
            '폭16 endpoint9개는 이전 inventory의 source closure digest가 null이지만 endpoint·선택 epoch·state 결속은 확인됐다. '
            '기존 행 receipt에는 해당 추론 source provenance가 남아 있다.',
            '\n**원본 전체 Validation의3seed 비교 — 재계산 완료**\n\n각 칸은 RMSE 평균±표본SD / MAE 평균±표본SD(ddof1)다. 서로 다른 데이터셋의 raw 오류는 합치지 않았다.']
    vals = []
    for ds in DS:
        row = [NAMES[ds]]
        for model, _ in ARMS:
            a = pick(endpoints, dataset=ds, model=model, metric='overall_rmse')
            b = pick(endpoints, dataset=ds, model=model, metric='overall_mae')
            row.append(f"{float(a['mean']):.4f}±{float(a['sample_sd_ddof1']):.4f} / {float(b['mean']):.4f}±{float(b['sample_sd_ddof1']):.4f}")
        vals.append(row)
    text += ['\n' + table(['데이터', *[a[1] for a in ARMS]], vals),
             '\nTaxi는 폭16 평균이 낮지만 seed42는 RMSE·MAE가 모두 악화한다. RAF는 폭16 RMSE가3seed 모두 악화한다. '
             'Intermittent는 폭16의 두 평균이 악화하고, AllAvailable은 MAE 평균만 소폭 낮다. '
             'CurrentOnly의 두 평균은 세 데이터셋에서 MLP4보다 높다. 단, MLP4가 모든 seed에서 우세한 것은 아니다. '
             'RAF seed42 MAE와 Intermittent seed62 두 지표에는 예외가 있다. '
             '전체·큰 수량 RMSE/MAE·전체/큰 수량 recorded positive-integer TimeNLL의 평균/SD와 모든 seed 절대·상대 차이는 '
             '`ENDPOINT_SUMMARY.csv`, `ENDPOINT_SEED_DELTAS.csv`에 남겼다.',
             '\n**Train과 Validation에서 같은 방향인가 — 확인 완료**\n\n'
             '두 split에서 직접 비교 가능한 것은 MLP4 대 AllAvailable seed42 가중 표본이다. '
             '아래는 AllAvailable−MLP4이며 음수는 낮은 오류다.']
    vals = []
    for ds in DS:
        row = [NAMES[ds]]
        for split in ['train', 'validation']:
            r = pick(pairs, dataset=ds, pair='all_available-width4', split=split,
                     scope='weighted_stratified_sample', axis='overall', group='all', weighting='row')
            row.append(f"{float(r['delta_rmse']):+.6f} / {float(r['delta_mae']):+.6f}")
        vals.append(row)
    text += ['\n' + table(['데이터', 'Train ΔRMSE / ΔMAE', 'Validation ΔRMSE / ΔMAE'], vals),
             '\nTaxi에서는 두 split 모두 낮지만 RAF에서는 모두 높다. Intermittent는 Train RMSE만 낮고 '
             'Validation 두 지표는 높다. 따라서 분기를 더 켜는 것을 공통 개선 원리로 채택할 근거가 없다. '
             'Train은 전수가 아닌 층화 표본의 포함확률 역수 가중 집계이며 이를 실제 전수 재평가로 부르지 않는다.',
             '\n**공통으로 보였던 오류와 구성 차이 — 확인 완료**\n\n'
             '기존 고정 조건을 유지했다. 하락은 log1p(last)−log1p(history mean)<−log(2), '
             '중간은 그 차이가 ±log(2) 이내다. 낮은 정답 수량은 Train 경계의 첫 구간 '
             '(Taxi q≤7, RAF q≤2, Intermittent q≤2)이다. 미래 정답을 사용한 이 구간은 사후 진단이며 '
             '모델의 입력이나 예측 시 routing 조건으로 사용할 수 없다.\n\n'
             'MLP4 seed42 전체 Validation의 하락−중간 MAE 차이를 비교했다. '
             '공통 cell에서 두 조건의 가중 질량 중 작은 값을 사용하고 같은 cell 구성으로 재가중했다. '
             '엄격 비교는 개체×정확 정답 수량×활성 분기 수를 맞췄다. 동일 개체의 전체 분포나 '
             '변화 폭까지 완전히 통제한 것은 아니다. 일치하지 않는 행은 제외되므로 원래 전체 표본과 다른 대상이다.']
    vals, match_records = [], []
    for ds in DS:
        kw = dict(dataset=ds, model=ARMS[0][0], seed=42, scope='full_rows', focus='lowest_truth_bin',
                  comparison='drop-middle', weighting='row', predictor='model')
        coarse = pick(matches, **kw, scheme='truth_bin')
        exact = pick(matches, **kw, scheme='entity_exacttruth_branches')
        vals.append([NAMES[ds], f"{float(coarse['delta_mae']):+.6f}", f"{float(exact['delta_mae']):+.6f}",
                     f"{exact['n_other']} / {exact['n_middle']}", exact['entities_other']])
        match_records.append(dict(dataset=ds, crude_delta_mae=float(coarse['delta_mae']),
                                  standardized_delta_mae=float(exact['delta_mae']),
                                  drop_n=int(exact['n_other']), middle_n=int(exact['n_middle']), entities=int(exact['entities_other'])))
    text += ['\n' + table(['데이터', '정답 구간만 맞춘 ΔMAE', '엄격 공통범위 ΔMAE', '하락/중간 행', '공통 개체'], vals),
             '\nTaxi·RAF에서는 부호가 반전했고 Intermittent는 +0.004682로 작아졌다. '
             '개체 균등 가중의 Intermittent 차이는 −0.011684로 부호도 바뀐다. '
             '폭16 세 seed에서도 엄격 공통범위의 Taxi/RAF MAE 차이는 모두 음수다. '
             '이는 하락이 오류를 줄인다는 인과 결과도, 전체 하락 오류가 없다는 결과도 아니다. '
             '눈에 띈 집계 패턴을 세 데이터셋의 보편적 결함으로 확대할 수 없다는 반례다. '
             '최저 수량 Train의 엄격 공통범위는 RAF4/3행·3개체, Intermittent1/1행·1개체로 매우 작다. '
             '나머지 조합·no-support190행도 삭제하지 않고 CSV에 보존했다.',
             '\n**현재 관측을 더 직접 쓰면 충분한가 — 반례 확인 완료**\n\n'
             '관측 마지막 수량과 관측 평균을 그대로 예측하는 결정적 참조를 같은 저장 행에서 계산했다. '
             '학습 모델의 새 추론이 아니며 두 참조에 확률 시간 head가 없으므로 TimeNLL은 N/A다. '
             '최저 정답 수량·하락 조건에서 마지막 수량 참조는 Taxi MAE2.352448 대 MLP4 10.753174, '
             'RAF1.000000 대4.754346으로 낮다. 그러나 Intermittent는0.648148 대0.508261로 더 높다. '
             '관측 조건만으로 선택한 전체 하락 집단에서는 마지막 수량 참조의 RMSE·MAE가 세 데이터셋 모두 MLP4보다 높다. '
             '전체 Validation의 Intermittent 마지막값 참조는 MAE0.564733으로 MLP4 0.638480보다 낮지만 '
             'RMSE1.707803으로1.578487보다 높다. 현재값 의존만 늘리는 것이 두 지표의 공통 해법은 아니다.',
             '\n**공통으로 남은 사실과 남지 않은 설명 — 확인 완료**\n\n'
             'MLP4 seed42 Validation의 Train고정 큰 수량 집단(Taxi q>3449, RAF q>200, Intermittent q>187)은 '
             '각각 전체의0.955%·0.747%·1.322%이지만 SSE의26.016%·68.908%·29.851%를 차지한다. '
             '다만 큰 수량 평균 편향은 Taxi−239.524151, RAF−279.933500, Intermittent+2.546120으로 방향이 다르다. '
             '“모두 큰 수량을 과소예측한다”, “log1p가 원인이다”, “이력이 현재값을 소실시킨다”는 설명은 지지되지 않는다. '
             '소수 큰 오류와 다수 일반 오류의 절충은 공통 관찰이지만 그 자체로 어느 백본 연산을 바꿀지는 식별하지 못한다. '
             '큰 수량의 예측 불가능성을 증명한 결과도 아니다.',
             '\n**기존 시도와 구별되는 가설 하나 — 설계 후보 / 효과 미확인**\n\n'
             '가설은 **현재 관측 임베딩을 두 encoder 이후에서 별도로 재사용하면, 문맥 표현의 이점을 유지하면서 '
             '수량 수준 오류를 줄일 수 있는가**다. 소스 범위 감사에서 정확 중복을 찾지 못했다. '
             'SharedW(10/17)와 폭8+8 순차 MLP(10/16)는 이미 제안됐으므로 새 가설에서 제외했다. '
             '입력전 CrossFusion·PointwiseFusion, h1을 읽는 CurrentOnly, 후기 hidden FFN, 과거값을 읽는 '
             'SuccessorMemory와도 정보 경로가 다르다. 상세 범위와 SHA는 `DUPLICATE_AUDIT.json/.md`에 있다.\n\n'
             '기술적 후보는 기존 log1p 시간·수량의 input projection 출력(위치 임베딩 전) e0를 보존해 '
             'MLP 백본의 두 encoder 출력 h2 뒤·기존 LMM 앞에 더하는 것이다. 동일 용량·위치 대조를 둔다.\n\n'
             '```text\n고정 공통 백본: 기존 MLP4 (세 데이터셋 동일)\n'
             '후보: z = LMM(h2 + P(e0))\n대조: z = LMM(h2 + P(h2))\n'
             'P: bias 없는 64→8→64 GELU, 출력 투영0초기화, 두 arm 각각1024가중치\n'
             '기존 저장 MLP4: z = LMM(h2)\n```\n\n'
             '폭8은 향후 계약 검토를 위한 단일 제안값이며 검색·구현·학습한 값이 아니다. '
             '변경 축은 추가 용량 자체가 아니라 후기 보정이 읽는 정보의 출처다. '
             '1024개는 같은 명목 파라미터 수다. e0는두 입력에서 온 저차원 임베딩이고 h2는 문맥화된 표현이므로 '
             '함수 공간과 gradient 경로까지 같은 대조는 아니다. 그 차이가 의도한 정보 출처 비교에 포함된다. '
             '공통 input/log1p·loss·head·메모리 모듈은 유지한다. '
             '기존 encoder에도 입력 residual이 있으므로 정보 소실을 전제하지 않는다. '
             'LMM에 들어가는 상태가 바뀌면 검색 결과도 달라질 수 있으므로 메모리 효과를 통제한 주장도 하지 않는다. '
             '초기 특징을 뒤층에서 다시 읽는 발상 자체는 알려져 있다 '
             '([DenseNet 원논문](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)). '
             '이 논문의 이미지 결과는 현재 시계열에서의 효과 근거가 아니며 학술적 새 원리라고 주장하지 않는다.',
             '\n**학습 진입 여부와 반증 기준 — 보류**\n\n'
             '현재 자료는 위 가설의 비중복 설계를 설명할 수 있지만, 세 데이터셋 공통 효과를 기대할 '
             '직접 근거는 부족하다. 특히 구성 정합 후 공통 하락 오류가 유지되지 않고, 마지막값 참조에도 반례가 있다. '
             '따라서 이번 판단은 **자동 학습에 진입하지 않고 여기서 분석을 마치는 것**이다. '
             '연구 전체가 불가능하다는 판정은 아니다.\n\n'
             '향후 사용자가 이 가설의 실험을 선택한다면 같은 MLP4 기반·동일 seed·동일 예산의 두 새 arm을 비교해야 한다. '
             '후보가 기존 MLP4뿐 아니라 같은 용량의 P(h2) 대조보다 Validation RMSE·MAE3seed 평균이 '
             '세 데이터셋 모두 낮아야 정보 출처의 실용적 근거가 남는다. seed별 악화·일반/큰 수량 절충·시간 지표를 모두 남긴다. '
             '후보와 대조가 같이 좋아지면 후기 추가 용량의 효과일 수 있으며 e0 경로의 기여로 부르지 않는다. '
             'Taxi만 좋아지거나 한 지표만 좋아지면 공통 백본 채택 근거가 아니다. 고정 Validation 선택과 '
             '검증 계획을 계약으로 정한 뒤에야 실행을 검토한다. 이번에는 그 계약·모델 구현·학습을 수행하지 않았다.',
             '\n**방법과 한계 — 검토 완료**\n\n'
             '4,314개 기술통계·2,788개 표준화 비교는 모두 보고하며, 유의성 검정은0회다. '
             'seed3개 SD는 재학습 변동의 기술통계이며 독립 표본 기반 신뢰구간이 아니다. '
             '기존 결과와 이번 조건 선정은 탐색적이며 분석 계획 파일을 새 계산 전에 고정한 사실을 '
             '사전등록이나 Test 맹검으로 해석하지 않는다. 진실 수량에 조건을 건 분석은 사후 선택·구성 효과가 있으므로 '
             '인과 해석하지 않는다. 수량·개체를 맞춘 공통 범위에서도 연속적 변화 폭이나 모든 이력 특성이 같아지는 것은 아니다.\n\n'
             '현재 수치 계산은 허용목록의 Train·Validation 파일만 사용했다. 기록을 추가할 Notion 페이지를 확인할 때 '
             '기존 Cap/Test 요약이 함께 표시됐다. 이를 분석 입력이나 가설 채택 근거에 넣지 않았으며 새 Test 예측 파일은 열지 않았다. '
             '과거에도 Test/Cap 결과를 알고 있었으므로 전 과정의 Test 미노출·독립 검증을 주장하지 않는다.\n\n'
             '독립 수치 검산은 생산 분석 코드를 실행하지 않고33개 cohort의52,206개 수치를 다시 계산했다. '
             '불일치0건, 최대 절대차1.819e−12였으며18개 짝 비교149,231개 target의 신원·정답·이력·가중치도 일치했다. '
             '전체 Validation endpoint36개의 평균/표본SD144개도 기존 저장 요약과 일치했다. '
             '이 검산은 수치 일관성을 확인한 것이며 Runtime 통제·원인 식별이나 새 모델의 효과를 증명하지 않는다.\n\n'
             'ARS의 통계·방법 오류11종을 점검했다(11/11): 집계 부호 반전, 집단→개별 일반화, 표본 선택, '
             '정답 조건화, 집단 빈도, 극단 오류의 평균 회귀, 누락 자료, 다중 탐색, 분석 경로 선택, '
             '상관→인과, 역방향 인과다. 특히 부호 반전과 공통 표본 제한을 본문 결론에 반영했다. '
             '별도 p값이나 임의의 유의성 등급을 만들지 않았다.',
             '\n**남은 작업 순서 — 사용자 결정 대기**\n\n'
             '1. 완료: 원본 네 구조의 전체 Validation과 가용 Train 행을 함께 분석하고 공통 원인 가설의 반례를 확인했다.\n'
             '2. 완료: 프로젝트 내 기능 중복이 확인되지 않은 후기 관측 경로 한 개와 동일 용량 대조를 문서로 남겼다.\n'
             '3. 사용자 결정 대기: 근거가 제한된 이 설계 후보에 한정된 실험을 할지 결정한다. '
             '이번 결과만으로 학습을 권고하거나 다음 실행을 예약하지 않았다. 기존 CNN·모델·checkpoint·원본 결과는 그대로다.']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n')
    decision = dict(status='analysis_complete_training_deferred', scope='uncapped_saved_train_validation_only',
        common_backbone_failure_identified=False, naive_drop_overprediction_survives_composition_check=False,
        matched_low_quantity_comparisons=match_records,
        hypothesis=dict(name='late_reuse_of_current_observation_embedding',
                        duplicate_audit='no exact functional duplicate found in bounded inspected sources',
                        evidence='design-level only; common error mechanism not established',
                        base='MLP4 common across datasets', candidate='LMM(h2+P(e0))', control='LMM(h2+P(h2))',
                        proposed_P='bias-free64→8→64 GELU/output-zero-init;1024weights each', implemented=False),
        new_training_count=0,new_inference_count=0,remote_execution_count=0,cnn_modified=False,
        local_heldout_prediction_files_opened=0,incidental_existing_notion_cap_test_history_visible=True,
        statistical_tests=0,fallacy_review_coverage='11/11',
        recommendation='Do not auto-start training from this evidence; user decides after reading results')
    (OUT / 'DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report_sha256': hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest(), 'decision':decision['status']}))


if __name__ == '__main__':
    main()
