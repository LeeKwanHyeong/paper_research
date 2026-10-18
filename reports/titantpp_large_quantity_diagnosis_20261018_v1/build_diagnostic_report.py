"""Assemble source-bound diagnostic tables; never import models or access Test."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
W4='titantpp_history_mlp'
NAMES={'yellow_trip_hourly':'Taxi','raf_spare_parts':'RAF','intermittent_frozen_5000':'Intermittent'}
ORDER=list(NAMES)

def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def one(data,**keys):
    found=[r for r in data if all(str(r.get(k))==str(v) for k,v in keys.items())]
    assert len(found)==1,(keys,len(found))
    return found[0]

def run():
    cap=rows('CAP_VALIDATION_SUMMARY.csv');uncap=rows('UNCAPPED_QUANTITY_METRICS.csv')
    transitions=json.loads((OUT/'CAP_HISTORY_TRANSITIONS.json').read_text())['summaries']
    table=[]; rawtable=[]; ref=[]
    for ds in ORDER:
        hit=one(cap,dataset=ds,model=W4,group='target_at_cap')
        body=one(cap,dataset=ds,model=W4,group='target_below_cap')
        tr=one(transitions,dataset=ds,split='validation')
        table.append({'dataset':ds,'cap':float(hit['cap_value']),'ceiling_share_percent':100*float(hit['population_fraction_mean']),
            'ceiling_prediction_mean_3seed':float(hit['prediction_mean_mean']),
            'ceiling_bias_mean_3seed':float(hit['bias_prediction_minus_truth_mean']),
            'ceiling_rmse_mean_3seed':float(hit['qty_rmse_mean']),'ceiling_mae_mean_3seed':float(hit['qty_mae_mean']),
            'below_rmse_mean_3seed':float(body['qty_rmse_mean']),'below_mae_mean_3seed':float(body['qty_mae_mean']),
            'ceiling_sse_share_mean_percent':100*float(hit['sse_share_of_total_mean']),
            'observed_last_ceiling_mae':tr['last_reference_at_cap']['mae'],
            'next_ceiling_given_previous_ceiling_percent':tr['conditional_next_cap_percent_given_previous_cap'],
            'new_ceiling_after_below_percent':tr['cap_targets_with_previous_below_cap_percent']})
        tail=one(uncap,dataset=ds,model=W4,seed=42,split='validation',scope='full_rows',predictor='model',quantity_group='q_gt_train_q99')
        mid=one(uncap,dataset=ds,model=W4,seed=42,split='validation',scope='full_rows',predictor='model',quantity_group='train_q3_lt_q_le_train_q99')
        tailref=one(uncap,dataset=ds,model=W4,seed=42,split='validation',scope='full_rows',predictor='observed_last_reference',quantity_group='q_gt_train_q99')
        rawtable.append({'dataset':ds,'train_q99':float(tail['train_q99']),'tail_count':int(tail['n']),
          'tail_share_percent':float(tail['represented_share_percent']),'tail_sse_share_percent':float(tail['sse_share_percent']),
          'q3_exceeded_sse_share_percent':float(tail['sse_share_percent'])+float(mid['sse_share_percent']),
          'tail_bias':float(tail['bias']),'tail_rmse':float(tail['rmse']),'tail_mae':float(tail['mae']),
          'last_reference_tail_rmse':float(tailref['rmse']),'last_reference_tail_mae':float(tailref['mae'])})
        for model in {r['model'] for r in cap}:
            a=one(cap,dataset=ds,model=model,group='overall')
            ref.append({'dataset':ds,'model':model,'reference':'last_observed_quantity','model_rmse_mean_3seed':float(a['qty_rmse_mean']),
               'model_mae_mean_3seed':float(a['qty_mae_mean']),'reference_rmse':tr['last_reference_overall']['rmse'],
               'reference_mae':tr['last_reference_overall']['mae'],
               'delta_rmse':float(a['qty_rmse_mean'])-tr['last_reference_overall']['rmse'],
               'delta_mae':float(a['qty_mae_mean'])-tr['last_reference_overall']['mae']})
    for name,values in [('OFFICIAL_CAP_HEADLINES.csv',table),('PRE_CEILING_HEADLINES.csv',rawtable),('OFFICIAL_LAST_REFERENCE_COMPARISON.csv',ref)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    lines=['# 큰 수량 예측 오류: 현재 기준 데이터와 상한 적용 전 자료','',
      '**결론:** 세 데이터셋에 공통인 “큰 수량은 예측할 수 없다”는 설명은 지지되지 않는다. 현재 기준 데이터의 Taxi에서는 상한 집단의 MAE가 더 낮고, RAF에서는 지속적인 과소예측, Intermittent에서는 직전 관측을 그대로 쓰는 참조 대비 손실이 확인된다. log 수량 손실과 실제 수량 RMSE/MAE의 목표 차이는 코드로 확인됐지만 단일 원인으로 확정하지 않는다.','',
      '## 범위와 근거','',
      '- 현재 기준 데이터는 Train Q3 상한을 적용한 기존 Cap 데이터다. 이전 자료는 “상한 적용 전 데이터”라 부른다. 기존 파일·계약의 기술적 경로와 SHA는 바꾸지 않았다.',
      '- 새 수치 분석은 저장된 Train·Validation만 사용했다. 현재 기준 MLP4/16/AllAvailable/CurrentOnly의 전체 Validation36개와1,214,916개 저장예측 행을 집계했다. 같은 대상이 모델·seed별 반복되므로 독립 표본 수로 부르지 않는다.',
      '- 상한 적용 전 전수 Validation은 MLP4seed42 및 MLP16세seed다. Train은 MLP4/AllAvailable seed42 가중 층화표본이며 추정값이다. 누락 Train 행·다른 seed를 생성하지 않았다.',
      '- 새 모델 학습·추론·checkpoint 평가·원격 호출·Test 예측 열람은0이다. 실행 중인 Cap 외부27개와 기존 CNN·loss·head·모델·결과·실패 기록을 변경하지 않았다.','',
      '## 1. 현재 기준 데이터: 큰 수량 오류가 항상 더 큰가','',
      '아래는 같은 MLP4의3seed평균이다. 상한 집단은 q==TrainQ3, 나머지는 q<TrainQ3다. 정답으로 나눈 사후 진단이며 이 표만으로 편향 원인이나 예측 시 선택 규칙을 주장하지 않는다.','',
      '| 데이터 | 상한 | 상한 대상 비율 | 그 집단의 평균 예측 | 상한 MAE | 나머지 MAE | 상한 RMSE | 나머지 RMSE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {NAMES[r['dataset']]} | {r['cap']:g} | {r['ceiling_share_percent']:.2f}% | {r['ceiling_prediction_mean_3seed']:.3f} | {r['ceiling_mae_mean_3seed']:.3f} | {r['below_mae_mean_3seed']:.3f} | {r['ceiling_rmse_mean_3seed']:.3f} | {r['below_rmse_mean_3seed']:.3f} |")
    lines += ['',
      '- Taxi에서는 큰 수량 집단의 두 오류가 오히려 낮다. RAF는 실제10에서 평균7.156을 예측하며, 상한집단25.04%가 전체SSE의48.40%를 차지한다. 네구조·모든seed 예외는 별도CSV에 보존한다.',
      '- Intermittent MLP4는 상한집단 평균20.068로 소폭 과대다. seed42/52/62 편향은 +0.1692/−0.0519/+0.0872이며 공통 과소예측이 아니다. MLP16·AllAvailable·CurrentOnly의3seed평균은 과소 방향이므로 구조와seed차이를 보존한다.',
      '- 상한집단 비율은23.33/25.04/33.51%로 현재 문제를 단순히 “희귀한1%”라 설명할 수 없다. 원래 상한 이상 수량이 같은 값으로 합쳐진 영향과 이산 동률·split 분포 차이가 있다.','',
      '## 2. 큰 수량이 갑자기 생기는가: 관측 가능한 직전 값','',
      '| 데이터 | 직전이 상한일 때 다음도 상한인 비율 | 상한 정답 중 직전은 상한 미만인 비율 | 상한집단 직전값 참조 MAE |','|---|---:|---:|---:|']
    for r in table:
        lines.append(f"| {NAMES[r['dataset']]} | {r['next_ceiling_given_previous_ceiling_percent']:.2f}% | {r['new_ceiling_after_below_percent']:.2f}% | {r['observed_last_ceiling_mae']:.6f} |")
    lines += ['',
      '분모가 서로 다르다. 두 번째 열은 관측 조건이고 세 번째·네 번째는 미래 정답 상한 집단을 사후 선택한 통계다. 직전 값 하나의 불확실성은 전체 이력의 예측 불가능성을 증명하지 않는다.',
      '현재 기준 Intermittent 전체 Validation에서 직전값 참조 RMSE0.360281/MAE0.122316은 네 MLP의모든12모델·seed보다 낮다. 네구조 중 평균이 가장 낮은 AllAvailable도0.414230/0.261810이다. 학습모델의Test결과나외부27개결과로확대하지 않는다. Taxi·RAF에서는 전체RMSE가 모든12모델·seed에서 직전값보다낮으므로 세데이터공통으로직전값으로대체하는결론도아니다.',
      '이 결과는 현재 관측에 있는 지속 수준 신호를 모델이 더 잘 활용할 여지가 있음을 보여주는 비교다. Encoder·메모리·수량목표·정규화·공동시간손실 중 어디서 차이가 생겼는지는 분리하지 못했다.','',
      '## 3. 상한 적용 전: 극단1%와 상위중간 수량의 차이','',
      'MLP4 seed42 전체Validation의 고정TrainQ99초과 집단이다. 현재기준데이터와절대점수차이·개선율을계산하지않는다.','',
      '| 데이터 | TrainQ99 | Q99초과 수 / 비율 | 전체SSE 기여 | 평균편향(예측−정답) | Q3초과 전체SSE 기여 |','|---|---:|---:|---:|---:|---:|']
    for r in rawtable:
        lines.append(f"| {NAMES[r['dataset']]} | {r['train_q99']:g} | {r['tail_count']:,} / {r['tail_share_percent']:.3f}% | {r['tail_sse_share_percent']:.2f}% | {r['tail_bias']:+.3f} | {r['q3_exceeded_sse_share_percent']:.2f}% |")
    lines += ['',
      'Taxi의큰수량79개중65개(82.28%),Intermittent1,141개중1,137개(99.65%)는직전도Q99초과였다. RAF는50개중14개(28%)만그랬고33개(66%)는 q+1이직전q+1의2배를넘었다. 그러나RAF의반복큰수량14개도편향−268.05로과소예측하여갑작스런급등만의문제는아니다.',
      '큰실제수량만고르면정상적인조건부예측도과소방향으로보일수있다. 반대로Intermittenttail에는과대예측이있어세데이터공통의과소예측·입력정보소실을단정할수없다. Q3초과영역이SSE94~96%를차지하므로극단1%뿐아니라그아래상위중간수량의절대오차도중요하다.','',
      '## 4. 확인된 공통 계산 요인: log 손실과 평가 지표의 차이','',
      '활성 수량 head는 a=Linear(h), z_hat=softplus(a), p=expm1(z_hat), Lq=(z_hat−log1p(q))²다. Train표준편차로이손실을나누지않고출력p를Q3상한으로자르지도않는다. 전체Validation실수량RMSE로checkpoint를선택해도학습gradient는log공간의손실에서계산된다.',
      '수식 예시로 q9→예측4와 q99→예측49는 log손실이 모두 (ln2)²≈0.480453이지만 실수량제곱오차는25와2500이다. log손실은큰절대오차보다비율차이를중시하는성질이있다. 이것은실제backbone의모든gradient가항상작다는주장이아니다.',
      '이상적인 수량손실 단독해는 exp(E[log(1+Q)|X])−1이며 raw조건부평균과다를수있다. raw제곱오차는평균,절대오차는중앙값을목표로한다는차이는 [공식 지표 문서](https://scikit-learn.org/stable/modules/model_evaluation.html#which-scoring-function-should-i-use)와일치한다. 로그역변환관련 [저자 교재](https://otexts.com/fpp3/toolbox.html)도참조했다. 구체적인해와미분은이번코드를대입한직접수식전개이며실제과소예측원인의검증이아니다.',
      '입력log1p는단조·가역이므로범위가줄었다는이유만으로관측정보가사라졌다고볼수없다. 입력변환·수량손실·Encoder표현·공동시간손실의효과를분리한대조가필요하며이번분석은그대조학습을하지않았다.','',
      '## 판단과 남은 순서','',
      '1. **현재 실험 유지 — 진행 중.** 승인된외부27개학습·평가는그대로두고동일목표에서백본별차이를확인한다. 이번분석으로공통loss/head를바꾸거나새학습을추가하지않는다.',
      '2. **현재 오류 진단 — 완료.** RAF상한과소예측·Intermittent지속수준참조대비손실·Taxi반례를분리한다. 데이터부족/모델복잡도/log1p단일원인/예측불가능성은확정하지않는다.',
      '3. **외부 결과 후 비교 — 다음 작업.** 같은Train·Validation관측조건에서신규외부모델도같은오류가있는지확인한다. 특정backbone만의문제인지공통학습목표·관측불확실성과겹치는지판단할근거로사용하며독립인과검증으로부르지않는다.','',
      '상세수치: CAP_VALIDATION_PER_SEED.csv/SUMMARY.csv,UNCAPPED_QUANTITY_METRICS.csv/LARGE_HISTORY.csv,CAP_HISTORY_TRANSITIONS.csv,OFFICIAL_LAST_REFERENCE_COMPARISON.csv. 전체seed예외·가중표본·Runtime차이를보존했다. 기술적수식과소스라인은LOSS_HEAD_ANALYSIS.md,검산은TRANSITION_QA.json과후속SYNTHESIS_QA.json을본다.','']
    text='\n'.join(lines)
    observed=rows('CAP_OBSERVABLE_PREVIOUS_CAP_SUMMARY.csv')
    block=['### 미래 정답으로 고르지 않은 대조: 직전 관측이 상한인 경우','',
      '이번에는 예측 전에 알 수 있는 last_q==cap을 조건으로 고정했다. 같은 MLP4의3seed평균이며, 아래 결과는 target_q==cap으로 고른 앞 표와 다른 대상이다.','',
      '| 데이터 | 대상 수 | 모델 RMSE / MAE | 직전값 참조 RMSE / MAE |','|---|---:|---:|---:|']
    for ds in ORDER:
        a=one(observed,dataset=ds,model=W4,condition='last_at_cap')
        block.append(f"| {NAMES[ds]} | {int(a['count']):,} | {float(a['model_rmse_mean']):.6f} / {float(a['model_mae_mean']):.6f} | {float(a['last_reference_rmse_mean']):.6f} / {float(a['last_reference_mae_mean']):.6f} |")
    block += ['',
      'Taxi·RAF에서는 네구조×3seed의12개 모두 모델이 직전값 참조보다RMSE는 낮고MAE는 높았다. 큰 관측 이후 값이 내려오는 일부 사례에 대응하는 것이 제곱오차에 유리해도 절대오차에는 불리할 수 있다. 이 결과를 단순히 큰 수량을 못 배웠다고만 설명하기 어렵다.',
      'Intermittent에서는 직전값이상한인29,052개중99.10%가다음에도상한이었다. 여기서모델12개모두RMSE·MAE가직전값보다높다. 관측가능한지속수준조건에서도예측손실이있다는기술적근거다. 원인이특정백본연산인지공통손실·정규화·최적화인지는분리되지않았다.',
      'last_q<cap의나머지집단에서는Taxi12개모두두지표개선,RAF는11개개선이고CurrentOnlyseed52의MAE가예외다. Intermittent는12개모두RMSE개선·MAE악화다. 두조건의대상·정답·순서를전36Validation의1,214,916개반복행에서loader와정확히맞췄다.', '']
    text=text.replace('## 3. 상한 적용 전:', '\n'.join(block)+'\n## 3. 상한 적용 전:')
    text=text.replace('Taxi에서는 큰 수량 집단의 두 오류가 오히려 낮다.', 'Taxi의3seed평균에서는 큰 수량 집단의 두 오류가 낮다. 다만 seed42의RMSE는 상한6.078908 대 나머지6.022325로 반대다.')
    text += '\n검증 상태: ANALYZED — 저장 예측의 수치 재계산과 source SHA 확인이다. 모델 실험의 재현 실행이나 인과 검증을 수행한 상태가 아니다.\n'
    replacements={
        '네구조×3seed의12개':'네 구조 × 3seed의 12개',
        '네구조·모든seed':'네 구조·모든 seed',
        '현재기준데이터':'현재 기준 데이터',
        '상한집단':'상한 집단',
        '큰실제수량만고르면정상적인조건부예측도과소방향으로보일수있다.':'큰 실제 수량만 고르면 정상적인 조건부 예측도 과소 방향으로 보일 수 있다.',
        '원인은':'원인은',
        '검증 상태: ANALYZED — 저장 예측의 수치 재계산과 source SHA 확인이다. 모델 실험의 재현 실행이나 인과 검증을 수행한 상태가 아니다.':'검증 상태: ANALYZED. 저장 예측의 수치 재계산과 source SHA 확인을 수행했으며, 모델 실험 재실행이나 인과 검증을 수행한 것은 아니다.'}
    for a,b in replacements.items():text=text.replace(a,b)
    (OUT/'REPORT.md').write_text(text)

    decision={'status':'descriptive_analysis_complete','common_single_cause_established':False,
      'official_cap_large_group_always_harder':False,'unpredictability_established':False,'log_input_as_single_cause_established':False,
      'loss_metric_estimand_mismatch_confirmed':True,'intermittent_last_quantity_reference_better_than_all12_cap_models_on_validation':True,
      'current_cap_external27_unchanged':True,'new_training_count':0,'new_model_inference_count':0,'remote_calls':0,'held_out_predictions_opened':0,
      'next':'Complete already-running external27 and compare same observed-context errors, no automatic new architecture or objective change.'}
    (OUT/'DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'official':table,'pre_ceiling':rawtable,'decision':decision},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
