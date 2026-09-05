# Instacart raw 이력과 encoder 상태의 예측 정보 비교

**완료 / 로컬 — raw 이력에는 다음 수량 신호가 남아 있지만, 고정한 두 동일 용량 probe에서는 `h`가 raw보다 더 잘 예측했다. Raw 이력은 frozen Hard-LMM의 남은 오차도 설명하지 못했다.** 따라서 이번 범위에서는 Instacart의 공통 성능 저하를 encoder의 이력 정보 소실로 설명할 근거가 없다. 이는 raw 이력에 추가 정보가 전혀 없거나 Instacart 개선이 불가능하다는 뜻은 아니다.

## 질문별 결론

| 질문 | 확인 결과 | 판단 |
| --- | --- | --- |
| Raw 이력에 다음 수량 신호가 남아 있는가? | 상수 예측 대비 direct log-MSE가 47.741~48.288% 감소했다. | 예 |
| Raw 이력이 현재 `h`보다 잘 예측하는가? | Separate-key에서 raw의 direct log-MSE가 `h`보다 3.693% / 2.222% 높았다. 원본도 같은 방향이었다. | 아니오 |
| Raw 이력이 frozen 모델의 남은 잔차를 설명하는가? | Separate-key에서 상수 보정 대비 0.073% / 0.576%, `h` 대비 0.186% / 0.305% 악화했다. | 아니오 |
| Encoder 병목 근거가 확보됐는가? | 12개 사전 비교 중 통과한 cell이 없고 원본 checkpoint에서도 같은 방향이었다. | 아니오 |

첫 번째 결과는 데이터가 완전히 무신호가 아님을 보여준다. 두 번째와 세 번째 결과는 **고정한 접근성 검사 안에서는 raw에만 남아 있고 `h`에서 사라진 예측 신호를 찾지 못했다**는 뜻이다. 따라서 Instacart에서 encoder를 더 크게 만들거나 raw 이력을 다시 우회 입력하는 변경을 다음 공통 Backbone 후보로 바로 연결하지 않는다.

## 비교 범위와 계약

- 대상은 `paper_research/master`의 기준 commit `815f3b3acdfcc124ad5f02edd9fc4b75c95eb211`과 기존 original/separate-key seed42 checkpoint다.
- 이전 8,192개 information-access 진단에서 제외된 input-only train pool 65,536개 series를 사용했다. Target이나 checkpoint를 읽기 전에 `3 <= H <= 32`를 적용해 65,525개 고유 series와 fold 32,657 / 32,868개를 고정했다. `H>32` 11개는 자르지 않고 제외했다.
- `raw64`는 관측 사건을 오래된 순서부터 최신 순서로 놓고 `[log1p(delta_t), log1p(quantity)]` 32쌍을 최신 위치에 맞춘 64차원 입력이다. 첫 관측 사건의 경계 gap은 포함하고, 미래 target gap·수량과 padding 값은 포함하지 않는다.
- `h64`는 같은 observed-only 이력을 받은 encoder의 마지막 관측 사건 상태이며 memory 검색 전 값이다. 이력 길이만 담은 `H_only64`와 label 독립 난수 `sham64`도 모두 64차원이다.
- 모든 입력에 동일한 two-fold series OOF decoder를 적용했다. 하나는 ridge linear, 다른 하나는 입력 64차원과 고정 tanh 특성 128개를 결합한 ridge다. 표준화와 적합은 항상 반대 fold에서만 수행했다.
- 주 판정 target은 `log1p(다음 수량) - frozen base log prediction`이다. Direct `log1p(다음 수량)`은 raw 이력 자체의 신호를 확인하는 보조 결과로만 사용했다.
- [사전 계약](../../contracts/hard_lmm_instacart_raw_history_access_v1.md)과 input identity·fold·source hash를 결과 접근 전에 고정했다. Validation, held-out test, 5090, 모델 학습, 파라미터 갱신은 사용하지 않았다.

J-extreme 일부 행은 앞선 시간 진단에서 target error가 읽힌 적이 있다. 그러나 이번 `H<=32` 표본 필터, raw-vs-h 비교, probe 용량과 판정 기준은 그 오류를 사용하지 않고 고정했다. 이 범위는 계약 JSON에도 명시했다.

## Raw 이력 자체의 예측력

아래 값은 다음 수량 `log1p`를 직접 예측한 OOF MSE이며 낮을수록 좋다. 상대 개선은 raw를 후보로 계산했다.

| Checkpoint | Decoder | 상수 | raw64 | h64 | raw vs 상수 | raw vs h64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Separate-key | Linear | 0.485317 | 0.253620 | **0.244587** | +47.741% | **−3.693%** |
| Separate-key | Tanh128 | 0.485317 | 0.250969 | **0.245514** | +48.288% | **−2.222%** |
| 원본 | Linear | 0.485317 | 0.253620 | **0.244724** | +47.741% | **−3.635%** |
| 원본 | Tanh128 | 0.485317 | 0.250969 | **0.245396** | +48.288% | **−2.271%** |

Raw64가 상수와 이력 길이만 담은 대조군을 두 decoder에서 크게 이겼으므로 관측 이력에 수량 신호는 남아 있다. 그러나 `h64`가 두 checkpoint·두 decoder에서 모두 raw64보다 낮은 MSE를 보였다. 이 결과는 `h`가 raw 이력의 모든 정보를 보존한다는 정보이론적 증명이 아니라, 이번 probe가 읽을 수 있는 다음 수량 신호가 encoder 뒤에서 약해졌다는 근거가 없다는 뜻이다.

## Frozen 모델의 남은 오차

아래는 각 frozen base prediction의 log 잔차를 OOF로 보정한 결과다. 상대 개선이 음수면 raw64 보정이 비교 대상보다 나쁘다.

| Checkpoint | Decoder | 상수 보정 MSE | raw64 MSE | h64 MSE | raw vs 상수 | raw vs h64 | raw / h64 body MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate-key | Linear | 0.244856 | 0.245035 | **0.244580** | −0.073% | −0.186% | 3.345821 / **3.342912** |
| Separate-key | Tanh128 | 0.244856 | 0.246267 | **0.245517** | −0.576% | −0.305% | 3.357348 / **3.351710** |
| 원본 | Linear | 0.244798 | 0.245123 | **0.244705** | −0.132% | −0.171% | 3.348645 / **3.344133** |
| 원본 | Tanh128 | 0.244798 | 0.246402 | **0.245375** | −0.655% | −0.419% | 3.358163 / **3.350063** |

Separate-key linear에서 raw64가 `H_only64`를 pooled MSE 기준 0.027% 이긴 경우가 유일한 양의 값이었지만, 사전 최소 1%에 못 미쳤고 fold0 방향, 동시 bootstrap 하한, body MAE 조건을 모두 충족하지 못했다. 나머지 raw-vs-control 비교도 통과하지 못했다.

Raw-vs-`h`는 네 조합 모두 두 fold에서 음수였다. 10,000회 shared series bootstrap의 동시 하한도 separate-key linear −0.000850, separate-key Tanh128 −0.001437, 원본 linear −0.000830, 원본 Tanh128 −0.001752였다. 네 sham 검사는 모두 정상이다. 따라서 separate-key encoder 병목과 두 checkpoint 공통 encoder 병목을 모두 `false`로 판정했다.

보조적으로 separate-key의 frozen base 자체 log 잔차 MSE는 0.245634다. `h64` direct linear 재적합은 0.244587로 0.426% 낮고 body MAE는 3.344636에서 3.342100으로 0.076% 낮았다. 작은 후처리 보정 가능성은 있지만 기존 5% 성능 목표를 설명할 크기는 아니며, 이 보조 비교는 사전 병목 판정에 포함하지 않았다.

## 해석

이번 결과는 두 극단적인 설명을 모두 배제한다.

- Instacart raw 이력이 아무 정보도 갖지 않는 것은 아니다. 관측된 시간·수량 이력만으로 상수보다 다음 수량을 훨씬 잘 예측했다.
- 현재 증거로 encoder가 그 정보를 버린다고 보기도 어렵다. 같은 차원의 `h`가 raw보다 direct target을 더 잘 예측했고 raw는 이미 남은 모델 오차를 줄이지 못했다.

따라서 **Instacart의 우선 병목을 일반적인 encoder 정보 소실로 두지 않는다.** 현재 입력으로 설명되지 않는 조건, 수량 분포와 point loss의 불일치, 더 복잡한 순서 상호작용 중 무엇이 남은 오차를 만드는지는 아직 구분되지 않았다. 이 구분 없이 Taxi에서 일부 효과가 있었던 시간 bias나 더 큰 adaptive Backbone을 Instacart까지 공통 적용할 근거는 없다.

## 검증과 증적

- 관련 단위·계약 테스트 **62개를 통과**했다. Raw 최신 정렬, target·padding 차단, 짧은 이력, series fold 분리, 동일 용량, sham guard, 양 fold·bootstrap·body 조건, frozen source/checkpoint hash를 확인했다. [pytest.xml](pytest.xml).
- CPU에서 original/separate-key checkpoint를 동결 추론했고 65,525개 행을 모두 처리했다. 두 checkpoint의 state hash는 추론 전후 동일하다. [execution_manifest.json](execution_manifest.json).
- 독립 감사에서 저장된 OOF prediction으로 **40개 metric cell과 16개 10,000회 bootstrap cell**을 다시 계산하고 artifact·checkpoint hash를 확인했다. [independent_audit.json](independent_audit.json).
- 세부 수치는 [decoder_metrics.csv](decoder_metrics.csv), [raw64_comparisons.csv](raw64_comparisons.csv), 전체 분석은 [analysis.json](analysis.json), 최종 판정은 [evidence_decision.json](evidence_decision.json)에 있다. 표 내보내기는 [build_summary_tables.py](build_summary_tables.py), 독립 검산은 [audit script](../../scripts/audit_hard_lmm_instacart_raw_history.py)로 재현한다.
- 대용량 input/probe cache와 OOF prediction은 git에서 제외된 `search_artifacts/hard_lmm_instacart_raw_history_20260905/`에 보존하며 manifest가 hash를 고정한다. 기존 미추적 `scripts/`는 변경하거나 포함하지 않는다.

`analysis.json`의 `log_vs_body_direction_conflict` 필드는 raw body MAE가 더 크면 log-MSE 개선이 음수인 경우에도 `true`로 기록한다. 이는 계약 문구보다 넓은 기록 방식이다. 모든 해당 비교는 MSE·fold·bootstrap 조건만으로도 이미 실패하므로 판정은 변하지 않으며, 위 설명에서는 이 값을 독립 실패 원인으로 사용하지 않았다.

## 남은 작업 순서

**현재 기준선 · 완료 — Instacart raw 이력 접근성 판정 / `paper_research/master`, 로컬**

- Raw 이력의 직접 예측 신호와 `h`의 접근성을 분리해 확인했다. Encoder 정보 소실 가설은 이번 범위에서 지지되지 않으므로 raw 우회 입력이나 encoder 확장 후보는 보류한다.

**다음 작업 — Instacart 수량 분포·목적함수 병목의 단일 가설 확정 / `paper_research/master`, 로컬**

- 현재 입력과 `h`가 다음 수량의 중심은 읽지만 frozen 잔차를 거의 설명하지 못한다는 결과를 기준으로, 조건부 수량 분포의 비대칭·분산을 point log-MSE가 충분히 다루지 못한다는 가설을 먼저 검토한다.
- 새 결과를 보기 전에 대상 train series, frozen `h`, 동일 fold, 후보 head·loss 하나, 상수·동일 용량 대조군과 body MAE·RMSE·tail MAE·proper score 판정 기준을 고정한다. 이 가설이 지지되지 않을 때만 달력·series 정적 정보처럼 예측 시점에 알 수 있는 추가 입력을 별도 가설로 검토한다.
- 완료 조건은 Backbone을 바꾸기 전에 Instacart의 다음 후보가 수량 head/loss인지 추가 입력인지 구분할 수 있는 근거를 얻는 것이다.

**병렬 가능 — Taxi의 separate-key 개선 보존 경로 검토 / `paper_research/master`, 로컬**

- Taxi에서는 elapsed-age 후보가 원본 대비 개선됐지만 separate-key의 성능을 보존하지 못했다. Instacart 진단과 파일·데이터·판정 계약이 독립적이므로, separate-key retrieval과 시간 bias가 서로 상쇄된 위치를 별도로 확인할 수 있다.
- 두 데이터셋의 병목이 같은지 가정하지 않고 각각 단일 후보를 확정한 뒤에만 dataset-adaptive Backbone의 공통 routing 조건을 설계한다.

**조건부 승인 필요 — 확정 후보 구현·검증 / `paper_research/master`, 로컬 및 필요 시 5090**

- 위 두 로컬 진단 중 하나가 사전 기준을 통과할 때만 해당 후보를 별도 모델 경로로 구현하고 단위·계약 테스트를 수행한다. 이후 CUDA/e1과 seed42 e300의 구체적인 실행 계약을 제시한다.
- 이번 진단만으로 새 Backbone 구현, 5090 실행, validation 또는 held-out test를 시작하지 않는다.

위 두 로컬 진단은 서로 독립이어서 병렬 진행할 수 있다. 각 데이터셋의 후보가 정해진 뒤 adaptive routing과 실제 성능 검증은 두 결과에 의존하므로 직렬로 진행한다.
