# Instacart 미사용 train series의 시간·이력 비교 집단 검증

**완료 — 입력과 body 범위의 비교 가능성을 먼저 확인했고, 이후 고정된 표본에서 같은 시간 가설의 오류 연관 기준을 통과했다.** Separate-key의 시간 배치 차이(J)가 큰 집단은 작은 집단보다 body 수량 MAE가 **10.3201%** 높았다. 따라서 경과시간을 encoder에 전달하는 단일 후보를 검토할 근거가 생겼다. 특정 encoder 결함이나 새 구조의 성능 개선이 입증된 것은 아니다.

## 현재 기준선과 범위

- 대상: `paper_research/master`, 소스 기준선 `c4ee1140dcaa2b866e31064725d66c6bc00e867f`. Instacart train 내부 진단이며, 원본 Hard-LMM의 채택 상태는 유지한다.
- 이전 [시간·이력 진단](../hard_lmm_temporal_diagnostic_20260905/README.md)은 Instacart의 이력 길이 불균형 때문에 결론을 보류했다. 이번에는 그 진단의 7,848개 series를 전부 제외하고 비교 설계를 고정했다.
- 같은 가설: 사건 번호로 표현한 과거 시점과 실제 경과시간으로 표현한 과거 시점의 차이가 클수록, 입력 조건이 비슷한 이력에서도 body 수량 오류가 높은가?
- J는 정규화된 실제 과거 시점과 사건 번호상의 과거 시점 간 RMS 차이다. 첫 관측의 경계 바깥 gap을 제외하며, 기존 관측 feature 정의를 그대로 재사용했다.
- encoder·readout·K/V·top-4·temperature·head·loss를 변경하지 않았다. 원본과 separate-key의 기존 seed42 checkpoint를 동결해 CPU에서 추론했다. 5090 접속, 재학습, validation/test 평가, push는 수행하지 않았다.

## 오류 접근 전에 고정한 설계

[계약](../../contracts/hard_lmm_instacart_balanced_v1.json)의 SHA256은 `d0378dd94802fa60c1061c8976bf2763c48d8ae3515d718878a966633b9c6ce7`이다. 표본 수, 해시 선택 규칙, 구간 경계 계산법과 모든 판정 기준을 새 입력값과 오류를 보기 전에 고정했다.

1. **메타데이터로만 표본 선택.** Train 206,209개 series 중 이전 진단의 7,848개를 제외했다. 남은 198,361개 중 관측 수 H≥3과 다음 train 사건을 확보할 수 있는 139,722개가 적격이었다. 각 series에서 해시가 가장 작은 관측 구간 하나를 고른 뒤, series 해시 순서로 65,536개를 한 번만 선택했다. 원본의 native key 정렬과 전체 target index를 보존했으며, 이전 8,192행의 index·이력 길이 대응도 검증했다.
2. **선택된 관측 행만 입력으로 읽기.** 관측 행의 수량·gap만 수집했다. 선택된 미래 target 행은 읽기 대상에서 제외했다. Train/validation/test 분리 후 train 행만 사용했다.
3. **입력 gate 고정.** 서로 다른 series로 나눈 두 fold에서 반대 fold의 J 25·75백분위수를 low/high 기준으로 사용했다. 비교 cell은 `정확한 관측 수 H × 평균 log 수량 3구간 × 수량 변화 RMS 2구간 × 평균 내부 gap의 log 2구간`이다. H를 기존 2구간 대신 정확하게 맞춘 것이 matching 설계의 변경점이다. Cell별 두 집단의 가중 질량은 작은 집단의 행 수로 맞췄다.
4. **Body gate 재검증.** 입력 gate 통과와 hash 고정 이후에만 전체 후보 중 J 극단값에 해당하는 32,814개 target의 수량을 읽어 기존 body≤25 여부를 확인했다. 기존에 허용된 cell만 유지하거나 탈락시켰고, body 구성에 따라 가중치를 다시 고정했다. 유지율의 분모에는 입력 단계에서 탈락한 cell의 body 적격 행도 포함했다.
5. **오류 계산.** 두 gate가 통과한 후에만 두 checkpoint를 복원했다. 최종 28,119개의 동일한 target과 고정된 cell·가중치를 사용했다. 결과를 본 뒤 표본·구간·기준을 변경하지 않았다.

각 cell·집단은 최소 20행/5개 series, 각 fold·집단은 최소 100행/20개 series를 요구했다. Fold별 후보 유지율은 50% 이상, 네 입력 조건의 최대 절대 표준화 평균 차이(SMD)는 0.25 이하여야 한다. 모든 기준은 이전 진단과 같다.

## 비교 가능성 결과

행마다 서로 다른 series이므로 아래 행 수와 series 수는 같다. H는 같은 길이의 cell 안에서 비교했고, 가중 H 분포도 두 집단에서 같다.

| 단계 | Fold | Low / High 행 | 전체 적격 극단값 행 | 유지율 | 최대 절대 SMD | H의 절대 SMD | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| 입력 | 0 | 6,810 / 7,918 | 16,318 | 90.2562% | 0.111643 | 0 | 통과 |
| 입력 | 1 | 6,692 / 8,045 | 16,496 | 89.3368% | 0.110659 | 0 | 통과 |
| Body | 0 | 6,423 / 7,566 | 15,546 | 89.9846% | 0.107851 | 3.45e-16 | 통과 |
| Body | 1 | 6,363 / 7,767 | 15,803 | 89.4134% | 0.109676 | 3.55e-16 | 통과 |

입력 단계의 100개 cell 중 body 재검증에서 99개를 유지했다. Body 단계에서 가중 H 평균은 fold0 3.8332, fold1 3.8247이다. 나머지 입력 차이는 허용 기준 안에 있지만 완전히 제거되지는 않았다. 가장 큰 잔여 차이는 수량 변화 RMS였다.

## 고정 표본의 오류 결과

아래 값은 body 수량의 가중 MAE다. 증가율은 각 모델 안에서 `(High − Low) / Low`로 계산한 **집단 간 오류 차이**이며 모델 변경의 개선율이 아니다.

| 모델 | 범위 | Low J MAE | High J MAE | High − Low | 증가율 |
|---|---|---:|---:|---:|---:|
| Separate-key | 전체 | 3.137812 | 3.461639 | 0.323826 | **10.3201%** |
| Separate-key | Fold 0 | 3.158283 | 3.394798 | 0.236515 | 7.4887% |
| Separate-key | Fold 1 | 3.117794 | 3.527003 | 0.409209 | 13.1250% |
| 원본 Hard-LMM | 전체 | 3.143242 | 3.464695 | 0.321453 | 10.2268% |
| 원본 Hard-LMM | Fold 0 | 3.161449 | 3.399400 | 0.237951 | 7.5266% |
| 원본 Hard-LMM | Fold 1 | 3.125437 | 3.528547 | 0.403111 | 12.8977% |

Separate-key의 MAE 차이에 대한 series bootstrap은 고정 seed `20260905`, 500회 모두 유효했다. 각 fold 안에서 series를 재추출하고 cell·가중치는 유지했다. 차이의 하위 5% 지점은 **+0.260547**, 중앙값은 +0.323533이다.

기존 판정 조건인 ① 두 gate 통과, ② separate-key 전체 오류 차이 ≥5%, ③ separate-key 양 fold 양수, ④ 원본 양 fold 양수, ⑤ bootstrap 하위 5% 지점 >0을 모두 충족했다. 보조 지표인 절대 log 잔차도 separate-key의 전체 high 집단에서 9.1980% 높았다. 부호가 있는 log 잔차 차이는 전체 +0.000304로 작고 fold별 방향도 달라, 일정한 과대·과소 예측 보정을 제안하는 근거는 아니다.

## 판단과 해석 한계

**시간 표현을 바꾸는 후보의 계약을 검토할 수 있다.** 이번 고정 설계에서는 이력 길이 불균형을 해소한 뒤에도 실제 시간 배치와 오류의 연관성이 유지됐다. 원본에도 같은 관계가 있어 separate-key 검색만의 현상으로 보기 어렵다.

- 이 series들은 이전 진단에서만 미사용이다. 두 모델의 학습에서는 사용됐으며 checkpoint는 기존 validation으로 선택됐다. 새로운 데이터에 대한 일반화 성능으로 해석하지 않는다.
- Series당 구간 하나를 선택한 이번 모집단은 이전 target 중심 8,192행과 다르며, 가중 평균 이력도 약 3.83개로 짧다. 이전 4.489%와 이번 10.320%의 차이를 이력 길이 matching만의 효과라고 볼 수 없다. 긴 이력·다른 데이터셋까지 확대 해석하지 않는다.
- 관측된 네 조건의 균형을 정해진 수준으로 맞춘 결과다. 남은 입력 차이, 관측하지 않은 조건, 사건 과정 자체의 예측 난이도는 남을 수 있다. 연관성이 encoder 결함의 인과 증거나 relative-time attention의 우월성을 뜻하지 않는다.
- 결론은 body≤25와 비교 가능한 J 극단 집단에 한정된다. 제외된 cell, 중간 J, tail까지 같은 관계가 있다고 주장하지 않는다.
- Bootstrap은 고정된 cell·가중치 조건의 불확실성을 본다. 새 표본 선택과 matching 설계 전반의 불확실성까지 추정하지 않는다.
- 5% 기준은 후보 검토를 위한 오류 연관 기준이다. 새 모델의 기존 채택 조건(원본 대비 body MAE 5% 이상 개선, RMSE·tail MAE 악화 각각 2% 이하, Time NLL 증가 0.01 이하)을 대체하지 않는다.

## 검증과 재현

새 단위·계약 테스트 **20개를 통과**했다. 미래 target 값 오염 방지, native series 정렬과 원본 Dataset 대응, body에서 허용된 label만 읽기, 정확한 H matching, 전체 후보 유지율 분모, body 이후 cell 추가 금지, gate 선행 조건, frozen hash 변경 감지, 두 모델·양 fold 판정 기준을 검증했다.

기존 추출 경로와 관측 행만 입력하는 경로의 첫 64개 수량 logit 차이는 두 모델 모두 0이었다. 추론 전후 checkpoint state hash도 동일했다. 최종 실행 manifest는 179개 소스·데이터·중간 결과 hash를 검증했다. 독립 재계산 결과는 [independent_audit.json](independent_audit.json)에 기록한다.

- [입력 설계](input_design.json), [body 설계](body_design.json), [오류 분석](analysis.json)
- [메타데이터 선택 기록](selection_manifest.json), [입력 hash 고정](design_manifest.json), [입력 gate](input_gate.json), [body gate](body_gate.json)
- [최종 추론 기록](diagnostic_manifest.json), [실행 상태](execution_manifest.json), [단위·계약 테스트](pytest.xml)
- 원시 selection·관측 이력·features·assignment·labels·predictions는 `.gitignore` 대상인 `search_artifacts/hard_lmm_instacart_balanced_20260905/`에 보존하며, 위 manifest가 hash를 고정한다. 기존 미추적 `scripts/`는 변경·추가하지 않았다.

원시·결과 경로가 없는 체크아웃에서 아래 순서로 실행한다. 각 명령은 이전 단계의 통과와 hash를 확인한다. 실패한 설계의 경로를 지우고 조건을 바꿔 재시도하는 용도로 사용하지 않는다.

```bash
python3 -m pytest simple_lab_test/search/tests/test_hard_lmm_instacart_blind_inputs.py simple_lab_test/search/tests/test_hard_lmm_instacart_balanced_analysis.py simple_lab_test/search/tests/test_hard_lmm_instacart_balanced_runner.py -q
python3 paper/scripts/run_hard_lmm_instacart_balanced.py --phase design
python3 paper/scripts/run_hard_lmm_instacart_balanced.py --phase body
python3 paper/scripts/run_hard_lmm_instacart_balanced.py --phase diagnose
```

## 남은 작업 순서

**다음 작업 — Hard-LMM encoder의 경과시간 표현 후보 하나와 계약 확정 / 로컬**
- 이번에 확인된 시간 배치와 오류의 연관성을 설명하는 최소 변경 후보 하나를 정한다. 기존 readout 실험과 겹치지 않도록 encoder 내부에서 경과시간이 전달되는 경로를 대상으로 한다.
- 변경 위치, 시간 단위·범위·padding·짧은 이력 처리, 초기 출력 동일성, 유지할 모델 구성과 성능 판정 기준을 명시한다. Separate-key 대비 추가 효과와 Taxi의 기존 개선 보존 기준도 결과 확인 전에 수치로 고정한다.

**승인 필요 — 확정 후보 구현과 단위·계약 검증 / `paper_research/master`, 로컬**
- 후보가 확정되면 별도 모델 경로에서 초기 동등성, gradient, 시간·target·padding 누출 방지, finite 계산과 저장·복원을 검증한다. 완료된 변경을 기존 미추적 `scripts/`와 분리해 독립 커밋한다.

**승인 필요 — CUDA와 실제 데이터 e1 실행 검증 / 5090**
- 로컬 계약 검증 이후 GPU·Runtime·데이터 checksum과 전송 소스 무결성을 확인하고 CUDA 테스트, Taxi·Instacart e1 순으로 진행한다. 완료 조건은 정상 실행과 계약 준수이며 e1 결과로 성능 채택을 판단하지 않는다.
