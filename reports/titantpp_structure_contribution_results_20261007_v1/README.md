# 이력 보정의 수량 이득은 재현됐고, prototype과 시간 이득은 조건에 따라 달랐다

기존 **MLP8 구조 대조 54조건**의 학습·세 소유 root 종료·선택/마지막 checkpoint 원본 회수와 SHA 검증이 완료됐다. 이번 보고서는 그 최종 **Validation** 결과로 논문의 기여 범위를 정리한다. 진행 중인 **CNN+GRU54 검색부 파일럿**은 별도 후보로 검토한다.

수량 성능에서 가장 일관된 근거는 **현재 상태와 직전 관측 상태를 명시적으로 결합하는 이력 보정 구조**다. 보정 추가 비교는 전체 RMSE에서 18/18쌍 개선됐다. 다만 파라미터 증가도 포함한 효과이며, 시간 성능·큰 수량·prototype의 추가 이득까지 모두 개선됐다는 뜻은 아니다.

핵심 판단은 다음과 같다.

- **이력 보정을 중심으로 기여를 좁힐 근거가 있다.** 전체 수량 RMSE 평균 감소는 Taxi 약13~14%, Intermittent 약3~9%, RAF 약0.8%다. 동일 예산의 현재 상태 보정과 비교한 결합 모델도 Taxi·RAF 모든 seed에서 개선했다.
- **prototype의 보편적 필요성은 확인되지 않았다.** 이력 보정에 prototype을 추가하면 Taxi·Intermittent 평균 전체 RMSE가 각각 0.46%, 2.92% 악화했다. RAF에서는 0.12% 개선했다.
- **시간 손해를 필수 결과로 유지해야 한다.** Taxi는 보정 추가 후 전체 TimeNLL이 평균53~70% 악화했고 각 비교의 모든 seed에서 악화했다. 수량·시간 공동 우월성을 주장할 수 없다.
- **검색 파일럿은 비용 측면에서 실행을 이어갈 근거가 있다.** 5090 합성 측정의 가변 검색은 비용 한도를 통과했다. 마지막 확보 관측에는 평균 검색만 진행 중이므로 새 검색법의 정확도 개선 가능성은 아직 판정할 수 없다.

[전체·큰 수량 6지표, 3seed 평균/표본SD와 54조건 선택 epoch](VALIDATION_TABLES.md) · [논문 기여 구성안](CONTRIBUTION.md) · [반올림 없는 파생 수치](ANALYSIS.json) · [원본 계보와 검증 기록](PROVENANCE.json)

## 1. 이번 54조건이 확인한 질문

같은 인코더·출력부·데이터·학습/선택 절차에서 **명시적 직전 상태 입력**과 **정적 prototype 검색**이 각각 수량 예측에 도움이 되는지, 같은 파라미터 예산의 일반 신경망으로 대체했을 때도 차이가 남는지를 확인했다. 기준 이력 보정은 Validation에서 선정한 폭8이며, CNN+GRU54의 구조 대조가 아니다.

| 구조 | 실제 개입 | 추가 개입 파라미터 |
|---|---|---:|
| P0_H0 | prototype 검색·명시적 보정 없음 | 0 |
| P1_H0 | prototype 검색 | 4,096 |
| P0_H1 | 명시적 이력 보정 | 12,288 |
| P1_H1 | prototype 검색 + 명시적 이력 보정 | 16,384 |
| PF_H1 | 동일 예산 일반 FFN + 명시적 이력 보정 | 16,384 |
| P1_HC | prototype 검색 + 동일 예산 현재 상태 보정 | 16,384 |

P0도 공통 인과적 인코더와 persistent token 총2,048파라미터를 유지한다. H0/HC에도 인코더가 요약한 과거 정보가 남으므로 각각 “메모리가 없는 모델”, “과거 정보가 없는 모델”로 표현하면 안 된다.

H1의 8개 분기는 모두 **현재 hidden state와 직전 관측 hidden state**를 입력한다. 1/2/4/8/16/32/64/128은 분기를 활성화하는 관측 수 경계이며 서로 다른 시점의 lag를 입력하는 구조가 아니다. 보정된 상태는 수량·시간 head 양쪽에 전달된다. P1은 학습된 정적 prototype64개 중 cosine 유사도 top4의 평균을 더하며 추론 중 메모리를 갱신하지 않는다. RAF에서는 길이 제한으로 마지막 분기가 비활성이다.

근거: [동결 구조 계약](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/execution_contract.json:903), [H1 실제 입력](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/source/models/TPPs/CountAwareTitanHistoryWidth.py:61).

## 2. 전체 수량 RMSE는 이력 보정이 가장 일관되게 개선했다

아래는 seed42/52/62의 **평균 ± 표본SD**다. 모두 각 조건의 최초 최소 전체 수량 RMSE Validation epoch에서 읽었다. 굵은 값은 이번 여섯 구조 중 데이터별 평균 최저이며, 사후 최종 모델 재선택을 의미하지 않는다.

| 구조 | Taxi | Intermittent | RAF |
|---|---:|---:|---:|
| P0_H0 | 88.9766 ± 0.5011 | 1.777327 ± 0.065439 | 34.2422 ± 0.1471 |
| P1_H0 | 90.5093 ± 0.5207 | 1.720767 ± 0.044778 | 34.2303 ± 0.1273 |
| P0_H1 | **77.3773 ± 1.8224** | **1.617101 ± 0.008638** | 33.9788 ± 0.0450 |
| P1_H1 | 77.7352 ± 0.6807 | 1.664286 ± 0.009488 | **33.9380 ± 0.0706** |
| PF_H1 | 79.2546 ± 1.2300 | 1.758884 ± 0.124080 | 34.1251 ± 0.0744 |
| P1_HC | 88.8457 ± 1.2669 | 1.728073 ± 0.060651 | 34.2733 ± 0.3269 |

| 데이터 | P0에서 이력 보정 추가 | 개선 seed | P1에서 이력 보정 추가 | 개선 seed |
|---|---:|---:|---:|---:|
| Taxi | 전체 RMSE 13.0364% 감소 | 3/3 | 14.1136% 감소 | 3/3 |
| Intermittent | 9.0150% 감소 | 3/3 | 3.2823% 감소 | 3/3 |
| RAF | 0.7693% 감소 | 3/3 | 0.8540% 감소 | 3/3 |

여기서 18/18쌍은 `P0_H1 vs P0_H0` 및 `P1_H1 vs P1_H0`의 세 데이터×3seed다. **추가 보정 모듈과 파라미터를 함께 넣은 효과**다. 순수한 과거 정보만의 효과를 더 좁혀 보는 근거는 아래 HC 동일 예산 대조다.

원본: [최종 Validation 요약](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/Validation_summary.json:1411), [54조건 원본 CSV](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/Validation_per_seed.csv).

## 3. 사전 동일 예산 기준은 통과했지만, prototype의 보편적 우월성은 지지하지 않는다

고정 기준은 **P1_H1이 PF_H1과 P1_HC 모두보다, 같은 최소2개 데이터의 모든3seed에서 전체 RMSE를 개선하는가**였다. Taxi와 RAF가 통과했고 Intermittent는 통과하지 않았다. 시간 손해 허용치를 사후로 만들어 판정을 바꾸지 않았다.

| 데이터 | P1_H1 vs PF_H1 평균 RMSE 감소 | 개선 seed | P1_H1 vs P1_HC 평균 RMSE 감소 | 개선 seed | 두 대조의 모든 seed 개선 |
|---|---:|---:|---:|---:|---|
| Taxi | 1.9172% | 3/3 | 12.5055% | 3/3 | 충족 |
| Intermittent | 5.3783% | 2/3 | 3.6912% | 2/3 | 미충족 |
| RAF | 0.5484% | 3/3 | 0.9782% | 3/3 | 충족 |

Intermittent는 seed52에서 HC에, seed62에서 PF에 졌다. 같은 예산의 두 대조를 동시에 이긴 seed는42뿐이다. PF 대조는 같은 hidden context/예산을 사용하지만 초기 residual 분포가 다르다. HC의 현재 hidden에도 인코더 이력이 남는다. 따라서 통과 결과는 **지정한 대조 설계 안의 구조 패키지 차이**다.

prototype을 **없애는 것**과 같은 예산의 **FFN으로 바꾸는 것**은 서로 다른 질문이다. P1_H1은 일반 FFN 대조보다 평균 RMSE가 좋지만, prototype 없는 P0_H1보다 항상 좋지는 않다.

| 데이터 | P1_H1 vs P0_H1 평균 전체 RMSE | P1_H1 개선 seed | 해석 |
|---|---:|---:|---|
| Taxi | 0.4625% 악화 | 1/3 | prototype 추가의 일관된 이득 없음 |
| Intermittent | 2.9179% 악화 | 0/3 | 이번 전체 RMSE에서는 prototype 없는 보정이 유리 |
| RAF | 0.1202% 개선 | 3/3 | 수치 개선이 있으나 효과 크기는 작음 |

2×2 RMSE 상호작용 `q11−q10−q01+q00`은 Taxi −1.174790±2.980753, Intermittent +0.103745±0.034334, RAF −0.028933±0.051723이다. 음수 seed는 각각2/3,0/3,2/3이다. 용량 변화가 포함된 대조이고 seed 간 방향도 같지 않으므로 **보편적 시너지**를 주장할 근거가 부족하다.

근거: [사전 성공 기준](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/execution_contract.json:1198), [최종 기준 판정](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/Validation_summary.json:3748), [같은 seed별 전체 차이](DERIVED_CONTRASTS.csv).

## 4. 큰 수량과 시간 결과를 함께 보면 결론이 더 좁아진다

| 데이터·비교 | 수량 선택 epoch의 평균 전체 TimeNLL | 시간 손해 |
|---|---|---|
| Taxi: P0_H0 → P0_H1 | 0.817976 → 1.252906 | +53.17%, 3/3seed 악화 |
| Taxi: P1_H0 → P1_H1 | 0.724678 → 1.229372 | +69.64%, 3/3seed 악화 |
| Intermittent: P0_H0 → P0_H1 | 0.365886 → 0.384389 | +5.06%, 2/3seed 악화 |
| RAF: P0_H0 → P0_H1 | 3.592166 → 3.553547 | 평균 개선, 큰 수량 TimeNLL은 3/3seed 악화 |

Taxi는 수량 RMSE 개선과 시간 손해가 함께 나타났다. 이는 수량 RMSE로 선택한 상태에서의 관측 결과이며, 시간 악화가 불가피하거나 출력부 계산 오류 때문이라는 근본 원인 결론은 아니다.

큰 수량에서도 반대 근거가 남는다. Taxi의 P1_H1 평균 큰 수량 RMSE는376.6596, PF_H1은366.7086으로 P1_H1이2.7136% 나쁘다. RAF에서 H1을 추가하면 큰 수량 MAE는 P0/P1에서 각각1.269%/0.938% 악화하고, 큰 수량 TimeNLL도 각각3.645%/3.851% 악화한다. 전체 수량 RMSE 개선을 큰 수량 전반의 우월성으로 옮길 수 없다.

Intermittent P1_HC의 큰 수량 TimeNLL은 seed42/52/62 각각 약0.031906/29.688931/0.002160이다. 평균9.907666은 seed52의 불리한 결과에 크게 영향을 받는다. 그 결과를 제외하지 않았고, 평균만으로 안정성을 주장하지 않는다. Taxi 큰 수량 TimeNLL은 대부분 near-zero이므로 백분율 비교 대신 원래 값과 절대 차이를 보존했다.

전체·큰 수량 RMSE/MAE/TimeNLL **모두**의 평균·표본SD 및 같은 epoch 차이는 [상세 결과표](VALIDATION_TABLES.md)에 있다. 큰 수량 Validation 행은 Taxi79, Intermittent1141, RAF50개이며 seed 수로 행 수를 늘리지 않았다.

## 5. 논문 기여는 ‘명시적 직전 상태 보정의 효과와 적용 범위’가 현재 가장 적합하다

현재 결과가 지지하는 중심 문장은 다음과 같다.

> 관측 이력의 현재 상태와 직전 상태를 명시적으로 결합하는 보정 구조를 수량 예측에 적용하고, prototype 검색 유무와 동일 파라미터 예산의 일반 신경망 대조를 통해 수량 성능 개선 및 시간 성능 손해가 나타나는 조건을 분석한다.

논문에서는 **방법의 정확한 구조**, **같은 seed의 통제된 대조로 확인한 효과**, **데이터·지표별 적용 한계** 순서로 구성하는 것이 적합하다. 기존 대표 P1_H1의 사전 판정을 유지하면서 P0_H1의 좋은 결과를 함께 공개해야 한다. P0_H1을 이 결과를 본 뒤 곧바로 공정하게 선정된 새 대표 모델로 바꾸지는 않는다.

일반 MLP 보정 자체가 새로운 연산이라는 주장은 근거가 없다. “최초 제안”과 방법론적 참신성은 선행연구 대조가 필요하고, 다른 backbone으로의 일반화는 이54조건으로 확인되지 않았다. 이번 보고서는 외부S2P2·SOTA 우월성이나 학회 채택 가능성을 판정하지 않는다. 구체적인 문단 구성과 주장별 근거·추가 조건은 [Contribution 구성안](CONTRIBUTION.md)에 정리했다.

## 6. 검색부 파일럿은 비용 가능성이 확인됐고 정확도 가능성은 미확인이다

현재 확보된 마지막 파일럿 관측은 **2026-10-07 09:57:02 KST**다. 이 수치를 기존54조건의 최종 결과로 합산하지 않는다.

| 서버 | 실제 관측 KST | 완료/실패/진행/대기/미확정 | 현재 방식·seed | 실제 저장/best epoch |
|---|---|---|---|---|
| 5090 | 10월7일 09:57:02 | 0/0/1/4/0 | top4 평균·42 | 94/80 |
| 5080 | 자체 학습 관측 없음 | 0/0/0/5/0 | 미시작 | 첫 새 checkpoint 없음 |

당시 5090 평균 검색 E80의 전체 Validation RMSE/MAE/TimeNLL은84.878426/26.735983/1.205846이다. 이는 진행 중인 **기준선**이며 adaptive의 성과가 아니다. 해당 fit 최근10epoch 중앙값49.587초 기준, best 미갱신·속도 유지 시 E120까지 약21분29초, E300까지 약2시간50분이었다. 관측시각 기준 조건부 추정이며 전체 큐·회수 ETA가 아니다.

5080은 마지막 관측에서 기존54 원본 회수를 기다렸다. 이번10:08:43 KST의 회수 완료로 그 선행 증거는 확보됐지만, 파일럿 자체5080 qualification·실행은 별도 허가/영수증으로 확인해야 한다. 이 보고서 작성 과정에서 새 qualification이나 학습을 시작하지 않았다.

### 계산비용 판단

5090의 합성 native 측정(batch128, warmup5·실측20, 표본 중앙값)에서 모든5방식은 사전 비용 한도를 통과했다. adaptive의 평균 검색 대비 증가율은 다음과 같다.

| 입력 길이 | 추론 시간 | 학습 step 시간 | Peak allocation | 사전 비용 적격 |
|---|---:|---:|---:|---|
| T64 | +8.43% | +4.82% | +3.16% | 충족 |
| T256 | +2.43% | +1.74% | +2.59% | 충족 |

한도는 각각+20%/+30%/+20%이며 두 길이 모두 충족해야 한다. 합성 프로파일로 전체 학습·실서비스 지연·다른 GPU 비용을 확정할 수는 없다. 그러나 “검색 기준 조절이 계산비용 때문에 곧바로 부적격일 것”이라는 우려는 이 측정 범위에서 완화됐다.

### 방법론 판단

다섯 비교는 평균→유사도 가중합→시간·수량 점수 분리→이력 가변 비중→동일 예산 일반 검색의 효과를 구분하는 데 적합하다. 검색 대상은 **학습된 정적 prototype64개**다. 실제 과거 사건을 bank에 저장하는 모델로 설명하면 안 된다. 시간·수량 점수는 과거32개 관측 사건의 요약으로 query를 만든다는 의미이며, 미래 target을 사용하는 것이 아니다.

고정 비중512개와 가변 비중521개의 차이에는 gate9개 추가 효과도 포함한다. 가변 검색 대 일반 검색은 둘 다521개지만 query/점수/temperature의 설계 묶음 비교다. MLP8의 이력 보정 결과를 CNN+GRU54 adaptive의 기여 증거로 바로 옮길 수 없다.

사전 판정은 **5090 adaptive가 mean·fixed·generic 각각보다 전체·큰 수량 RMSE 모두 strict 개선하고 비용 한도를 충족하는가**다. weighted 대비 변화와 같은 수량 선택 epoch의 MAE·TimeNLL도 필수 보고한다. mean만 이기거나 fixed/generic에 지면 이력 가변 검색의 추가 효과는 지지되지 않는다.

따라서 현재 판단은 **“기술·비용 측면에서 계속 검증할 가치가 있지만, 예측 개선 가능성이 높다고 말할 결과는 아직 없다”**이다. 통과하더라도 Taxi·seed42 Validation 탐색 근거다. 두 GPU는 같은 seed 반복이며 두 seed나 독립 평가가 아니다. 확대학습은 자동 실행 범위가 아니다.

근거: [09:57 파일럿 관측](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contextual_retrieval_pilot_dual_20261007_v1/hourly/20261007T005659314481Z_1791334623041507000.md), [native 비용 원본과 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contextual_retrieval_pilot_dual_20261007_v1/NATIVE_QUALIFICATION_5090.md:20), [파일럿 고정 판정 계약](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/execution_contract.json:746).

## 7. 결과를 읽는 기준과 완료 증거

54조건은 fresh fit이며 기존 CNNGRU나 시간 head 원본을 재사용한 조건이 아니다. AdamW LR0.001/weight decay0.01, batch128/clip1, min40/max300/patience40 및 strict 최초 finite 최소 full Validation raw 수량 RMSE 선택을 유지했다. TimeNLL을 잘하는 다른 epoch로 지표를 섞지 않았다. 시간 단위는 Taxi hour, Intermittent week, RAF month이므로 데이터 사이 TimeNLL 값을 같은 척도의 순위로 평균하지 않는다.

| 소유 root | 고유 조건 | 실제 terminal 관측 KST | 소유 프로세스 종료·원본 SHA |
|---|---:|---|---|
| 원래5080: Taxi·RAF | 36 | 10월6일 23:45:19 | 검증 완료 |
| 원래5090: Intermittent42/52 | 12 | 10월7일 07:00:41 | 승인된 예약 경계 종료·검증 완료 |
| 이관5080: Intermittent62 | 6 | 10월7일 10:06:56 | 검증 완료 |

5090의 첫 예약 seed62 claim에서 발생한 FileExistsError는 계약에 따른 이관 경계 종료다. 원래 failure/status를 보존하고 새 수치적 학습 실패로 세지 않았다. 학습 실패0·미확정0이며 canonical54조건을 각각 한 번 집계했다. Intermittent seed62 GPU가 바뀐 사실을 보존하며 이종 GPU 실행 시간을 구조 효율로 혼합하지 않았다.

최종 완료 영수증은 **10월7일 10:08:43 KST**에 작성됐다. 세 archive·manifest 및 선택/마지막 binary와 소형 기록의 로컬 SHA 검증이 완료됐다. 이는 binary를 다시 실행한 CPU 재추론 감사와 다르다. [완료 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/completion_receipt.json), [소유권 mapping](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/analysis/summary_identity_mapping.json).

완료된 구조54 관측 자동화 `titantpp-54`는 승인된 종료 조건을 충족해 삭제했다. 진행 중인 검색 파일럿 자동화 `titantpp-5-gpu`는 유지했다. [종료 기록](AUTOMATION_COMPLETION.json).

평가 범위는 Validation 개발 분석이다. 과거 Test 노출 계보가 있으므로 청정한 사전 등록·독립 확인이라고 표현하지 않는다. 이번 분석에서 Test·예측 행·Validation/Test 혼합 성과를 열지 않았고 선택을 바꾸지 않았다. 표본SD는 seed 간 변동이며 통계적 유의성·모집단 신뢰구간을 증명하지 않는다. 외부 동일 조건 비교, CPU binary 재추론 감사, Test 및 새 독립 평가는 별도 미완료다.

## 남은 작업 순서

**검색부 파일럿의 방식별 종료 결과를 판정한다 — 진행 중 / 외부 실행 결과 대기**
- 대상은 별도 CNN+GRU54 Taxi·seed42 파일럿이다. 기존 사전 정확도·비용 규칙으로 adaptive의 추가 효과를 판단한다.
- 5080 실행은 기존54 완료 증거 뒤에 자신의 qualification·실행 영수증을 확인하는 승인된 절차를 유지한다. 기존54조건을 다시 학습하지 않는다.

**기여의 방법론적 차별성을 문헌과 비교해 좁힌다 — 다음 작업**
- 명시적 직전 상태 보정과 관측 수별 활성화가 기존 잔차 보정·이력 입력 방식과 어떻게 다른지 확인한다.
- 숫자가 좋은 것과 새로운 방법인 것을 구분해 논문 문구를 확정한다. 현재 보고서는 전체 manuscript를 수정하지 않았다.

**대표 모델과 새 독립 평가 계약을 확정한다 — 다음 작업 / 위 판단 이후**
- 기존 P1_H1의 결과·선택을 보존하고 P0_H1의 유리한 결과도 공개한다. 새 대표 전환이나 파일럿 확대는 별도 설계/승인 범위다.
- 전체·큰 수량 수량 지표와 시간 손해를 같은 선택 절차로 보고할 외부 대조·독립 평가를 정한다.
