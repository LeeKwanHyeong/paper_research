# TitanTPP 공유형의 기여와 실험 근거

작성 기준일: 2026-10-20. 내부 연구 판단과 사람이 논문을 작성할 때 사용할 근거 원장이다. 확정된 원본 3seed 결과, 완료된 5090 효율 측정, MAC 정확도 중간 Validation을 구분한다. 본문 수치는 낮을수록 좋으며 평균과 표본표준편차를 사용한다.

## 현재 결론

현재 중심에 둘 기여는 **관측 이력의 관계 표현을 공유하는 Quantity-Aware backbone과, 그 구조 전체가 보이는 수량 정확도 및 계산 효율**이다. 동일 구조를 원본 RAF·Taxi·Intermittent에 적용했다. 외부 비교군 대비 확정 3seed Test는 Taxi·Intermittent의 평균 RMSE·MAE 개선을 지지하며, RAF는 RMSE가 외부 최저와 근접하고 MAE는 더 높다.

Titans-MAC 대비 효율은 세 데이터 모두 같은 5090에서 실측됐다. 정확도는 완료된 RAF 3seed와 Taxi 2seed의 Validation에서 공유형이 두 지표 모두 낮다. 따라서 **현재 완료된 Validation의 우세는 확인됐지만, 세 데이터 전체의 MAC 대비 Test 우세까지 기정사실로 두지는 않는다.** MAC9 종료·Validation 재현·고정 Test 비교 후 정확도 주장의 범위를 확정한다.

## 기여별로 무엇을 확인했는가

| 기여 | 근거 실험과 대상 | 어떻게 확인했는가 | 현재 방어 가능한 범위 |
|---|---|---|---|
| 관측 이력의 관계 표현을 재사용하는 공유형 backbone | M1 구현 사실, A1 원본 세 데이터의 동일 Shared15 구조 학습 | 첫 encoder block 출력의 현재·직전 관측 hidden을 공유 projection으로 변환하고 8개 출력 분기가 재사용한다. 기존 입력·head·손실·메모리를 유지한 전체 구조의 성능을 비교했다. | 실제 구조와 적용 결과. 공유 하나만이 개선 원인이라는 분리 인과 주장이나 새 메모리 원리는 주장하지 않는다. |
| 수량 예측의 실증적 경쟁력 | A1 외부6종과 공유형의 원본 3seed Validation/Test | 같은 대상·공통 입력/head/학습 예산에서 Validation RMSE로 checkpoint를 고정하고 Test RMSE·MAE를 비교했다. 평균·표본SD와 seed 예외를 함께 확인했다. | Taxi·Intermittent 평균 개선, RAF의 지표별 경쟁력. 모든 데이터·지표·seed 우세는 아니다. |
| MAC 대비 계산 시간과 학습 메모리 절감 | E1 동일5090 비용 측정36개 | 같은 batch128·Train target·공통 head·Runtime, MAC 온라인 갱신을 유지해 학습 step과 평가 batch를 시간 반복3회 측정했다. | 측정 조건의 실행 속도·학습 peak 메모리·파라미터 절감. 평가 peak 메모리는 증가했다. |
| MAC 대비 정확도와 효율의 동시 개선 가능성 | A2 진행 중 MAC9 직접 정확도 대조 + E1 완료된 비용 측정 | A2에서는 입력·head·손실·학습/선택 조건을 맞춘 MAC을 fresh 학습한다. 완료된 선택 Validation만 같은 seed로 비교했다. | 현재는 유망한 중간 근거. 전체9와 고정 Test 확인 후 확정한다. 비용 측정 자체가 정확도 개선을 입증하지는 않는다. |

### M1 공유형의 실제 구조 — 완료된 코드 사실

- 첫 encoder block을 지난 현재 hidden64와 직전 **관측** hidden64를 관계 입력으로 사용한다. backbone 전체가 두 사건만 보는 구조는 아니다.
- bias 없는 공유 입력 projection `128→15`와 exact GELU 하나를 `15→64` 출력 분기8개가 재사용한다. 출력 projection은0으로 초기화한다.
- 관측 prefix 수가 `1·2·4·8·16·32·64·128`보다 클 때 해당 분기가 가용하다. 모든 분기는 같은 현재·직전 관계를 사용하므로 서로 다른8개 lag를 읽는다고 설명하면 안 된다.
- 활성 분기 출력을 합친 뒤 항상8로 나누고 첫 block 출력에 더해 두 번째 block으로 전달한다. padding을 관측으로 세지 않고 withheld 경계를 반영한다.
- 기존 causal encoder2개 block, hidden64, attention4head, persistent memory16과 후단 static Hard-LMM prototype64/top4 평균 retrieval은 유지한다. MAC의 온라인 메모리 갱신을 새 공유형 메모리로 바꿨다는 설명과 구분한다.
- 전체 backbone의 정확도와 비용이 현재 비교의 대상이다. 공유 자체의 순수 효과만을 주장하지 않으므로 원본 용량 일치 대조를 논문 진행의 필수조건으로 추가하지 않는다.

코드 사실은 M1에 연결된 `METHOD_FACTS.md`와 해당 코드 경로에서 확인했다. 그 문서 작성 당시의 효율 미측정 항목은 오늘 완료된 E1이 후속 근거다. 과거 문서는 덮어쓰지 않는다.

## A1 외부 비교군과 확정된 원본 3seed 정확도 — 완료

비교군은 RMTPP·THP·NHP·SAHP·S2P2 matched-head·AttNHP matched-head다. 논문 핵심 비교는 이 외부6종과 현재 TitanTPP 공유형으로 구성한다. 내부 MLP16·CNN 비교를 필수 기여로 두지 않는다.

**실험 방식:** seed42·52·62, 원본 uncapped 데이터, 시간·수량 log1p 입력, 공통 수량/시간 head, 수량 logMSE+시간NLL 가중치1, 공통 학습 예산을 유지했다. 전체 fullValidation raw 수량 RMSE의 최초 strict finite 최소 epoch를 선택하고 MAE도 같은 checkpoint에서 계산했다. 고정한 모델의 Test를 평가했으며 Test로 checkpoint를 다시 고르지 않았다.

### 전체 수량 Test 결과

외부 최우수는 지표별로 정한다. Intermittent의 두 기준이 서로 다른 모델이라는 점을 표에 명시했다.

| 데이터 | 공유형 RMSE | 외부 최저 RMSE | 공유형 상대 변화 | 공유형 MAE | 외부 최저 MAE | 공유형 상대 변화 |
|---|---:|---:|---:|---:|---:|---:|
| RAF | 39.775389 ± 0.186122 | S2P2 39.772935 ± 0.160914 | +0.0062% | 10.150475 ± 0.098714 | RMTPP 9.906347 ± 0.037018 | +2.4644% |
| Taxi | 107.084863 ± 11.559813 | S2P2 115.541513 ± 6.402269 | −7.3191% | 34.323697 ± 3.020096 | S2P2 36.835852 ± 2.381474 | −6.8199% |
| Intermittent | 2.966841 ± 0.248890 | S2P2 3.215355 ± 0.274250 | −7.7290% | 1.197329 ± 0.036735 | THP 1.312281 ± 0.027045 | −8.7597% |

상대 변화는 `(공유형 평균−외부 최저 평균)/외부 최저 평균×100`이다. 양수는 악화, 음수는 개선을 뜻한다.

수량 우세는 평균 기준이다. Taxi Test RMSE seed52는 S2P2보다1.5238% 높고, Intermittent seed42는 S2P2보다0.2795% 높다. Taxi MAE는 S2P2 대비 세 seed 모두 낮고 Intermittent MAE는 THP 대비 세 seed 모두 낮다. RAF RMSE는42·52에서 높고62에서 낮으며 MAE는 RMTPP 대비 세 seed 모두 높다. 본문에는 평균±표본SD, 부록에는 모든 seed와 차이를 두는 구성이 적절하다. 일부 seed를 제외하지 않는다.

### 큰 수량의 보조 근거

Train에서 고정한 큰 수량 기준은 RAF `q>200`, Taxi `q>3449`, Intermittent `q>187`이다. Test의 같은 부분집합에서 공유형 평균 RMSE·MAE는 세 데이터 모두 외부 지표별 최우수 S2P2보다 낮다.

| 데이터 | 부분집합 target 수/seed | 공유형 RMSE / MAE | 외부 최저 S2P2 RMSE / MAE |
|---|---:|---:|---:|
| RAF | 39 | 405.165825 / 356.589070 | 406.404871 / 360.721446 |
| Taxi | 83 | 546.872240 / 431.967496 | 587.448847 / 486.680913 |
| Intermittent | 243 | 15.713591 / 14.632456 | 21.134082 / 19.016264 |

이는 전체 수량 결과에 덧붙일 보조 근거다. 특히 RAF 부분집합은39개이므로 전체 분포나 드문 급등을 해결했다고 확대하지 않는다. 큰 수량 표본의 조건이나 분할은 Test를 보고 재설정하지 않는다.

### 무엇이 공정한 비교를 뒷받침하는가

- 같은 입력·head·목적함수·예산을 둔 통제 실험은 backbone 전체를 비교하기 위한 설계다. 공통 조건을 사용했다는 사실 자체를 결함으로 취급하지 않는다. 비교 범위는 해당 구현이며 외부 원논문의 고유 head·목적함수·모든 튜닝을 능가한다는 뜻은 아니다.
- 원본9 학습과 native Validation9 재현이 종료됐고, 별도 RAF/Taxi Test6 및 Intermittent Test3 결과가 확정됐다. Intermittent 최종 epoch5/5/11과 고정 Test 모델의 canonical tensor 동일성도 확인했다.
- 현재 공유형은 확인된 Train 학습→Validation 선택→고정 Test 평가로 기술한다. 이번 설정을 Test에 맞춰 변경했다는 근거 없이 과거 다른 실험의 Test 튜닝 설명을 적용하지 않는다. 현재 해석 정정 C1을 우선한다.
- 공유형 학습·Validation은5090/cuDNN92000, 기존 공유형 Test·외부 학습은5080/cuDNN91900이다. host/Runtime을 결과와 함께 공개한다.
- Test TimeNLL이 세 데이터 모두 외부 최저는 아니다. 시간 결과를 보조 지표로 공개하되 기여의 중심은 Quantity-Aware 수량 예측으로 둔다.

## A2 Titans-MAC 직접 정확도 대조 — 진행 중

2026-10-20 20:33:20 KST 관측의 완료 MAC5개만 사용했다. RAF는42·52·62, Taxi는42·62이며 모든 기존 비교군도 같은 seed 구성으로 맞췄다. 학습 worker 실제exit0·선택 epoch·저장 Validation reference의 연결과 평균/표본SD를 독립 검산했다.

| 데이터와 seed 범위 | 공유형 Validation RMSE / MAE | MAC Validation RMSE / MAE | 현재 판정 |
|---|---:|---:|---|
| RAF42/52/62 | 33.967318 / 9.246784 | 34.293567 / 9.279310 | 완료3seed 평균에서 공유형 두 지표가 낮음 |
| Taxi42/62 | 77.171992 / 24.971024 | 86.985373 / 28.406990 | 완료2seed 평균에서 공유형 두 지표가 낮음 |
| Intermittent | 비교 미집계 | 완료 MAC 없음 | 미확정 |

현재 실험은 기존9개 fresh MAC 학습을 공통 입력/head/손실/학습 예산/Validation 선택으로 맞춘 event adapter 대조다. legacy 시간head·joint 선택 MAC 결과를 섞지 않는다. 전체9 학습 종료→native Validation9 전체 재현→선택 모델 Test9→원본 회수와 SHA 검증 뒤에 전체 직접 비교를 판정한다. 새 MAC Test는5090이며 기존 공유형 Test의5080과 Runtime 차이를 명시한다.

현재 방어할 수 있는 판단은 **완료된 Validation 범위에서 공유형이 MAC보다 평균 수량 오류가 낮다**는 것이다. “세 데이터의 MAC 대비 정확도 우월성이 이미 확정됐다”는 판단은 다음 결과를 기다린다. 이후 결과가 같은 방향이면 E1과 연결해 정확도와 효율의 개선을 함께 뒷받침할 수 있다.

## E1 동일 5090의 효율 측정 — 완료

두 구조를 같은 RTX5090·Python3.12.13·torch2.11.0+cu130·cuDNN92000·4threads·batch128에서 비교했다. 데이터마다 Train target4,736개를 고정해 양 모델/경로/반복이 공유했다. 예열5batch 뒤 측정32batch의 중앙 계산 시간을 구하고 fresh process 시간 반복3회의 평균±표본SD를 계산했다. RAF padded 길이84, Taxi·Intermittent256이다.

| 데이터 | 학습 step MAC/공유형 시간비 | 평가 batch MAC/공유형 시간비 | 학습 peak allocated 변화 | 평가 peak allocated 변화 | 전체 파라미터 Shared / MAC |
|---|---:|---:|---:|---:|---:|
| RAF | 7.862 ± 0.192배 | 6.984 ± 0.118배 | −81.18% | +9.55% | 88,451 / 111,302 |
| Taxi | 9.395 ± 0.142배 | 12.116 ± 0.220배 | −75.39% | +297.87% | 99,459 / 122,310 |
| Intermittent | 9.243 ± 0.104배 | 11.627 ± 0.115배 | −75.39% | +297.87% | 99,459 / 122,310 |

시간비가1보다 크면 공유형이 빠르다. 메모리 변화는 `(공유형/MAC−1)×100`이며 양수는 공유형이 더 큰 경우다. 파라미터 감소는 RAF20.53%, Taxi·Intermittent18.68%다.

**이 실험이 속도 차이를 뒷받침하는 방식:** 같은 GPU·batch tensor·target·공통 head와 초기 조건을 맞추고, CUDA 동기화 경계를 둬 같은 경로의 시간을 비교했다. MAC의 온라인 메모리 갱신을 끄지 않았다. 학습 측정에는 forward/head/loss/backward/clip/AdamW를, 평가 측정에는 forward/head/loss와 MAC 온라인 갱신을 포함했다. CPU→GPU 전송은 계산 시간에서 제외하고 별도로 보존했다.36개 비용 worker를 포함한41개 child와 controller의 실제exit0, 원본230개 파일 SHA를 확인했다.

**해석 범위:** fresh seed42의 짧은 scratch 비용 실험이며 시간 반복3회는 학습3seed가 아니다. 고정 scientific checkpoint의 속도, 수렴까지 총학습비용, 같은 정확도를 달성하는 비용, 순수 inference-only 지연 또는 모든 길이의 복잡도는 측정하지 않았다. 학습 메모리는 감소했지만 평가 allocated 메모리는 증가했으므로 “모든 경로에서 메모리 절감”으로 기술하면 안 된다.

## 심사 질문과 대응할 근거

| 예상 질문 | 현재 근거에 맞는 대응 | 추가 결과가 필요한 경우 |
|---|---|---|
| MAC보다 정확하고 빠른가? | 효율은 E1에서 완료. 정확도는 A2 완료 Validation에서 공유형 우세. | MAC9 고정 Test까지 확인한 뒤 전체 정확도 주장을 확정. |
| 공유가 성능 개선의 원인인가? | 현재 목적은 전체 backbone 비교이며 성능과 비용을 함께 제시한다. 공유만의 인과 기여로 설명하지 않는다. | 공유 효과만 분리해 주장하려는 경우에 한해 원본 용량 대조가 필요. 현재 필수 실험으로 추가하지 않음. |
| 기존 메모리보다 새 메모리가 좋은가? | 공유형은 기존 static prototype memory를 유지한다. 기여는 이력 관계를 공유하는 backbone 구성과 측정 결과다. | 새 메모리 원리 또는 특정 MAC 요소의 인과 효과를 주장하려면 별도 대조가 필요하나 현재 주장에 포함하지 않음. |
| 일부 데이터나 seed만 고른 것 아닌가? | 세 데이터 전체와 모든 seed를 공개하고 RAF/시간/평가 메모리의 불리한 결과도 남긴다. | A3 추가2seed 완료 후 기존3seed를 보존한5seed 표로 평균·변동·예외를 재확인. |
| 시간 예측도 개선됐는가? | 기여는 Quantity-Aware 수량 예측이다. 시간 head/가중치는 유지했고 TimeNLL을 숨기지 않는다. | 일관된 시간 우세를 현재 주장하지 않으므로 새 시간 최적화 실험을 필수로 추가하지 않음. |
| Test로 모델을 다시 선택했는가? | Validation 선택 규칙·고정 checkpoint·native 재현·Intermittent 최종 tensor 동일성이 실행 증적이다. | 근거 없는 Test 튜닝 단정을 추가하지 않음. 새로운 데이터의 일반화 실험과도 구분. |

## 남은 작업 순서

**MAC9의 직접 정확도 비교를 닫는다 — 진행 중**
- 대상은5090의 기존 MAC9다. 학습 종료, native Validation9, 고정 Test9, 원본 회수·검증을 마치고 A2의 판정을 갱신한다. 이 문서 작성 중 원격 조회·학습·평가를 새로 실행하지 않았다.

**추가2seed의 종료·회수 연결과5seed 비교를 마무리한다 — 진행 중 / 연결 복구는 승인 필요**
- 대상은5080의 공유형+외부6종 추가seed72·82, 총42개다. 마지막 실제 관측19:33:17 KST는27개 학습 완료·3개 진행·12개 대기였다. 이는 현재 상태나 현재 서버 유휴 증명이 아니다.
- 17:55:13 KST 종료 대기 SSH 장애 원문과 ACTION_REQUIRED는 보존돼 있다. 학습 실패와 구분하며 자동 재연결하지 않았다. 회수 연결을 복구한 뒤 전체126개 Train/Validation/Test worker와 최종 원본 검증을 거쳐 기존3seed를 덮어쓰지 않는5seed 비교를 만든다.

**5090 종료 후 실제 잔여 작업의 분할 여부를 결정한다 — 다음 작업 / 승인 필요**
- [분할 실행 제안](REDISTRIBUTION_PLAN.md)에 완료·유휴 확인, 미시작 작업만 배정, 중복 방지, host 차이 기록, 원래 기한 유지 조건을 정리했다. 현재 분할 건수나 job 소유권은 정하지 않았고 scheduler·claim·실행 코드는 변경하지 않았다.

**논문 근거 표와 부록의 확정 범위를 정리한다 — 다음 작업**
- A1+E1을 현재 근거로 사용하고 A2/A3 완료 후 추가한다. 본문용 평균±표본SD, 부록용 seed별 결과/예외, 공통조건/Runtime, 효율 측정 범위를 준비한다. 새로운 실험이나 알고리즘 탐색을 추가하지 않는다.

## 근거 파일

| ID | 원본 경로 | 역할 |
|---|---|---|
| M1 | `reports/titantpp_shared_paper_preparation_20261020_v1/METHOD_FACTS.md` | 구조·적용 위치·mask·초기화와 코드 연결 |
| A1 | `reports/titantpp_shared_paper_preparation_20261020_v1/RESULT_TABLES.json` 및 `.md` | 확정3seed Validation/Test·큰 수량·시간·seed별 결과 |
| A1-G | `reports/titantpp_shared_external_extra42_20261020_v1/PAPER_START_GATE.json` | 최종 선택과 고정 Test 모델의 canonical tensor 동일성 |
| A2 | `reports/titans_mac_accuracy9_20261020_v1/interim_validation/20261020T113320610203+0000/RESULT.json` 및 `.md` | 완료 MAC5개와 같은 seed의 Validation 비교 |
| A3 | `reports/titantpp_shared_external_extra42_20261020_v1/LATEST_PROGRESS.json` 및 `ACTION_REQUIRED.json` | 추가2seed 마지막 관측과 미해결 연결 장애 |
| E1 | `reports/titantpp_shared_efficiency_5090_20261020_v2/report.md`, `analysis.json`, `verification.json` | 현재 공유형과 MAC의 동일5090 효율 실측 |
| C1 | `reports/titantpp_shared_paper_interpretation_correction_20261020_v1/CORRECTION.md` | Test 사용·기존 효율·전체 backbone 비교의 현재 해석 정정 |

이 파일들과 새 문서의 SHA는 `EVIDENCE_INDEX.json` 및 `MANIFEST.json`에 남긴다. 기존 실험·실패·해석 원본은 수정하지 않았다.
