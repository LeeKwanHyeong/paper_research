# Shared15 방법 사실표

## Material Passport

- 작성일: 2026-10-20.
- 목적: 사람이 논문을 작성할 때 확인할 코드 사실·평가 범위·제약의 목록.
- 참고 절차: academic-research-suite / experiment-agent의 근거와 해석 구분.
- 확인 범위: 아래 코드·기존 근거 원장·로컬 종료 gate. 이번 문서 작업에서 학습·추론·원격 실행·수치 재집계는 하지 않았다.
- 지위: 사실 정리. 투고용 방법 본문, 초록, 서론, 새로운 독창성 주장이 아니다.

## 근거 등록부 — 현재 기준선

아래 근거 ID는 이 폴더의 세 문서에서 공통으로 사용한다. 기존 파일은 변경하지 않았다.

| ID | 근거 | 사용 범위 |
|---|---|---|
| E1 | [기존 논문 근거 원장](../titantpp_shared_external_extra42_20261020_v1/PAPER_EVIDENCE_LEDGER.md) | 기존 3seed 결과·예외·해석 한계·효율 미측정 항목. 원장의 E1/E2와 혼동하지 않도록 이 문서의 ID로만 사용한다. |
| V1 | [투고 지침 확인 기록](../titantpp_shared_external_extra42_20261020_v1/VENUE_REQUIREMENTS.md) | 2026-10-20 확인한 PAKDD 2027 작성·제출 정책. 이 작업에서 외부 페이지를 다시 확인하지 않았다. |
| G1 | [최종 선택 동일성 gate](../titantpp_shared_external_extra42_20261020_v1/PAPER_START_GATE.json) | Intermittent 최종 epoch5/5/11 및 canonical tensor가 기존 고정 Test 모델과 일치함. |
| G2 | [원본 raw9 최종 학습·Validation receipt](../../search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1/retrieval/final_receipt.json) | 완료9·Validation 재현9·scientific worker exit0 18·3seed·Test 미실행인 원본 학습 작업의 종료 범위. |
| G3 | [원본 raw9 회수 receipt](../../search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1/retrieval/5090/receipt.json) | 회수 파일227개. 새 Test 실행을 뜻하지 않는다. |
| C1 | [Shared15 코드](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanSharedHistoryMLP.py:93) | 공유 보정의 projection·GELU·마스크·초기화·상속 경로. |
| C2 | [관측 인덱스와 보정 삽입 위치](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanMultiLagDetail.py:77) | 관측 수 임계값·padding 처리·withheld 경계·두 encoder block 사이 보정. |
| C3 | [기존 폭16 이력 MLP](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanHistoryWidth.py:61) | 8개 독립 128→16→64 보정 분기. 내부 구조 대조군. |
| C4 | [공통 시간·수량 입력](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTPP.py:435) | 관측 시간 간격과 수량의 log1p 특징. |
| C5 | [공통 수량 예측·손실](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTPP.py:770) | softplus 위치·expm1 복원·직접 log-MSE. |
| C6 | [목표 차단과 다음 사건 평가](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/core.py:60) | 목표 수량 차단·목표 위치 write 차단·직전 관측 hidden에서 예측. |
| C7 | [기존 encoder 구성](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTPP.py:1037) | 2개 block·4개 attention head·persistent memory16·causal 설정. |

## 최종화 상태 — 완료

| 확인 항목 | 확인된 값 | 경계 |
|---|---|---|
| G1 | `status=PASS`, `canonical_tensors_compared=true`, `paper_may_start=true` | 논문 준비 자료의 시작 조건이다. 채택 가능성·독창성·통계적 유의성을 판정한 gate가 아니다. |
| Intermittent seed42/52/62 | 최종 선택 epoch5/5/11, 각 `matches=true`, `validation_replay_passed=true` | 기존 임시 고정 Test 모델과 최종 선택 가중치의 동일성 확인. Test 재선택이나 새 Test 평가가 아니다. |
| 원본 raw9 | G2 `status=complete`, `new_fit_count=9`, `validation_replay_count=9`, `held_out_test_evaluated=false` | 학습·Validation 회수 작업과 별도 Test 작업을 구분한다. |
| 원장의 이전 미확정 표시 | E1 작성 당시에는 Intermittent 최종 선택 미확인 | 그 과거 원장은 그대로 두고 G1의 후속 확인으로 연결한다. 원장 전체의 모든 문장이 갱신됐다고 표현하지 않는다. |
| 추가 seed72·82 | 완결된 5seed 집계는 이 자료에 포함하지 않음 | 기존 3seed를 임의로 5seed 결과로 바꾸지 않는다. |

G1 SHA-256: `e107aa615b23603859e519d93d482ce92b8741d18b8deb064ea02318cbfd80ca`.
G2 SHA-256: `5f0deda74a3e606cd7a5819f3a15c55cf8c24105f5708abcf1dfc0f3ae7412d1`.
G3 SHA-256: `40306294085be88fdfd538177647120ee53aea549f60f3b0e2978b82a250e535`.

## 보정 구조 — 코드에서 확인

| 항목 | 확인된 구현 | 근거 | 확대하면 안 되는 해석 |
|---|---|---|---|
| 적용 위치 | 기존 두 encoder block 중 첫 block 출력에 보정을 더하고 두 번째 block으로 전달한다. | C1의 상속·`multilag_detail` 대체, C2 `_encode_base` | 전체 encoder·attention·출력 head를 새 구조로 교체한 것은 아니다. |
| 보정 입력 | 현재 64차원 hidden과 바로 전 **관측**의 64차원 hidden. 합쳐서128차원 관계 입력에 해당한다. | C1 `forward`의 `current`, `previous`, gather | 원시 입력의 두 사건만 사용하는 전체 모델로 설명하지 않는다. 보정 입력은 첫 encoder block을 지난 상태다. |
| 공유 projection | bias 없는 128→15 projection 하나. 구현은 가중치를 현재/직전 부분으로 나눠 각각 선형 계산 후 합산한다. | C1 `input_projection`, `F.linear` | 8개 분기에 서로 다른 입력 projection을 두는 구조가 아니다. |
| 비선형 | 합산한15차원 관계에 exact GELU(`approximate="none"`)를 한 번 적용한다. | C1 `forward` | 학습 가능한 별도 gate·정규화·RevIN을 추가하지 않는다. |
| 출력 분기 | bias 없는15→64 projection 8개가 같은 관계 기저를 재사용한다. | C1 `output_projections` | 분기마다 서로 다른 과거 사건 내용을 읽는다고 설명하지 않는다. |
| 분기 가용 조건 | 관측 prefix count가1·2·4·8·16·32·64·128보다 **클 때** 해당 분기를 활성화한다. | C1 `mode="local"`, C2 `available` | 임계값1은 관측1개에서 활성화된다는 뜻이 아니다. 서로 다른 실제 lag 조회도 아니다. |
| 분모 | 분기 출력을 합친 뒤 활성 분기 수와 무관하게 항상8로 나눈다. | C1 반환식 | 활성 분기 수에 따른 평균 또는 정규화라고 부르지 않는다. |
| 마스크 | padding은 관측 수를 늘리지 않는다. withheld valid 위치는 가용성의 기준을 초기화한다. 목표 위치는 memory write에서 제외된다. | C2 `observed`, `reset_count`, `local_count`; C6 | 결측·목표 위치를 일반 관측처럼 세지 않는다. |
| 초기화 | 입력 projection은 기본 Linear 초기화, 출력 projection은0 초기화. `fork_rng(devices=[])` 안에서 생성한다. | C1 `__init__` | 기존 checkpoint를 가져온 warm start라고 설명하지 않는다. |
| retained 구조 | 기존 causal encoder2개 block, hidden64, attention head4개, persistent memory16과 후단 static Hard-LMM prototype64/top4 산술평균 retrieval을 유지한다. | C2 `encode_task_states`, C7, E1 §4 | 새 prototype 학습 원리·새 head·온라인 기억 갱신을 기여로 추가하지 않는다. |

수식에 대응하는 코드 사실: `z_i = GELU(W_c h_i + W_p h_prev)`, `Δh_i = (1/8) Σ_b a_ib V_b z_i`.
`W=[W_c,W_p]`의 모양은15×128, 각 `V_b`는64×15다. `a_ib`는 위의 관측 prefix·마스크 가용 조건이며 실제 시간 간격 감쇠계수가 아니다.

## 비교에서 유지한 조건 — 기존 기록

| 항목 | 고정 조건 | 근거 |
|---|---|---|
| 자료 | 상한 미적용 원본 RAF·Taxi·Intermittent. Train·Validation·Test 시간 순서 분할. Cap 결과와 합치지 않음. | E1, G2 |
| 입력 | 시간 간격과 수량 모두 관측값을 log1p 변환. 목표 수량은 차단. | C4, C6 |
| 수량 head | `u=softplus(linear(h))`, `q_hat=expm1(u)` | C5 |
| 손실 | 수량의 직접 log-MSE와 시간 NLL의 계수1 결합. `lambda_tail=0`, 수량 손실 추가 정규화 없음. | E1 §4, C5·C6 |
| 시간 head | 관측 양의 정수 간격 likelihood를 사용하는 이분산 lognormal duration head. | E1 §4 |
| 예산·선택 | batch128, 최대300/min40/patience40, AdamW lr0.001, weight decay0.01, clip1. 전체 Validation raw 수량 RMSE의 최초 엄격한 유한 최소 epoch를 고정. MAE도 같은 checkpoint. | E1 §4, G2의 원본 작업 |
| 평가 | 전체/Train 고정 큰수량의 RMSE·MAE·recorded positive-integer TimeNLL. 평균±표본SD(ddof1), seed별 악화 포함. | E1 |
| 외부 비교 | RMTPP·THP·NHP·SAHP·S2P2·AttNHP의 프로젝트 내 공통 head 연결 구현. | E1 §4 |

외부 모델의 원논문 native head·native marked likelihood·모든 튜닝을 재현했다는 뜻은 아니다. S2P2·AttNHP의 matched-head adaptation과 학습·평가 Runtime 차이는 별도로 공개해야 한다.

현재 수량 head는 log1p 공간의 softplus 위치에서 `expm1`로 복원하는 **점예측**이며 log-MSE로 학습한다. 완전한 수량 확률분포의 NLL을 학습하는 head가 아니다. 시간의 recorded positive-integer lognormal duration NLL과 수량 log-MSE를 함께 최적화했다는 사실만으로 원논문의 marked intensity likelihood를 학습했거나 시간·수량의 완전한 결합 생성모델이 우세하다고 표현하지 않는다(C5·C6, E1).

## 파라미터 계산과 측정 경계 — 구분

| 구분 | 코드상 확인 | 아직 확인하지 않은 것 |
|---|---|---|
| Shared15 보정 | `128×15 + 8×15×64 = 9,600` | 전체 모델이9,600개라는 해석 |
| 기존 MLP16 보정 | `8×(128×16 + 16×64) = 24,576` | Shared15와 같은 함수 공간이라는 해석 |
| 보정 파라미터 감소 | `(24,576−9,600)/24,576 = 60.9375%` | 전체 모델 파라미터 감소율·속도·메모리 감소율 |
| 계산 재사용 | 공유 입력 projection·GELU를 분기가 재사용 | 같은 장비·배치·길이에서의 latency·throughput·GPU/CPU memory·학습비용 실측 |
| 분리 대조 | 코드에는 prefix별 활성 파라미터를 맞춘 독립 `(15,5,5,5,5,5,5,5)` 모드가 있음 | 그 원본 데이터 대조 결과가 이번 논문 근거 표에 있다는 주장. Cap 결과를 원본 대조로 대신 쓰는 것도 불가. |

C1 SHA-256: `eee18d3abe199f0c71d80da35acf01cf19acd8f56291ff7cfd33fd86d8bc82a2`.
C2 SHA-256: `4bcbecdadf921df21a4e7fb379962eb4ecec2cfeb2f8ff1edccbb6b418807eae`.
C3 SHA-256: `2a903290810070bdfe72a355a98ff09790b02858a6853e94a7a4a87336afc6f7`.

## 작성 경계 — 유지

- V1이 기록한 PAKDD 2027 정책에 맞춰 도구 지원은 사실·근거·결과표·빈 작성 구조에 한정한다. 본문·초록·서론·novelty 문장을 생성하지 않는다. 사람이 작성한 문법 교정과 사용 공개 요구는 별도다.
- G1 통과는 이 작성 경계를 해제하지 않는다.
- 이 문서 자체는 독창성·투고 적합성·규정 준수·채택 가능성 판정이 아니다.
