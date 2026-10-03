# 기존 Titans-MAC 결과와 후속 Benchmark 필요 범위

분석일: 2026-09-28. 대상: `paper_research / codex/hard-lmm-causal-qkv`의 로컬에 보존된 **validation 증적**.
현재 구조 비교는 **2026-09-28 10:55 KST 저장 snapshot**을 기준으로 한다. 이번 작업은 원격 진행 상태 조회, 학습, checkpoint 재평가, held-out 열람을 수행하지 않았다.

## 1. 결론 — 분석 완료

**기존 결과를 활용할 수 있지만, 현재 TitanTPP의 백본 기여를 주장하려면 ‘새 외부 Benchmark만 추가’하는 것으로는 아직 충분하지 않다.**

1. 원본 Titans 메모리 계열인 B1/MAC 실험은 이미 존재한다. 구현과 과거 메모리 작동 분석을 버리거나 처음부터 새로 만들 필요가 없다.
2. RMTPP·THP·NHP·SAHP는 현재와 같은 head/선택 기준으로 3개 데이터 × 3 seeds = **36조건**이 완료돼 있다. B·Full **18조건**과 함께 재사용한다. 이 네 비교군을 일괄 재학습할 이유는 이번 감사에서 발견되지 않았다.
3. 과거 MAC를 현재 Full/MLP와 **백본만 다른 직접 비교**로 쓰기에는 학습 시간 손실, checkpoint 선택·조기 종료 규칙, MAC seed 간 안정화 정책이 다르다. 현재 규칙으로 MAC 비교가 필요하다면 **3개 데이터 × 3 seeds = 9조건**을 새 계약으로 보완하는 범위가 가장 명확하다. 기존 B·Full을 다시 학습하는 18조건을 덧붙이지 않는다. 이 9조건은 실행 승인이나 비용 산정이 완료된 계획이 아니다.
4. 진행 중인 Full/MLP/수준/변화/정적 검색 제거의 미완료 조건은 기존 승인 범위에서 마무리한다. 이번 분석을 근거로 Full 개조나 Gate 신설을 먼저 요구하지 않는다.
5. 이후 외부 비교는 기존 기획의 S2P2·FlexTPP·수량 특화 비교군을 검토한다. 기존 480시간·88~100회 제안은 현재 실행 계약이 아니며 자동 재사용하지 않는다.

MAC를 과거 탐색 결과로만 남기고 ‘원본 Titans 대비 개선’ 주장을 하지 않는다면 MAC 9조건은 논리적 필수가 아니다. 그러나 사용자가 검토 중인 **Titan 원본 메커니즘과 비교하는 백본 논문 방향**에서는 빠진 핵심 대조군이다. 이것을 새로운 외부 모델 여러 개로 대체할 수는 없다.

## 2. 과거 MAC는 무엇을 보여 주었나 — 완료된 결과의 해석

### A. 과거 seed42: 메모리의 효용은 존재하지만 균일하지 않았다

아래는 당시 B0와 B1의 validation 비교이며, 현재 head/선택 기준 결과와 합치지 않는다. 변화율은 `100 × (B1 / B0 − 1)`이다.

| 데이터 | 당시 B0 수량 RMSE | 당시 B1/MAC 수량 RMSE | RMSE 변화 | 해석 |
|---|---:|---:|---:|---|
| Taxi | 162.1927 | 116.9266 | −27.91% | 과거 조건에서 개선 근거가 있다. |
| Intermittent | 1.7224 | 1.9062 | +10.67% | body/전체 MAE 개선과 RMSE·극단 tail 악화가 공존한다. |
| RAF | 36.5764 | 37.1872 | +1.67% | 수량 RMSE 우위가 없다. 현재 3개 데이터 비교에는 포함되지 않는 과거 데이터다. |

동일한 B1 checkpoint에서 온라인 갱신을 껐을 때와 비교한 MAE 차이(full minus no-update)는 Taxi −18.6029, Intermittent −0.2840, RAF 0이었다. 이는 **그 checkpoint가 학습한 메모리 경로가 예측에 실제 영향을 주었다는 증거**다. 메모리 없는 모델을 처음부터 학습한 제거 실험은 아니다. RAF는 해당 validation 구성에서 이전 segment의 쓰기를 읽을 기회가 없어 온라인 갱신 효과가 0이었다. 메모리가 언제나 무용하다는 결론으로 옮길 수 없다.

과거 B1/B0의 완료 epoch당 시간 비는 Intermittent 5.45배, Taxi 6.86배, RAF 7.23배였다. 당시 source·Runtime·학습 길이의 측정값이며, 현재 5090 병렬 실행과의 비용 우열이나 향후 MAC 9조건 ETA로 사용하지 않는다.

출처: [과거 원본 비교표](../../search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1/comparison/metrics.csv), [메모리 감사](../../paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/decision.json), [과거 비용](../../paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/historical_cost.csv). 이번 독립 재계산: [historical_seed42.csv](historical_seed42.csv).

### B. 안정화된 seed52·62: 정상 완료했으나 과거 seed42와 합칠 수 없다

MAC는 Instacart에서 내부 associative gradient가 비유한 값이 되는 문제가 있었고, 이후 **메모리 내부 gradient clipping=1**을 추가했다. 외부 optimizer의 gradient clipping과 별개인 내부 학습 정책 변경이다. 과거 unbounded seed42와 이 두 seed를 같은 모델 설정의 3-seed 결과로 합치지 않는다. 실패한 Instacart 첫 시도 및 복구 실패 기록도 유지한다.

| 데이터 | MAC 수량 RMSE, seed52·62 평균 ± 표본 SD | 실제 선택 epoch, 52 / 62 | 실제 완료 epoch, 52 / 62 |
|---|---:|---:|---:|
| Taxi | 154.3737 ± 23.8637 | 3 / 6 | 43 / 46 |
| Intermittent | 2.0804 ± 0.2486 | 213 / 214 | 253 / 254 |
| Instacart | 6.0669 ± 0.0103 | 37 / 24 | 77 / 64 |
| RAF | 36.9737 ± 1.5612 | 2 / 15 | 42 / 55 |

8개 summary·history·launch contract를 읽고 기존 CSV와 수치 및 checkpoint digest가 일치하는지 확인했다. 모든 선택 epoch가 실제 history의 validation joint-objective 최소와 일치한다. 표준편차는 ddof=1이며 신뢰구간이 아니다. MAC의 과거 `time_nll`은 **clamped time loss**로 현재의 관측 시간 확률질량 NLL과 비교하지 않는다.

검토한 로컬 증적에는 동일 clipping 정책의 seed42 장기 실험 완료 근거가 없다. seed42의 full-epoch 안정성 사전 검사는 장기 학습을 대신하지 않는다. 현재 3개 데이터의 비교 범위는 총 6개의 안정화된 과거 MAC 결과이며, RAF 2개는 별도 과거 증거로 보존한다.

출처: [안정화 사건 기록](../../paper/results/titantpp_mac_inner_stability_incident_20260831/README.md), [seed52 계약](../../paper/contracts/count_aware_titantpp_mac_stable_seed52_5090_v1.md), [8조건 명세 및 지표](stable_mac_conditions.csv), [2-seed 재계산](stable_mac_two_seed_summary.csv).

## 3. 현재 실험과 맞는 부분, 맞지 않는 부분 — 비교 가능성 확인 완료

| 비교 항목 | 과거 안정화 MAC | 현재 B·Full·MLP / 공통 head 비교군 | 재사용 판단 |
|---|---|---|---|
| 데이터 및 split 원본 SHA | 동일 3개 데이터 | 동일 | 6개 MAC 조건 모두 일치 |
| history 최대 길이·lookback | Taxi256/168, Intermittent256/520, Instacart64/52 | 동일 | 설정 일치; 숫자는 최대 sequence 길이 / lookback 값 |
| train 및 validation target 수 | 기존 population | 기존 population | 6개 조건 모두 일치 |
| 관측 입력 / 수량 손실 | 시간·수량 이력 / log1p 수량 MSE | 같은 정보 범위 / log1p 수량 MSE | 주요 문제 설정 재사용 가능 |
| batch / learning rate / max-min-patience | 128 / .001 / 300-40-40 | 동일 | 수치 상한이 같아도 실제 종료 규칙은 아래 차이 있음 |
| 시간 head와 손실 | `legacy_clamped_rmtpp` | lognormal duration + 관측 정수 구간 질량 | **불일치: 학습 중 공유 encoder에 전달되는 gradient도 달라짐** |
| 시간 scale | 모든 데이터 3 | Taxi1 / Intermittent3 / Instacart7 | Taxi·Instacart 불일치 |
| 선택·조기 종료 monitor | validation joint objective | validation raw quantity RMSE | **불일치: 선택 epoch와 실제 학습 노출이 달라짐** |
| MAC 내부 gradient 정책 | 안정화52/62 clip1, 과거42 unbounded | 아직 현재 규칙의 MAC 결과 없음 | seed42를 단순 결합할 수 없음 |
| 초기화·batch prefix | 과거 source의 기록 | 현재 campaign의 동결 초기화 및 prefix 감사 | 현재 모델과의 동일 tensor/prefix를 입증한 것으로 취급하지 않음 |

특히 과거 MAC의 **8조건 모두 joint 선택 epoch와 기록된 raw-RMSE 최소 epoch가 달랐다.** 예를 들어 Taxi seed52는 joint 선택 epoch3, 저장 history 내 RMSE 최소 epoch41이다. epoch별 RMSE를 다시 골라 적으면 학습 손실·조기 종료·checkpoint 보존 문제까지 해결되는 것이 아니다. 이 진단값을 새 공식 선택 결과나 실제 replay로 보고하지 않았다.

따라서 과거 MAC 수량 RMSE의 단위가 같다는 이유만으로 현재 Full보다 열등한 백본이라고 결론 내릴 수 없다. 과거 결과를 별도 프로토콜의 탐색 표로 제시하는 것은 가능하지만, **백본 효과를 분리하는 주 비교표에는 현행 조건을 맞춘 MAC가 필요**하다.

동일 data/split byte SHA와 설정·개수는 확인했지만, 과거 MAC의 target identity와 batch prefix를 원자료에서 새로 재구성하지 않았다. [6조건 대조표](mac_current_compatibility.csv).

## 4. 이미 확보한 외부 비교는 어느 정도인가 — 재사용 가능

다음은 현재 관측 시간 head, 동일 raw-RMSE checkpoint 선택을 적용한 **validation 3-seed 평균 ± 표본 SD**다. 기존 54조건의 summary·history·endpoint 기록을 다시 대조했다. 외부 모델의 원래 head·모델별 최적 튜닝을 재현한 순위표는 아니다.

| 모델 | Taxi RMSE | Intermittent RMSE | Instacart RMSE |
|---|---:|---:|---:|
| B | 90.5093 ± 0.5207 | 1.7806 ± 0.0439 | 5.8796 ± 0.0166 |
| Full | 78.7774 ± 1.0305 | 1.8140 ± 0.0854 | 5.8859 ± 0.0162 |
| RMTPP | 94.7283 ± 3.4006 | 2.5499 ± 0.1604 | 5.8775 ± 0.0213 |
| THP | 107.5977 ± 4.5040 | 2.7534 ± 0.1784 | 5.8582 ± 0.0253 |
| NHP | 318.8148 ± 5.3011 | 12.8755 ± 1.0405 | 6.7613 ± 0.0538 |
| SAHP | 120.5273 ± 1.9996 | 7.0298 ± 0.7673 | 5.8776 ± 0.0067 |

이 표는 다음을 뒷받침한다.

- Taxi에서는 Full의 수량 개선이 기존 외부 인코더 비교에서도 반복된다. 그러나 Full의 시간 NLL은 B 0.7247 대비 1.1822로 악화됐고, THP/NHP/SAHP의 시간 점수보다도 높다. 수량과 시간을 동시에 개선했다는 주장은 안 된다.
- Intermittent에서는 B·Full 모두 현재 네 외부 인코더보다 수량 RMSE가 낮지만, Full은 B보다 나쁘다. Full의 시간 NLL 0.4714도 B 0.3846 및 RMTPP 0.2748보다 높다. **백본 전체의 유망성**과 **Full 보완의 필요성**은 다른 질문이다.
- Instacart에서 Full은 B 및 RMTPP·THP·SAHP보다 평균 RMSE가 높다. 외부 최신 모델을 추가하기 전부터 확인된 반례다.
- 이번 복합 성능 gate는 B 대비 9개 데이터×seed에서 모두 미통과다. 실행·재평가 성공과 성능 기준 통과를 혼동하지 않는다.

기존 B·Full 18조건은 현재 core ledger의 재사용 항목과 선택 checkpoint 상태 SHA 및 지표가 모두 같다. 따라서 현재 MLP 등과 위 외부 인코더의 비교도 공통 조건이 유지되는 범위에서 연결할 수 있다. 파라미터 수·사전 튜닝 이력까지 모두 같다는 뜻은 아니다.

출처: [기존 완료 결과](../local_detail_3seed_final_20260927_v1/condition_results.json), [기존 최종 보고](../local_detail_3seed_final_20260927_v1/final_report.md), [재계산 표](current_comparable_summary.csv).

## 5. Full과 MLP에 대해 지금 결정할 수 있는 범위 — 일부 완료, 일부 대기

아래는 **저장 snapshot의 같은 seed끼리** 비교한 수량 RMSE다. 새로운 원격 조회 결과가 아니다.

| 데이터 | 비교 가능한 seed | Full | MLP | 의미 |
|---|---|---:|---:|---|
| Taxi | 42·52·62 | 78.7774 | 79.7111 | Full 평균 RMSE가 약1.17% 낮지만 MLP 시간 NLL이 더 낮다(1.0387 대 1.1822). Full의 통계적 유의성·전반 우월성을 선언하지 않는다. |
| Intermittent | 42만 | 1.7925 | 1.5785 | MLP가 약11.94% 낮다. 나머지 seed의 최종 판정은 이 snapshot으로 못 한다. |
| Instacart | 42만 | 5.8714 | 5.8787 | 차이가 작다. 완성된 3-seed 우열은 아니다. |

Taxi 정적 검색 제거는 3-seed RMSE 79.8454로 Full 78.7774보다 약1.36% 높지만, MAE는 25.2754로 Full 25.3623보다 낮고 시간 NLL도 낮다. 정적 검색의 모든 지표 우위를 주장할 수 없다. 두 기능을 모두 제거한 조건 없이 메모리와 이력 보완의 상호작용까지 확정하지 않는다.

현재 core 저장 증적은 B·Full 재사용18 + 신규완료17 = 35/54조건이다. 미완료19조건은 누락/실패로 바꾸지 않고 대기 범위로 남긴다. 기존 외부36조건과 합치면 완료된 고유 조건은 71개다. 이는 서로 다른 모델의 학습 조건 수이며 독립 데이터셋/통계 표본 수가 아니다.

따라서 **추가 Full 개선을 먼저 시작해야 한다는 결론도, MLP를 최종 채택했다는 결론도 아직 아니다.** 현 승인 실험을 마치고 사용할 모델·유지할 주장을 결정하는 것이 먼저다. [같은 seed의 비교](full_component_pairs.csv).

## 6. 필요한 후속 범위를 가장 작게 정리한다

**현재 내부 비교의 최종 결과와 주장 범위 확정 — 진행 중 / 완료 대기**
- 대상: 승인된 5080·5090 core 캠페인. 현재 실행과 원본 증적을 유지한다.
- Full·MLP·제거 조건의 3-seed 결과, 시간 성능, 비용, 반례를 함께 확정한다. Full 전체 우월성 대신 백본 전체·특정 데이터 조건의 효과를 주장할지도 결정한다.
- 완료 조건: 미완료/실패를 포함한 최종 표와 주장별 지원/반례가 정리돼야 한다.

**MAC를 현재 프로토콜에 연결할 계약 정리 — 다음 작업; 실제 학습은 승인 필요**
- 대상: `paper_research`의 Titans-MAC 비교 구현·새 실행 계약. 현재 원본97개 소스나 진행 중인 Runtime을 수정하지 않는다.
- 원본 메커니즘 대조 주장을 유지한다면 clip1 안정화 정책을 명시한 MAC 3데이터×3seed의 최대9조건을 산정한다. head/loss/loader/선택/조기종료를 현재 공통 규칙과 맞추고 인과성·메모리 상태·padding 검증을 재사용 및 보완한다.
- 완료 조건: 변경되는 wrapper 범위, 서버, Runtime, source, 비용 상한, 실패 처리 및 과거 증거와의 관계가 검토 가능한 상태여야 한다. 현재 B·Full18과 외부36을 자동 재학습하지 않는다.

**역할이 겹치지 않는 외부 비교를 확정 — 다음 작업; 실제 학습은 승인 필요**
- 대상: 기존 후속 기획의 S2P2(최근 TPP 인코더), FlexTPP(시간·수량 결합 분포), TimeMixer-event(수량 중심 예측), DeepRenewal-event(Intermittent 간격·크기 모델).
- 이들은 현재 기록에서 실행 완료된 결과가 아니라 기획된 후보다. 이번 작업에서 최신 문헌/구현을 새로 검증하거나 해당 구성을 승인하지 않았다. 동일 head 비교와 native 계열 비교를 별도 표로 둔다.
- 최종 모델의 튜닝 예산도 같이 고정한다. 외부 모델에만 탐색을 주거나 기존 B·Full을 무조건 재튜닝하지 않는다. 단순한 수량 persistence/최근 이력 평균 같은 낮은 비용 참조가 논문 범위에 필요한지도 결과 계약에서 확인한다.
- 완료 조건: 각 비교군이 어떤 주장을 검증하는지와 최소 실행표·비용이 연결돼야 한다. 480시간의 과거 제안을 현 계약에 합산하지 않는다.

**최종 평가 규칙 잠금과 held-out 적격성 확인 — 이후 작업 / 별도 승인 필요**
- validation으로 정한 모델과 평가 규칙을 잠근 후 기존 held-out 사용 이력의 메타데이터를 검토한다. 깨끗한 미사용 test라고 가정하지 않는다.
- 본 분석은 held-out 성능을 열람하지 않았고 test 실행을 승인하지 않는다. validation 반복 탐색 결과만으로 최종 일반화나 PAKDD 채택 가능성을 확정하지 않는다.

공통 계약과 동일 기준선을 다루므로 이번 감사는 현재 세션에서 직렬로 수행했다. 향후 독립 모델의 구현은 계약이 고정된 뒤 분리 가능하며 GPU 병렬 배정은 새 자원 계약에서 판단한다. 커밋·Push·공용 Runtime 배포는 수행하지 않았다.

## 검증과 재현

- 모델을 import하지 않는 표준 라이브러리 분석 스크립트 [analyze.py](analyze.py)로 원본 CSV/JSON을 읽고 집계를 재계산했다.
- 검증한 파일 해시는 [sources.json](sources.json), 확인 항목은 [verification.json](verification.json), 기계 판독 결론은 [analysis.json](analysis.json)에 있다.
- source·contract·history·summary·endpoint 간 일관성 검사이며, 이번 CPU 작업이 기존 GPU replay를 다시 수행한 것은 아니다. checkpoint 바이너리나 held-out 원자료를 열지 않았다.
- 기존 연구 파일을 수정하지 않고 이 보고서 폴더만 생성했다. 모델 학습 코드, 실행 계약, 자동화 및 진행 중인 서버는 변경하지 않았다.
