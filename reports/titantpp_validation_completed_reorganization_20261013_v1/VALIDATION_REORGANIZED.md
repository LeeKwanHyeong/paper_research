## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-13T17:46:25.211662+09:00
- Verification Status: ANALYZED
- Version Label: completed_validation_reorganization_v1

## 수량 중심으로 완료 Validation을 정리한다 — 완료

주 지표는 전체 raw 수량 RMSE다. 전체 MAE와 큰 수량 오류를 함께 보고하고 TimeNLL은 보조 지표로 남긴다. 이번 정리는 선택checkpoint를 다시 고르거나 원래 과학 판정 기준을 바꾸지 않는다.

| 실험 묶음 | source | 완료 상태 | 확보된 결과 |
|---|---|---|---|
| 이력 보정: FULL/NONE/current-only | source129 | 27/27 완료 | 회수된 선택 endpoint·전체/큰 수량 여섯 지표 |
| Instacart: CNN+GRU/MLP8/MLP12 | source126 | 9/9 완료 | 회수된 선택 endpoint·여섯 지표 |
| 신규 CNN·GRU 구성요소 대조 | source131 | 32/36 완료 | 완료32의 선택history·전체 RMSE/MAE/TimeNLL |
| 외부6종 | 기존 동결 source | 기존 완료 참조 | 공통 출력부의 역사적 Validation 비교 |

신규 실험의 로컬 근거는 17:37:02 KST의 단일 관측이다. 기존 5080의 Taxi·RAF24개는 08:27:20 KST 종료 캐시를 사용한다. 서로 다른 실험의 fresh fit·source·GPU·Runtime 결과를 섞어 평균내지 않는다. 아래는 seed42/52/62의 산술평균±표본SD(n-1)이며, ensemble 성과나 신뢰구간·유의성 검정은 아니다.

## 신규 Taxi·RAF 네 구조를 비교한다 — 완료

| 자료 | 구조 | 전체 RMSE | 전체 MAE | 보조 TimeNLL |
|---|---|---:|---:|---:|
| Taxi | NONE | 89.305 ± 1.151 | 28.683 ± 0.652 | 0.851 ± 0.079 |
| Taxi | CNN | 86.650 ± 2.099 | 27.793 ± 0.500 | 0.782 ± 0.082 |
| Taxi | GRU | 84.478 ± 1.646 | 26.710 ± 0.150 | 1.060 ± 0.128 |
| Taxi | CNN+GRU | 81.923 ± 5.027 | 25.816 ± 1.135 | 1.318 ± 0.733 |
| RAF | NONE | 34.224 ± 0.118 | 9.264 ± 0.059 | 3.595 ± 0.112 |
| RAF | CNN | 34.431 ± 0.073 | 9.218 ± 0.028 | 3.497 ± 0.139 |
| RAF | GRU | 34.211 ± 0.226 | 9.159 ± 0.025 | 3.578 ± 0.160 |
| RAF | CNN+GRU | 34.182 ± 0.234 | 9.181 ± 0.039 | 3.505 ± 0.097 |

Taxi FULL은 NONE보다 평균 RMSE가 8.266%, MAE가 9.998% 낮고, CNN·GRU 단독보다도 평균 RMSE가 낮다. 다만 seed42에서는 CNN·GRU 단독보다 높다. RAF FULL의 평균 RMSE는 네 구조 중 최저이며 NONE 대비 0.121% 낮다. RAF의 평균 MAE는 GRU가 최저다. 작은 평균 RMSE 개선도 관측된 개선으로 보고하며, 모든 seed·모든 지표의 일관된 우위와는 구분한다.

## 신규 Intermittent는 완료된 seed42만 비교한다 — 일부 완료

| 구조 | 선택/종료 epoch | 전체 RMSE | 전체 MAE | 보조 TimeNLL |
|---|---:|---:|---:|---:|
| NONE | 30/70 | 1.710699 | 0.745611 | 0.319725 |
| CNN | 28/68 | 1.645086 | 0.676601 | 0.358743 |
| GRU | 14/54 | 1.799217 | 0.713528 | 0.279343 |
| CNN+GRU | 15/55 | 1.666661 | 0.715963 | 0.618713 |

이 seed에서는 CNN의 RMSE가 최저이고 FULL은 NONE·GRU보다 낮다. Intermittent의 3seed 네 구조 최종 평균은 아직 미확정이다. 완료된 NONE/CNN 3seed와 GRU/FULL 1seed를 같은 완성 평균 비교로 제시하지 않는다. 진행 중인 epoch의 잠정 best는 완료된 선택 결과에 넣지 않는다.

## 기존 이력 보정27은 다른 질문의 완료 근거다 — 완료

대조는 FULL/NONE/current-only이며, CNN-only/GRU-only 구성요소 대조와 다르다. 같은 prototype backbone에서 해당 보정 경로의 효과를 검토했다. current-only의 hidden에는 prefix 이력이 포함되므로 완전한 무이력 대조가 아니다.

| 자료 | 대조 | 전체 RMSE | 전체 MAE | 보조 TimeNLL |
|---|---|---:|---:|---:|
| Intermittent | CNN+GRU | 1.696 ± 0.042 | 0.706 ± 0.015 | 0.410 ± 0.182 |
| Intermittent | NONE | 1.741 ± 0.028 | 0.745 ± 0.028 | 0.343 ± 0.040 |
| Intermittent | current-only | 1.756 ± 0.057 | 0.753 ± 0.023 | 0.348 ± 0.024 |
| RAF | CNN+GRU | 34.171 ± 0.212 | 9.202 ± 0.080 | 3.548 ± 0.063 |
| RAF | NONE | 34.296 ± 0.085 | 9.268 ± 0.033 | 3.589 ± 0.129 |
| RAF | current-only | 34.311 ± 0.101 | 9.367 ± 0.166 | 3.469 ± 0.101 |
| Taxi | CNN+GRU | 80.409 ± 2.983 | 25.635 ± 1.022 | 1.272 ± 0.165 |
| Taxi | NONE | 87.612 ± 1.417 | 28.276 ± 0.468 | 0.954 ± 0.112 |
| Taxi | current-only | 88.001 ± 1.376 | 28.197 ± 0.328 | 0.821 ± 0.226 |

원래 Q1 개발 기준은 Taxi·Intermittent의 각 3seed에서 FULL이 두 대조보다 전체 RMSE를 엄격하게 개선해 충족했다. RAF는 평균 개선이 있지만 seed52 반례가 있다. 이는 신규 구성요소 대조의 joint_path_support, 즉 FULL이 NONE/CNN/GRU 모두보다 각 3seed에서 엄격하게 개선되는 자료가 최소2개여야 한다는 기준과 다르다. 신규 joint 기준은 Taxi·RAF의 반례로 미충족이며, 이 판정이 관측된 평균 개선을 취소하는 것은 아니다.

## 회수된 이력 FULL의 큰 수량 결과를 보존한다 — 완료

| 자료 | 큰 수량 RMSE | 큰 수량 MAE | 보조 큰 수량 TimeNLL |
|---|---:|---:|---:|
| Intermittent | 8.297 ± 0.655 | 6.467 ± 0.436 | 9.24e-05 ± 0.000153 |
| RAF | 327.624 ± 6.654 | 280.782 ± 7.508 | 3.504 ± 0.208 |
| Taxi | 394.761 ± 42.161 | 306.032 ± 42.707 | 8.06e-16 ± 1.4e-15 |

전체 수량 개선이 큰 수량 개선까지 보장하지는 않는다. 기존 이력 Intermittent·RAF에서 FULL의 큰 수량 RMSE·MAE 평균은 두 대조보다 나쁘다. Taxi의 큰 수량 이득도 일관되지 않는다. 이 표는 기존 source129 원본이며, 아직 회수되지 않은 신규 source131의 큰 수량 값을 대신하지 않는다. 시간 단위는 데이터셋별로 다르므로 서로 합쳐 평균내지 않는다.

## Instacart 세 구조를 별도 완료 결과로 정리한다 — 완료

| 구조 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 보조 전체 TimeNLL |
|---|---:|---:|---:|---:|---:|
| CNN+GRU | 5.877 ± 0.009 | 3.987 ± 0.004 | 24.438 ± 0.201 | 21.782 ± 0.250 | 2.811 ± 0.009 |
| MLP12 | 5.880 ± 0.017 | 3.991 ± 0.003 | 24.668 ± 0.225 | 22.103 ± 0.265 | 2.807 ± 0.005 |
| MLP8 | 5.885 ± 0.013 | 3.991 ± 0.003 | 24.635 ± 0.337 | 22.024 ± 0.419 | 2.807 ± 0.004 |

CNN+GRU는 전체·큰 수량의 수량 오차 평균이 세 구조 중 최저이며, 전체 이득 폭은 작다. seed52의 MLP12 우위와 전체 TimeNLL 손해도 남긴다. Instacart에는 NONE/current-only 대조가 없으며, 서로 다른 파라미터 예산의 구조 비교다. 이 수치를 보정 경로 제거 효과로 해석하지 않는다.

## 신규 FULL의 외부6종 대비 위치를 확인한다 — 역사적 참조

| 모델 | Taxi 전체 RMSE | RAF 전체 RMSE |
|---|---:|---:|
| 신규CNN+GRU | 81.923 ± 5.027 | 34.182 ± 0.234 |
| RMTPP | 94.728 ± 3.401 | 34.550 ± 0.251 |
| THP | 107.598 ± 4.504 | 34.745 ± 0.320 |
| NHP | 318.815 ± 5.301 | 36.273 ± 0.319 |
| SAHP | 120.527 ± 2.000 | 35.419 ± 0.213 |
| S2P2 | 106.354 ± 1.914 | 33.995 ± 0.126 |
| AttNHP | 127.129 ± 8.992 | 34.877 ± 0.068 |

Taxi는 기존 외부6종보다 평균 RMSE·MAE가 낮다. RAF의 평균 RMSE는 5종보다 낮고 S2P2보다 0.552% 높다.

과거 내부 MLP4의 RMSE는 Taxi79.711, RAF33.951로 신규 CNN+GRU보다 낮다. 외부6종에 대한 좋은 성능 위치와 모든 TitanTPP 변형 중 최고라는 주장은 구분한다. 외부군은 공통 입력·출력부·손실로 적응한 구현이므로 원 논문의 native head·최적 튜닝 전체 대비 우월성을 뜻하지 않는다. 대상·주요 학습·선택 조건은 맞지만 전체 source·Runtime·실행 세대가 같은 fresh 외부 재비교는 아니다. 최신 FULL의 선택 history와 기존 외부 selected endpoint라는 증거 단계 차이도 남긴다.

## 남은 작업을 완료 상태에 맞춰 재정렬한다

**신규 Intermittent 네 구조를 확정한다 — 진행 중 / 외부 작업 대기**
- 기존 dispatcher의 GRU52·GRU62가 자연 종료한 뒤 각 FULL이 진행된다. 완료 후 같은 3seed의 네 구조를 비교한다. 이미 완료된 조건은 재학습 대상으로 나열하지 않는다.

**신규 원본과 여섯 Validation 지표를 확정한다 — 다음 작업**
- Immutable 원본 archive·manifest·selected/last binary를 회수해 SHA를 확인하고, endpoint/summary의 전체·큰 수량 RMSE·MAE·TimeNLL을 같은 선택 epoch로 묶는다. 현재 선택 history에는 큰 수량 세 지표가 없으며 val_tail_count/aux_loss로 대체하지 않는다.

**수량 중심 원고의 주장과 표를 정합한다 — 다음 작업**
- 평균 수량 이득, seed별 예외, 작은 RAF·Instacart 이득, 큰 수량 손해를 함께 남긴다. TimeNLL은 보조 결과로 보존하고 시간 개선을 위한 새 학습을 필수 과제로 추가하지 않는다. 신규 구성요소와 기존 이력·Instacart·외부 참조는 별도 패널로 연결한다. 이번에는 열린 main.tex를 편집하지 않았다.

**고정 모델의 Test 비교를 준비한다 — 이후 작업 / 실행 승인 필요**
- 현재36조건과 외부 모델의 완료 여부·동결 checkpoint·전처리·평가 대상·지표, 기존 Test 원본 재사용 가능성을 먼저 계약으로 확정한다. 기존 Test 노출 이력을 유지하며 새 독립 평가로 부르지 않는다. 이번 Test 열람·평가 및 추가 학습은0이다.

## 검증과 출처

완료32의 선택 history SHA·terminal manifest 결속, 실제 exit0·원격 terminal416파일을 확인했다. 기존27/9의 endpoint·원본 SHA와 평균·표본SD 재확인, 독립 하위 감사를 통과했다. 이력27·Instacart9의 CPU binary 재추론 감사는 미완료이며 새 독립 평가도 아니다. 3seed SD는 유의성 검정이 아니다. 과거 A100/RTX 물리 GPU 이관과 seed 간 Runtime 차이를 보존한다. 신규 구성요소 pipeline wrapper 자체 OS exit는 직접 관측하지 않았으며, 시간·효율을 서로 다른 GPU 실행과 합산하지 않는다.

17:37 시간별 관측은 helper1회, 원5080 캐시 SSH0, 진행 중인 원5090·새5080의 status SSH 각1회였다. 재실행·추가 원격·deploy/qualification·학습·추론·회수·Runtime/인증 변경은0이다. 자동화는 ACTIVE를 유지한다.

출처 rawSHA는 validation_summary.json의 provenance_sources에 기록했다. 완료된 선택 행은 completed_selected_metrics.csv에, 신규 완료32의 원값과 미완료 경계는 같은 JSON에 있다.
