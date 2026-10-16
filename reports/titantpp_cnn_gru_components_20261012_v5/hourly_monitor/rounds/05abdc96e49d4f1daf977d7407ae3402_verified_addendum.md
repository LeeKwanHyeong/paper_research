# CNN·GRU36 시간별 관측 보완 보고

**저장 모델과 실제 종료를 관측한다 — 진행 중**

실제 관측: 2026-10-12 21:15:47 KST. 전체36 중 종료3·진행4·대기29.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 3 / 24 | 1 | 20 |
| 5090 | 0 / 12 | 3 | 9 |

| GPU | dataset·seed·arm | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 종료 | 87 / 47 | 종료 | — |
| 5080 | Taxi·42·CNN만 | 종료 | 101 / 61 | 종료 | — |
| 5080 | Taxi·42·GRU만 | 종료 | 85 / 45 | 종료 | — |
| 5080 | Taxi·42·CNN+GRU | 진행 중 | 77 / 46 | 8분 | 3시간20분 |
| 5090 | Intermittent·42·보정 없음 | 진행 중 | 31 / 30 | 3시간12분 | 22시간8분 |
| 5090 | Intermittent·42·CNN만 | 진행 중 | 31 / 28 | 3시간6분 | 22시간34분 |
| 5090 | Intermittent·42·GRU만 | 진행 중 | 14 / 14 | 7시간0분 | 50시간1분 † |

- 종료3개는 dispatcher completed·actual child exit0·각13개 terminal 파일 원격SHA(선택/최종binary·endpoint replay 포함)로 검증했다. 원관측 worker 종료시각은 NONE19:28:48/CNN19:35:55/GRU20:05:48 KST이며 새 종료시각으로 바꾸지 않는다. 로컬 binary 원본 회수·최종 집계는 미완료다.
- 진행4개는 각각 최근10개 완료epoch의 finite 양수 elapsed_seconds 중앙값만 사용했다. best는 최초 strict finite 최소 fullValidation 전체 raw 수량RMSE epoch이며 최종 selected binary는 진행fit에서 아직 생성 전이다. 조기종료 목표 min(300,max(40,best+40)); 원관측시점 기준이며 best 갱신/속도/부하 변화에 따라 바뀐다. 미시작/다른fit 속도 복사0, 전체36 큐 ETA 미확정이다.
- †5090 GRU300epoch 시나리오50시간1분은 남은 조건기한33시간24분보다 길다. 현 기한 내300epoch 완료를 보장하지 않으며 기한 연장/학습 실패로 해석하지 않는다.
- 원래 round의 5080 완료0·세 대조 저장 전 표시는 지난회차의 MEDIUM 보고기 집계 오류가 남은 것이다. 동일 raw snapshot 종료 증적으로 이 별도 보고서를 보완했다. 원본 round/코드/봉인/학습/자동화 설정은 변경하지 않았다.
- helper1회·live host별SSH1회·추가SSH0·Test0·원본회수0. pipeline wrapper 자체OS exit는 직접 관측하지 않았다. 모든36개 종료 전이므로 이 자동화를 유지한다.

**종료 집계를 보완하고 최종 원본·논문 근거를 확정한다 — 다음 작업**

- 종료 증적을 반영하는 보고기 수정·검증이 필요하다. 시간별 관측→36개 자연종료 확인→원본회수 범위/계약 확인→immutable archive/manifest·선택/최종binary 로컬SHA 확정→여섯Validation지표·모든seed/대조 평균/표본SD→Results·Methods·주장 정합 순서다.

원본 회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/05abdc96e49d4f1daf977d7407ae3402.json
rawSHA: 0e471f8a02be5329af7793eecfcb4197eb3f66fceeb59d2aa5f3a2127e01ac31
