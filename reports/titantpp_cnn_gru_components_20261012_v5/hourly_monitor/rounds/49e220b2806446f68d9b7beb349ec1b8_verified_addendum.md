# CNN·GRU36 시간별 관측 보완 보고

**실제 종료와 현재 checkpoint를 확인한다 — 진행 중**

실제 관측: 2026-10-13 00:17:11 KST. 전체36 중 종료7·진행4·대기25.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 7 / 24 | 1 | 16 |
| 5090 | 0 / 12 | 3 | 9 |

| GPU | dataset·seed·arm | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 종료 | 87 / 47 | 종료 | — |
| 5080 | Taxi·42·CNN만 | 종료 | 101 / 61 | 종료 | — |
| 5080 | Taxi·42·GRU만 | 종료 | 85 / 45 | 종료 | — |
| 5080 | Taxi·42·CNN+GRU | 종료 | 86 / 46 | 종료 | — |
| 5080 | Taxi·52·보정 없음 | 종료 | 108 / 68 | 종료 | — |
| 5080 | Taxi·52·CNN만 | 종료 | 77 / 37 | 종료 | — |
| 5080 | Taxi·52·GRU만 | 종료 | 122 / 82 | 종료 | — |
| 5080 | Taxi·52·CNN+GRU | 진행 중 | 52 / 35 | 21분 | 3시간48분 |
| 5090 | Intermittent·42·보정 없음 | 진행 중 | 68 / 30 | 10분 | 18시간57분 |
| 5090 | Intermittent·42·CNN만 | 진행 중 | 67 / 28 | 5분 | 19시간25분 |
| 5090 | Intermittent·42·GRU만 | 진행 중 | 32 / 14 | 3시간52분 | 47시간6분 † |

- 새 종료 Taxi52 GRU는2026-10-12 23:28:27 KST 실제exit0, 저장122/best82다. 현재5080은 Taxi52 CNN+GRU52/best35로 진행 중이다. 기존 완료fit 원관측 종료시각도 보존하며 새로운 RTX 관측시각으로 바꾸지 않는다.
- 완료7개는 실제worker exit0·dispatcher completed의 manifest/exit 원문SHA 연결·계약/job·terminal manifest 각13파일 원격SHA(선택/최종binary·endpoint replay 포함)로 검증했다. 원격완료와 로컬immutable 원본회수·최종Validation 집계는 별개다.
- 진행4개 ETA는 각fit 자신의 최근10완료epoch finite 양수 elapsed_seconds 중앙값만 사용했다. best는 최초strict finite 최소 fullValidation 전체 raw 수량RMSE epoch다. 진행fit의 최종selected binary는 아직 생성 전이다. 조기종료 목표 min(300,max(40,best+40)), best 미갱신·속도 유지 조건이며 best/속도/부하 변화시 달라진다. 전체36 큐 ETA 미확정이다.
- †5090 GRU만의300epoch 대안47시간6분는 남은 조건기한30시간23분을 넘는다. 현 기한 내300epoch 완료는 보장되지 않으며 실제timeout 또는 기한연장 승인으로 해석하지 않는다.
- 기존round의 완료0·종료fit 저장 전은 보고 집계 오류가 남은 표시다. 같은 raw snapshot의 종료 증적으로 이 별도 보고서를 보완했다. 원본round·봉인·코드·학습·자동화 설정은 변경하지 않았다.
- helper1회·livehost별 읽기전용SSH1회·추가SSH0·Test0·원본회수0·endpoint 성과 열람0. pipeline wrapper 자체OS exit는 직접 관측하지 않았다. 모든36개 종료 전이므로 자동화를 유지한다.

**종료 집계를 보완하고 원본·논문 근거를 확정한다 — 다음 작업**

- terminal 증적을 반영하는 보고기 수정·검증이 필요하다. 시간별 관측→36개 자연종료 확인→원본회수 범위/계약 확인→immutable archive/manifest·선택/최종binary 로컬SHA 확정→여섯Validation지표·모든seed/대조 평균/표본SD→Results·Methods·주장 정합 순서다.

원본 회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/49e220b2806446f68d9b7beb349ec1b8.json
rawSHA: 3ddba4d79c84b11c1b7b09b5aa90ad106c425ebc6e6d45f9c67752b33de1f4ed
