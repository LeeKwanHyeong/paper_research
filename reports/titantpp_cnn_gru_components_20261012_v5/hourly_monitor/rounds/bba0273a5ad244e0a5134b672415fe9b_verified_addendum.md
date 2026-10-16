# CNN·GRU36 시간별 관측 보완 보고

**실제 종료와 현재 checkpoint를 확인한다 — 진행 중**

실제 관측: 2026-10-12 23:17:55 KST. 전체36 중 종료6·진행4·대기26.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 6 / 24 | 1 | 17 |
| 5090 | 0 / 12 | 3 | 9 |

| GPU | dataset·seed·arm | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 종료 | 87 / 47 | 종료 | — |
| 5080 | Taxi·42·CNN만 | 종료 | 101 / 61 | 종료 | — |
| 5080 | Taxi·42·GRU만 | 종료 | 85 / 45 | 종료 | — |
| 5080 | Taxi·42·CNN+GRU | 종료 | 86 / 46 | 종료 | — |
| 5080 | Taxi·52·보정 없음 | 종료 | 108 / 68 | 종료 | — |
| 5080 | Taxi·52·CNN만 | 종료 | 77 / 37 | 종료 | — |
| 5080 | Taxi·52·GRU만 | 진행 중 | 109 / 82 | 11분 | 2시간40분 |
| 5090 | Intermittent·42·보정 없음 | 진행 중 | 56 / 30 | 1시간10분 | 20시간24분 |
| 5090 | Intermittent·42·CNN만 | 진행 중 | 55 / 28 | 1시간6분 | 20시간44분 |
| 5090 | Intermittent·42·GRU만 | 진행 중 | 26 / 14 | 4시간55분 | 48시간6분 † |

- 새 종료: Taxi52 CNN은22:16:49 KST 실제exit0, 저장77/best37; 보정 없음은22:29:46 KST 실제exit0, 저장108/best68이다. GRU109/best82는 진행 중이고 CNN+GRU는 GRU 자연종료 뒤 시작하도록 대기한다. 이전 Taxi42 네 종료의 원관측시각도 보존했다.
- 완료6개는 실제 worker exit0·dispatcher completed 원문SHA 연결·계약과 job·terminal manifest 각13파일 원격SHA(선택/최종binary·endpoint replay 포함)로 검증했다. 원격 종료와 로컬 원본 회수·최종 Validation 집계는 별개다.
- 진행4개 ETA는 각fit 최근10개 완료epoch finite 양수 elapsed_seconds 중앙값만 사용했다. best는 최초strict finite 최소 fullValidation 전체 raw 수량RMSE epoch다. 진행fit의 최종 selected binary는 아직 생성 전이다. 조기종료 목표 min(300,max(40,best+40)), best 미갱신·속도 유지 조건이며 부하/속도/best 변화에 따라 달라진다. 전체36 큐 ETA는 미확정이다.
- †5090 GRU만의300epoch 대안48시간6분는 남은 조건기한31시간22분을 넘는다. 현 기한 내300epoch 완료는 보장되지 않으며 실제timeout 또는 기한연장 승인으로 해석하지 않는다.
- 기존 round의 완료0·종료fit 저장 전은 보고 집계 오류가 남은 표시다. 동일 raw snapshot 종료 증적으로 이 별도 보고서를 보완했다. 원본round·봉인·코드·학습·자동화 설정은 변경하지 않았다.
- helper1회·live host별 읽기전용SSH1회·추가SSH0·Test0·원본회수0·endpoint 성과 열람0. pipeline wrapper 자체OS exit는 직접 관측하지 않았다. 전체 종료 전이므로 자동화를 유지한다.

**종료 집계를 보완하고 원본과 논문 근거를 확정한다 — 다음 작업**

- terminal 증적을 반영하는 보고기 수정·검증이 필요하다. 시간별 관측→36개 자연종료 확인→원본회수 범위/계약 확인→immutable archive/manifest·선택/최종binary 로컬SHA 확정→여섯Validation지표·모든seed/대조 평균/표본SD→Results·Methods·주장 정합 순서다.

원본 회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/bba0273a5ad244e0a5134b672415fe9b.json
rawSHA: 7061b14a2a53c30af0a9dd262278fbd910a62f9b38babfc1da064f77013e8057
