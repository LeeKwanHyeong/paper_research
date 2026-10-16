# CNN·GRU36 시간별 관측 보완 보고

**실제 종료와 현재 checkpoint를 확인한다 — 진행 중**

실제 관측: 2026-10-13 03:20:03 KST. 전체36 중 종료11·진행4·대기21.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 8 / 24 | 3 | 13 |
| 5090 | 3 / 12 | 1 | 8 |

| GPU | dataset·seed·arm | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 종료 | 87 / 47 | 종료 | — |
| 5080 | Taxi·42·CNN만 | 종료 | 101 / 61 | 종료 | — |
| 5080 | Taxi·42·GRU만 | 종료 | 85 / 45 | 종료 | — |
| 5080 | Taxi·42·CNN+GRU | 종료 | 86 / 46 | 종료 | — |
| 5080 | Taxi·52·보정 없음 | 종료 | 108 / 68 | 종료 | — |
| 5080 | Taxi·52·CNN만 | 종료 | 77 / 37 | 종료 | — |
| 5080 | Taxi·52·GRU만 | 종료 | 122 / 82 | 종료 | — |
| 5080 | Taxi·52·CNN+GRU | 종료 | 197 / 157 | 종료 | — |
| 5080 | Taxi·62·보정 없음 | 진행 중 | 72 / 72 | 28분 | 2시간37분 |
| 5080 | Taxi·62·CNN만 | 진행 중 | 72 / 36 | 3분 | 2시간38분 |
| 5080 | Taxi·62·GRU만 | 진행 중 | 35 / 32 | 51분 | 6시간8분 |
| 5090 | Intermittent·42·보정 없음 | 종료 | 70 / 30 | 종료 | — |
| 5090 | Intermittent·42·CNN만 | 종료 | 68 / 28 | 종료 | — |
| 5090 | Intermittent·42·GRU만 | 종료 | 54 / 14 | 종료 | — |
| 5090 | Intermittent·42·CNN+GRU | fit worker 실행 중·첫 저장 전 | 저장 전 / 미확정 | 미확정 | 미확정 |

- 종료는 실제worker exit0·dispatcher completed의 manifest/exit 원문SHA 연결·계약/job·terminal manifest 모든 파일 원격SHA(선택/최종binary·endpoint replay 포함)로 검증했다. 완료fit의 원관측 종료시각은 JSON에 보존했고 새로운 live 관측시각으로 바꾸지 않았다. 원격완료와 로컬immutable 원본회수·최종Validation 집계는 별개다.
- 진행fit ETA는 각fit 자신의 최근 최대10완료epoch finite 양수 elapsed_seconds 중앙값만 사용했다. 표본 수와 실제값은 JSON에 기록했다. best는 최초strict finite 최소 fullValidation 전체 raw 수량RMSE epoch이며, 진행fit의 최종selected binary는 아직 미확정이다. 첫 저장 전 fit은 내부 단계와 ETA도 미확정이며 다른 fit 속도를 차용하지 않는다. 조기종료 목표 min(300,max(40,best+40)), best 미갱신·속도 유지 조건이고 best/속도/동시부하 변화에 따라 달라진다. 미시작fit/다른GPU 속도 복사0, 전체36 큐 ETA 미확정이다.
- 기존round의 완료0·종료fit 저장 전은 보고 집계 오류가 남은 표시다. 같은raw snapshot의 종료 증적으로 이 별도 보고서를 보완했다. 원본round·봉인·코드·학습·자동화 설정은 변경하지 않았다.
- helper1회·livehost별 읽기전용SSH1회·추가SSH0·Test0·원본회수0·endpoint 성과 열람0. pipeline wrapper 자체OS exit는 직접 관측하지 않았다. 모든36개 종료 전이므로 자동화를 유지한다.

**종료 집계를 보완하고 원본·논문 근거를 확정한다 — 다음 작업**

- terminal 증적을 반영하는 보고기 수정·검증이 필요하다. 시간별 관측→36개 자연종료 확인→원본회수 범위/계약 확인→immutable archive/manifest·선택/최종binary 로컬SHA 확정→여섯Validation지표·모든seed/대조 평균/표본SD→Results·Methods·주장 정합 순서다.

원본 회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/ce597c5a226041efa6bd3d05656b2895.json
rawSHA: ea67af5c8a921228805656f45213b0a34006e2c1aedb0fa524fcc869173384c7
