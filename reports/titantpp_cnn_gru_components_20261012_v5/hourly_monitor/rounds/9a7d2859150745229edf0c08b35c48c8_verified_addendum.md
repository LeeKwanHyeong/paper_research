# CNN·GRU36 시간별 관측 보완 보고

**원격 종료·진행을 원문으로 확인한다 — 진행 중**

실제 관측시각: 2026-10-12 20:12:10 KST. 고유36 ID 중 원격 종료3·진행4·대기29.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 3 / 24 | 1 | 20 |
| 5090 | 0 / 12 | 3 | 9 |

| GPU | dataset·seed·구조 | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 원격 종료 | 87 / 47 | 종료 | 종료 |
| 5080 | Taxi·42·CNN만 | 원격 종료 | 101 / 61 | 종료 | 종료 |
| 5080 | Taxi·42·GRU만 | 원격 종료 | 85 / 45 | 종료 | 종료 |
| 5080 | Taxi·42·CNN+GRU | 진행 중 | 7 / 4 | 33분 | 4시간 23분 |
| 5090 | Intermittent·42·보정 없음 | 진행 중 | 18 / 16 | 3시간 7분 | 23시간 7분 |
| 5090 | Intermittent·42·CNN만 | 진행 중 | 18 / 10 | 2시간 41분 | 23시간 35분 |
| 5090 | Intermittent·42·GRU만 | 진행 중 | 8 / 5 | 6시간 28분 | 51시간 4분 † |

- 원격 종료3개는 dispatcher 완료ID·실제 worker exit0·terminal manifest의 모든13개 파일 SHA(선택/최종 binary·endpoint replay 포함)로 별도 검증했다. 저장/best는 manifest에 연결된 checkpoint receipt/history의 첫 strict finite 최소 fullValidation raw 수량RMSE epoch다. endpoint 성과 수치는 열지 않았다. 로컬 binary 원본 회수는 미완료다.
- 원래 report의 완료0/세 fit 저장 전 표시는 종료한 fit의 exposure.json이 live checkpoint receipt 이후 변경된 것을 전체 checkpoint 미확정으로 취급하여 terminal 근거를 집계하지 못한 보고기 오류다. 원본 round/봉인/과학 코드/학습은 변경하지 않았다.
- 진행4fit의 최종 selected binary는 아직 생성 전이다. ETA 표본은5080 FULL7,5090 NONE10/CNN10/GRU8 완료epoch이며 fit별 최근 최대10 finite 양수 elapsed_seconds 중앙값만 사용한다. 조기종료 목표 min(300,max(40,best+40))이며 관측시점 기준 조건부 예상이다. 전체36 큐 ETA 미확정이다.
- †5090 GRU의300epoch 시나리오51시간4분은 실제 남은 조건기한34시간28분을 넘는다. 현 계약 기한 내300epoch 완료를 보장하지 않으며 기한 연장/학습 실패로 해석하지 않는다.
- 이번 helper1회, host별 읽기전용SSH1회; 추가SSH0·학습변경0·Test0·원본회수0. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다. 자동화는 유지한다.

**보고 집계를 바로잡고 종료·원본·논문 근거를 확정한다 — 다음 작업**

- 종료 증적을 우선하는 보고기 집계 수정·검증이 필요하다. 이후36개 자연종료 관측→기존 승인범위와 별도 회수계약 확인→immutable 원본 archive/manifest·선택/최종binary 로컬SHA 확정→여섯Validation지표·모든seed/대조의 평균/표본SD→Results·Methods·주장 정합 순서다. 새 학습/검색/평가는 추가하지 않는다.

원본회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/9a7d2859150745229edf0c08b35c48c8.json
원본회차 rawSHA: 5e636274b731b7cb60c26b7642f24c741cc9fe67abe2b5257452a6708f84918e
