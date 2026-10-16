# 5080 실제 학습 시작 관측

**프로세스 판정 수정과 실제 검증을 마친다 — 완료**

- 로컬 고유55개 검증, 실제 Train/Validation 입력계약24호출과 독립 QA가 통과했다. 과학 source131·학습/선택조건·최초168시간 절대기한은 유지했다.
- 5080의 native 네 구조 검증, 동시3합성worker 및 supervised qualification campaign의 실제 종료0와 GPU idle을 확인했다. pipeline wrapper 자체OS exit를 직접 관측했다고 주장하지 않는다.
- v2/v3/v4의 실패 원본·claim/실제exit를 보존했다. v4에서 정상 종료한 CNN PID가 proc에서 사라진 뒤 NVML에 남는 관측 지연을 확인했고, 현재 정확한 소유기록이 있는 PID만 최대5초·25회로 재조회한다. 외부/미확인/재사용PID는 거부한다.

**5080 Taxi·RAF24개 학습을 이어간다 — 진행 중**

- train phase 시작 2026-10-12 18:28:26 KST, 실제 관측 2026-10-12 18:29:34 KST.
- 실제 supervised dispatcher PID41374, 세 GPU worker의 정확한 argv·PID/PPID·고정 GPU UUID·lease·claim을 확인했다.

| 대상 | 상태 | 실제 GPU PID | 저장 epoch | history 기준 best |
|---|---|---:|---:|---:|
| Taxi seed42 보정 없음 | 진행 중 | 41390 | 1 | 1 |
| Taxi seed42 CNN만 | 진행 중 | 41392 | 1 | 1 |
| Taxi seed42 GRU만 | 진행 중 | 41394 | 저장 전 | 미확정 |

- 초기 관측의 best는 저장 history에서 계산한 최초strict finite 최소 fullValidation raw수량RMSE epoch다. 최종 selected binary는 아직 없으며 종료 후 검증한다. 학습 완료는0/24이다.
- 현재 세 arm 모두 종료한 뒤 FULL1개를 실행하고, Taxi52→62→RAF42→52→62 순서로 같은3+1 wave를 수행한다. 전체큐 ETA는 미확정이다. 초기 확인 뒤 지속 polling은 하지 않는다.

**5090 Intermittent12개를 시작한다 — 외부 작업 대기**

- 사용자 별도 시작 지시 전SSH·건강조회·배포·qualification·학습 모두0이다. 별도host 승인과own native gate를 요구한다.

**모든 종료·원본·여섯Validation 지표를 확인한다 — 다음 작업**

- 고유36ID·실제worker/dispatcher exit0·마지막phasecomplete·GPU/PID부재·모든selected/last binary·manifest/archive·로컬SHA 확인 후모든seed/대조·평균/표본SD를 집계한다.
- 이후구성요소근거를판정하고 CNN+GRU54 원고와필요한최소개선을연결한다. Test·검색·새loss/head실험은 별도범위다. 열린main.tex는 이번운영수정에서편집하지않았다.

**세 초기학습 모두 첫 저장을 확인한다 — 완료 / 이후큐는 진행 중**

- 마지막 초기 관측은 2026-10-12 18:32:30 KST이다. 동일세GPU PID·argv·lease·claim과각조건checkpoint/history/timing SHA를 확인했고 지속조회는 종료한다.

| Taxi seed42 arm | 저장 epoch | history 기준 best |
|---|---:|---:|
| 보정 없음 | 5 | 4 |
| CNN만 | 5 | 4 |
| GRU만 | 2 | 2 |

- 최종선택 binary는종료후생성·검증한다. 현재학습완료는0/24이고FULL은현wave종료대기다. 관측시각을새시각으로변경하지않는다.
