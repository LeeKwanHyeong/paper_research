# TitanTPP 5080·5090·A100 시간별 관측

**세 GPU를 하나의 시간별 모니터로 연결한다 — 완료**

- 사용자 요청: “5080/5090/A100 한시간 스케쥴러 만들어두자”. 기존 A100 heartbeat를 수정했습니다. 별도 중복 모니터를 생성하지 않았습니다.
- 자동화 ID: `titantpp-cnn-gru-a100-seed42`; 표시명: **TitanTPP 5080·5090·A100 시간별 모니터**; 상태: ACTIVE; 간격: 1시간.
- 매회 각 대상의 실제 KST 관측시각, 완료·실패·진행·대기·미확정, 데이터·모델·seed, 저장/best epoch, 조건부 ETA를 짧은 한국어 표로 알립니다.

| 대상 | 캠페인 | 조건 수 | 병렬 수 |
|---|---|---:|---:|
| 5080 | 원래 Taxi·RAF12 완료 증거 재사용 + Intermittent seed62 폭8/12 | 과거12 + 이관2 | 1 |
| 5090 | Intermittent × MLP 폭8/12 × seed42/52 | 4 | 1 |
| A100 | Taxi·Intermittent·RAF × CNN+MLP16/GRU54/CNN+GRU54 × seed42 | 9 | 최대3 |

**실행 계약과 실제 저장 증거를 읽는다 — 진행 중**

- 폭8/12 bundle: `search_artifacts/titantpp_history_width8_12_dual_20261004_v1`; 계약 `8434041a6a301995a8d22baa77d5ca0da4eb2cd942d091a8aa8cef379406c48d`; 동결114소스 `41a4a4ff79e5e17d907b9ee8966723004516ee2e42608d31c7f1a871f0acd3e1`.
- A100 bundle: `search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1`; 소유 Pod `okbl76hxfr7a9d`; 계약 `46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae`; 동결117소스 `40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd`; worker4 `7d961867f612f86e1b78346630764d7624472532f596aba133a3bf3d24da2f26`.
- current·계약·과학소스·소유 Pod가 달라지면 해당 캠페인의 과거 대상으로 접속하지 않습니다. 한 서버의 실패는 다른 서버 관측을 막지 않습니다.

프로젝트 root `/Users/igwanhyeong/PycharmProjects/paper_research`에서 다음 명령을 회차마다 각각 한 번 실행합니다.

```sh
/usr/local/bin/python3 paper/scripts/observe_titantpp_capacity_handoff_hourly.py --host 5080
/usr/local/bin/python3 paper/scripts/observe_titantpp_capacity_handoff_hourly.py --host 5090
/usr/local/bin/python3 search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/control/hourly_once.py
```

- 미종료 대상마다 읽기 전용 SSH 한 번을 수행합니다. 별도 observer·SSH·manager 호출 또는 반복 polling을 하지 않습니다.
- 소유 supervisor/worker 명령·PID와 GPU UUID/PID, 실제 저장 epoch와 소형 기록 SHA·terminal·selected/last Validation replay를 대조합니다. 서버가 선언한 완료와 검증한 완료를 구분합니다.
- 완료가 검증된 5080/5090, 삭제가 확인된 A100은 보존된 terminal 관측을 재사용하고 원래 관측시각을 유지합니다. 조회 실패를 학습 실패로 간주하거나 과거 결과를 새 관측으로 표시하지 않습니다.
- 최근 최대10개 실제 epoch 중앙값과 best 미갱신 시 `min(300,max(40,best_epoch+40))`으로 조건부 ETA를 표시합니다. 미시작 모델/대기시간이 불확실하면 전체 큐 ETA는 미확정입니다. 종료 마감은 ETA가 아닙니다.
- 학습 관측은 Validation 전용입니다. 사용자가 승인한 Intermittent6 완료 후 Test 후속 처리는 아래 별도 실행기를 통해 수행합니다. A100 Test는 접근하지 않습니다. 원본 binary SHA 회수와 CPU 재추론 감사는 구분합니다.

**미시작 seed62 두 조건의 소유권을 5080에 연결한다 — 완료**

- 사용자2026-10-04 승인에 따라 미시작 Intermittent seed62 폭8·12만 이관했습니다. 원래5090 실행 중 seed42와 다음seed52는 계속 유지합니다.
- [이관 계약과 실행 기록](../../search_artifacts/titantpp_history_capacity_handoff_20261004_v1/README.md)의 운영 계약 SHA는 `e98a59a9f32c04be44f61eba0653920a564b6696ed0be134cb9a07b518603f80`입니다. 부모 계약·114개 과학 소스·자료 SHA·학습·손실·선택 기준·36h/168h 마감은 유지합니다.
- 5080 자체 native qualification,5090 두 O_EXCL 예약 claim과 reservation SHA,5080 training permit을 모두 확인한 뒤 별도 tmux에 연결했습니다. 실제 저장 epoch의 확인 여부는 관측 기록으로 구분합니다.
- 5090이 소유한 네 조건을 마친 뒤 예약 첫claim에서 종료하는 정확한 `FileExistsError`만 의도한 이관 경계로 처리합니다. 다른 실패나 원래seed62 run 생성은 문제로 보고합니다. 원래 failure/status는 보존합니다.
- 원래5080 완료12 + 새5080두 + 5090네 =18조건이며 과학 조건은 추가하지 않았습니다. 다른 GPU의 학습 속도를 복사하거나 같은 조건을 두 번 집계하지 않습니다.

**Intermittent 여섯 조건이 모두 완료되면 동결된 Test를 실행한다 — 외부 작업 대기**

- [승인 기록](../titantpp_capacity_test_handoff_20261004_v1/approval.json)에 따라 매회 관측 이후 `/usr/local/bin/python3 paper/scripts/finalize_titantpp_capacity_test.py`를 한 번 호출합니다. 미완료 상태에서는 로컬 pending만 기록하며 원격 평가를 시작하지 않습니다.
- 새5080두·5090네의 terminal과 SHA를 검증·회수하고, 동일 선택 checkpoint로 full Validation6을 먼저 확인합니다. 모두 통과한 뒤에만 원래 Test6을 평가합니다. 전체·큰 수량 RMSE·MAE·Time NLL을 seed별로 기록하고 raw prediction은 쓰지 않습니다.
- 이 호출은 관측기 oneSSH 규칙과 구분된 승인된 원본 회수·평가 작업입니다. 실행/배치 lock과 receipt로 중복 시작·자동 retry를 막으며 별도 평가의 timeout·메모리 상한을 따릅니다. 기존 Test 분할 재평가이며 새 독립 자료나 Test에 의한 checkpoint/폭 선택으로 표현하지 않습니다.

**마지막 대상과 원본 회수를 확인하고 모니터를 종료한다 — 다음 작업**

- 종료된 서버의 반복 접속을 끝내고 실행 중인 대상만 계속 확인합니다.
- 폭8/12의18조건과 A1009조건·서버 terminal, A100 원본 SHA 회수·비용보고·Pod 삭제/목록부재와 승인된 Intermittent6 Test 후속 처리 완료까지 모두 확인한 뒤 최종 관측 보고서를 남기고 이 통합 모니터만 삭제합니다.
- 미확정 상태·회수·비용 문제가 있으면 모니터를 유지하고 원인과 필요한 조치를 알립니다. 자동 재시작·재개·임대·프로세스 종료·Pod 삭제를 하지 않습니다.
- A100 총상한 $100, 실제 작업비 상한 $33.0050935159·회수 여유20·자동충전 없음과 로컬 마감 **10/5 15:17:45 KST**를 유지합니다. provider stop 요청 **16:17:45 KST**의 적용/readback은 미확인입니다. 기존 소유 manager의 실제 PID/명령과 최근 기록을 대조합니다.
- 기존 PAUSED 모니터와 다른 자동화·채팅 archive 상태는 변경하지 않았습니다.

저장된 전체 지침은 [모니터 프롬프트](scheduler_prompt.txt), 실제 앱 설정은 [생성·갱신 기록](scheduler_receipt.json)에 있습니다. A100의 과거 단독 설정은 이전 증적으로 보존하며 이 통합 지침이 우선합니다.
