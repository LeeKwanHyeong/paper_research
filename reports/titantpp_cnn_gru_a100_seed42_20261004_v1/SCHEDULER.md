> 2026-10-04 최신 요청으로 이 heartbeat를 **5080·5090·A100 통합 시간별 모니터**로 확장했습니다. 현재 실행 지침과 전체 종료 조건은 [통합 SCHEDULER](../titantpp_three_gpu_hourly_monitor_20261004_v1/SCHEDULER.md)를 따릅니다. 아래는 A100 단독 설정 당시의 원본 이력입니다.

# CNN·GRU A100 seed42 시간별 관측

**새 캠페인에 시간별 모니터를 연결한다 — 완료**

- 사용자의 A100 시간별 진행·종료 예상 보고 요청을 적용했습니다. 자동화 ID는 `titantpp-cnn-gru-a100-seed42`이고 ACTIVE입니다.
- 기존 종료한 모니터를 재생성하거나 다른 PAUSED 자동화를 변경하지 않았습니다.
- 소유 Pod `okbl76hxfr7a9d`, 계약 `46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae`, 과학117소스 `40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd`, worker4 `7d961867f612f86e1b78346630764d7624472532f596aba133a3bf3d24da2f26`가 대상입니다.
- Taxi·Intermittent·RAF × CNN+MLP16·GRU54·CNN+GRU54 × seed42, 총9조건입니다. 기존 MLP16은 재사용하며 Instacart는 후속입니다.

**실제 실행 증거를 한 번만 관측한다 — 진행 중**

프로젝트 root `/Users/igwanhyeong/PycharmProjects/paper_research`에서 아래 기존 관측기를 회차당 한 번만 실행합니다.

```sh
/usr/local/bin/python3 search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/control/hourly_once.py
```

- 계약·current·소유 Pod를 먼저 확인합니다. 변경 시 이전 대상으로 접속하지 않습니다.
- 읽기 전용 SSH 한 번과 저장된 manager 기록·소유 PID 조회를 수행합니다. 별도 SSH·observe.py·manage.py 호출과 자동 재시작·재개·임대는 하지 않습니다.
- 실제 KST 관측시각, 9조건 완료·실패·진행·대기·미확정, GPU 작업·병렬 수와 저장/best epoch를 한국어 표로 알립니다. 과거 snapshot을 새 관측으로 바꾸지 않습니다.
- 완료는 terminal scientific_success·소형 기록 SHA·selected/last 전체 Validation replay를 대조한 경우입니다. 원본 회수·CPU binary 감사는 별도입니다. Test 성능·예측을 열지 않습니다.
- ETA는 각 fit 자체 최근 최대10개 epoch 중앙값과 best 미갱신 시 `min(300,max(40,best_epoch+40))`을 사용합니다. 미시작 모델의 자체 실측/대기시간이 없으면 전체 큐 ETA는 미확정입니다. best와 병렬 수가 바뀌면 ETA도 달라집니다.

**원본·비용·종료를 확인한 뒤 모니터를 끝낸다 — 다음 작업**

- 총상한 $100(작업80·회수20), 자동충전 없음입니다. 현재 잔액에서 회수20을 남겨 실제 작업상한은 $33.0050935159입니다. 계정 잔액은 이 Pod 청구서가 아닙니다.
- 보수요율 $1.5969444444/h, 실제 로컬 마감 2026-10-05 15:17:45 KST를 유지합니다. provider stop 요청 16:17:45 KST의 적용/readback은 미확인입니다. 마감은 ETA가 아닙니다.
- 기존 소유 manager가 원본 SHA 회수 후 이 Pod만 stop/delete하고 목록 부재를 확인합니다. Mac 절전·통신 장애 중 과금 종료는 보장되지 않습니다. manager 부재·회수/비용 문제는 사용자 조치 필요로 보고하며 heartbeat가 manager를 재시작하지 않습니다.
- 9조건과 서버 terminal, 원본 SHA 회수·비용보고·Pod 삭제/목록부재를 모두 확인한 뒤 최종 관측 보고서를 남기고 이 자동화만 삭제합니다. CPU binary 감사가 남으면 따로 미완료로 표시합니다.

전체 저장 프롬프트는 `scheduler_prompt.txt`, 실제 생성 receipt는 `scheduler_receipt.json`에 보존했습니다.
