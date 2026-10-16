# 이관 이후 시간별 관측

활성 자동화는 기존 `titantpp-54` 하나를 유지한다. 사용자가 요청한 시간별 보고를 유지하며 이관 후 현재 대상은 `search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/current.json`으로 확인한다. 부모 포인터와 원래 5080 terminal cache는 변경하지 않는다.

각 회차에 프로젝트 root에서 아래 두 명령을 독립적으로 각각 한 번 실행한다.

```bash
/usr/local/bin/python3 search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/control/monitor.py --host 5080
/usr/local/bin/python3 search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/control/monitor.py --host 5090
```

5080은 새 소유 seed62 여섯 조건을 관측하고, 원래 완료36 cache는 실제 원 관측시각을 보존해 재사용한다. 5090은 기존 root의 seed42/52 소유12조건을 관측한다. 예약6 claim의 원본 SHA와 source run 부재를 확인한 뒤에만 이관된 조건을 제외한다. 각 live 서버에 SSH1회이며 verified terminal cache는 SSH0이다. 조회 실패는 이번 조회 실패로 보고하며 학습 실패로 세지 않는다.

실제 supervisor/timeout/worker argv·PID·PPID, GPU UUID/PID, 과학/운영 SHA, checkpoint receipt/history/timing/exposure, 실제 저장/best epoch를 대조한다. 저장 전 상태를 과거 진척과 혼동하지 않는다. 조건별 자체 완료 epoch 최근 최대10개 중앙값으로 best 미갱신·현재 속도 유지 가정의 종료 epoch `min(300,max(40,best+40))`와 최대300 시나리오를 나눈다. 미시작 속도와 전체 큐 ETA는 미확정으로 둔다. 마감·endpoint/retrieval/CPU 감사 시간은 ETA와 별도다.

두 관측 이후 다음 명령을 정확히 한 번 실행한다.

```bash
/usr/local/bin/python3 search_artifacts/titantpp_architecture_capacity_handoff_20261006_v1/control/finalize_once.py
```

54 scientific terminal·전체 Validation replay·원본 SHA·세 소유 root의 실제 종료가 검증되지 않으면 로컬 pending만 남긴다. 검증된 뒤에만 기승인 원본 회수·같은 선택 epoch의 Validation 집계를 한 번 수행한다. exclusive lock/claim·실패를 보존하고 자동 retry를 하지 않는다. 원래 5090의 정확한 예약 경계 종료만 행정 종료로 인정한다.

원래 5080의 원 관측시각, 원래 5090의 실제 관측시각, 새 5080의 실제 관측시각을 구분하고 각 canonical ID를 한 번 집계한다. 모든54조건·세 root 종료·원본 manifest/archive SHA·최종 Validation 보고까지 완료된 뒤 이 자동화만 삭제한다. Test·CPU binary 재추론 감사·독립 평가가 별도 미완료임을 명시한다. 다른 자동화·PAUSED monitor·Pod·다른 서버 작업·chat archive는 변경하지 않는다.
