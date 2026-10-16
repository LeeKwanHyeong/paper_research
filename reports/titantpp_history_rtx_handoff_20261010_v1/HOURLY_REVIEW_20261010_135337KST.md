# CNN+GRU54 이력 보정 정기관측 — 2026-10-10 13:53 KST

5080 실제 관측: **2026-10-10 13:53:36 KST**. 5090 실제 관측: **2026-10-10 13:53:37 KST**. A100과 Instacart는 로컬 보존 증거를 재사용하며 원관측시각을 바꾸지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9, 원본 614개 로컬 SHA 검증; 기존 observer/SSH/재학습0 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료18·진행3·대기6·실패0·미확정0. A10015와 RTX5080 RAF62 세 조건 완료 |
| 고정 검색 | 대기 | 공통 mean·자원 계약 및 논문 포함 여부 정리 전 자동 실행0 |

| GPU / 데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300epoch 잔여 | 완료 epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080 / RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080 / RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080 / RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 이력 보정 full | 진행 중 | 8/4 | 6.22시간 (종료 E44) | 50.43시간 | 621.68초 (8epoch) |
| 5090 / Intermittent·42 | 보정 제거 | 진행 중 | 18/16 | 3.07시간 (종료 E56) | 22.79시간 | 290.97초 (10epoch) |
| 5090 / Intermittent·42 | 현재 상태 전용 | 진행 중 | 18/6 | 2.33시간 (종료 E46) | 23.43시간 | 299.10초 (10epoch) |

5090의 Intermittent seed52/62 × 세 arm6개는 미시작 대기이며 저장/best epoch와 ETA는 없다. 전체 큐 ETA는 미확정이다. ETA는 각 fit·물리 GPU 자체의 최근 최대10 완료 epoch finite elapsed_seconds 중앙값으로 계산했다. best 미갱신 및 현재 속도 유지 시 min(300,max(40,best+40)) 종료와300epoch 시나리오를 분리했다. 5090 full의300epoch 약50.43시간은 조건36시간 lease를 초과하는 시나리오이며 실행 한도·예산을 늘리지 않는다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행 시간으로 구조 효율을 단정하지 않는다.

## 종료와 원본 보존

- 5080은3/3 scientific terminal, 실제 worker·supervised dispatcher exit0, pipeline_complete 및 마지막 phase complete, actual owner/GPU PID부재와 qualification/admission 연결 검증을 통과했다. pipeline OS exit코드는 직접 관측하지 않았다는 구분을 유지한다.
- 5080 immutable 원본 회수는 기승인 finalizer 내부에서1회 수행했고 manifest/archive 및 binary 원본 SHA 검증을 통과했다. receipt raw SHA: `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd`. 전체27의 최종 Validation 집계는 아직 pending이다.
- 5090은PID13255/13261/13267 세 worker가 실제 동시 실행 중이다. 정확 argv·PPID·GPU UUID/PID·lease·claim·checkpoint/history/timing을 검증했다. live이므로 원격 원본 회수0이다.
- 이번 도우미 실행은1회, 양host 읽기 전용 관측 SSH각1회다. 승인 원본 보존은 관측과 구분된 finalizer remote2이며, 추가 deploy/qualification/launch/observer/SSH/retry/resume/kill/claim삭제0이다.
- A10015는 보존된 완료 원본을 한 번씩 집계했다. 실제 종료 원관측12:17:18 KST 및 cached epoch 원관측12:25:00 KST를 유지한다. 의도된 행정 인계 FileExistsError·실제 dispatcher/pipeline exit1·failure/claim은 수치 실패로 바꾸지 않는다.
- 삭제된 A100 Pod/volume의 로컬 cleanup receipt와 parent393 원본을 검증했다. 삭제된 두 Pod의 SSH/API재조회0이며 비용은 누적 추정$32.36242173899348, 최종 청구 미확인이다.

## 해석과 다음 순서

현재는 진행·종료 증거 보고다. 전체27 원본과 종료 확정 후 동일 최초 strict finite 최소 fullValidation 전체 raw수량RMSE 선택 epoch에서 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표,3seed 평균/표본SD 및Q1을 집계한다. 수량 개선과 함께 큰 수량·시간 손해를 보존하며 현재 완성된 일부 조건만으로 전체 성공을 확정하지 않는다. RAF42 full/current_only의 물리A100 이관과 cross-seed GPU/Runtime 차이를 명시하고 host 시간을 혼합하지 않는다.

이력 보정27이 끝나면 원본·종료 확정 → Q1/여섯 지표 판정 → CNN+GRU54 중심 Results 및 Instacart/Methods/주장 정합 순서다. 이후 검색 포함 여부와 공통 계약, 미노출 평가·CPU 감사 계약, 관련연구/References/Appendix, 컴파일·투고 점검을 진행한다. 고정 검색은 계속 대기한다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다. Test/held-out/mixed 성과는 열지 않았다.

현재 사용자 조치 필요false다. 전체 완료 조건 미충족이므로 titantpp-instacart-63을 유지한다.

## 관측 원본 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791608018912577000.json`
- 묶음 raw SHA256: `e6c67f04fc0858368b13832b905592d09fa24393ded57d2a819374ea0a32bede`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `84297da7ca048232b9bb4d25f1f69f7afcf16399d39881acc3457296c5d5fd35`
