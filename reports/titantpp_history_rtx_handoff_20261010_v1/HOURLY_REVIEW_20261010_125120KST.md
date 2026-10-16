# 이력 보정 RTX 분담 정기관측 — 2026-10-10 12:51:20 KST

5080·5090의 실제 관측시각은 모두 2026-10-10 12:51:20 KST다. Instacart와 종료한 A100은 로컬 원본 증적을 재사용했으며 원관측시각을 변경하지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9, 원본 614개 로컬 SHA 검증; 원격 조회 0 |
| CNN+GRU54 이력 보정 | 진행 중 | 27개 중 완료 16, 진행 5, 대기 6, 실패 0, 미확정 0 |
| 고정 검색 추가 효과 | 대기 | 공통 mean·자원 계약 정리 전 자동 실행 0 |

| GPU | 데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300 epoch 잔여 | epoch 시간 중앙값 |
|---|---|---|---|---|---|---|---|
| 5080 | RAF · 62 | full(이력 보정) | 진행 중 | 32 / 20 | 9.4분 | 1.51시간 | 20.23초 (10 epoch) |
| 5080 | RAF · 62 | 보정 제거 | 진행 중 | 65 / 26 | 10초 | 38.8분 | 9.90초 (10 epoch) |
| 5080 | RAF · 62 | 현재 상태 전용 | 완료 | 62 / 22 | 종료 | 종료 | — |
| 5090 | Intermittent · 42 | full(이력 보정) | 진행 중 | 2 / 1 | 6.70시간 | 51.22시간 | 618.76초 (2 epoch) |
| 5090 | Intermittent · 42 | 보정 제거 | 진행 중 | 5 / 5 | 3.32시간 | 24.47시간 | 298.66초 (5 epoch) |
| 5090 | Intermittent · 42 | 현재 상태 전용 | 진행 중 | 5 / 4 | 3.27시간 | 24.70시간 | 301.48초 (5 epoch) |

5090의 Intermittent seed52·62 × 세 방식 6개는 미시작 대기이며 저장 epoch·ETA는 없다. 전체 큐 ETA는 미확정이다. ETA는 해당 fit·물리 GPU의 최근 최대 10개 완료 epoch만 사용한 조건부 시나리오다. best 미갱신 종료 epoch는 min(300,max(40,best+40))이며, 300 epoch 시나리오와 구분했다. Intermittent의 표본은 2~5 epoch로 적다. 5090 full의 300 epoch 시나리오는 36시간 조건 lease를 초과하므로 허용된 실행시간 예측이나 예산 연장이 아니다. 원본 회수·CPU 감사 시간은 ETA에 포함하지 않았다.

실행 증적: 5080 진행 worker PID 4060/4066, 5090 PID 13255/13261/13267의 정확 argv·PPID·고정 GPU UUID/PID·lease·claim·checkpoint/history/timing을 검증했다. 5090 세 worker의 동시 GPU 점유를 확인했다. qualification/admission은 과학 학습 수량에 포함하지 않았다.

A100 완료 15는 보존 원본을 로컬 검증해 한 번씩 집계했다. 실제 종료 관측은 12:17:18 KST, cached epoch 관측은 12:25:00 KST다. 삭제된 두 Pod와 전용 volume은 다시 조회하지 않았다. 누적 추정 비용은 $32.36242173899348이며 최종 청구는 미확인이다. RAF42 full/current_only의 물리 A100 이관과 cross-seed GPU/Runtime 차이를 유지한다.

새 RTX 두 host의 finalizer는 pending, 원격 회수 0, 사용자 조치 필요 false다. 전체 scientific terminal·actual dispatcher exit·owner 부재가 모두 충족되지 않았으므로 자동화를 유지한다. pipeline OS exit코드는 직접 관측하지 않았으며 exit0으로 표기하지 않는다.

성능 판정: 현재 관측은 진행 증적이며 최종 Q1 판정은 대기다. 동일 최초 최소 fullValidation 전체 수량 RMSE 선택 epoch의 전체/큰 수량 RMSE·MAE·TimeNLL과 3seed 평균/표본 SD를 모두 보존한다. 큰 수량 또는 시간 손해는 제외하지 않는다. Test·held-out·Validation/Test 혼합 성과는 열지 않았다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.

## 증적 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791604280628062000.json`
- 관측 묶음 raw SHA256: `3f54389e0086772776744fe8c44948b81e9dff9bdd9d48a3e8b479d01ae853ba`
- 5080 snapshot raw SHA256: `7192632d4af5de660bad28e0b61852f4c18ee68d9133ac2a501c0cfabb34fffb`
- 5090 snapshot raw SHA256: `07914b25cc5e45a78108b36b3fdc7e60e7cc7d83ed8da2f805462a6da4983e48`
- 도우미 실행 1회, live host별 읽기 전용 SSH 1회, 재배포·재시작·추가 관측·Pod API 0.
