# 학습 진행 상태 — 2026-10-01 08:59 KST

5080 RAF는 08:55:56에 24/24조건의 학습과 selected/last validation 재평가를 마쳤다. 이번 조회에서 소유 프로세스와 GPU compute process가 없음을 확인했다. 원본 checkpoint·source 회수와 최종 CPU 감사는 남아 있다.

5090 Instacart 추가 TPP는 4/6조건 완료다. seed42·52의 S2P2/AttNHP는 완료됐고, S2P2 seed62가 19epoch(현재 best10)까지 저장됐으며 이후 AttNHP seed62가 대기한다. 소유 worker와 GPU 및 실제 배치 진행을 확인했고 신규 오류는 없다.

현재 best가 갱신되지 않으면 S2P2는 50epoch에 종료하며, 최근 246.62초/epoch 기준 약 2시간 7분 뒤인 11:06경이다. 마지막 AttNHP가 기존 seed의 51~112epoch와 실측 속도에 따른 시나리오로 끝날 경우 전체 학습은 13:33~16:22경이다. 이 범위는 보장이나 최대시간이 아니며 best 갱신과 실행 속도에 따라 늦어질 수 있다. 재평가·회수·최종감사 시간은 별도다.

- [5080 관측](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_raf_5080_20260930_v1/monitor/20260930T235907531118Z/snapshot.json)
- [5090 비교 기록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20260930T235903944489Z/comparison.json)

다음 작업은 RAF 원본 회수·CPU 감사이며, 5090의 마지막 두 조건은 기존 큐에서 계속 진행한다. 새 학습·재시작·스케줄러 생성은 하지 않았다.
