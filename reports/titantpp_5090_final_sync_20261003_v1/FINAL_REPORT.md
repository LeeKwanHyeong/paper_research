# 5090 후속 실험 원본 동기화·CPU 감사 완료

작성: 2026-10-03 10:04:11 KST

Instacart 9/9조건의 학습 완료, 원본 동기화, CPU checkpoint 감사를 모두 확인했다. 이번에는 마지막 Deep Renewal seed62의 원본 130파일(동결 source109 포함)·checkpoint 2개를 새로 회수하고 감사했다. 기존 8조건은 이전 감사 결과를 재사용하고 현재 보존 원본·archive·checkpoint SHA를 재확인했다. 총 18개 checkpoint와 18개 selected/last validation replay 기록이 연결된다.

| 모델 | Seed | 종료 / 선택 epoch | 학습 | 원본 | CPU 감사 |
|---|---:|---:|---|---|---|
| Current-only | 42 | 70 / 30 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| All-available | 42 | 70 / 30 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| Deep Renewal | 42 | 53 / 13 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| Current-only | 52 | 55 / 15 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| All-available | 52 | 55 / 15 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| Deep Renewal | 52 | 53 / 13 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| Current-only | 62 | 41 / 1 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| All-available | 62 | 75 / 35 | 완료 | SHA 검증 | 통과 (기존 감사 재사용) |
| Deep Renewal | 62 | 86 / 46 | 완료 | SHA 검증 | 통과 (신규) |

마지막 조건의 선택 epoch46 validation은 MAE 4.014112496, RMSE 5.778649846, 시간 NLL 2.846077006다. 동일 선택 epoch의 세 지표이며 test 결과가 아니다.

CPU 감사는 파일·tensor SHA, strict 모델 로드, optimizer/RNG 복원, source·초기화·선택 epoch·전체 train/validation exposure와 저장된 재평가 기록을 대조했다. 모델 forward, 새 학습, 새 replay, held-out/test 열람은 수행하지 않았다. 원격 읽기 전용 수집만 수행했으며 서버 Runtime과 프로세스는 변경하지 않았다.

Deep Renewal은 원고 비교에서 제외된 상태를 유지한다. 이번 작업은 연구 원본 보존과 감사 완료이며 논문 주장을 바꾸지 않는다. 이전 RunPod의 별도 CPU 감사 미완료 항목이나 신규 A100 routing/placement 실험에는 적용하지 않는다.

- 원계약 canonical SHA: `08f7e9ed402ba74fde099321f039b78de3a11b0bd93cba27492b1a84454f6a6c`
- 동결 source109 closure SHA: `989153f9eea2dac6a790835ab1b7c502fc9c0461829800915a6c602166b6ea92`
- [신규 회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_5090_final_sync_20261003_v1/retrieved/5090/retrieval_receipt.json)
- [신규 CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_5090_final_sync_20261003_v1/retrieved/5090/terminal_audit.json)
- [9조건 원본·감사 연결 목록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_5090_final_sync_20261003_v1/registry.json)
- [조건별 CSV](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_5090_final_sync_20261003_v1/condition_registry.csv)

**남은 작업**: 5090 동기화 범위에는 미완료 항목이 없다. 기존 RunPod의 별도 미완료 CPU 감사는 별도 작업으로 남는다.
