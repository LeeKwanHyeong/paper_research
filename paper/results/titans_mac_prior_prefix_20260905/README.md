# B1 prior-prefix 출력 read 실행 증적

- 저장소/작업 브랜치: `paper_research/codex/b1-prior-prefix-read`.
- [기계 판정 계약](../../contracts/titans_mac_prior_prefix_v1.json), [설명 계약](../../contracts/titans_mac_prior_prefix_v1.md).
- 현재 논문 T0 기준선은 그대로 유지한다. 이 디렉터리는 한 후보의 별도 연구 증적이다.
- `frozen_references.json`: 기존 T0/RMTPP/THP 3 datasets × 3 seeds = 27 validation 행. 원수치, 출처 summary/hash, checkpoint hash를 보존한다. 이번 fresh B1과 후보의 cap300 계약에 대응한다. 역사적 reference를 새로운 paired rerun으로 표현하지 않는다.
- `local_contract_tests.log`, `local_contract_tests.xml`: 최종 통합 207 passed, CUDA 전용 4 skipped, 실패/오류 0. 실행 명령과 검사 source hash를 로그에 기록했다. 아래 47개·22개 부분 검증을 포함한 총수다.
- `local_execution_tests.xml`, `local_execution_validation.json`: launcher/package/CUDA cost 판정기 단위 테스트 47 passed, 실패/오류/skip 0.
- `local_comparison_tests.xml`: frozen reference·paired seed·지표 판정 및 실패 경계 테스트 22 passed, 실패/오류/skip 0.
- GPU 실행과 성능 평가의 상태는 별도 실행 감사 및 5090 phase proof로 판단한다. 로컬 통과 수치나 합성 benchmark는 성능 개선 증거가 아니다.

## 고정된 판정

1. B1 추가 효과: Instacart body MAE 1% 이상 개선, 전체 dataset body MAE 악화 1% 이하, RMSE 악화 없음, tail 2%/time loss .01 이내.
2. T0 기존 기준: 세 dataset 각각 body MAE 5% 이상 개선, RMSE/tail 악화 2% 이하, time loss .01 이내.
3. 공통 raw RMSE: 세 dataset 각각 T0보다 엄격히 감소하고 전체 MAE 악화 2% 이하.
4. 외부 TPP: RMTPP/THP 각각에 대한 RMSE 우위는 별도 판정하며 1–3 통과와 혼동하지 않는다.

Seed42의 1–3이 모두 통과해야 seeds52/62로 확장한다. e1은 정상 실행·처리 건수·학습 경로·저장/복원만 확인한다. Held-out은 사용하지 않는다.
