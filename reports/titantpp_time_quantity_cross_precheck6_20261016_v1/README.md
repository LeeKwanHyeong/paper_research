# 시간·수량 Cross Attention pre-check 실행 기록

## 실제 학습 시작 — 완료

`EXPERIMENT_PLAN.md`의 두 신규 후보 × RAF·Taxi·Intermittent × seed42, 총6개를 시작했다.
5080/5090 각각3병렬이며 시작 관측에서는 완료0·진행6이다.

- 구현: paper_research 저장소 `codex/contextual-retrieval-pilot-20261007`, 코드 `743691743837c0b1b66460923565c64f09659982`.
- `LOCAL_CHECKS.json`: 로컬 합성·학습 연결25개 검사 통과.
- `PREPARED.json`: 고정 실행 계약 및 source snapshot.
- `DISPATCHED_5080.json`, `DISPATCHED_5090.json`: native GPU/Runtime/full-batch 검증과 launch 응답.
- `START_VERIFIED.json`, `observations/`: 실제 worker와 GPU 점유 확인. 시작 관측을 후속 현재 상태로 재사용하지 않는다.
- 원본 bundle: `search_artifacts/titantpp_time_quantity_cross_precheck6_20261016_v1`.
- Notion: https://app.notion.com/p/3fabbe40561380e591e9e66ff02bb190 의 같은 제목 기록.

## 학습 종료와 Validation 비교 — 다음 작업

새 scheduler는 없다. 사용자 요청 시 아래 도구로 각 host를 한 번 관측한다.
이미 시작한 root에서 prepare 또는 deploy-start를 다시 실행하지 않는다.
실패/timeout이면 원문을 확인하며 자동 재시작하지 않는다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_cross_fusion_precheck --mode status --host 5080
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_cross_fusion_precheck --mode status --host 5090
```

각 host의 final_receipt complete 및 scientific worker3·dispatcher actual exit0 후에만
해당 host의 원본을 한 번 회수한다. 미완료 상태에서 collect를 호출하지 않는다.
회수 archive/manifest와 원본 checkpoint·summary SHA를 확인한 뒤 저장된 Validation을 집계한다.
종료 후 도구의 실제 회수·최종 집계 실행은 아직 수행하지 않았으며 그 단계에서 결과를 확인한다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_cross_fusion_precheck --mode collect --host 5080
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_cross_fusion_precheck --mode collect --host 5090
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_cross_fusion_precheck --mode compare
```

기존 CNN3·이력 MLP16의 seed42 저장 Validation 참조는 bundle에 별도로 보존했다.
재학습·재추론은 하지 않는다. Test0, 1-seed, Intermittent의 GPU/cuDNN 차이 및
전단 메모리 미검증 범위를 유지한다. 기존 실험과 합쳐 3seed 성능으로 표현하지 않는다.
