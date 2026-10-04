# TitanTPP 연구 코드와 재현성 인덱스

이 문서는 2026-10-04의 후속 연구 상태와 사용자 결정을 연결한다. 루트 README의 v0.7은
`567196e5`의 별도 validation freeze이며, 이후 MLP·관측 시간 likelihood·raw
quantity RMSE checkpoint 선택·기존 test 재평가와 동일한 실험이 아니다.

## 코드와 실제 실행본

- `models/`, `paper/scripts/`, `paper/contracts/`, `simple_lab_test/search/tests/`는
  현재 개발 코드와 계약 테스트다. 이를 과거 checkpoint에 그대로 적용하지 않는다.
- `snapshots/*/manifest.json`은 실제 캠페인의 source·계약·운영 코드의 파일별 SHA,
  경로, 크기와 캡처 시각을 담는다. `objects/<sha256>`는 원본 바이트다.
- 데이터·checkpoint·예측은 Git에 넣지 않는다. 원래 로컬/서버 보관 위치와 SHA는
  각 실행 계약 및 [정리 전 목록](../../reports/titantpp_scm_cleanup_20261003_v1/initial_inventory.json)에
  남아 있다. SHA 목록은 백업이 아니다. 원본과 외부 데이터 접근권한이 있어야
  전체 실험을 재현할 수 있다.
- 캡처된 승인·permit·서버 주소는 당시 실행의 증거다. 복원은 실행이나 재임대
  승인이 아니며, 만료된 permit으로 재실행하지 않는다.

## 현재 기준선과 예정 실험

**2026-10-04 사용자 결정: 후속 실험의 기본 기준선은 History MLP 폭16이다.**
현재 선택은 [기준선 포인터](current_baseline.json)와
[결정 기록](../../reports/titantpp_width16_baseline_decision_20261004_v1/README.md)을 따른다.
이는 실험 비교 기준의 선택이며, 모든 데이터·지표에서 최저 오차라는 판정은 아니다.
기존 폭4와 동결 결과는 비교 대조군으로 보존한다.

| 구분 | 상태 | 변하는 내용 | 유지할 기준 |
|---|---|---|---|
| History MLP 폭 4 | 보존 대조군 | 분기 `128→4→64`, 보정 6,144개 | 기존 checkpoint와 해당 source 재사용 |
| History MLP 폭 16 / seed42·52·62 | 후속 기본 기준선 확정 · 완료된 12조건의 Validation·Test 검증 | 분기 `128→16→64`, 보정 24,576개 | 기존 마스크, 고정 `/8`, GELU, zero output, 출력부·손실·선택 기준; 기존 source·checkpoint 재사용 |
| 폭 8·12 | 구현·계약 검증 완료; 5080 학습 중, 5090 체크포인트 전송 승인 대기 | 보정 파라미터는 각각 12,288·18,432개 | Taxi·Intermittent·RAF ×3seed, 총18조건. Instacart 후속; 폭4/16 재사용 |
| A100 routing/placement | 완료 · 별도 seed42 탐색 | recent4_attention / post_block, 3데이터×seed42 | 동결 source와 계약으로 해석; 3seed 결과와 구분 |
| Encoder 1 CNN＋중간 GRU | 대조 설계·파라미터 검산 완료; 구현 필요 | E1 event Q/K/V causal CNN; MLP16 대신 GRU54, 64차원 잔차 유지 | MLP16/CNN+MLP16/GRU54/CNN+GRU54; 27개 신규 조건은 별도 후속 계약 |

현재 전용 모델·실행기는 폭4/8/12/16을 구분하며 기존 폭4/16의 초기 상태와
identity를 유지한다. [폭8·12 실행 상태](../../reports/titantpp_history_capacity_launch_20261004_v1/README.md)가
실제 시작 여부의 기준이다. 폭16 모델 ID는
`titantpp_history_mlp_width16`이며, 전용 실행기의 factory 연결을 사용한다.
기존 `titantpp_history_mlp` ID는 폭4 재현용으로 유지한다. 네 데이터의 3seed
평균·표본 표준편차와 기존 Test 재평가는
[전체 비교 보고서](../../reports/titantpp_completed_validation_test_20261004_v1/README.md)에
있다. Test를 본 뒤 정한 후속 후보를 새 독립 평가로 표현하지 않는다.

CNN＋GRU는 [설계 검토](../../reports/titantpp_cnn_gru_review_20261003_v1/README.md)의
방향이며 현재 A100 routing 실험과 다르다. 앞선
[CNN 보정 모듈 대체 제안](../../reports/titantpp_causal_cnn_design_20261003_v1/README.md)도
역사적 대안으로 보존한다. CNN/GRU와 폭 확대를 한 실험으로 묶어 효과를 단정하지
않고 각 요소의 대조 조건을 먼저 정한다.
이전 GRU r=24의 근접 파라미터 비교는 폭4 기준 제안이다. 폭16을 기준으로 하는
폭16 기준 [새 CNN·GRU 설계](../../reports/titantpp_cnn_gru_width16_design_20261004_v1/README.md)는
GRU54 보정24,732개와 MLP16 보정24,576개를 비교한다. 이는 근접한 파라미터 대조이며
작은16차원 상태나 순수 recurrence만의 효과를 검증하는 계약은 아니다.
[채택·시간 진단](../../reports/titantpp_width_time_adoption_20261004_v1/README.md)은
전수 validation과 train/validation 표본 진단을 분리한다.

## 동결본 확인과 복원 — CPU·오프라인, 학습 실행 없음

저장소 root에서 실행한다. Python 표준 라이브러리만 필요하다.

```sh
python3 paper/reproducibility/archive.py verify paper/reproducibility/snapshots/width16_seed42
python3 paper/reproducibility/archive.py verify paper/reproducibility/snapshots/width16_seed52_62
python3 paper/reproducibility/archive.py verify paper/reproducibility/snapshots/routing_a100_seed42
python3 paper/reproducibility/archive.py verify paper/reproducibility/snapshots/historical_untracked_sources
python3 paper/reproducibility/archive.py restore paper/reproducibility/snapshots/width16_seed52_62 /private/tmp/titantpp-width16-restored
```

복원 대상은 존재하지 않아야 한다. 복원된 bundle에는 당시 `frozen_source/`
(또는 `source/`), 계약과 운영 코드가 원래 상대경로로 생긴다. 파일들은 자동으로
실행되지 않는다. 세 캠페인의 과학 source 336개와 별도 지원 파일을 보존했다.
`historical_untracked_sources`는 Git에서 제외한 중복 폴더에 있던 Python·license
4,511개 경로를 143개 고유 바이트로 보존한다. 이것은 완전한 결과 archive가 아니다.

`capture_campaigns.py`는 명시된 원본 bundle이 로컬에 있을 때 새 캡처를 만드는
도구다. 기존 캡처를 덮어쓰지 않는다. 모니터 최신 상태·원본 데이터는 수집하지 않는다.
원래 실험에는 GPU·데이터·checkpoint·서버 runtime이 추가로 필요하다. 현재 Mac
Python 3.12/PyTorch 2.14 CPU 테스트 통과를 서버 x86/PyTorch 2.11의 초기화 SHA나
GPU 수치 재현 성공으로 해석하지 않는다. 과거 계약의 runtime qualification을 따른다.

## master 통합과 브랜치 보존 — 완료

1. **현재 연구 변경 고정 — 완료:** `codex/hard-lmm-causal-qkv`에서 의존 소스,
   계약·테스트, 동결본, 연구 문서를 커밋한다. 실행 중인 remote bundle은 수정하지 않는다.
2. **통합 범위 검토 — 완료:** fetch로 원격 master `567196e5`를 확인하고 활성 정리본
   `533f45bc`의98커밋을 fast-forward 범위로 확정했다.
3. **별도 연구 이력 보존 — 완료:** 아래3개 branch의 고유12커밋은 과거 시간 손실·
   선택 기준을 사용하는 대안이다. 현재 과학 코드에 채택하지 않고 원래 전체 이력을
   `archive/20261003/*` tag로 보존했다.
4. **master 통합·원격 반영 — 완료:** Git 파일만으로 만든 clean checkout의254개
   CPU 테스트를 통과한 `b05c26e2`를 `paper_research/master`와 origin에 반영했다.
   미추적 core ablation 계약 의존성은 원본 바이트의 tracked fixture로 해결했다.
   원격13개 tag의 SHA 확인 후 로컬 codex12개·원격2개를 예상 tip 조건으로 정리했다.

| 현재 HEAD에 없는 별도 이력 | 고유 커밋 | 처리 |
|---|---:|---|
| `codex/b1-prior-prefix-read` | 9 | `archive/20261003/b1-prior-prefix-read` |
| `codex/raw-rmse-baseline-completion` | 1 (`16ebcc8b`) | `archive/20261003/raw-rmse-baseline-completion` |
| `codex/hard-lmm-dual-timescale` | 2 (`c6562495`, `ca8823e6`) | `archive/20261003/hard-lmm-dual-timescale` |

기존 commit·tag·원본 산출물은 보존한다. branch 이름 정리는 연구 이력 삭제가 아니다.
복원 방법과 원격 SHA·삭제 검증은 [master 통합 보고서](../../reports/titantpp_master_integration_20261003_v1/README.md)를
따른다. 커밋은 보존 단위이며 모든 후보의 과학적 타당성이나 논문 채택을 의미하지 않는다.
전체 GPU 재현·모든 과거 스크립트 실행은 이번 검증 범위가 아니다.
