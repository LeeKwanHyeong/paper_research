# TitanTPP 연구 코드와 재현성 인덱스

이 문서는 2026-10-03의 후속 연구 상태를 연결한다. 루트 README의 v0.7은
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

| 구분 | 상태 | 변하는 내용 | 유지할 기준 |
|---|---|---|---|
| History MLP 폭 4 | 기존 기준선 | 분기 `128→4→64`, 보정 6,144개 | 기존 checkpoint와 해당 source 재사용 |
| History MLP 폭 16 / seed42 | 승인·실행 증거 보존 | 분기 `128→16→64`, 보정 24,576개 | 기존 마스크, 고정 `/8`, GELU, zero output, 출력부·손실·선택 기준 |
| 폭 16 / seed52·62 | 승인된 후속 큐 증거 보존 | 4데이터×2seed, 5080·5090의 seed42 종료 후 시작 | 과학 소스 112/113 동일, seed-aware 실행기만 변경 |
| 폭 8·12 | 다음 작업: 미구현·미실행 | 보정 파라미터는 각각 12,288·18,432개 | 폭별 identity·계약·초기화 검증 후 별도 실행 계약 필요 |
| A100 routing/placement | 별도 캠페인 | recent4_attention / post_block, 3데이터×seed42 | 아래 동결 계약으로만 해석 |
| Encoder 1 CNN＋중간 GRU | 다음 작업: 설계 후보 | E1에 인과적 CNN; 중간 MLP 대신 순환 보정; 64차원 잔차 유지 | CNN/GRU 잠재 폭·파라미터 예산·삽입 위치·비교군 미확정 |

현재 `CountAwareTitanHistoryWidth.py`는 폭4/16만 허용한다. 8·12 계획을 현재
실행 가능한 옵션으로 오인하지 않는다. 기존 seed42 결과의 우열이나 test 접근
여부를 이 SCM 정리에서 다시 분석하지 않았다. test를 본 뒤 정한 후속 후보를
새 독립 평가로 표현하지 않는다. 세 seed가 모두 끝나고 원본 검증이 완료된 뒤
평균·표본 표준편차를 집계한다.

CNN＋GRU는 [설계 검토](../../reports/titantpp_cnn_gru_review_20261003_v1/README.md)의
방향이며 현재 A100 routing 실험과 다르다. 앞선
[CNN 보정 모듈 대체 제안](../../reports/titantpp_causal_cnn_design_20261003_v1/README.md)도
역사적 대안으로 보존한다. CNN/GRU와 폭 확대를 한 실험으로 묶어 효과를 단정하지
않고 각 요소의 대조 조건을 먼저 정한다.

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

## master 통합 순서

1. **현재 연구 변경 고정 — 완료:** `codex/hard-lmm-causal-qkv`에서 의존 소스,
   계약·테스트, 동결본, 연구 문서를 커밋한다. 실행 중인 remote bundle은 수정하지 않는다.
2. **통합 범위 검토 — 다음 작업:** 로컬 master `567196e5` 대비 기존 94커밋과 이번
   정리를 검토한다. 현재 DAG상 fast-forward 가능하지만 원격 최신 여부는 fetch 전 미확인이다.
3. **별도 연구 이력 결정 — 다음 작업:** 아래 3개 branch의 고유 12커밋을 필요성·
   충돌 기준으로 검토한다. 현재 branch와 중복 또는 불일치하는 연구 코드를 일괄 병합하지 않는다.
4. **master 통합·원격 반영 — 다음 작업:** 검증된 범위로 별도 통합 checkout에서
   확인 후 `paper_research/master`에 정리한다. 이번 로컬 정리에서 master·origin은 바꾸지 않는다.

| 현재 HEAD에 없는 별도 이력 | 고유 커밋 | 처리 |
|---|---:|---|
| `codex/b1-prior-prefix-read` | 9 | branch 유지, 비교 검토 대기 |
| `codex/raw-rmse-baseline-completion` | 1 (`16ebcc8b`) | branch 유지, 비교 검토 대기 |
| `codex/hard-lmm-dual-timescale` | 2 (`c6562495`, `ca8823e6`) | branch 유지, 비교 검토 대기 |

기존 분기·원본 산출물은 삭제하지 않는다. 커밋은 보존 단위이며 모든 후보의 과학적
타당성이나 논문 채택을 의미하지 않는다. 전체 GPU 재현·모든 과거 스크립트 실행은
이번 검증 범위가 아니다.
