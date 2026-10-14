# 선택 checkpoint 고정 진단의 실행·회수 기록

2026-10-14, paper_research. 사용자가 승인한 Train·Validation·Test raw 오류와 개별 예측 산출이다. 새 학습, checkpoint 재선택, 통계 재추정, 원고 변경은 하지 않았다. 수치 분석과 독립 QA는 별도 보고서로 구분한다.

## 실제 실행과 재사용 — 완료

선택 모델은 CNN·CNN+GRU 18개와 외부 비교군 54개, 총72개다. CNN은 TitanTPP의 CNN만 사용하는 구성요소 대조이며 일반적인 독립 CNN 모델을 뜻하지 않는다. 외부 비교군은 RMTPP, THP, NHP, SAHP, S2P2 matched head, AttNHP matched head다.

| 구분 | 모델·분할 조건 수 | 실제 증적 |
|---|---:|---|
| 5080 v3 신규 추론 | 136 | 모든 worker exit0, pipeline·supervisor 실제 OS exit0 |
| 5080 v2 성공한 Validation 재사용 | 14 | 기존 receipt·exit·계약·개별 예측 SHA 그대로 보존 |
| 5090 v2 완료 결과 재사용 | 12 | 모든 worker exit0, pipeline·supervisor 실제 OS exit0 |
| 고유 진단 조건 | 162 | 중복 없는 (모델ID, split) |
| 기존 외부 비교군 Test 재사용 | 54 | 기존 예측을 재실행하지 않고 분석에 연결 |
| 최종 비교 대상 | 216 | 72개 모델 × Train·Validation·Test |

162개 진단 조건의 예측 기록은 총14,031,168행이며 Train10,991,904·Validation2,429,832·Test609,432행이다. 같은 대상에 여러 모델을 적용한 기록 수로, 독립 표본 수를 뜻하지 않는다. 1,812개 Parquet part에 개별 예측을 저장했다.

## 선택·입력·Runtime 결속

선택 checkpoint와 source, 전처리, 원래 입력, target 모집단을 고정했다. 각 실행은 모델 tensor의 전후 SHA, target 차단·batch 일치 검사, native Runtime과 GPU UUID를 확인한다. Validation과 Test는 기존 지표를 재현해야 통과한다. Train raw 오류는 고정 모델의 학습 구간 내 진단이며 학습 중 온라인 log 수량 손실과 구분한다.

Intermittent seed42·52의 구성요소는 5090/cuDNN92000, seed62는 5080/cuDNN91900이다. 같은 seed의 CNN·CNN+GRU는 같은 host에서 비교한다. 외부 모델은 각각의 과거 학습 source와 checkpoint를 유지하므로 동일 source·Runtime으로 새로 학습한 대조로 해석하지 않는다.

Runtime receipt에는 직접 저장된 resolved Python 경로가 없다. 실제 worker argv와 SHA가 고정된 평가 코드의 추론 전후 `sys.executable.resolve()` 검사 및 실제 종료 증적으로 확인한다. 저장된 경로 측정값으로 바꾸어 표현하지 않는다.

## 원본과 실패 증적 보존

v1은 process 환경의 NVRTC builtins 라이브러리 경로 누락으로 Validation에서 중단했다. v2는 기존 native 라이브러리 경로를 사용했으며, 5080 외부 모델의 첫 Validation에서 과거 source의 프로젝트 경로 탐색에 필요한 빈 `sample_data` 디렉터리 누락을 발견했다. v3는 새 격리 경로에 이 빈 디렉터리만 복원했다. 과학 source와 모델, 설치된 Runtime 바이트는 변경하지 않았다.

v1·v2의 claim·failure·stdout/stderr와 실제 exit1 증적을 삭제하지 않았다. v2의 성공 결과는 SHA를 고정해 재사용했으며, 완료한 5090 추론을 다시 실행하지 않았다. 실제 Test로 도구를 디버깅하지 않았다. 기존 학습 root와 claim을 변경하지 않았고, 삭제된 A100 Pod/volume에 접근하지 않았다.

## 회수와 포인터

- [실행 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3/execution_receipt.json): `5e5ba273ab545aee222d51c38804f6a6c3d44c2835c76f111e8819f7348ff157`
- [5080 원시 파일 manifest](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3/native/5080/retrieval_manifest.json): 2,273개 파일, SHA `8cddec8ade4988effea54d59f1139ca329894e4ac25c2a6a0c7dddd98c671136`
- [5090 원시 파일 manifest](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3/native/5090/retrieval_manifest.json): 353개 파일, SHA `c251596db8a4484cf0f6431c0bce34693145efecc4f12d35d628ff3d7516760c`
- 5080 회수 archive: 921,627,487bytes, SHA `4f802b80469e04b023d73c611cc87c270cee59bd5d089624863d88e5c0c8c1eb`, actual transport exit0, timeout=false.

개별 예측은 각 native 디렉터리의 `results/<모델ID>/<split>/predictions/part-*.parquet`에 있고, 해당 split의 `predictions_manifest.json`이 행 수와 SHA를 결속한다. 실행 성공은 진단 수치의 인과적 설명이 확정됐다는 뜻이 아니다. 최종 수치와 주장 범위는 `ROW_ANALYSIS.json/md`, 독립 검증은 `FINAL_QA.json/md`에서 확인한다.
