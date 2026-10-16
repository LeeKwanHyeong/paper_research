# paper_research 작업 안내

이 문서는 코드 탐색과 로컬 점검의 시작점이다. 실험 실행 승인이나 현재 서버 상태를
대신하지 않는다. 작업 전 [세션 규칙](TEST_SESSION_PROTOCOL.md) 0절을 확인한다.

## 어디에서 작업할 것인가

| 경로 | 역할 | 정리·수정 원칙 |
|---|---|---|
| `models/`, `data_loader/`, `utils/` | 현재 모델과 공통 코드 | 과거 실험의 실제 실행본과 구분한다. 기존 CNN 기준선은 변경하지 않는다. |
| `paper/scripts/` | 계약 준비, 실행, 관측, 회수, 분석 도구 | 파일이 있다는 이유로 실행하지 않는다. 해당 실험의 계약·승인·현재 receipt부터 확인한다. |
| `paper/tests/` | 모델·운영 계약 검사 | 합성/모의 입력인지 확인한 뒤 필요한 검사만 실행한다. 일부는 Git에서 제외된 로컬 bundle을 요구한다. |
| `scripts/` | 저장소 관리 도구 | 실험 코드와 분리한다. `check_local_sources.py`는 아래 설명대로 구문 검사만 수행한다. |
| `reports/<실험 또는 분석명>/` | 계획, 실행 증거, 분석, 실패 이력 | SHA로 연결된 파일의 이름·내용을 일괄 정리하지 않는다. 과거 `SCHEDULER`는 현재 실행 지시가 아니다. |
| `search_artifacts/` | 데이터·선택 checkpoint·동결 실행본·회수 원본 | 기존 Git 제외 정책을 유지한다. 삭제·이동하지 않는다. Git 커밋은 원본 데이터 백업이 아니다. |
| `paper/reproducibility/` | 과거 소스 복원과 이력 연결 | 고정 snapshot과 현재 개발 코드를 구분한다. |
| `paper/*draft*/`, `paper/*submission*/` | 원고 작업본 | 코드 정리와 과학적 주장 수정은 별도 작업이다. |

## 로컬에서 코드 구문 확인하기

저장소 루트에서 다음 명령을 실행한다. Python 표준 라이브러리와 Git만 사용하며,
모델 import·학습·추론·원격 연결·bytecode 저장을 하지 않는다.

```sh
python3 scripts/check_local_sources.py
python3 scripts/check_local_sources.py --changed
```

첫 명령은 안내된 개발 소스 경로의 Git 추적 파일과 ignore되지 않은 새 Python 파일을,
두 번째는 그중 현재 HEAD 대비 변경 파일을 검사한다. `reports/`와
`search_artifacts/`의 과거 소스 복사본은 기본 점검 대상에서 제외한다.
통과는 구문이 유효하다는 뜻이며, 모델 동작·수치 재현·GPU 검증을 대신하지 않는다.

새 관리 도구 자체는 다음 합성 검사로 확인한다.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

## 실험 파일을 정리할 때

1. **현재 기준선 확인:** 코드와 실험 계약이 사용하는 실제 source revision을 구분한다.
   완료된 실험은 기준 자료로만 남기고 재실행 목록에 넣지 않는다.
2. **변경 대상 분리:** 공통 과학 코드, 새 실험 운영 코드, 분석 보고서, 원고를 나눈다.
   기존 봉인 파일을 고치는 대신 필요한 변경을 별도 경로와 새 버전으로 검증한다.
3. **검증 후 기록:** 해당 변경의 합성·계약 검사와 기존 파일 보존 여부를 기록한다.
   테스트가 원본 bundle에 의존하면 새 checkout에서도 실행된 것처럼 설명하지 않는다.
4. **커밋 범위 확인:** `git diff --check`와 staged 목록을 확인하고 용도별로 커밋한다.
   새 학습·공용 Runtime 변경·Push는 코드 정리의 자동 후속 단계가 아니다.

## 2026-10-16 정리 기준 자료

- [저장소 정리 기록](reports/repository_cleanup_20261016_v1/README.md): 미커밋 파일 목록,
  바이트 보존, 로컬 검사와 커밋 범위를 기록한다.
- [최근 MLP 재검토](reports/titantpp_history_mlp_reassessment_20261016_v1/REPORT.md):
  직전 분석의 결과 문서다. 새 학습의 승인이나 실행 상태를 의미하지 않는다.
- [과거 소스 복원 안내](paper/reproducibility/README.md): 2026-10-04 시점의 기준선과
  준비 상태를 포함한다. 당시 진행 상태를 현재 상태로 읽지 않는다.

준비 후 보류되거나 다른 실험으로 대체된 코드도 연구 이력으로 보존한다.
예를 들어 `CountAwareTitanMomentCNN.py`와 `nonlinear_raf9` 도구의 보존 커밋은
그 후보를 현재 채택하거나 다시 시작한다는 뜻이 아니다.
