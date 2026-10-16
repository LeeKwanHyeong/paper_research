# CAP9 평가 연결부 사전검사 — 차단됨

집계 수정·Validation 비교 완료 후, 사용자가 승인한 기존 CAP9 평가 helper를 한 번 실행했다. 서버 접속 전에 `bind()`의 로컬 계약 검사에서 `Selector/reference is not Validation` 오류로 종료됐다. CAP9 native Validation·Test는 모두 미실행이다. 이 실패는 학습 실패나 성능 문제가 아니다.

## 확인된 원인

`advance_titantpp_gru_cap9_test.py`는 학습 종료 원본 `endpoint_replays.json`의 `selected` 객체를 그대로 `validation_reference`로 사용한다. 원본은 `evaluation_scope="validation_only"`와 `held_out_test_evaluated=false`로 split을 표현하지만 평가기는 `split="validation"`을 요구한다. 9개 모두 같은 형식 불일치다.

## 필요한 수정안 — 검토 가능, 미적용

원본의 두 Validation 전용 표기를 확인한 뒤 새 파생 참조 객체에만 split을 명시한다.

```python
require(endpoint.get('evaluation_scope') == 'validation_only'
        and endpoint.get('held_out_test_evaluated') is False,
        'Selected endpoint is not Validation-only')
reference = {**endpoint, 'split': 'validation'}
# j.update(..., validation_reference=reference, ...)
```

위와 같은 메모리 내 변환과 파생 계약 SHA 재계산만 적용했을 때, 기존 평가기의 실제 `verify_contract()`가 9개 전체를 통과했다. 원본 계약·선택 binary·수치·epoch·대상·source·평가기 코드는 변경하지 않았다. 로컬 확인 증적은 `evaluation_preflight_failure.json`에 보존했다.

실패한 평가 bundle의 `contract.json`, `advance.claim`, `failure.json`은 그대로 보존한다. 재진행하려면 별도 후속 평가 경로에 이 연결부 수정과 회귀검증을 적용해야 한다. 이번 사용자의 수정 범위는 집계 도구였으며, 기존 운영 규칙은 평가 도구 자동 수정·재봉인·실패 후 자동 재시도를 금지한다. 따라서 새로운 평가 연결부 변경과 후속 실행은 승인 전 실행하지 않았다.

## 남은 작업 순서

**평가 연결부의 split 표기를 보완한다 — 승인 필요**
- 새 파생 참조에만 위 변환을 적용하고, 실제 원본 형식을 재현한 회귀검증을 추가한다. 기존 실패 bundle과 원본을 보존한다.

**고정 checkpoint로 CAP9 평가와 비교를 마친다 — 승인 후 다음 작업**
- 별도 평가 경로에서 native Validation9를 재현하고 전부 통과하면 Test9를 실행한다.
- 기존 CNN/FULL18 Test를 재사용해 데이터셋별 3seed 평균·표본SD와 seed별 차이를 비교한다.
- 원본과 결과 SHA 및 실제 종료 증적 검증이 끝난 뒤 해당 자동화만 종료한다.
