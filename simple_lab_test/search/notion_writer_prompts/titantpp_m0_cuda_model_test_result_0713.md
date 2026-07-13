다음 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

위치:
- `5. Model Design Enhancement > Enhancement & Validation History`
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 기존 `TitanTPP M0 CUDA Model Test (2026-07-13)` 페이지에 결과를 추가

결과 artifact:
- local path: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_m0_cuda_model_test_0713`
- server path: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_m0_cuda_model_test_0713`
- final log: `logs/run.log`
- first-attempt diagnostic: `attempt_1/run.log`
- summary: `model_test_summary.json`, `model_test_summary.csv`
- completion marker: `MODEL_TEST_SUCCESS`

실험 시간:
- 최초 실행 시작: `2026-07-13 11:09:04 KST`
- 최초 실행 종료: `2026-07-13 11:09:06 KST`, exit code `1`
- 재실행 시작: `2026-07-13 11:11:44 KST`
- 최종 완료: `2026-07-13 11:11:46 KST`, exit code `0`
- 최초 실행부터 최종 완료까지: 약 `2분 42초`
- 최종 model-test runtime: 약 `2초`
- 실행 서버/tmux: `5090` / `titantpp_m0_cuda_modeltest_0713`

실행 결과:

| 항목 | 결과 |
| --- | ---: |
| status | `success` |
| device | `cuda` |
| model / candidate | `TitanTPP / small_lmm` |
| hidden shape | `[4, 16, 64]` |
| total NLL | `3.902806` |
| marker NLL | `2.495406` |
| time NLL | `1.407400` |
| magnitude loss | `0.212801` |
| legacy value loss | `0.0` |
| supervised steps | `4` |
| quantity prediction mean | `35.847954` |
| time prediction mean | `0.638614` |
| parameter count | `78,111` |
| quantity decoder | `direct_log_qty` |
| normalization | `global` M0 |
| legacy value-by-mark output | `null`, 비활성 |

Runtime contract 확인:
- forward와 direct quantity reconstruction이 RTX 5090 CUDA에서 완료됨
- total/marker/time NLL, magnitude loss, quantity/time prediction 모두 finite
- final log에서 NaN, Inf, Traceback, Error 없음
- `MODEL_TEST_SUCCESS` 생성 확인
- `value_by_mark_shape=null`로 legacy mark-residual decoder가 동시에 활성화되지 않았음을 확인
- marker CE와 time NLL은 유지되고 direct magnitude loss가 별도 출력됨

첫 시도 실패와 조치:
- 첫 시도는 모델 오류가 아니라 `nvrtc: failed to open libnvrtc-builtins.so.13.0` 환경 오류로 종료됨
- 라이브러리는 ai_env의 `nvidia/cu13/lib`에 설치되어 있었으나 동적 링크 경로에 포함되지 않았음
- 실행 스크립트의 `LD_LIBRARY_PATH`에 해당 경로와 ai_env lib를 명시
- 동일한 코드와 옵션으로 재실행해 exit code `0` 확인
- 첫 실패 artifact는 `attempt_1` 아래에 보존
- final run에는 kernel cache 디렉터리 생성 경고가 한 건 있었지만 kernel 실행과 결과 생성은 정상 완료됨

해석:
- M0 shared magnitude-context와 direct log2-quantity decoder의 5090 CUDA runtime gate는 통과
- 이번 결과는 synthetic interface 검증이므로 M0의 실제 데이터 성능이나 V2 대비 우위를 의미하지 않음
- 다음 단계는 Instacart top-20 e1 fixed-split smoke에서 train-only global statistics, 학습 backward, checkpoint, validation/test artifact 경로를 확인하는 것

문체:
- 첫 실패를 생략하지 말고 환경 원인과 수정 후 성공을 짧게 구분
- CUDA 실행 성공과 모델 성능 검증을 같은 의미로 표현하지 않음
- 불필요한 수식어와 과장된 결론 없이 확인된 결과 중심으로 작성
