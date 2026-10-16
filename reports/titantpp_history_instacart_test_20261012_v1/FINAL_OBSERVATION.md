# 선택 checkpoint36개의 기존 Test 최종 비교 완료

2026-10-12 09:32:08 KST에 Validation36 → Test36 파이프라인이 실제 exit0로 종료됐다.
09:32:21 KST 관측에서 완료를 확인했고, 원본 929개 파일의 archive·manifest 및
로컬 SHA를 검증했다. 개별 평가 worker72개와 합성 native probe6개의 실제
exit0, 평가 소유 PID·GPU PID 부재를 확인했다. 기존 학습·원본·Runtime·인증은
변경하지 않았다.

사용자가 승인한 기존 Test 최종 비교다. Validation에서 선택한 checkpoint와
epoch를 고정했으며 재학습·Test 재선택·Test 튜닝을 하지 않았다. 기존 Test
생성·접근 이력 때문에 새 미노출 독립 확인으로 분류하지 않는다.

## 전체 raw 수량 RMSE: 3seed 평균 ± 표본 SD

|데이터|full CNN+GRU54|보정 제거 / MLP8|current-only / MLP12|같은 seed에서 두 대조 모두 개선|
|---|---:|---:|---:|---|
|Taxi|123.0831 ± 28.0153|171.7657 ± 35.1589|156.6933 ± 30.0009|42·62, 2/3|
|Intermittent|2.779257 ± 0.145831|2.834064 ± 0.199906|3.097933 ± 0.272916|52·62, 2/3|
|RAF|39.90883 ± 0.32668|39.38908 ± 0.09197|39.43374 ± 0.11421|0/3|
|Instacart|5.969777 ± 0.004763|5.982445 ± 0.011238|5.976099 ± 0.018595|42·62, 2/3|

위 표의 첫 세 데이터는 이력 보정 연구의 두 대조이고, Instacart는 별도 구조
비교의 MLP8·MLP12다. 서로 같은 대조로 해석하지 않는다. 모든 수치는 Test이며
선택 기준은 사전에 고정된 Validation 전체 수량 RMSE다.

**이력 보정에서 모든3seed·양대조를 개선한 Test 데이터는0개다.** Taxi·Intermittent의
평균 개선은 관찰됐지만, Validation에서 충족한 엄격한 개발 일관성이 Test에도
유지됐다고 주장할 수 없다. RAF에서는 세 seed 모두 full의 전체 RMSE가 두 대조보다
나쁘다. Instacart는 MLP8에3/3, MLP12에2/3 개선이며 seed52가 MLP12보다 나쁘다.

## 손해와 단순 기준선을 함께 보존한다

- 이력 보정18쌍(full 대 두 대조×3dataset×3seed)에서 전체 RMSE는8개 개선·10개
  손해, 전체 TimeNLL은6개 개선·12개 손해였다. 전체 RMSE의 Validation→Test
  부호 반전은8쌍이다. 큰 수량 RMSE는10개 개선·8개 손해, 큰 수량 MAE는8개
  개선·10개 손해였다. 여섯 지표의 모든 paired 차이는 최종 보고서에 남겼다.
- Taxi·Intermittent·Instacart full의 평균 전체 TimeNLL은 각 두 대조보다 나쁘다.
  수량의 평균 개선으로 시간 손해를 가리지 않는다. 기록된 TimeNLL의 정의와
  clamped 시간 head는 동결 과학 source를 유지한다.
- RAF 전체 RMSE는 입력 이력 평균 기준선39.35791보다 나쁘다. Taxi·RAF·Instacart의
  큰 수량 RMSE는 직전 수량 반복보다 나쁘다. Instacart의 전체 RMSE는 입력 이력
  평균5.980200 대비 약0.01042(약0.17%) 개선에 그치며, 큰 수량에서는 같은 평균
  기준선이 더 낫다. 수량 기준선에 임의 TimeNLL을 부여하지 않았다.
- 단순 평균은 모델에 주어진 lookback·sequence cap 안의 관측 이력 평균이다.
  미래 target나 평가 이후 사건을 넣은 평균이 아니다.

## 원본·종료 및 평가 계약

- 평가 GPU: RTX5080 `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`.
- Runtime: 기존 ai_env, Python3.12.13, Torch2.11.0+cu130, CUDA13.0, cuDNN91900,
  NumPy2.4.4, Polars1.39.3. 모든 조건을 같은 평가 Runtime에서 추론했다.
- 같은 dataset/seed의 세 arm을 최대3개 동시에 추론했다. 원래 A100/RTX 학습
  출처와 RAF42의 source lineage는 조건별 원본에 그대로 유지한다. 동시 실행의
  wall time을 구조 효율 결과로 해석하지 않는다.
- 36개 전체 Validation의 여섯 지표는 사전 `math.isclose(rel_tol=1e-5,abs_tol=1e-5)`
  기준을 통과했고, 대상 수·대상/정답 digest 및 실제 종료를 검증했다. 전역 gate
  이전에는 Test 추론을 시작하지 않았다. RAF 전체 부모의 Train/Validation은
  원래 학습 export와 일치했다. 원래 학습 통계를 다시 적합하지 않았다.
- 로컬·archive·manifest의 모든 원본 SHA가 일치했다. 기존 파일 덮어쓰기0이다.
  원고 편집·새 학습·검색·미노출 자료 다운로드·Runpod 작업은 실행하지 않았다.
- 별도의 전체 CPU binary 재추론 감사와 새 미노출 독립 평가는 미완료다.

## 직접 증적

- [36조건 원수치·3seed 평균/표본 SD](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/test_conditions.json)
- [여섯 지표·대조별 차이·방향 반전·단순 기준선](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/TEST_FINAL_COMPARISON.md)
- [분석 완료 receipt](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/completion_receipt.json)
- [로컬 929개 원본 검증](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/originals/LOCAL_VERIFICATION.json)
- [실제 종료 시각·봉인 SHA](RUN_STATUS_COMPLETE.json)
- [사전 QA](PRELAUNCH_QA.md), [원본 회수 QA](FINALIZER_QA.md), [실제 종료·수치 독립 검증 QA](FINAL_TEST_QA.md)

canonical 계약 SHA `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`;
Validation gate rawSHA `6e908b9fe01c608373d189eca7d7fc3c0b02f3d648d0e7e3e907fc90d2e24751`;
원본 archive rawSHA `bf7aa41942420966dc612cd1b8a8806733cf38abeb373dea127558782dbfb3ca`;
분석 완료 rawSHA `d21d137190e98552d07c5b9e2d8fc31151846e39c782ad907ff9a1a3cf644dbd`.

다음 작업은 이 Test 결과와 예외를 기준으로 논문의 수치·Methods·주장 범위를
맞추는 것이다. 고정 검색은 실행 대기를 유지하며, 이를 현재 손해의 해결책으로
자동 시작하지 않는다.
