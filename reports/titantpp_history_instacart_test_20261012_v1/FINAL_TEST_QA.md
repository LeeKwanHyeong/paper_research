# 기존 Test 최종 비교36 — 실행 후 독립 QA

## Review Result: PASS

검토시각: 2026-10-12 09:39 KST. 대상은 승인된 이력 보정27 + Instacart9의 동결 checkpoint 평가다. PASS는 원본·실행·집계의 정합 판정이며, 모델 우위나 논문 채택 가능성 판정이 아니다. 기존 Test 접근 이력은 유지한다.

### CRITICAL

- 없음.

### HIGH

- 없음.

### MEDIUM

- 없음.

### LOW

- 없음.

## 검토 범위와 수행 경계

- `search_artifacts/titantpp_history_instacart_test_20261012_v1/`의 contract, input/operation integrity, 선택 checkpoint36, native qualification6, Validation/Test native receipt72, 실제 worker exit72, 전역 Validation gate, pipeline complete/OS exit, originals manifest/archive/verified, 최종 분석 JSON/Markdown/completion을 읽기 전용으로 검토했다.
- 이 검토에서 SSH·GPU/CPU 재추론·재학습·원격 회수·분석 finalizer 실행은 하지 않았다. 원본 binary는 역직렬화 없이 stdlib SHA로만 읽었다. 이번 변경 파일은 이 QA 보고서 하나다.
- 사전 소스/계약 테스트 80 PASS 및 finalizer 합성 정상/변조/symlink 검증은 PRELAUNCH_QA.md와 FINALIZER_QA.md에 별도 보존돼 있다. 실행 후 검증은 원본과 분석 수치의 독립 재계산으로 수행했다.

## 실제 종료와 평가 순서

|증적|KST 또는 결과|
|---|---|
|마지막 Validation worker 실제 exit0|2026-10-12 09:23:29.248336|
|36개 전체 Validation gate 생성|2026-10-12 09:23:29.257104|
|첫 Test 추론 시작|2026-10-12 09:23:29.303300|
|pipeline 실제 OS exit0 관측|2026-10-12 09:32:08.693799|
|실제 Validation/Test worker 정상 종료|36 + 36 = 72|
|native CUDA 합성 자격 검증|6개 경로 PASS, 실제 probe exit0|
|원격 종료 회수 증적|own process/GPU PID 부재, 재시도 없음|

모든 Validation receipt·실제 종료 rawSHA를 gate와 대조했고, 모든 Test 시작이 gate 생성 이후이며 gate rawSHA가 일치함을 확인했다. 모든 worker의 PID/PPID, Python 경로를 포함한 argv, job/split, 실제 시작·종료 순서를 native receipt와 대조했다. pipeline OS exit는 별도 실제 관측 증적에 의해 확인됐으며 단순 pipeline_complete만으로 추정하지 않았다.

## 원본 무결성과 실행 환경

- 계약 canonical SHA: `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`.
- 활성 operation integrity rawSHA: `22db16310ed0b1af96b3fda32980f46095e387bc5c36e51531b2b964d14d3888`.
- immutable input493 및 operation5 파일의 stdlib rawSHA 일치.
- 선택 binary36의 SHA·state SHA·selected epoch·initial state·과학 source closure를 계약/native/분석에 대조했다. 새 선택·재학습 표시는 모두 false다.
- archive rawSHA: `bf7aa41942420966dc612cd1b8a8806733cf38abeb373dea127558782dbfb3ca`.
- manifest rawSHA: `b25c0e2c6e39bc8f15d2f7569ad81ed1587c8d1fdfcb907fd1bb451ef135ff27`.
- tar929개 정확 member 집합·regular file·안전한 상대 경로와 각 member SHA를 확인했다. tar 내용, originals/verified, bundle에 게시한 파일의 SHA를 929개 전수 삼중 대조하여 일치했다. 로컬 회수 증적의 기존 파일 덮어쓰기 수는 0이다.
- native6와 실제 평가72 모두 5080 UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, Python3.12.13, torch2.11.0+cu130, numpy2.4.4, polars1.39.3, CUDA13.0, cuDNN91900과 일치했다. 실제 평가72는 float32·TF32 false·deterministic true·threads4/inter-op1과 일치했다.
- native 자격 검증6은 real_data_loaded false, checkpoint_reads0, optimizer_updates0와 실제 pinned Runtime 확인 true다. local CPU 합성 검증과 구분했다.

## 수치 및 동일 대상 검증

- 36조건 모두 Validation 원참조와 재현 여섯 지표가 계약의 상대/절대 허용오차 1e-5 이내이고 모집단 수가 일치한다. 원참조와 재현 수치는 별도로 보존됐다.
- dataset/split별 모든 arm·seed의 target/truth digest·rawSHA·대상 수가 일치한다. Test 수는 Taxi8327, Intermittent88019, RAF5226, Instacart578387로 계약과 일치한다.
- 큰 수량 경계는 계약의 train 경계이며 `raw_quantity > threshold`다. 36개 모델 및 last/mean 기준선의 전체/큰 수량 216개 셀에서 SSE/count→RMSE, absolute-error-sum/count→MAE, 해당 모델의 NLL-sum/count→TimeNLL을 독립 재계산했다.
- last/mean 기준선은 dataset별 모든 arm·seed에서 동일하다. 중첩 구조에도 TimeNLL을 넣지 않았다.
- 평균/표본SD 216개, paired delta 288개, 개선/악화/부호반전 집계 48개를 분석 함수 호출 없이 Python stdlib로 독립 재계산하여 정확히 일치했다. 조건36, 비교쌍24, 각 그룹 seed42/52/62가 빠짐없다.
- 최종 completion에 결속된 source182 및 analysis3 rawSHA가 모두 일치한다. Test 결과36과 native metrics 및 baseline이 정확히 일치한다.

## 불리한 결과와 주장 경계

|자료|비교 대상|Test 전체 RMSE 개선 seed 수|반드시 함께 기록할 결과|
|---|---|---:|---|
|Taxi|보정 제거 / current-only|각 2/3|seed52는 두 대조보다 악화. 전체 TimeNLL은 두 대조 대비 3/3 악화|
|Intermittent|보정 제거 / current-only|각 2/3|seed42는 두 대조보다 악화. full 전체 TimeNLL 평균32.7176, 대조17.8020 / 23.2380|
|RAF|보정 제거 / current-only|각 0/3|전체 RMSE 및 큰 수량 RMSE·MAE가 두 대조 대비 3/3 악화|
|Instacart|MLP8 / MLP12|3/3 / 2/3|MLP12 대비 seed52 RMSE 악화. full 전체 TimeNLL 평균2.8390, 대조2.8327 / 2.8323|

이력 보정의 기존 개발 지지 조건인 동일 자료에서 두 대조 모두에 3seed strict 전체 RMSE 개선은 이번 Test에서 충족한 dataset이 없다. 따라서 기존 Validation 판정을 Test의 일관된 개선으로 확대하면 안 된다. 평균 이득과 seed별 재현성은 구분해야 한다.

단순 기준선 손해도 제외하지 않는다. RAF 과거 평균 전체 RMSE39.3579는 full 평균39.9088보다 낮다. Taxi 직전 수량의 큰 수량 RMSE506.9763은 full 평균747.8639보다 낮다. Instacart의 큰 수량 RMSE는 full23.8959보다 last23.0313·mean21.0990이 낮다. Instacart full 전체 RMSE 평균5.9698의 이득은 MLP8 5.9824·MLP12 5.9761·과거 평균5.9802 대비 작다.

분석 JSON/Markdown은 `test_is_independent:false`, 유의성 미확립, Test 선택·조정 없음과 과거 접근 이력을 명시한다. 미노출 새 평가, 통계 유의성, 외부 우월성, 모든 데이터 일반화 증거로 표현하지 않는다. 원래 학습 GPU/Runtime 차이를 보존하며 이번 공통5080 추론을 원래 학습 시간의 동일 환경 비교로 바꾸지 않는다. current-only의 prefix 이력 및 구성요소 원인분리 한계도 해결된 것으로 간주하지 않는다.

## 수용 기준

- [x] 승인된36개 조건 및 frozen checkpoint/epoch/source 유지.
- [x] 전체 Validation36 실제 정상 종료와 gate 생성 뒤 Test36 시작.
- [x] worker72 및 pipeline 실제 OS 정상 종료와 native6 검증.
- [x] 모든 원본929 및 manifest/archive rawSHA 검증.
- [x] 최종 집계의 평균·표본SD·paired delta·방향 반전 독립 재계산.
- [x] 불리한 seed, 시간·큰 수량 손해, 단순 기준선 우위 보존.
- [x] 기존 Test 평가와 새 독립 평가를 구분.

## 남은 작업 순서

**Test 결과에 맞춰 논문 주장을 정합화한다 — 다음 작업**
- 부모 세션에서 승인 범위에 따라 기존 Validation 개발 증거와 이번 기존 Test 결과를 분리해 반영한다. RAF 악화·Taxi/Intermittent seed 반전·시간 손해·단순 기준선 우위를 결과와 한계에 포함한다.

**추가 연구 여부를 판단한다 — 다음 작업**
- 검색 실험이나 모델 재조정은 이 QA의 완료 조건이 아니다. 새 비교·미노출 평가가 필요하면 별도 계약과 승인 범위에 따라 판단한다. CPU binary 감사 역시 이번 GPU Validation 재현과 구분한다.
