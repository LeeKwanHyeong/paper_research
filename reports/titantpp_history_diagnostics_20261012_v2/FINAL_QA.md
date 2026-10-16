# 저장 모델 진단 계측 v2 최종 QA

## Review Result: PASS

**상태: 실제 평가·원본·수치 검증 및 수정 보고서 검토 완료.** 고정 선택 checkpoint27개에 대한 Validation27 → Test27, 총54개 평가의 실제 worker exit 0과 pipeline 실제 OS exit 0, 원본 회수, 로컬 SHA 및 기존 평가 재현을 확인했다. 이는 저장 모델의 관측 진단 완료이며, 독립 일반화 검증·인과 경로 실험·새 학습 완료를 뜻하지 않는다.

### CRITICAL

- 없음.

### HIGH

- 없음.

### MEDIUM

- 없음.

### LOW

- 없음. 최종 문서의 두 LOW 정정은 아래 재확인 절에 기록한 대로 해소됐다.

## 요약 및 수정 이력의 보존

v1은 메타데이터 membership 집합을 `None`으로 덮어쓴 코드 결함으로 첫 Validation 세 worker가 실패했다. v1의 부분 시도3·Validation 완료0·Test 완료0·실제 pipeline exit1·claim/failure와 기존 PRELAUNCH_QA를 보존했다. 해당 결함은 별도 v2 root와 새 instrumentation 봉인으로 수정했다. **v2의 완료54는 v1의 부분 시도3을 지우지 않는다.** 원래 과학 source129·데이터·선택 checkpoint27·원래 Test 결과는 변경하지 않았다. 수정 근거는 `../titantpp_history_diagnostics_20261012_v1/REPAIR_QA.md`에 기록했다.

QA가 변경한 파일은 이 `FINAL_QA.md`다. QA는 추가 SSH·모델 재추론·학습·checkpoint 재선택·원본 수정을 수행하지 않았다. 실제 추론 및 원격 회수는 root 담당자가 수행했으며, QA는 회수된 로컬 증적과 집계 값을 독립 검증했다.

## 고정 원본과 실행 증적

| 항목 | 값 |
|---|---|
| v2 canonical contract | `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d` |
| operation integrity rawSHA | `236ad7a57096bd91811bf8ad0cc1a6b389c43b3272ac8520687683b7b8458682` |
| 과학 source129 closure | `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a` |
| frozen science revision | `57da14a2a7a5687e1bdef7d3164f3e8c0051da59` |
| 수정 evaluator rawSHA | `087a181e00f46bb4a36f37ce3df9929a75094c39a7f6359bebba4c03c5364d25` |
| 원본 archive rawSHA | `4efb0453b5590eac18e5bcbfa29062970c9c5c291f8672cc14ad32ddce24c3f7` |
| 원본 manifest rawSHA | `e2dd3217e2657d9fbabc9589c1d53bd9dad0707f2eeb8d7af1763a0248e65588` |
| 원본 manifest 파일 수 | 746 |
| GPU UUID | `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6` |
| Python | `/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12` |
| analysis script rawSHA | `1b02541d33c26f1146f3a7bad770b81acebe26f14023cd0c6f97eed00d70e306` |
| analysis completion rawSHA | `f689243b21dcd556af81b33117f994191bc6b2d0a139634153f187172560f5ea` |
| diagnostic_conditions rawSHA | `8590a30c122818c94c4f11f196b63095ef6bcb42e9fedf4fd8ac1bee33462b6f` |
| DIAGNOSTIC_REPORT rawSHA | `39edeecce72d223434c2f408eede9fc9727f9d30e3bfe1cfdda2e74f32b929b1` |

증적 root는 `search_artifacts/titantpp_history_diagnostics_20261012_v2`다. 원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v2_5080`이며, 원래 Test parent는 `titantpp_history_instacart_test_20261012_v1`다. v2 명칭과 v1 schema 형식의 공존은 의도된 것이며 canonical pin으로 실행을 구별했다.

`originals/LOCAL_VERIFICATION.json`의 원관측 확인 시각은 **2026-10-12 10:53:58.455070 KST**다. QA는 archive의 746개 regular member를 다시 해시하고, materialized root와 `originals/verified` 두 사본의 모든 파일 SHA를 manifest와 대조했다. archive·manifest 자체 SHA도 일치했다.

## Acceptance checklist

- [x] 3 dataset × 3 seed × 3 arm의 선택 checkpoint27과 source129를 보존. 원본 manifest 및 기존 선택 SHA 결합 확인.
- [x] native synthetic 3개 route 실제 exit0, checkpoint 읽기0·실제 데이터 로드0, hooked/unhooked 원래 출력 bitwise equality, finite, qualification PASS 확인.
- [x] Validation27 실제 worker exit0 및 원래 Validation 지표·대상 재현 뒤 전역 gate 확정.
- [x] 마지막 Validation actual exit `1791769637.7873802` ≤ gate 생성 `1791769637.7953472` ≤ 첫 Test 시작 `1791769638.1601312`. 각 Test의 gate rawSHA 일치.
- [x] 전체54 worker의 정확 argv·PID·PPID·job/split·실제 exit0·timeout false·supervisor_aborted false 결합 확인.
- [x] 고정 GPU UUID, Torch/CUDA/cuDNN/NumPy/Polars Runtime 일치. 저장 tensor SHA와 평가 후 state SHA 일치.
- [x] 대상 identity·truth/history digest가 원래 native reference와 정확히 일치하고 동일 dataset/split의 모든 seed·arm 사이에서도 일치.
- [x] 전체/큰 수량의 RMSE·MAE·TimeNLL 여섯 지표를 원래 native reference와 허용오차 1e-5로 독립 비교.
- [x] 54개 telemetry의 169,290개 strata cell에서 count·SSE·SAE·signed error·TimeNLL 합과 저장된 RMSE/MAE/NLL, feature count·mean·joint covariance 재계산 확인.
- [x] full9 × 양 split에는 실제 CNN/GRU norm이 존재. 대조18 × 양 split에는 해당 full 이력 norm을 가짜0으로 넣지 않았음을 확인.
- [x] 분석18개 대조 × 양 split, 112,860개 contrast cell의 ΔSSE/N전체와 bias² + centered error variance 재구성을 독립 검증. 각 차원의 count/key가 arm 사이에 같음.
- [x] 실제 pipeline OS exit0, 마지막 complete phase, owner process·GPU PID 부재를 보존된 종료/회수 receipt에서 확인. 원격 추가 조회 없음.
- [x] exclusive archive 회수와 로컬 두 사본·archive member·manifest SHA 검증 통과. 기존 원본 덮어쓰기 없음.
- [x] analysis completion에 결합된 output 파일 SHA와 실행 분석기 SHA 일치. 전체27 조건·두 대조·두 split 보존.
- [x] 신규 학습0·경로 개입0·parameter update0·새 checkpoint 선택0, 현재 Test의 기존 노출 이력 및 비독립성 유지.

## 재현된 대상 수

아래 N은 각 seed·arm에 공통인 평가 대상 수이며, seed·arm별 반복 평가를 합쳐 독립 표본 수로 해석하지 않는다.

| 자료 | Validation 전체 / 큰 수량 | Test 전체 / 큰 수량 |
|---|---:|---:|
| Taxi | 8,268 / 79 | 8,327 / 83 |
| Intermittent | 86,285 / 1,141 | 88,019 / 243 |
| RAF | 6,690 / 50 | 5,226 / 39 |

## 분석기 품질과 수치 해석

`paper/scripts/analyze_titantpp_history_diagnostics.py`는 stdlib만 사용한다. 명시적 `--contract-sha256` pin을 요구하고, failed pipeline/worker root를 완료 보고로 바꾸지 않으며, 원본 검증·실제 종료·전역 Validation gate가 선행돼야 한다. toy test11개를 QA가 독립 실행해 모두 통과했다(0.003초). 원래 과학 source의 재추론이나 새 평가를 수행하지 않는다.

ΔSSE/N전체·ΔSAE/N전체·ΔTimeNLL합/N전체는 동일 대상의 오류 합 분해다. RMSE 차이 자체를 합산하지 않으며 서로 다른 partition을 합치지 않는다. tail_by_series는 큰 수량 subset에 대한 전체 분모 기여라는 점을 구분했다. bias²와 centered error variance는 조건 내 오차의 항등 분해이며 모델 불확실성·seed 분산·원인 분리를 뜻하지 않는다.

확인된 핵심 사례는 다음과 같다.

- **Taxi seed52:** 큰 수량83건의 네 series가 양 대조 모두에 대해 full의 큰 수량 SSE 손해를 만든다. full의 해당 series 평균 오차는 약 −741~−1,177로 음의 bias가 나타난다. 이는 실제 손해 위치와 오차 방향이며 CNN 또는 GRU의 인과 효과를 단정하지 않는다. 다른 seed의 이득도 보고서와 전체 JSON에서 보존했다.
- **Intermittent seed42:** Test 전체 TimeNLL 손해는 보정 제거 대비 약 +47.1766, current-only 대비 약 +51.2108이다. 기록된 목표 gap 구간별 손해·이득 합과 실제 μ/σ 관측을 제시했다. 다음 gap은 사후 설명 변수이며 이를 예측 전 gating 정보로 사용할 수 없다. 시간 head에서 작은 σ가 동반됐다는 관측을 원인 또는 calibration 검증으로 확대하지 않는다.
- **RAF:** 큰 수량39건은 모두 Train 최대 입력 이력14 이내이며, 15~19개 입력 구간에는 해당 큰 수량 대상이 없다. 따라서 “Train보다 긴 이력이 큰 수량 손해의 주원인”이라는 설명은 이번 분해로 지지되지 않는다. 큰 수량39건 중 Validation에도 등장한 series의32건이 손해 대부분을 설명하고, Validation에 없는 series7건은 seed/대조에 따라 이득 또는 손해다. “새 series만의 문제”로 좁힐 수도 없다.

모든 norm은 부호 없는 벡터 크기다. 공동 모멘트 기반 covariance는 관측 연관성이며, 어떤 CNN/GRU 경로가 수량을 증가·감소시켰는지 또는 해당 구성요소가 필요한지를 증명하지 않는다. 희소 큰 수량 대상은 series 안에서 종속될 수 있으므로 유의성·독립 표본·외부 일반화 주장으로 확대하지 않는다.

## 남은 작업 순서

**관측 진단을 논문 주장과 연결 — 다음 작업**
- Taxi·Intermittent의 seed 반전과 RAF의 큰 수량 bias를 제한된 결과로 설명한다. 새 독립 평가 또는 구성요소별 인과 검증으로 표현하지 않는다. 현재 요청 범위의 진단 계측은 완료됐다.

**실제로 필요한 최소 후속 실험 결정 — 다음 작업**
- 이번 결과로 지지되지 않은 긴 이력 원인 설명을 제거하고, 시간 head·큰 수량 손해·구성요소 분리 중 논문 주장에 꼭 필요한 질문만 남긴다. 추가 학습·경로 개입·새 자료 평가는 현재 진단 실행과 별도 범위로 정리한다.

QA 보고 작성 KST: 2026-10-12T10:57:44+09:00

보고서 구간 선택의 MEDIUM 결함은 해소됐다. 이전 Intermittent 상세 표가 bin2=(31,46]을 선택했던 문제를 실제 경계값으로 bin1=(2,31]을 찾도록 수정했다. 수정본의 두 대조 joint 표는 각각 N=33,203이며, 모든3 seed×3 arm의 feature 표9개도 같은 범위와 N=33,203을 표시한다. 단순 숫자 bin index 가정 대신 물리적 경계를 찾는 회귀 테스트까지 포함해 11/11을 QA가 독립 실행했다.

원래 analysis/claim/completion은 `analysis_revisions/v1`에 보존됐고 `REVISION_REASON.json`의 네 원본 SHA를 독립 대조했다. 수정 전후 `rows`, `contrasts`, `source_files`가 정확히 같으므로 전체54 결과·모든 계층 수치·원본은 변하지 않았다. 새 report/completion/JSON/claim SHA와 수정 분석기 SHA를 검증했다. 이 수정 과정에서 GPU/모델 평가를 다시 실행하지 않았다. RAF tail39 중 within39·above0의 실제 사실과 긴 이력 단독 원인 해석의 제한도 수정 보고서에 반영됐다.


## 추가 원인 후보의 독립 수치 확인

- Intermittent seed42의 `2 < q ≤ 31` 대상33,203건 중 비절단33,183건·절단20건이다. 비절단 구간의 전체 ΔMSE 기여는 보정 제거 대비1.0250073/current-only 대비0.7384018이며, 절단 구간은0.00331736/0.00601540이다. 전체 수량에서 절단 대상의 ΔMSE 기여는0.00132488/−0.00053924로 전체1.8559154/0.6202821에 비해 작다. 따라서 이 사례의 전체 반전을 길이 cap 절단이 주로 설명한다는 주장은 지지되지 않는다.
- 같은 full Test 시간 head의 σ 평균0.0395892는 보정 제거0.1068280/current-only0.0945537보다 작고 실제 TimeNLL은62.5721210이다. 작은 scale과 큰 TimeNLL의 동반은 확인했으나 scale 변경 효과나 원인은 별도 개입 없이 확정하지 않는다.
- Taxi seed52의 큰 수량 CNN Q 잔차/base 평균0.6458219, GRU 보정/base 평균0.4151165는 좋은 seed42의0.6604018/0.4354696보다 크지 않다. 따라서 최대 norm 또는 norm 크기 하나로 반전을 설명할 수 없다.

위 값은 이미 검증한 실제 telemetry/contrast의 합계와 평균에서 독립 재계산했으며 모델 재실행은 없다.


## 최종 문서의 좁은 검토

대상은 같은 폴더의 README.md·FOCUSED_INTERPRETATION.md·FOCUSED_NUMBERS.json·RUN_STATUS_COMPLETE.json 및 RTX 인계 PAPER_WORK_ORDER.md다. 추가 SSH·모델 평가·큰 원본 전수 재검증 없이 기존 실제 분석의 필요한 집계와 문서만 대조했다.

- 중대한 수치·주장 오류 없음. Taxi 두 series의 실제 전체 분모 기여3819.703053694935와 첫 series bias−1160.8231312144887에 대한 본문 반올림3819.703/−1160.823, 손해 비중82.7%가 일치한다. Intermittent gap>5의 순 TimeNLL 손해 비중60.20%/67.51%, σ 및 대상 수의 반올림도 일치한다.
- Markdown 상대 링크는 모두 존재한다. 완료54와 실패 v1 부분시도3의 구분, 기존 Test의 비독립성, 연관과 원인의 구분, 단일 가설→조건부 구성 실험→동결→미노출 확인의 의존순서와 추가 실행 승인 경계가 정합한다. 원고는 미수정 상태로 구분돼 있다.
- LOW 문서 정정: `README.md:26`의 v2 봉인 transport66을 실제67멤버로 맞춘다. v2에는 repair decision 파일이 추가됐다.
- LOW 시간 표현 정정: `../titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md:45`에서 10:51:06의 실제 pipeline OS exit와 그 뒤의 owner/GPU PID 부재 확인을 분리해 쓴다. 후속 부재 관측을 exit의 원시각으로 소급하지 않는다.

두 문서 정정은 실제 평가·집계 검증 판정에 영향을 주지 않는다. root 문서 담당자에게 전달했으며 QA는 다른 작업자의 파일을 수정하지 않았다.


## 최종 문서 재확인 — PASS

요청한 두 줄만 재확인했다. `README.md:26`은 실제 운영·참조·복구 판단 **67멤버**로 수정됐고, `PAPER_WORK_ORDER.md:45`는 **10:51:06 KST의 실제 pipeline exit0**과 **종료 후 별도의 소유 GPU/PID 부재 확인**을 구분한다. LOW 두 건 모두 해소됐다. 미해결 CRITICAL/HIGH/MEDIUM/LOW는 없으며 실제 평가·집계 및 최종 문서의 판정은 **PASS**다. 이번 재확인에서는 큰 원본 재검증·추가 모델 실행·SSH를 수행하지 않았다.
