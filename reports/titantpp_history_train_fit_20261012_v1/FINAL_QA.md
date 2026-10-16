# 선택 checkpoint Train 적합 상태와 기존 Validation·Test 비교 — 최종 독립 QA

## Review Result: PASS

상태: **완료**. 동일 선택 checkpoint27개의 실제 Train 추론 종료·원본 회수와 기존 Validation27/Test27의 재사용, 총81개 split 평가 항목의 집계를 독립 검증했다. 새 학습0, Validation/Test 재추론0, checkpoint 재선택0이다. PASS는 계약·증적·계산의 검증이며 모델의 우월성이나 원인 규명에 대한 판정이 아니다.

### CRITICAL
- 미해결 없음.
### HIGH
- 미해결 없음. 실제 회수를 막았던 argv 비교 결함은 아래에 보존하고 별도 고정 수정의 효과를 검증했다.
### MEDIUM
- 미해결 없음.
### LOW
- 미해결 없음.

## 검토 범위와 변경 파일

사전 QA의 범위에 실제 native qualification3, Train27 worker/supervisor/pipeline 기록, 새 원본559개, 재사용 Validation/Test 원본, 분석기와 최종81개 항목을 추가했다. QA가 변경한 파일은 이 폴더의 `PRELAUNCH_QA.md`, `FINAL_QA.md` 두 보고서뿐이다. QA의 소스·원고·`.agents` 수정0, SSH0, GPU 실행0이다. 설치된 oma-qa와 프로젝트 실험 규칙을 적용했고 Python CLI이므로 npm audit·브라우저 접근성 검사는 해당하지 않는다.

## 실제 종료와 고정 증적

- Train canonical: `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf`.
- 원래 operation rawSHA: `157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b` — 12개 원래 operation 파일 무변경.
- evaluator rawSHA: `00144fc8329e68b4467c5308ed24544f2f657b08fe2896a065bc4bf3229d2cf2`.
- 실제 Train worker27 모두 exit0, 실제 pipeline OS exit0: **2026-10-12 11:47:48 KST**. 이후 회수 시점의 own process/GPU PID 부재 검증을 종료 시각과 혼동하지 않았다.
- GPU UUID: `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`; 기존5080 ai_env, torch2.11.0+cu130/CUDA13/cuDNN91900 유지.
- archive rawSHA: `26cd5a94ac02efcebdb8eb063e4397acc9c426cdd1289f35446dbb5f37523ed7`.
- manifest rawSHA: `705e863e6cbcbdab71a6f6ec105d747318ee7ce3146ddc55d6070e59869e3600`.
- 분석 JSON rawSHA: `afd2bfd8a8fefb4a3bc8b3c20b8cb54537e4a7ccf0de5495c8d427c4084cfa75`.
- 분석 Markdown rawSHA: `c102dad941eb3088fa5849b3420414e29210c27d7bf16ac1fd4b534e1a728b24`.
- analysis completion rawSHA: `8fadf88dd1983f74dbebc23118bcade67893a26800c4dac0af6f26e6b2f6f9e2`.
- 원고 rawSHA: `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354` 유지.

## 실제 발견된 운영 결함과 수정 검증

원래 `remote_finalize.py:85`는 launcher의 `[python, script, ...]`와 worker `sys.argv=[script, ...]`를 전체 동일 비교했다. 이 **HIGH 회수 차단 결함**을 사전 QA가 발견하지 못했다. 실제 추론은 이미 정상 종료했고, 회수 claim/archive 생성 전 실패했다. `RETRIEVAL_CHECK_FAILURE.json`과 별도 관측 원문을 보존했다.

원래 봉인 `control/finalize.py`는 변경하지 않았다. 별도 수정 `remote_finalize_argv_repair.py` rawSHA `0eeb2ba86fe0b116654241dd6e131e92dcd901f2501241144b70bbd6bcfd71c4`는 아래 관계를 확인하고 전체 launch argv의 고정 Python/script/root/job/split/device 검사를 유지한다.

```python
child_exit['argv'] == launch['argv']
output['argv'] == launch['argv'][1:]
```

실제27개의 PID/PPID/argv/exit0 기록과 negative controls를 대조했다. 수정의 자기 SHA, 이전 claim/archive 부재, exclusive repair source/admission, 원래 종료·무변경 검사와 회수 claim을 검토했다. 실제 귀환 archive에 별도 수정 source/admission이 포함됐고 원래 operation 봉인이 유지됐다. 실패 은폐·원래 봉인 교체·새 학습·추론·worker retry는 없다.

## Acceptance criteria

- [x] 선택27 checkpoint byte SHA·state SHA·selected epoch·dataset/arm/seed가 원래 계약 및 재사용54 receipt와 일치한다. frozen source129 및 data 변경0.
- [x] native synthetic3 모두 실제 exit0·hook 비간섭 bitwise equality 통과. native 검사에서 선택 checkpoint 접근0, 실제 dataset 로드0.
- [x] 실제27 worker의 exact argv/PID/PPID, supervisor exit, timeout/abort 부재, pipeline 실제 OS exit0 및 receipt 간 SHA 연결을 확인했다.
- [x] 최종 retrieval receipt의 own_processes_absent/own_gpu_pids_absent와 고정 GPU/Runtime을 확인했다. QA가 추가 원격 조회하지 않았다.
- [x] 새 archive559 member 각각의 stdlib SHA, root 및 originals/verified 사본1,118개와 manifest를 독립 검증했다. LOCAL_VERIFICATION의 verified와 일치한다.
- [x] completion에 연결된 새 source_files1,122개와 재사용 source_files1,496개를 각각 stdlib SHA로 검증했다. 기존 Validation/Test의 원관측시각을 새 Train 시각으로 갱신하지 않는다.
- [x] Train 대상은 frozen eligible event 기준 Taxi38,393/Intermittent393,824/RAF25,779다. target identity·truth quantity SHA가 dataset별9모델에서 같고 계약과 일치한다. Train 첫 event 제외 및 target split을 유지한다.
- [x] selected model.eval/inference_mode, shuffleFalse/drop_lastFalse B128 평가다. 이동 중인 학습모델의 무작위 배치 log loss 평균과 이번 raw RMSE/MAE를 같은 지표로 취급하지 않는다. 기존 Train raw 성과 재현이라고 주장하지 않는다.
- [x] actual Train receipt의 전체·tail 지표를 telemetry와 대조했다. 98,397개 strata cell의 RMSE=sqrt(SSE/n), MAE=SAE/n, TimeNLL=sum/n을 재계산했고 885,573개 feature 통계의 평균·분산·두 공분산을 대조했다.
- [x] 27행×3split=81항목의 metrics/baselines가 원래 receipt와 객체 단위로 일치한다. 각 항목의 checkpoint/state/epoch, target/truth SHA와 전체/tail 목표 평균도 원문과 같다.
- [x] 9 dataset/arm group의3seed 평균·표본SD를 전체/tail raw RMSE·MAE·bias·TimeNLL에 대해 재계산했다.
- [x] full 대 두 대조18쌍×3split의 delta와 수량 구간·시간 간격·tail series별 순 SSE 기여를 원문에서 재현했다.
- [x] 기존 population_diagnosis rawSHA `e2bd494fc4519904fae8fce0049ede45bb4a842771a63be4e31ff7886fb07508`의 대상 연결을 검증한 뒤만 target SD/범위를 사용한다. 오류 분산으로 목표 분산을 대신하지 않는다.
- [x] 실제 frozen raw quantity 변환에 finite upper cap이나 raw_scale 재곱셈이 없다. 실수식 등가성과 float32 bitwise equality를 구별한다. Intermittent42 Test 최소σ0.0044993771은 floor0.001보다 크며 hard floor가 작동했다는 증거로 쓰지 않는다.
- [x] 집중 테스트: evaluator6개 PASS(9.793초), 분석기13개 PASS(0.008초), 회수 argv 회귀4개 PASS(0.001초). AST parse PASS. QA 테스트는 선택 checkpoint를 추가 추론하지 않는다.

## 수치와 해석 정합

아래는 full 모델이며 순서는 **Train / Validation / Test**다. 전체81항목 및 두 대조는 `analysis/TRAIN_FIT_REPORT.md`와 JSON에 모두 보존한다.

|대상|원문에서 대조한 핵심 수치|허용되는 해석|
|---|---|---|
|RAF seed42/52/62|Train tail RMSE461.98/468.43/458.95, bias−354.74/−364.10/−350.86|큰 수량 오차와 음의 평균오차가 Train부터 존재한다. 평가 구간만의 악화로 설명할 수 없다.|
|Taxi seed52|tail RMSE413.31 / 370.76 / 1017.49, tail bias−199.26 / −109.26 / −888.41|Train의 tail 오차에 더해 Test에서 이 seed의 악화가 크게 관측된다.|
|Intermittent seed42|전체 TimeNLL0.2193 / 0.6187 / 62.5721, tail quantity bias−1.9414 / −0.1661 / +12.9104|Test 시간 분포 적합의 악화와 수량 과대예측 방향을 분리해 다뤄야 한다.|

목표 tail로 조건을 건 bias만으로 underfitting이나 모델 용량 부족을 증명하지 않는다. split별 대상 분포·이력 구성이 다르므로 Train–Test 차이를 순수 과적합이나 보정의 인과 효과로 단정하지 않는다. 현재 Test는 이미 노출된 진단 대상이며 새로운 독립 평가가 아니다. 세 dataset을 모두 수량 과소예측으로 묶지 않는다.

## 남은 작업 순서

**현재 관측으로 지지되는 원인 후보와 제한을 정리 — 다음 작업**
- 신규 Train27과 재사용54의 완료 증적을 기준으로 RAF의 Train tail 적합, Taxi52의 평가 악화, Intermittent의 시간 분포 문제를 각각 기술한다. 순수 underfit·과적합·구성요소 인과로 확대하지 않는다.

**추가 계측을 최소 계약으로 분리 — 승인 필요**
- RAF H1의9모델×Train/Validation=18pass와 Intermittent H2의9모델×Train/Validation=18pass는 아직 미실행 후보다. 현재 QA PASS가 추가 실행 승인은 아니다. 추가 학습은0으로 계획하되 향후 학습 요청은 별도 계약이 필요하다.
- H2의48점수셀은 원래 full 셀 중복을 포함하며 새 교차셀24개다. 관측 출력 교환의 진단 범위를 넘어 새 배포 모델이나 구성요소 인과 실험이라고 부르지 않는다.

## 최종 설명 문서와 후속 순서의 좁은 추가 QA — PASS

README, RUN_STATUS_COMPLETE, EXISTING_EVIDENCE_DIAGNOSIS, MINIMUM_FOLLOWUP_CONTRACT 및 RTX handoff의 PAPER_WORK_ORDER를 실제 완료 후 문서로 대조했다. 원본/native 전수 검증은 반복하지 않았고 추가 SSH·추론0이다.

- 진단 문서의 full9×3split×5수량 지표135개 표기와 Intermittent3seed×3split의 TimeNLL/평균σ18개를 authoritative 분석 JSON에 허용 반올림 오차 이내로 대조했다. Taxi52/62의 대조별 tail RMSE와 RAF의 input-only `last`/`mean` 기준선 숫자도 원문과 같다. 과거 평균 기준선을 learned mean prototype의 성과와 혼동하지 않는다.
- RAF Train tail 실제 평균433.8878, 예측 평균69.7844~83.0262, tail 목표 SD293.7987 및 두 기준선 tail RMSE426.5618/412.1775를 확인했다. Taxi와 Intermittent의 split별 목표 평균/모집단 SD·N도 같은 분포 원문과 일치한다. raw target/prediction 평균과 로그 공간 값의 규모를 혼합하지 않는다.
- 실제 pipeline exit의11:47:48과 이후 원본 회수·소유 PID 부재 확인은 별개다. RUN_STATUS_COMPLETE의 기록 시각12:00:35를 모델 종료 시각으로 바꾸지 않는다. 기존 v2 Validation/Test의 관측 시각과 결과는 재사용 원문으로 유지한다.
- H1 RAF9×Train/Validation18pass가 첫 추가 실행 후보이고, H2 Intermittent9×Train/Validation18pass는 별도 후보다. 둘 다 새 학습0·새 Test0·현재 미실행이다. H2의48 paired 셀/새 교차24셀, 고정 정수 likelihood와 원래 두 셀 재현 조건이 일치한다.
- PAPER_WORK_ORDER의 원고 정합 단계3은 현재 숫자·예외 반영을 두 후보 완료와 독립적으로 시작할 수 있게 설명한다. 후속 학습6/18fresh·구성 대조·검색은 필요성 판단과 별도 계약을 거치는 조건부 작업이며 현재 신규 학습 권고0과 모순되지 않는다.
- 검토 문서의 로컬 Markdown 참조가 모두 존재한다. 미해결 C/H/M/L 없음.

검토 시점 문서 rawSHA:

|문서|SHA256|
|---|---|
|README.md|`d3b2cff58c8c9194e9edd2e438575f8c1b07dfaab21cf2090a431a295a7e2671`|
|RUN_STATUS_COMPLETE.json|`b92d66ebd6f8211da77b5275ed3eb2ffc458c64f9277d69a570ffc166e8873ae`|
|EXISTING_EVIDENCE_DIAGNOSIS.md|`6bcfcf3300b2841a4cbe9405639a563417f1b144d7fc5d0c89b960b498cd07e8`|
|MINIMUM_FOLLOWUP_CONTRACT.md|`f0764e88add92c72ef613517e3d7e818994267d17f6bf2caf6a4e3332c9a96d6`|
|PAPER_WORK_ORDER.md|`4c1aa3d6e1c67320edecc22f7725be393d836c56820a2af2b51a770532cbfbfc`|
