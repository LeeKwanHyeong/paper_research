# 선택 checkpoint Train 적합 상태 평가 — 사전 QA

## Review Result: PASS

상태: 계약·로컬 구현·회귀 테스트·봉인 검토 완료. 실제 native qualification, Train27 평가, 종료 및 원본 회수는 이 사전 판정에 포함하지 않으며 별도 최종 QA에서 확인한다.

### CRITICAL
- 없음.
### HIGH
- 없음.
### MEDIUM
- 없음.
### LOW
- 없음.

## 검토 범위와 변경 파일

검토 대상은 `paper/scripts/evaluate_titantpp_history_train_fit.py`, `paper/scripts/run_titantpp_history_train_fit_pipeline.py`, 같은 reports 폴더의 준비·배포·launch·finalize 스크립트, 선택27 계약과 부모 v2 원본이다. QA가 작성한 파일은 이 보고서뿐이다. 소스·과학 모델·원고·`.agents` 파일 변경, SSH, GPU 실행은 수행하지 않았다.

프로젝트 `.codex/agents/qa-reviewer.toml`의 설정은 `gpt-6-astra`/`high`다. 설정상 모델과 실제 실행 모델에 관한 추정은 구분한다. 프로젝트에 명시된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 존재하지 않아 설치된 `oma-qa/resources/execution-protocol.md`를 적용했다. `.agents` 수정 금지와 지정한 결과 소유권에 따라 보고서는 이 경로에 남긴다. Python CLI 작업이므로 npm audit·브라우저 접근성 검사는 적용 대상이 아니다.

## 고정 계약과 source 간선

- Train canonical: `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf` — canonical JSON을 독립 재계산했다.
- operation rawSHA: `157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b` — 12개 파일 전부 확인했다.
- evaluator rawSHA: `00144fc8329e68b4467c5308ed24544f2f657b08fe2896a065bc4bf3229d2cf2` — 현재 코드와 배포본 일치.
- 부모 v2 canonical: `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d`.
- 부모 archive rawSHA: `4efb0453b5590eac18e5bcbfa29062970c9c5c291f8672cc14ad32ddce24c3f7`; manifest rawSHA: `e2dd3217e2657d9fbabc9589c1d53bd9dad0707f2eeb8d7af1763a0248e65588`.
- 부모 archive746개 member, root/verified 두 로컬 사본1,492개, analysis completion 및 연결된 output SHA를 stdlib로 독립 검증했다.
- 새 input359개 SHA와 재사용 Validation27/Test27의 receipt54개·telemetry54개 SHA를 검증했다. 새 계약의 jobs27·datasets·sources가 부모와 객체 단위로 동일하다.
- 원고 SHA `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354` 유지 확인.

## Acceptance criteria

- [x] 사용자 최신 “이거 먼저 진행하자”에 따른 동일 선택 checkpoint Train 적합 상태 추론. 신규 학습0, 경로 개입0, 모델 재선택0, Val/Test 추가 추론0.
- [x] 3 dataset × 3 arm × 3 seed 모두 포함. 데이터·checkpoint·source129·Runtime·Train에서 고정한 큰 수량 경계 유지.
- [x] 원래 Train 대상은 Taxi38,393/Intermittent393,824/RAF25,779. frozen loader의 first event 제외 및 target split 규칙, target identity·quantity SHA 결합을 확인했다.
- [x] 원래 학습 로그는 `model.train()` 및 배치별 parameter update 중 얻은 log objective 평균이다. selected checkpoint의 raw 수량 적합 평가와 혼동하지 않는다. 부모 manifest에 동일 선택 checkpoint Train 평가 산출물은 없다.
- [x] deterministic eval, inference_mode, requires_grad false, shuffle false, drop_last false, B128, worker0. Train/Validation export만 materialize하고 Test 대상은 추론하지 않는다.
- [x] frozen 모델의 원래 target withholding·시간 likelihood·raw quantity transform을 유지. 출력/모델 tensor/source/input 사후 무변경 확인 경로 존재.
- [x] full/current-only/보정 제거를 동일 대상에 비교하며, current-only hidden의 prefix 의존성과 norm의 부호 없는 성격을 유지한다.
- [x] 로컬 fabricated CPU 테스트6개 PASS(9.793초). 실제 frozen loader membership/context, 실제 worker 실행 경로, split·byte·population negative control, 원래3 route에서 hooked/unhooked 출력 bitwise equality 및 state 보존을 포함한다. 실제 선택 checkpoint를 이 테스트에서 추론하지 않았다.
- [x] AST parse 통과. native synthetic3가 실제 Train27 전에 실행되며 실제 exit0·finite·hook 비간섭·target withholding을 통과해야 다음 단계로 간다.
- [x] supervisor는 동일 dataset/seed의 세 arm을 최대3개 동시 실행하고 실제 argv/PID/PPID/exit를 보존한다. 계약 root·고정 Python·GPU UUID를 확인한다.
- [x] 환경의 과거 source/cache 값은 child_environment에서 새 격리 root로 덮어써 import/cache를 격리한다. 기존 Runtime·인증은 변경하지 않는다.
- [x] claim/실패/기존 root가 있으면 재실행·자동 retry/resume를 거부한다. 배포 archive 경로 traversal·symlink·중복 member를 검증하고 원본은 copy+SHA로 보존한다.
- [x] finalizer는 Train27 receipt 및 actual worker exit0, 실제 pipeline OS exit0, 종료 후 자체 PID/GPU PID 부재, native qualification, immutable input/op SHA를 요구한다. exclusive retrieval 및 archive 전체 member SHA를 확인한다.

## 실제 실행 이후 필요한 확인

**Train27 평가와 재사용54 증적을 합쳐 확인 — 진행 중**
- actual native3 qualification과 실제 worker27/pipeline exit0, 모든 원본 archive·manifest·로컬 SHA를 검증한다.
- Train raw RMSE/MAE·tail 및 bias를 동일 선택 모델의 Validation/Test와 비교하되 각 split의 다른 대상 분포와 원래 Test 노출 이력을 표시한다.
- 시간 head 좁은 분포와 수량 과소예측의 연관 진단을 인과 증명으로 확대하지 않는다. 추가 개입·학습은 별도 최소 계약의 실행 범위다.

## 실행 대기 중 로컬 분석기 보강 검토 — PASS

Train 실제 평가가 진행 중인 동안, 새 원본·실제 완료 판정을 만들지 않고 로컬 분석기와 기존 충분통계 설명만 좁게 재검토했다. `test_train_analysis.py`의 13개 테스트를 독립 실행해 모두 통과했다(0.008초). 기존 평가 코드/operation 봉인에 변경은 없다.

- `analyze_titantpp_history_train_fit.py:199`의 모집단 reader는 이미 고정된 `population_diagnosis.json` rawSHA `e2bd494fc4519904fae8fce0049ede45bb4a842771a63be4e31ff7886fb07508`, 과학 source, 기존 계약, 추론0 범위, dataset bytes를 검증한다. `:216` 이후 split별 target/truth-history digest와 count·평균을 실제 receipt에 결합한 뒤 기존 목표 수량 SD/범위를 가져온다. 오류 통계에서 목표 분산을 만들어내지 않는다.
- `EXISTING_EVIDENCE_DIAGNOSIS.md:31`의 cap/scale 설명을 frozen `CountAwareTPP.py:770`에 대조했다. `expm1(softplus(logit))`에 finite 상한·raw_scale 재곱셈은 없다. `:789`의 log MSE 분기는 tail-aux clamp 이전 반환하므로 현재 tail을 출력 cap 때문이라고 단정하지 않은 설명이 타당하다. 실수식 `exp(logit)`과 실제 float32 두 연산식의 bitwise 동등성을 구별했다.
- `EXISTING_EVIDENCE_DIAGNOSIS.md:47`의 Intermittent42 Test 최소 σ는 실제 telemetry `0.004499377105943636`으로 floor0.001보다 크다. frozen `CountAwareTPP.py:759`의 `floor+softplus(raw_scale)`에 대조했고, floor 수정의 해결 효과를 주장하지 않았다.
- bias²/MSE 비중을 원래 tail 통계에서 재계산했다. RAF Validation72.7490~74.5701%/Test76.9417~78.3186%, Taxi52 Validation8.6836%/Test76.2371%가 설명의 반올림과 일치한다. 목표 tail로 조건을 건 평균 오차만으로 underfitting·순수 과적합·인과 원인을 증명하지 않는 제한이 유지된다.
- H1 RAF9×Train/Validation의18pass와 H2 Intermittent9×Train/Validation의18pass는 미실행 후보로 표시했다. H2는6pair×2split×4셀=48점수셀(원래 full 셀 중복 포함), 새 교차24셀인 관측 출력 교환이다. 모델 구성요소 인과 실험이나 배포 가능한 새 모델로 표현하지 않았다.

미해결 CRITICAL/HIGH/MEDIUM/LOW 없음. 이 추가 검토는 로컬 분석 품질의 PASS이며, Train27 완료·회수·최종 수치 PASS를 선행 선언하지 않는다.

## 종료 후 회수 검사 오류와 독립 운영 수정 검토

원래 사전 검토에서 발견하지 못한 회수 검사의 결함이 실제 종료 후 드러났다. 원래 `remote_finalize.py:85`는 launcher의 `[python, script, ...]`와 Python worker의 `sys.argv=[script, ...]`를 전체 동일 비교해 정상 worker를 거부했다. 실패는 회수 claim/archive 생성 전에 발생했고 `RETRIEVAL_CHECK_FAILURE.json`에 보존됐다. Train 추론 실패나 학습 실패를 뜻하지 않는다.

**원래 결함: HIGH(회수 완료 차단), 별도 수정 검토: PASS.** 기존 봉인 `control/finalize.py`는 그대로 보존한다. 새 `remote_finalize_argv_repair.py` rawSHA `0eeb2ba86fe0b116654241dd6e131e92dcd901f2501241144b70bbd6bcfd71c4`의 diff를 검토했고 필요한 교정은 다음과 같다.

```python
child_exit['argv'] == launch['argv']
output['argv'] == launch['argv'][1:]
# 별도로 launch 전체 argv == 고정 Python + 고정 script/root/job/split/device 검증 유지
```

실제27개의 launch/exit/worker PID·PPID·argv 원문으로 위 관계와 exit0를 확인한 read-only observation을 검토했다. 새 회귀 테스트4개를 독립 실행해 PASS(0.001초)였고, 다른 executable/script/인자/identity drift를 통과시키지 않는다. 수정 코드는 자체 SHA를 확인하고 이전 claim/archive/repair admission이 없음을 요구하며, 모든 기존 종료·원본 검사를 유지한 뒤 exclusive repair source·admission·retrieval claim을 기록한다. 새 학습·추론·worker 재실행은0이다. 남은 미해결 C/H/M/L 없음. 실제 회수 및 로컬 archive 검증은 최종 QA에서 별도로 확인한다.
