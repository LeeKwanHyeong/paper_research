# 수량 목표와 시간 분포를 분리하는 최소 후속 계약 후보

이 문서는 **검토 가능한 후보이며 실행 승인서가 아니다.** 현 작업은 같은 선택 모델의 Train27 추론과 완료된 Validation27·Test27의 로컬 비교다. 아래 H1/H2 및 새 학습은 이번에 실행하지 않았다. Train 실제 종료·559원본의 archive/manifest·로컬 두 사본 SHA 및81개 split entry 분석 뒤 **H1 RAF의 log/raw 오류 추가 관측을 먼저, H2 Intermittent μ·σ 교차 점수를 별도 후속**으로 둔다. 원고 정합은 이 추가 실행을 기다리지 않고 현재 결과만으로 먼저 할 수 있다.

## 같은 선택 모델의 적합 상태를 기준선으로 둔다 — 완료

- 대상은 Taxi·Intermittent·RAF×full/보정 제거/current-only×seed42/52/62의 선택 checkpoint27개다. 현재 Train canonical contract는 `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf`, operation rawSHA는 `157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b`다.
- 2026-10-12 11:47:48 KST에 새 Train27의 actual worker/pipeline OS exit0을 확인했고 기존 v2 Validation/Test54의 같은 checkpoint·선택 epoch·state/source/target/truth-history SHA를 재사용해81개 split entry를 분석했다. Train은 eval/no_grad이며 학습 당시 batch loss와 다르다. prior selected-checkpoint Train 지표는 없으므로 과거 Train 수치 재현 완료라고 부르지 않는다.
- raw 전체/큰 수량 RMSE·MAE·bias와 원래 TimeNLL, input-only 직전 수량/과거 평균, 대상 수량 mean/SD·tail 비중을 같이 비교한다. 모든 좋은 seed와 대조를 유지한다. true tail의 조건부 평균 오차는 평균회귀·확률성으로도 생길 수 있으므로 Train에 남았다고 바로 용량/최적화의 underfitting으로 확정하지 않는다.
- 봉인된 raw decode에 유한 상한 cap·raw_scale 재곱셈은 없다. 현재 시간 계약은 모든27개 metadata에서 `positive_integer_round_clamp_v1`, `top_code=None`이다. [기존 증거·원문 확인](EXISTING_EVIDENCE_DIAGNOSIS.md)을 유지한다.

## 실제 적합 상태에 따라 우선순위를 정한다 — 완료

- **수량 H1을 먼저 검토한다.** RAF full은 Train tail205건 RMSE458.95~468.43·MAE351.00~364.10으로 직전 수량426.56/303.72, 과거 평균412.18/293.29보다 세 seed 모두 나쁘다. 평균 실제433.89 대비 예측69.78~83.03이고, 두 보정 대조도 Train부터 큰 수량을 충분히 재현하지 못한다. 전체 Train RMSE45.63~46.28은 두 단순 기준선보다 좋으므로 전체와 tail 목적의 상충이 현재 모델에 이미 있다. **평가 구간에서만 새로 생긴 RAF 실패라고 설명할 수 없으며, 그렇다고 용량/최적화 원인을 확정하지는 않는다.**
- **시간 H2는 별도 문제로 검토한다.** Intermittent42 TimeNLL은Train0.2193→Val0.6187→Test62.5721, 평균σ는0.06781→0.04684→0.03959다. 수량의 Train 평균 오차는 tail에서도 작으며 Test tail은 과대예측으로 바뀐다. 수량을 양의 편향으로 바꾸는 loss 개입과 시간 scale를 넓히는 개입을 한꺼번에 수행하지 않는다. 좋은 seed52/62와 두 대조도 포함해 location·scale 교차의 불리한 결과를 보존한다.
- **Taxi52는 지금 새 학습 대상으로 확대하지 않는다.** Train tail413.31은 두 대조·직전 수량보다 좋고, Val370.76→Test1017.49로 증가했다. 평균 tail 오차도Train−199.26→Test−888.41로 커졌다. 기존 series 분해·이미 검증한 split 분포를 원고의 사용 조건으로 기록하며 RAF와 같은 “Train부터 못 맞춤”으로 묶지 않는다.
- [전체 적합 상태](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/TRAIN_FIT_REPORT.md)와 [계산 원본](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/train_fit_conditions.json)에81개 split·모든 seed/대조·여섯 지표·raw bias·분포량·단순 기준선을 보존했다. 현재 새로운 평가 자료 일반화·순수 과적합·component 인과 필요성은 미확인이다.

## H1: log 수량 목표와 raw 수량 오류가 같은 방향인지 읽는다 — 후보 / 추가 실행 승인 필요

**질문과 대상**

- RAF의 현 모델이 Train부터 큰 수량 point를 잘 재현하지 못한다면, 같은 목표에서 log1p 손실은 양호한데 raw tail 손실만 나쁜지 확인한다. 이 관측은 목표 상충 후보를 좁히며 손실을 바꾸면 실제 개선된다는 증명은 아니다.
- RAF의 full·보정 제거·current-only×3seed, 같은 선택 checkpoint9개×Train·Validation의 **새18개 model-split pass**다. 새 학습0·새 Test 추론0·encoder/수량 예측 개입0·checkpoint 재선택0이다. Taxi로 바로 확대하지 않는다.

**동결할 과학·연산 계약**

- 현재 scientific source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`를 유지한다. 별도 계측 adapter만 새 operation seal로 분리한다. checkpoint9개의 raw/state SHA·선택 epoch·원래 eligible population digest·lookback/순서/drop_last·기존 수량 경계>200을 고정한다.
- frozen `target_outputs`가 이미 계산하는 **실제 `log_qty_loss`**를 동일 dtype·동일 목표에서 집계한다. 필요하면 실제 관측 logit의 같은 softplus와 목표 log1p로 signed log 오차를 읽기 전용으로 계산한다. 평균 raw 예측/logit에서 log 오차를 만들거나 별도 모델 forward를 호출하지 않는다.
- 전체·기존 수량 bin·tail의 N, 원래 raw SSE/SAE/bias와 log 오차합/제곱합을 충분통계로만 기록한다. 원별 예측·목표 배열은 저장하지 않고 Test 자료를 calibration/selector에 사용하지 않는다. 모델 eval/no_grad, 가중치·BN/기타 state·원본 파일 SHA 불변을 확인한다.

**통과 조건과 한계**

- 원래 Train/Validation의 대상·truth-history·raw 여섯 지표와 read-only output/state 불변을 먼저 재현한다. 원래 여섯 지표와 새 log 충분통계는 모든9개에서 finite여야 한다. 실제18worker exit0·pipeline OS exit0·immutable archive/manifest·로컬 SHA·독립 QA 뒤만 분석한다. claim/failure를 보존하고 자동 retry/resume하지 않는다.
- full−각 대조의3seed·전체와 tail 방향을 모두 기록한다. raw RMSE가 더 나쁘면서 log MSE가 더 좋거나 비슷한 경우 목표 상충의 관측으로 설명할 수 있으나, conditional mean·capacity·optimization·학습 gradient의 원인 하나를 확정하지 않는다. λ/LR/tail weight·seed 탐색0이며 수치 유의성 주장0이다.

## H2: 시간 location 교체와 scale 교체의 점수 효과를 분리한다 — 후보 / 추가 실행 승인 필요

**질문과 대상**

- Intermittent의 좁은 σ와 location 차이가 원래 정수 TimeNLL에 어떻게 함께 작용하는지 읽는다. scale만 교체했을 때와 location만 교체했을 때의 손해/이득을 별도로 본다.
- 같은 full·두 대조×3seed checkpoint9개×Train·Validation의 **새18개 model-split pass**다. 새 학습0·새 Test 추론0·encoder/실제 model forward 변경0이다. 매 paired event의 실제 μ·σ를 메모리 안에서 정확히 정렬하고 충분통계만 저장한다.

**동결할 과학·likelihood 계약**

- full−보정 제거, full−current-only의6pair를 두 split에서 모두 비교한다. 실제 scale는 각 모델의 기존 `0.001+softplus(raw_scale)`이며 임의 floor/temperature·σ grid·location offset·Test fit을 추가하지 않는다.
- `D=max(1,round(T))`, `T=3·exp(N(μ,σ))`, 현재 `top_code=None`의 봉인된 float64 `positive_integer_log_mass`를 그대로 쓴다. 일반 top-code K가 있는 계약에서는 마지막 code K가 survival 질량 `P(T>K−0.5)`임을 보존해야 하지만, 현재 계약에는 그런 cell이 없다. 연속 밀도·평균 μ/σ·표준화 잔차로 NLL을 대체하지 않는다.
- paired event에서 네 score만 계산한다: `Lff=NLL(μfull,σfull)`, `Lfc=NLL(μfull,σcontrol)`, `Lcf=NLL(μcontrol,σfull)`, `Lcc=NLL(μcontrol,σcontrol)`. 두 원래 셀은 기존 실제 model score를 재현하고, 새 교차 두 셀은 **head 출력의 counterfactual score**라고 명시한다. 수량 출력은 그대로다.
- 6pair×2split×4셀=**48 paired score cells**이며 원래 full score의 중복을 포함한다. 새 교차 cells는24개다. 전체와 기존 Train-derived 시간 gap bin에서 NLL 합·평균을 저장하고, scale 교체 `Lfc−Lff`, location 교체 `Lcf−Lff`, interaction `Lcc−Lfc−Lcf+Lff`를 함께 보고한다. 큰 gap 구간은 사후 outcome 구간이며 encoder 입력이나 hyperparameter 선택에 쓰지 않는다.

**통과 조건과 한계**

- 같은 target id/history/recorded gap·입력·checkpoint/state SHA를 강제하고 원래 두 셀의 원래 NLL/수량 지표·bitwise 비개입 forward를 먼저 확인한다. finite·실제18worker/pipeline exit0·원본 archive/manifest·SHA·독립 QA 조건은 H1과 같다. 불리한 scale/location/interaction 결과와 두 대조·모든 seed를 보존한다.
- 서로 다른 학습모델의 head 출력을 섞은 점수라 encoder 또는 head 학습의 인과 원인, CNN·GRU의 필요성, 배포 가능한 새 단일모델, 완료된 calibration을 증명하지 않는다. 특히 Train/Validation의 이 결과로 기존 Test 시간 반전이 해결됐다고 주장하지 않는다. 기존 집계의 평균 μ·σ로 이 paired score를 재구성할 수 없어 추가 추론이 필요한 후보이며 이번 실행은0이다.

## 실제 모델을 바꿔야 할 때만 새 학습 수를 정한다 — 조건부 / 별도 승인 필요

- **현재 권고는 새 학습0이다.** H1/H2 관측을 완료하거나 생략한 뒤 논문이 어떤 주장을 유지할지 먼저 정한다. saved-output score나 새로운 통계 집계를 fresh training 또는 calibration fit이라고 부르지 않는다.
- H1의 단일 수량 objective 변경을 실제 검증해야 한다면 가장 작은 matched 후보는 **RAF full×기존/변경 objective2×3seed=6fresh fits**다. 같은 GPU/Runtime·초기화·data·optimizer·selector·epoch 계약에서 기존 objective도 새로 학습한다. 이전 A100/RTX full3을 comparator로 재사용해 “새3개만으로 matched6”이라고 하지 않는다. 변경 objective의 정확한 수식·Train-only 정규화·상수·수치 안정성·시간 손해 판정은 새 승인 계약 전에 확정해야 한다.
- 이6개는 objective 변경의 효과를 확인한다. 보정 경로 이득까지 objective와 교차해 주장하려면 **3arms×2objective×3seed=18fresh fits**의 별도 확장 계약이 필요하다. 현재 필요하다고 판정하거나 실행하지 않는다. 기존 구성 경로18/RAF12 실험으로 자동 확대하지 않는다.
- H2의 출력 교차 score에는 optimizer가 없어 calibration fit0이다. 이후 scale/location 파라미터를 어떤 자료에 fit하면 그 횟수를 새 fit으로 세고 사용 split·학습 대상 파라미터·정규화·동결 encoder·후속 독립 평가를 별도로 승인한다. 현재 그런 parameter fit의 대상·값·횟수는 확정되지 않았고 실행하지 않는다.
- 어떤 후속 설계든 현재 노출 Test로 방향을 정했으면 개발 증거로 표기한다. 선택한 변경 완료 또는 생략 뒤 최종 checkpoint·전처리·지표·대조를 동결하고 미노출 평가 계약으로 이어간다. 신규 검색·Runpod·공용 Runtime 변경은 이 후보에 포함하지 않는다.

## 의존관계와 병렬 가능성을 정한다 — 다음 작업

1. **현재 기준선 — 완료.** Train27과 재사용Validation/Test54의81개 split 비교, 원본559개 SHA, 같은 selected checkpoint27와 eligible target/history 조인을 마쳤다. 완료 작업을 재실행하지 않는다. 최종 독립 QA 결과를 증적에 연결한다.
2. **현재 원고 정합 — 다음 작업.** 원고의 Test 예외/적용 조건 정합은 공통 `main.tex`에서 직렬 진행한다. 새로운 출력 개입·학습보다 이미 확인한 수치에 맞춘 주장 축소를 먼저 선택할 수 있다.
3. **추가 관측 계약 — 승인 필요.** H1을 우선 후보로, H2를 별도 후보로 둔다. H1과 H2는 dataset·추가 통계·질문이 달라 로컬 구현/계약 검토를 독립 병렬화할 수 있다. 실제 추가 GPU 추론은 각각 명시 승인 뒤 공통 자원·원본 seal·최대 worker·우선순위를 조율한다. 승인 전 자동 launch는0이다.
4. **변경 모델과 미노출 확인 — 조건부 / 별도 승인 필요.** 새 모델을 만드는 단일 가설 검증은 그 계약/자료/Runtime에 의존하므로 관측과 직렬이다. 마지막 모델 동결 후 미노출 확인을 정한다.
