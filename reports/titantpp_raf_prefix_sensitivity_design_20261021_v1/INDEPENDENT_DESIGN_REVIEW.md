# RAF 후속 백본 가설 독립 설계 검토

상태: 설계 검토 완료. 신규 학습·추론·checkpoint 로드·SSH·Test 열람·기존 파일 수정은 수행하지 않았다. 본문은 투고용 원고가 아니라 내부 설계 메모다.

## 판정

조건부 추천은 **공유 이력 보정 경로의 모든 활성 분기 조합에 같은 민감도 상한을 적용하는 Shared15** 한 가지다. 효과나 원인을 확인한 모델로 채택하는 것이 아니라, 검증 가능한 다음 가설로 좁힌 것이다. 현재 자료는 해당 경로의 실제 증폭이나 급락 원인을 입증하지 않는다. 선택 근거는 동일한 구조를 유지하면서 (1) 관측된 쌍의 변화에 대한 보정량 증폭, (2) 이력 길이에 따라 달라지는 출력 합의 규모를 직접 제한할 수 있고 기존 gate·lag·직접 전달·분모 변경과 구분되기 때문이다.

## 먼저 확인한 구조적 문제

`models/TPPs/CountAwareTitanSharedHistoryMLP.py`는 공유 입력 투영 U(15×128), 각 64×15인 출력 투영 W_b 8개를 쓴다. 모든 활성 분기가 동일한 현재·직전 hidden state 쌍을 읽는다. [1,2,4,8,16,32,64,128]은 실제 참조 lag가 아니라 활성 조건이다. 현재 코드의 보정은 정확히 다음과 같이 쓸 수 있다.

- z_i=[h_i;h_prev], h는 첫 encoder block 출력이다.
- k는 해당 위치에서 활성인 prefix 분기 수다.
- A_k=(W_1+…+W_k)/8, A_0=0.
- R_k(z)=A_k GELU(Uz).
- 두 번째 block 입력은 h_i+R_k(z_i)다.

따라서 RAF에서 여러 번의 과거 수량을 각각 복원하는 8-lag 모델이라고 설명하면 틀리다. 첫 encoder attention의 전체 관측 이력 처리와 이 국소 보정을 구분해야 한다. Train/Validation의 네 번째 분기 활성 비중 3.138%→25.262%는 A_4 사용 빈도의 변화이며, 그 자체로 A_4 오류를 증명하지 않는다.

저장된 현재 Shared3 Validation만으로 확인한 현상: 급락577행 MAE16.646806 대 RMTPP14.542938, 세 seed 모두 같은 방향. 큰수량50행은 Shared SSE의68.638%이고 RMTPP에 비해 큰수량 RMSE/MAE가 낮다. 이 두 구간은 상이한 미래 결과를 기준으로 나눈 사후 진단이다. 실제 입력으로 다음 급락 여부를 알 수 없다. Shared Train 개별 예측이 없으므로 Train–Validation 수량 오류로 과적합 판정도 하지 않는다.

## 서로 다른 대안 비교

| 대안 | 확인 가능한 개입 | 장점 | 비용·유지 복잡도 | 배제/보류 이유 |
|---|---|---|---|---|
| 모든 prefix의 이력 보정 민감도 제한 | 학습된 U·W가 만드는 전체 보정 경로의 gain 상한 | 입력·lag·width·분모·head·memory 유지, 별도 gate와 label 없음, 기전 반증 가능 | 새 parameter0, 작은 행렬의 prefix 누적·제곱합·scalar 연산; 코드/계약 변경은 작으나 수치 검증 필요 | 추천하되 tail 손실과 실제 gain 근거 부재를 인정 |
| 관측 전체의 robust 요약을 현재 표현과 결합 | 국소 쌍을 넘어 관측 분포 위치·일시적 spike 판별 정보를 제공 | 일회성 높은 관측과 지속 수준을 구분할 정보가 늘 수 있음 | 추가 집계·mask·입력경로·용량 대조, 여러 선택 및 튜닝 변수 증가 | LevelHistory/다중 이력/적응 gate와 기능 중복 위험, 이번에는 보류 |
| 급락/상승 gate 또는 현재 표현의 후단 직접 전달 | 관측 변화에 따라 이력 비중 조정 또는 우회 경로 | 직관적이며 표현 손실을 줄일 수 있음 | 추가 parameter/경로/초기화·용량 대조 필요 | AdaptiveHistory/LevelHistory/late observation의 재조합, 미래 급락을 선택적으로 알 수 없음. 추천 제외 |

ActiveBranchNorm처럼 활성 수로 나누거나, 분기 삭제·모든 분기 활성화로 바꾸는 안도 이번 가설에 합치지 않는다. 기존 `CountAwareTitanActiveBranchNorm.py`는 분모를 실제 활성 수로 바꿨고 `CountAwareTitanHistoryControls.py`는 current-only/all-after-first를 이미 구분했다. 기존 `CountAwareTitanLevelHistoryQKV.py`에는 입력 전이 RMS 기반 Q/K 포화가 있다. 이번 weight 기반 공동 보정 경로 상한은 적용 위치·대상·의존변수가 다르지만, 일반적인 안정화/정규화의 새로운 원리라고 주장하지 않는다.

## 단일 후보의 정확한 설계 제안

각 활성 prefix에 공통으로 적용할 단일 상한 L=1을 사전 고정한다. dataset/seed/Validation 지표로 탐색하지 않는다. 이 값은 보정 경로가 hidden pair 변화보다 큰 Lipschitz 상수를 갖지 않게 한다는 설계 기본값이다. 최적값이라는 실증 근거는 없다.

q_k = (1.25/L)^2 · sum(U^2) · sum(A_k^2)

s_k = rsqrt(max(1,q_k)), k=1,…,8; s_0=1

R'_k(z) = s_k A_k GELU(Uz)

각 위치의 기존 가용성으로 k를 구하고 weight-dependent s_k만 선택한다. 같은 k의 모든 샘플은 같은 s_k를 쓴다. 각 prefix 모두 동일한 L을 충족한다. 실제 활성 수를 분모로 사용하지 않으며 `/8`·가용 mask·누적 순서를 유지한다. 입력 수량·다음 정답·급락 label로 s_k를 계산하지 않는다. 기존 availability는 원래 k 선택과 같은 방식으로만 사용한다. 새 gate, 새로운 관측 범위, 차분 특징, prototype 변경은 추가하지 않는다.

이 식은 Frobenius norm으로 operator norm을 위에서 제한하는 **보수적 상한**이다. 정확한 spectral normalization이라고 이름 붙이지 않는다. exact GELU g(x)=xΦ(x)의 도함수는 Φ(x)+xφ(x)이고, 절댓값은 1+1/sqrt(2πe)<1.25 이하이므로 고정 k에서 ||R'_k(z)-R'_k(z')||₂≤L||z-z'||₂다. R'_k(0)=0이므로 ||R'_k(z)||₂≤L||z||₂도 성립한다.

이 보장은 다음을 포함하지 않는다: 전체 encoder/head의 Lipschitz 상한, raw수량 역변환 뒤의 오차 상한, k가 달라질 때의 연속성, 급락만 선택적 감소, 큰수량 이득 보존, 메모리 retrieval 결과 보존. first block/second block/head가 다른 방식으로 보상할 수 있으며 학습 공동최적화도 달라진다. 특히 큰수량에도 동일한 제한이 적용되므로 기존 tail 이득을 잃을 위험이 핵심이다.

## 독립 검토에서 수정한 중요 설계점

초안의 global s=max-prefix 방식은 폐기했다. 해당 batch에서 사용되지 않은 W5…W8도 global 최대 norm의 gradient를 받아 학습될 수 있어서, 원래 RAF의 미사용 분기 특성을 바꾸는 별도 기전이 섞이기 때문이다. per-prefix s_k는 A_k만 참조하며 j>k인 W_j로는 출력·상한 gradient가 흐르지 않는다. vectorized 구현에서도 선택된 s_k만 gradient가 흐르는지 검사한다. 이 점을 확인하지 않으면 설계 준비 PASS로 표시하지 않는다.

대안인 global scale의 stop-gradient는 실제 함수의 정확 gradient와 달라지고, optimizer 후 projection은 optimizer 동작까지 바꾸므로 이번 단일 백본 가설에 채택하지 않는다. 원래 `/8`과 branch eligibility를 유지하되 prefix마다 동일한 수학적 상한을 적용하는 것과 활성 수로 일괄 나누는 것은 다르다.

## 초기화·용량·연산

기존 Shared15와 같은 U 초기 tensor, W_b=0, common encoder/head/memory tensor, CPU/CUDA RNG를 유지한다. 추가 RNG draw가 없다. 기존 fork_rng 생성 순서를 보존해야 한다. W=0에서 모든 q_k=0,s_k=1이므로 초기 forward와 output-projection gradient가 기존과 같다. 실제로 해당 데이터에서 사용되지 않는 분기의 gradient가0으로 유지되는지도 반드시 확인한다. 처음 U gradient가0인 것은 기존 zero-output 초기화와 동일하며 결함으로 오해하지 않는다.

sqrt(norm²) 또는 SVD를 직접 미분하지 않고 제곱합+clamp_min(1)+rsqrt를 쓰면 zero division과 zero singular value 초기 문제를 피할 수 있다. 실제 gradient finite 여부는 아직 검증하지 않았다. stop-gradient denominator를 사용하면 수학적으로 동일한 목적의 정확 gradient가 아니므로 여기서는 사용하지 않는 설계다.

parameter9600 그대로, optimizer·학습률·weight decay·손실 그대로다. 추가 계산은 U의1920원소 및 8개 A_k의7680원소의 누적·제곱합 중심이다. batch/length별 새 MLP는 없다. 학습 step마다 상한을 갱신하고 추론에서는 같은 고정 weight로 8개 s_k를 한 번 계산할 수 있다. 원래 보정 결과에 위치별로 선택한 s_k를 곱하면 큰 중간 텐서를 새로 만들 필요가 없다. 실제 속도·메모리 수치는 측정 전 주장하지 않는다. Frobenius 상한은 spectrum이 분산된 행렬에 과하게 보수적일 수 있고 방향별 작동을 구분하지 못한다.

## 기전 확인과 반증 조건

학습 전 필요한 증거는 저장된 선택 weight의 U, A_1…A_8 norm 및 제안 상한의 실제 활성 여부다. 해당 dataset의 실제 사용 prefix에서 operator 기반 충분상한도 L을 넘지 않는다면, Frobenius 상한만 작동하는 것은 보수성에 의한 축소일 수 있다. 이 경우 실제 증폭을 제한한다는 근거가 약하므로 학습 진입을 보류한다. operator 기반 곱상한의 위반 자체도 실제 Jacobian 위반을 입증하는 것은 아니므로 필요조건 수준으로만 쓴다. 지금 검토에서는 checkpoint를 로드하지 않았다. 별도 분석에서 이를 조사하더라도 새 forward/평가와 구분한다. 상한이 모든 관련 checkpoint에서 비활성이면 현재 문제와의 연결은 약하다. gradient/forward 관측이 없으므로 norm만으로 실제 입력별 Jacobian이나 급락 인과를 확정하지 않는다.

승인 이후 하나의 새 후보를 평가할 때:
1. 제안 변화 외에는 원본 데이터·Shared15·log1pT/Q·공통 loss/time weight1·head·memory·optimizer·seed42/52/62·예산·RMSE선택을 고정한다. 저장 원본 Shared15가 주 비교다. 엄밀한 동일 GPU/Runtime의 인과 대조가 필요하면 변경 없는 Shared15 fresh 대조를 별도 승인 범위에 명시해야 하며, 과거 결과를 덮어쓰지 않는다. 기존 CNN 재학습은 필요 없다.
2. 전체 Validation RMSE·MAE를 함께 본다. 급락577행의 MAE/양의 signed error와 non-drop구간을 분해한다. 급락에서만 좋아지는지, 단순 전체 하향이동인지 구별한다.
3. q>200의 기존50행 RMSE·MAE·과소편향과 지속고수량/급증 구간을 함께 보고한다. 'tail 보존'은 Shared기준3seed 평균 tail RMSE·MAE가 더 높아지지 않는 것으로 사전에 정의할 수 있다. RMTPP보다 여전히 낫다는 것만으로 기존 이득을 보존했다고 말하지 않는다. 50행에 대한 결론이므로 seed별 예외와 개체 집중도를 보존한다.
4. history1/2/3–4/5–8/9+와 수량 구간을 교차 보고하되 사후 최선 구간을 새 성공 기준으로 삼지 않는다. k별 gain 상한 및 actual norm 로그를 남기되 그 로그가 예측 input/학습 gate가 되어서는 안 된다.
5. 추천된 기전의 강한 지지는 급락 MAE가 세 seed 모두 줄고, 전체 MAE 개선·RMSE 비악화 및 tail 평균 비악화가 함께 나타나는 경우다. 일부 seed만 개선하거나 tail 감소를 잃으면 tradeoff 결과로 보존한다. 이 조건은 기존 외부 비교군에 대한 전면 우세를 보장하지 않으며 Test는 새로 열지 않는다.
6. 상한이 작동하지 않았는데 개선하면 제안 gain 기전으로 설명할 수 없다. 상한이 작동하고 tail만 손해보면 균형 개선 가설을 기각한다. 전체 예측을 낮춰 drop만 좋아지면 선택적 급락 대응 근거가 아니다. history9+ 효과만으로 A_4 인과를 확정하지 않는다.

## 구현 전 검증 목록

현재 source/원본 tensor/RNG·初期출력 동일, zero/nearzero/상한경계에서 finite gradient, 모든 prefix의 해석식과 기존 곱셈식 일치, padding/withheld reset/length1에서 불필요 보정0, prefix 전체 bound 수치 검증, future input perturbation에 대한 causal mask, freeze/unfreeze/train/eval에서 함수 의미 동일, checkpoint identity와 정확 tensor key 검증. 가짜 급락 label을 입력으로 쓰는 항목은 없다.

## 근거와 한계

읽은 주요 근거: TEST_SESSION_PROTOCOL0·9·12; diagnosis REPORT.md, DATASET_AUDIT.json, RESULT.json의 scope/seed/target/causal 필드와 DETAILS.json의 loss-tail 및 history×quantity 분해; SharedHistoryMLP/ MultiLagDetail/ CoreAblation/ ActiveBranchNorm/ HistoryControls와 AdaptiveHistoryCNN/LevelHistoryQKV/late-path source의 관련 설계. 원래 실행 protocol `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 해당 경로에 없었으며 실제 확인 가능한 oma-architecture workflow/checklist를 읽었다. `.agents/` 수정 금지와 부모가 지정한 /private/tmp 경로를 준수했다.

이 검토는 기존 후보 전체에 대한 세계적 novelty 심사나 문헌 우월성 주장이 아니다. 확인한 기존 구현 범위에서 단일 구조 변화의 차이를 명시했다. 실제 gain 측정·구현·synthetic·native·학습이 남아 있다.

생성 산출물: `/private/tmp/raf_next_backbone_design_review_20261021.md` 하나.
