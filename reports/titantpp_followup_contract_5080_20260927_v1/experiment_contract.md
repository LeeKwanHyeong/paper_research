# TitanTPP 후속 validation 실험 계약 — 5080

작성일: 2026-09-27 · 대상 저장소: `paper_research` · 브랜치: `codex/hard-lmm-causal-qkv`

**비교 설계와 실행 수·비용 상한을 문서로 고정했다. GPU 실행은 승인되지 않았으며, 구현·합성 데이터 검증·실행 파일과 Runtime의 해시 확정이 다음 단계다.** 이 문서는 실행 명령이나 start permit을 대신하지 않는다.

기계 판독 계약은 [titantpp_followup_validation_5080_v1.json](../../paper/contracts/titantpp_followup_validation_5080_v1.json), 전체 조건표는 [run_matrix.csv](run_matrix.csv), 의존관계와 재사용 규칙은 [run_matrix.json](run_matrix.json)에 있다. 계약의 canonical SHA와 원문 SHA는 [contract_identity.json](contract_identity.json)에 따로 기록한다. 후속 구현에서 과학적 비교 범위를 바꿔야 하면 변경 이유와 실행 수·비용 차이를 먼저 기록하고 새 버전으로 확정한다. 빈 source/runtime/dispatch 해시를 임의로 채워 실행하지 않는다.

## 1. 현재 기준선과 이번 단계의 경계 — 완료

- **대상:** 로컬 `paper_research`와 5080의 읽기 전용 시스템 정보.
- HEAD는 `1de2c31e8e3febda7ff20d0527402ea9959b4098`이다. 기존 미커밋 변경을 보존했다. 이 HEAD만으로 현재 학습 코드를 재현할 수 있다고 주장하지 않는다. 확인한 주요 코드·결과 파일의 SHA를 계약에 기록했으며, 새 adapter의 전체 source closure는 아직 없다.
- 현재 근거는 [54조건·108 replay 최종 보고서](../local_detail_3seed_final_20260927_v1/final_report.md)의 **validation 결과**다. 기존 학습을 다시 수행하는 계약이 아니다. 원본 seed42 18조건, 5080 신규 24조건, 5090 원본 완료 8조건, 복구 4조건을 보존한다. 원래 멈췄던 RMTPP 시도의 부분 update 수 미확정이라는 한계도 소급해 제거하지 않는다.
- 이력 보완 모델은 Taxi에서 개선됐지만 Intermittent와 Instacart의 3-seed 평균에서는 B보다 좋지 않았다. Taxi에서도 시간 NLL이 악화됐다. 따라서 이번 주장은 모든 데이터·모든 지표의 우월성이 아니라 **어떤 조건에서 수준·변화 구조가 수량 예측에 도움이 되는가, 단순 MLP와 비교해 필요한가, 비용과 시간 예측 손실은 무엇인가**다.
- 5080 스냅샷 시각은 **2026-09-27 23:26 KST**다. RTX 5080의 compute process와 tmux 세션이 없었고 GPU 가용 메모리는 15,541 MiB, 디스크 여유는 약 683 GiB였다. 이는 당시 관측이며 예약이나 실행 가능성 보증이 아니다. [원본 스냅샷](host_snapshot_5080_verified.json)
- **이번에 수행한 것:** 문서·코드 읽기, 공개 소스의 버전 고정과 일부 파일 보존, 데이터 바이트 해시와 과거 test 산출물 메타데이터 확인, 조건표·계약 작성 및 문서 정합성 검증. 모델/runner 변경, 패키지 설치, CUDA qualification, 추가 학습, commit/Push는 수행하지 않았다.

## 2. 무엇을 비교하고 어떤 주장까지 허용할지 고정한다 — 완료

### A. 구조 효과를 확인하는 고정 설정 비교

3개 데이터 × seed 42·52·62 × 아래 6개 모델 = **54개 보고 조건**이다. 기존 설정의 학습률 0.001을 고정한다. 이 표에서 결과를 본 뒤 ablation별 설정을 다시 고르지 않는다.

| ID | 비교 조건 | Full 대비 달라지는 부분 | 확인할 질문과 주장 한계 |
| --- | --- | --- | --- |
| B | 기존 TitanTPP | 이력 보완 residual 없음 | 보완 구조의 전체 효과. B는 제거 실험의 기준이며 제안 모델로 다시 밀어붙이는 결정이 아니다. |
| Full | 현재 수준·변화 이력 보완 | 기준 후보 | 세부 구조의 이익이 단순히 파라미터 증가 때문인지 확인해야 한다. |
| MLP | 같은 파라미터 수의 인접 이력 MLP | 수준/변화 경로를 concat MLP로 대체 | 가장 중요한 용량 통제. MLP와 동등하거나 열세면 수준·변화 구조의 고유 우위는 지지되지 않는다. |
| Level | 수준 정보 경로만 | 변화 경로 제거 | 변화 경로의 추가 효과. 파라미터 수도 줄므로 의미 정보만의 순수 효과라고 단정하지 않는다. |
| Change | 변화 정보 경로만 | 수준 경로 제거 | 수준 경로의 추가 효과. 같은 용량 차이 한계를 적는다. |
| NoStaticLMM | Full에서 정적 Hard-LMM만 제거 | 검색 모듈을 identity로 대체 | 이력 보완이 있을 때 정적 검색이 필요한지 확인한다. encoder의 persistent memory 16개는 유지한다. |

현재 Full은 8개 분기가 모두 **직전 관측 이벤트**를 읽는다. 분기 활성 기준은 관측 이력 수 `> [1,2,4,8,16,32,64,128]`, residual 분모는 8이다. 서로 다른 8개 lag를 읽는 구조로 설명하지 않는다. 추가 gate나 온라인 메모리 쓰기는 없다. [현행 구현](../../models/TPPs/CountAwareTitanMultiLagDetail.py)

현재 상태를 h, 직전 상태를 p라 할 때 수준은 `(h+p)/2`, 변화는 `h-p`다. Full의 분기 출력은 `U[GELU(L·수준)+GELU(D·변화)]`이고 L·D·U는 bias가 없다. hidden 64, rank 4 기준 전체 추가 파라미터는 **6,144개**다. MLP는 분기마다 `W:128→4`, `V:4→64`의 `V GELU(W[h,p])`를 써서 추가 파라미터를 정확히 맞춘다. 두 조건의 활성 함수 폭과 연산량까지 동일하지 않으므로 실제 시간·메모리·연산량을 별도로 보고한다. Level/Change는 각각 4,096개이며, 쓰지 않는 파라미터를 붙여 같다고 하지 않는다.

수준·변화는 인접 상태 쌍의 가역적 좌표 변환이다. **새 정보를 만들어낸다는 주장은 허용하지 않는다.** 검증 대상은 제약된 비선형 구조의 효과다. Full은 동일 출력 U를 공유한 폭 8의 제약 MLP로도 쓸 수 있다는 점을 문헌·수식 설명에 유지한다.

메모리와 이력 보완의 **상호작용·시너지**는 이번 계약에서 주장하지 않는다. 이를 주장하려면 두 기능을 모두 제거한 네 번째 factorial 조건이 필요하다. 별도 이력 길이 gating 제거 조건도 이번 실행에 포함하지 않는다. 결과를 보고 이 조건들을 자동 추가하지 않는다.

### B. 외부 모델과 제한된 튜닝 예산으로 비교

| 모델/범위 | 입력과 출력 | 고정 구조·학습 방식 | 비교표에서의 정확한 이름과 한계 |
| --- | --- | --- | --- |
| B·Full / 3개 데이터 | 기존 과거 Δt·수량 → 동일 수량/시간 head | A와 동일 구조; 학습률 2개만 비교 | 고정 설정 A와 튜닝 후 B 표를 분리한다. |
| S2P2 / 3개 데이터 | 과거 `log1p(Δt), log1p(q)` → 마지막 관측 직후 상태 → 공통 head | hidden64, 2층, state P16, dropout0.1, 상대 시간, parallel scan | **S2P2-matched-head-event**. 원래 mark embedding과 native likelihood head를 그대로 평가한 논문 재현이 아니다. 실제 다음 시각까지 상태를 진행시켜 수량을 예측하면 누출이므로 금지한다. |
| FlexTPP / 3개 데이터 | 동일 과거 이벤트의 시간·수량 속성 → 다음 시간·수량의 결합 분포 | hidden64, 2층, 4 heads, FF256, dropout0.1, flow bins128. 시간→수량 순서 | **FlexTPP-native-family-event**. native 분포 계열을 사용하지만 관측 시간과 수량 지원 범위에 맞춘 adapter다. 정식 논문 전체 설정 재현이나 모든 공개 설정의 최적 성능이라고 부르지 않는다. |
| TimeMixer / 3개 데이터 | 동일 과거 Δt·수량의 이벤트 순서 → 다음 이벤트 수량 1개 | d64, FF128, 2층, scale 3개(2회×2 downsample), 이동평균25, 채널 혼합, dropout0.1 | **TimeMixer-event**. 원래 균등 간격 달력 시계열·다중 horizon 결과와 구분한다. 시간 head를 억지로 달지 않으며 시간 NLL은 N/A다. |
| DeepRenewal / Intermittent만 | 과거 양의 정수 간격·수량 → shifted NB 간격·수량 분포 | LSTM64×2, dropout0.1, native 분포식의 PyTorch 이식 | **DeepRenewal-PyTorch-event**. 다음 비영 수요 이벤트 예측이다. 전체 달력 기간의 0 수요까지 포함한 수요 예측 우월성은 이 실험으로 증명하지 않는다. |

S2P2는 [NeurIPS 2025 논문](https://papers.nips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html)의 공개 구현을 참고한다. FlexTPP는 [NeurIPS 2025 논문](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html)과 연결된 저장소를 사용하되 저장소 자체의 unofficial implementation 표기를 유지한다. TimeMixer는 [원 논문](https://arxiv.org/abs/2405.14616)과 공개 구현, DeepRenewal은 [원 논문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)과 GluonTS의 interval-size 구현을 참고한다. 논문에 쓴 모델명만 가져오고 변경점을 생략하지 않는다.

| 참고 구현 | 고정 commit | 확인한 핵심 내용 |
| --- | --- | --- |
| [S2P2](https://github.com/UCIDataLab/state_space_point_process/tree/c3933240f16a22b43d80d09bda272475526ff24b) | `c3933240f16a22b43d80d09bda272475526ff24b` | 상태공간 구현과 공개 Taxi 설정. 공개 설정의 hidden128·4층·lr0.01과 이번 공통 head용 축소 설정의 차이를 명시한다. |
| [FlexTPP](https://github.com/czi-ai/FlexTPP/tree/e2c77088392ae62b75be440fb0cfb33547c046c2) | `e2c77088392ae62b75be440fb0cfb33547c046c2` | 기본 모델/optimizer와 연속 속성의 조건부 분포·sampling 경로. 데이터별 공개 HPO 설정을 우리 데이터에 이미 최적인 것처럼 옮기지 않는다. |
| [TimeMixer](https://github.com/kwuking/TimeMixer/tree/e24610583b36fdd8c76cc17a8df4e65759a5f460) | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | 다중 scale mixing과 normalization. 이벤트 순서, mask 처리, 양의 수량 head는 adaptation이다. |
| [GluonTS DeepRenewal](https://github.com/awslabs/gluonts/tree/889a3df86a89a365880b4bc1488bcf4c039f265e/src/gluonts/mx/model/renewal) | `889a3df86a89a365880b4bc1488bcf4c039f265e` | 원 구현이 MXNet이며 interval·size 각각 `data-1`에 NB log-probability를 적용한다는 점. 0 기반 NB를 원 수량에 그대로 대입하지 않는다. |

확인 파일의 SHA와 읽기 실패까지 [upstream_source_receipt.json](upstream_source_receipt.json)에 보존했다. 이 파일 목록은 참고 자료이며 아직 실행용 전체 dependency/source closure가 아니다. upstream entrypoint를 그대로 호출하면 test 평가가 뒤따를 수 있으므로 후속 구현은 validation 전용 경로만 연결한다.

#### native adapter의 평가 규칙

- **FlexTPP:** 양수 변환은 `Δt=time_scale·exp(z_t)`, `q_cont+1=raw_scale·softplus(z_q)`로 고정한다. 시간 손실은 관측 정수 구간의 확률 질량이며, 수량은 raw 단위 Jacobian을 포함한 조건부 연속 밀도다. 수량의 +1 변환은 0 수량도 지원한다. 관측값을 자르지 않으며, 생성된 q_cont가 음수일 때만 0으로 제한한 뒤 점예측 평균을 낸다. 이 연속 밀도의 support는 q_cont > −1이고 이산 count 확률 질량으로 해석하지 않는다. 학습의 결합 분해에서 정답 시간을 조건으로 쓰는 것과, 수량 예측 시 미래 시간을 주는 것을 구분한다. 추론은 시간 분포에서 먼저 생성하고 관측 round/clamp/top-code를 적용한 뒤 수량을 생성한다. **정답 다음 시간을 수량 예측에 넣지 않는다.**
- 수량 점예측은 256개 고정 Sobol 쌍에서 얻은 **raw 수량 평균**이다. `exp(평균 log 수량)`으로 대체하지 않는다. 난수는 dataset·target identity·seed로 고정해 epoch/설정별 sampling 운으로 checkpoint가 바뀌지 않게 한다. 한 번에 16쌍씩 계산한다. selected checkpoint에서 canonical validation 앞 4,096개 이하 target을 512쌍으로 점검하고 RMSE 차이가 상대 0.1%를 넘으면 수치 평가 불일치로 남긴다. 결과를 보고 샘플 수를 바꿔 다시 순위를 고르지 않는다.
- **TimeMixer:** 관측 입력은 `Δt/time_scale, q/raw_scale`이고, 출력은 `raw_scale·softplus(output)`다. 손실은 train raw_scale로 나눈 raw 수량 MSE다. Full의 log 수량 MSE와 목적함수가 다르다는 것을 모델 전체 비교표에 표시한다. 같은 정보 범위로 비교하되 입력 변환까지 같다고 주장하지 않는다. 관측 history만으로 normalization·pooling·이동평균을 계산한다. padding이 평균이나 예측을 바꾸면 구현 검증 실패다.
- **DeepRenewal:** `간격=1+NB`, `수량=1+NB`이며 수량 예측은 해석적 평균 `1+μ`다. 마지막 target의 두 NLL만 학습한다. 원 논문의 calendar-horizon 생성 설정과 다른 target exposure를 숨기지 않는다. 모든 허용 train/validation 간격·수량이 양의 정수인지 실행 전에 확인한다. 이 support와 맞지 않으면 adapter 부적합으로 기록하며 target을 골라 제외하거나 분포를 자동 변경하지 않는다. MXNet을 5080에 설치하는 대신 식을 이식하되 NB 모수화·LSTM gate 순서·초기화·support를 합성 자료로 검증한 뒤에만 비교한다.
- 모든 모델은 관측 history를 encode하고 **같은 마지막 target 하나**를 학습한다. native 모델에만 history 내부 모든 이벤트를 추가 target으로 주지 않는다. 사용하지 않는 categorical/distribution parameter가 존재하면 total/active parameter 수를 모두 보고한다.

## 3. 데이터·학습·선택·반례 규칙을 고정한다 — 완료

| 데이터 | train target | validation target | 최대 관측 이벤트 | batch128의 epoch당 step | 시간 관측 |
| --- | ---: | ---: | ---: | ---: | --- |
| Taxi `yellow_trip_hourly` | 38,393 | 8,268 | 255 | 300 | 시간, 양의 정수 |
| Intermittent `intermittent_frozen_5000` | 393,824 | 86,285 | 255 | 3,077 | 주, 양의 정수 |
| Instacart `insta_market_basket` | 1,991,192 | 503,733 | 63 | 15,557 | 일, 30으로 top-code |

데이터 파일·split manifest의 **바이트 해시 6개가 기존 동결 계약과 일치**했다. train/validation target identity·quantity SHA는 기존 계약 값을 유지하며 실행 전에 다시 대조한다. 현재 단계에서는 parquet 행을 역직렬화해 성능을 계산하지 않았다. train 통계와 lookback은 [기계 계약의 datasets](../../paper/contracts/titantpp_followup_validation_5080_v1.json)에 완전하게 포함했다.

- 입력은 관측 완료된 Δt·수량·순서·mask다. item/user/category ID, 미래 달력 정보, 미래 시간·수량을 모델 입력에 추가하지 않는다. entity ID는 paired 분석의 군집 구분에만 사용한다.
- validation의 이전 관측을 나중 예측의 history로 쓰는지는 기존 동결 loader의 인과적 규칙을 그대로 따른다. 모델별 target/prefix 해시가 같아야 한다. test segment는 history에서도 제외한다.
- seed는 42·52·62, batch128, max300/min40/patience40, AdamW(weight decay0.01), gradient norm clip1, scheduler 없음, FP32, AMP/TF32/compile 없음으로 고정한다. warm start는 없다. 데이터 순서 RNG는 모델 초기화 RNG와 분리한다.
- checkpoint는 **매 epoch validation raw 수량 RMSE의 strict earliest finite minimum**이다. 모든 지표는 그 checkpoint를 함께 사용한다. 시간 NLL이 좋은 별도 checkpoint를 섞지 않는다. 마지막 epoch는 진단용 별도 endpoint이며 결과가 유리할 때 주 checkpoint를 대신하지 않는다.
- 수량 RMSE·MAE를 모든 모델에 보고한다. 공통 시간 점수는 관측 정수 구간의 질량으로 계산한다. 첫 구간은 `[0,1.5)`, 이후 `[k−0.5,k+0.5)`, Instacart의 마지막 30은 `[29.5,∞)`다. 연속 density NLL과 이 질량 NLL을 같은 순위표에 혼합하지 않는다.
- 수량 구간 경계는 train 기준 Taxi `[7,686,1562,3449]`, Intermittent `[2,31,46,187]`, Instacart `[8,20,25,35]`를 유지한다. 이력 구간은 64·128과 Instacart용 1·3·7·15·31을 유지한다. 추가 시간 간격, 최근 최대 8개 관측 log1p 수량의 평균 수준, 마지막 두 관측 log1p 수량의 절대 변화량 구간은 train quartile로만 정해 학습 전 해시를 고정한다. 이력이 2개 미만이면 변화량 미정으로 별도 표시하고 0으로 채우지 않는다. 동률 경계는 합치고 빈 구간도 표시한다.
- 3-seed 평균과 **표본 표준편차(ddof=1)**, 각 seed 값을 모두 기록한다. validation target 수를 독립 seed 수처럼 사용하지 않는다. 서로 단위가 다른 데이터의 raw RMSE를 그냥 평균내지 않는다.
- Full−B/MLP/Level/Change/NoStaticLMM의 paired 차이를 보고한다. Intermittent·Instacart는 oper_part_no 전체 계열 단위, Taxi는 계열 안에서 seq 순서의 연속 168 target 원형 블록 단위로 bootstrap 1,000회(seed20260927)를 수행한다. 같은 재표집 target을 모델과 세 seed에 공유하고 seed별 RMSE 차이의 평균에 대한 2.5/97.5 percentile 구간을 제시한다. 168 target을 반드시 달력 1주라고 해석하지 않는다. 독립 계열이 2개 미만인 entity bootstrap은 불확실성 미확인으로 표시한다. 유의성·p-value를 선언하지 않으며 이미 선택에 사용한 validation의 기술적 민감도 분석으로 한정한다. pairing에 필요한 target별 오차는 새로운 bridge에서 만들 수 있지만 test 예측은 만들지 않는다.
- 수량 개선·시간 악화·학습/추론 비용 증가를 함께 제시한다. 허용 가능한 시간 악화 폭을 결과를 본 뒤 만들어 성능 통과로 처리하지 않는다. 모델별 실패, 불리한 seed, 작은 표본의 tail, N=0 구간을 모두 남긴다.

### 2개 학습률 후보와 seed 역할

| 모델 | c0 | c1 | 선택 규칙 |
| --- | ---: | ---: | --- |
| B·Full·S2P2·TimeMixer·DeepRenewal | 0.001 | 0.0003 | seed42의 selected raw RMSE 최소; 완전 동일한 값은 c0 우선 |
| FlexTPP | 0.0008 | 0.0003 | 같은 규칙; 0.0008은 공개 기본 optimizer 설정을 참고 |

각 모델×데이터의 두 후보가 정상 종료해야 튜닝 셀이 완성된다. 선택한 설정을 바꾸지 않고 seed52·62에서 확인한다. seed42를 다시 학습하지 않는다. B·Full의 c0 결과는 A에서 재사용한다. c0가 선택되면 A의 seed52·62도 재사용한다. c1이 선택된 경우에만 새 seed52·62가 필요하다. 실패 후보를 숨기고 살아남은 설정을 동등한 2회 탐색의 승자로 보고하지 않는다.

동일한 후보 수를 주지만 연구 전 기간의 누적 튜닝 예산이 같아지는 것은 아니다. 구조·head·loss·초기화의 차이와 제한된 학습률 탐색을 명시한다. seed42는 튜닝에도 쓰였으므로 B 표의 3-seed 평균은 미사용 test에서의 일반화 추정치가 아니다.

## 4. 재사용·신규 실행·평가 횟수를 분리한다 — 완료

| 구분 | 보고/논리 조건 수 | 새 학습 수 | 새 selected/last replay |
| --- | ---: | ---: | ---: |
| A: B·Full, Taxi+Intermittent, 3 seeds | 12 | 0, 기존 5080 결과 재사용 | 0, 기존 24개 endpoint 기록 재사용 |
| A: B·Full, Instacart, 3 seeds | 6 | 6 | 12 |
| A: MLP·Level·Change·NoStaticLMM × 3 데이터 × 3 seeds | 36 | 36 | 72 |
| B: 16개 모델×데이터 셀의 2후보, seed42 | 32 | 26, A의 B·Full 6개 재사용 | 52 |
| B: 선택 설정의 seed52·62 | 32 | 20~32, B·Full 최대 12개 재사용 | 40~64 |
| **계획 전체가 완료될 경우** | 겹치는 보고 조건을 단순 합산하지 않음 | **88~100** | **176~200** |

B의 모델×데이터 셀은 B·Full·S2P2·FlexTPP·TimeMixer의 3개 데이터 15셀 + DeepRenewal의 Intermittent 1셀이다. 최종 B 보고표는 16셀×3 seeds = **48조건**이다. A의 54조건과 상당 부분 겹치므로 102개의 독립 학습으로 계산하지 않는다. 실행표에는 기존 재사용 12행과 신규/조건부 신규 최대 100행, 총 112행이 있다.

별도 평가 비용도 포함한다.

- **기존 selected checkpoint 12개에 validation bridge 각 1회:** 새 시간 구간·paired 오차를 동일한 평가 규칙으로 확보하고 기존 global 지표와 일치하는지 점검한다. 학습 수는 증가하지 않는다. bridge가 원본을 덮어쓰지 않는다. 원래 source·checkpoint가 없거나 해시/지표가 다르면 재사용 불가로 표시하며 새 학습으로 자동 대체하지 않는다.
- **FlexTPP selected checkpoint 최대 12개에 subset decoder 진단 각 1회:** 512쌍 사용 시의 수치 민감도를 점검한다. 전체 validation 1회로 세지 않는다.
- 따라서 새 **전체 validation endpoint 평가 최대 212회**(200+12), subset 진단 12회다. epoch마다 수행하는 validation, synthetic qualification, 성능 microbenchmark는 별도 로그·비용 항목이며 endpoint 수에 숨기거나 같은 결과를 독립 표본으로 세지 않는다.
- CUDA qualification은 10개 모델 중 9개×3 shape + DeepRenewal×Intermittent 1개 = **28개 shape**, 각각 synthetic 60 updates, 총 최대 **1,680 synthetic updates**다. 이는 실제 데이터 학습 100회와 별도다.

Instacart B·Full은 과거에 5090에서 수행했으므로 이번에 5080에서 6조건을 새로 수행한다. 동일 장비의 비교 기준을 얻기 위한 것으로, 기존 5090 결과를 실패나 불리한 결과라는 이유로 대체하는 것이 아니다. 기존 결과는 별도 lineage와 문맥으로 남기며 새 5080 평균에 섞지 않는다.

## 5. 비용 상한과 종료 규칙을 구체화한다 — 완료, 실행 승인 전 제안값

**제안 상한은 단일 5080 GPU 프로세스 점유 시간 합계 480시간이다.** GPU 사용률이 낮거나 실패한 시간도 비용에 포함한다. 실제 완료 ETA가 아니며, 전 조건 완료를 보장하지 않는다. 전기·장비 시간당 가격이 없어 금액으로 환산하지 않았다. 상세 계산은 [budget.json](budget.json)에 있다.

| 자원/구간 | 상한 | 포함 범위 |
| --- | ---: | --- |
| A 학습 | 192 GPU 시간 | 신규 42조건의 학습과 epoch validation |
| B 학습 | 264 GPU 시간 | 신규 튜닝/확인 최대 58조건의 학습과 epoch validation |
| 공통 검사·평가 | 24 GPU 시간 | qualification, 모든 endpoint/bridge/decoder 검사, latency·메모리 측정 |
| **합계** | **480 GPU 시간** | 구간 사이 자동 예산 이동·연장 없음 |
| 새 허가 이후 경과시간 | 720시간(30일) | 첫 GPU qualification 시작부터 준비·대기·중단 포함; 타이머 초기화 금지 |
| 학습 1회 | Taxi 3h / Intermittent 16h / Instacart 32h | 모델 프로세스 기동부터 학습/epoch validation 종료까지; replay는 공통 평가 예산 |
| qualification 1 shape | 30분 / synthetic 60 updates | 28개 전부 최대 시간을 써도 14h. 통과했다는 증거 없이 실제 학습 금지 |
| GPU | 시작 여유 ≥12 GiB; allocated ≤14 GiB, reserved ≤15 GiB | owned GPU process 1개; 타인 process 종료 금지 |
| 디스크 | 시작 여유 ≥100 GiB; 새 산출물 ≤80 GiB | checkpoint 파일당 ≤256 MiB; 원본 삭제로 공간 확보 금지 |

다음 값은 **완료 epoch 실측**이다. 모델 B→Full 순서로 Taxi 12.31→13.35초, Intermittent 117.09→127.77초이며 5080에서 측정됐다. Instacart 206.43→250.41초는 **5090 측정값**이다. 이를 5080의 실측으로 표기하지 않는다. [측정 원본](../local_detail_3seed_final_20260927_v1/three_seed_summary.json)

A의 새 ablation 비용·조기종료가 과거 Full과 같다고 놓고 Instacart의 5080 소요시간을 기존 5090의 1/1.5/2배로 가정하면 A만 각각 **128.8/169.0/209.2h**다. 이 배수는 측정값이 아닌 민감도 가정이다. 같은 속도 가정에서 모든 조건이 300 epoch를 채우면 A만 **505.7/688.0/870.3h**이므로 192h 상한을 넘는다. 외부 모델의 5080 epoch 시간은 아직 측정되지 않았다. **480h를 주면 모든 조건이 끝난다는 약속은 하지 않는다.**

상한은 중단 기준이다. 미완료가 생기면 planned/completed/failed/unrun을 전부 보고하고, 완성되지 않은 3-seed 셀을 완전한 평균으로 제시하지 않는다. 비용이 과도하면 실행 전에 과학적 질문과 조건표를 함께 줄인 새 계약을 만드는 것이 맞으며, 실행 중 불리한 모델·seed를 골라 빼거나 자동 연장하지 않는다.

실행 순서는 Instacart B/Full 동종 장비 anchor → MLP 용량 통제 → Level/Change → NoStaticLMM → A 감사 → B의 제한 튜닝과 확인이다. 각 ablation에서는 seed42→52→62, 각 seed에서 Taxi→Intermittent→Instacart 순서를 고정한다. 상세 dispatch 순서는 run matrix의 queue_order다. B는 각 셀의 c0/c1 이후 seed52/62 확인으로 진행한다. 다른 파일의 기록·문헌 점검은 독립 작업이지만, 공통 계약·source·GPU를 쓰는 구현 및 실행은 현재 세션에서 직렬 관리한다.

## 6. Runtime과 오류 처리를 고정한다 — 설계 완료, 구현·실행 검증은 다음 작업

- **서버:** 5080만. GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, index0. 5090으로 자동 이동하지 않는다.
- **새 root 제안:** `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_followup_5080_20260927_v1`. 조회 시 존재하지 않았으며 아직 생성하지 않았다.
- **Runtime 제안:** 위 root의 `runtime/bin/python3.12`. base Python은 `/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12`다. 기존 ai_env에 패키지를 설치하지 않고 독립 잠금 환경을 준비한다. 예상 core는 PyTorch2.11.0+cu130/CUDA13.0/cuDNN91900, NumPy2.4.4/Polars1.39.3이다. 이번 조회는 package metadata만 읽었으므로 실제 torch import·CUDA 실행 일치 확인은 미완료다.
- **확인한 준비 공백:** 5080에 Lightning·FrEIA·einops·MXNet이 없었다. 필요한 패키지/이식 범위를 구현 단계에서 정확히 잠근다. 외부 head를 단순 공통 head로 바꿔 놓고 native 비교가 완료됐다고 하지 않는다. MXNet native GPU 설치는 계획하지 않는다.
- tmux binary `/usr/bin/tmux`, session `titantpp_followup_5080_20260927_v1`을 새 허가에서만 사용한다. source/cache/output을 이전 run과 분리하며 종료된 run을 resume하지 않는다.
- fit마다 새 process를 사용한다. 5분 batch/evaluation 진척이 없으면 stack을 남기고 30분 없으면 소유 process만 중단한다. 이것은 과거 native 정지 원인이 확정돼 라이브러리 버그를 수정했다는 뜻이 아니다.
- OOM·NaN·fit 상한 도달은 실패/미완료로 보존하고 batch 축소·재시작·대체 seed 없이 처리한다. 독립된 사전 등록 조건은 남은 예산 내에서 진행할 수 있다. 공통 데이터/Runtime/source 불일치는 전체 dispatch를 중단한다.
- SSH 응답 실패는 상태 미확인이다. claim/tmux/PID를 확인하기 전 학습 실패로 단정하거나 다시 실행하지 않는다. GPU process·claim·tmux 실패 흔적을 삭제해 재시도하지 않는다.
- 옛 **2026-10-02 21:28:48 KST 마감과 남은 예산은 이전 실험 전용**이다. 이 새 실험에는 이월하지 않는다. 새 날짜·종료상한은 나중에 새 permit의 시작 시각으로 계산한다.

### 비용 측정의 동일 조건

28개 모델×데이터 shape의 seed42 selected checkpoint에서 canonical validation 앞 1,024개 이하 target으로 batch1·128 추론을 측정한다. 구조 비교 모델은 A의 c0, 외부 모델은 B에서 선택한 설정을 사용하며 가장 빠른 seed를 골라내지 않는다. warmup10회 후 50회 측정, CUDA 동기화 후 p50/p95와 peak allocated/reserved 메모리를 보고한다. FlexTPP의 256쌍 decoder를 포함하며 encoder만 재서 전체 추론 비용으로 쓰지 않는다. 데이터 로딩·전송 제외 여부를 표시한다. 이 측정도 공통 24h 평가 예산에 포함되며 별도 학습 조건은 아니다. 미지원 연산의 FLOPs를 추정 없이 정확한 값으로 채우지 않는다.

과거 재사용 행에 적힌 `state_sha256`는 tensor 상태 해시다. 직렬화된 checkpoint 파일의 바이트 SHA와 구분하고, 실제 bridge 전에 둘 다 확인한다. 이번 문서 작업에서 원격 checkpoint 파일 전체를 읽어 검증했다고 보고하지 않는다.

## 7. 과거 held-out 사용 이력을 구분한다 — 메타데이터 점검 완료, 최종 test 적격성 미확정

[heldout_provenance_audit.json](heldout_provenance_audit.json)은 과거 manifest 91개와 산출물의 이름·크기·연결 경로를 확인한 기록이다. `test_metrics` 등 성능 산출물 이름에 해당하는 고유 파일 경로 **4,094개**가 확인됐다. 관련 manifest record 78개는 중첩 campaign이 있으므로 독립 실험 78회로 해석하지 않는다. 현재 Taxi split manifest 경로와 연결된 record 15개, Instacart 경로와 연결된 record 41개가 있었고 현재 `intermittent_frozen_5000` 경로에서는 같은 연결을 찾지 못했다.

이 증거는 **과거 test 산출물이 생성됐다는 사실**을 뒷받침한다. 당시 파일 bytes가 지금과 완전히 같았는지, 누가 성능을 읽었는지, 모델 선택에 반영했는지는 확정하지 못한다. 최근 54조건이 validation 전용이라는 사실도 별개로 유지된다. 현재 Intermittent 경로의 일치 기록을 못 찾았다는 이유로 그 test의 미사용을 보증하지 않는다.

이번에 과거 test 예측·성능 파일과 validation/test 혼합 결과표는 열지 않았다. 다만 최초 split manifest의 넓은 메타데이터 미리보기에서 held-out **목표 분포 요약 필드가 우발적으로 노출**됐다. 그 값은 설계·선택에 사용하거나 이 계약에 복사하지 않았고, 이후 메타데이터 읽기는 허용 필드로 제한했다. 이 제한을 감사 기록에 남긴다.

따라서 기존 test를 ‘한 번도 사용하지 않은 최종 test’라고 부르지 않는다. **이번 계약은 validation-only로 진행 가능**하지만 최종 test를 열기 전에는 별도 provenance 판정과 평가 계약이 필요하다. 필요하면 실제로 미사용인 이후 시점/외부 데이터를 확보한다. 이미 노출 가능성이 있는 행을 재분할하거나 이름만 바꿔 새로운 test라고 하지 않으며, 기존 test를 validation에 편입하지 않는다.

## 8. 구현 단계에서 반드시 확인할 항목 — 다음 작업

**모델과 데이터 adapter를 계약대로 구현한다 — 다음 작업**
- 대상: 로컬 `paper_research`, `codex/hard-lmm-causal-qkv`의 기존 변경을 보존한 코드.
- MLP/Level/Change/NoStaticLMM 변형, S2P2 공통 head, FlexTPP 관측 시간·joint decoder, TimeMixer mask 처리, DeepRenewal 수식 이식을 준비한다.
- 완료 조건: 모델 이름뿐 아니라 정확한 입력·출력·손실·초기화·활성 파라미터 수가 계약과 일치하고, dependency/license 및 전체 source closure가 기록된다.

**합성 자료로 누출·선택·실패 처리를 검증한다 — 다음 작업**
- 대상: 로컬 CPU 테스트와 validation 전용 평가 경로.
- 미래 target/time 변경 불변성, padding 위치/개수 불변성, 이력 제한, withheld segment reset, 동일 seed의 초기 공통 tensor와 batch prefix, 실제 파라미터 수, 각 손실의 FP64 참조값, NB support, Flex CDF/Jacobian/decoder, strict earliest checkpoint, early-stop·last replay, 구간 합산을 점검한다.
- 과거 12개 재사용 anchor의 checkpoint/source availability와 해시를 점검하고 새 bridge의 경로를 준비한다. runner가 어떤 경로에서도 test를 호출하지 못하게 한다.
- 완료 조건: 계획한 비교와 실패 상태가 합성 테스트에서 확인되고 exact Runtime·source·dispatch SHA를 담은 실행용 부속 계약이 준비된다. 지금 문서 검사 통과를 모델 테스트 통과라고 표시하지 않는다.

**고정된 실행 패키지로 5080 qualification과 추가 validation을 시작한다 — 승인 필요**
- 대상: 위 새 5080 root/독립 Runtime/GPU1개. 승인 요청은 구현·검증 결과와 실행용 해시, 최대100회/480GPU시간/720경과시간이 구체화된 뒤 한 번에 한다.
- 새 permit으로 synthetic CUDA qualification을 통과한 조건만 진행한다. 이후 ETA가 필요하면 실제 완료 epoch 속도와 남은 조건을 근거로 조건부 추정한다. 합성 update 속도를 전체 학습 ETA로 바꾸지 않는다.
- 완료 조건: 모든 planned 조건의 완료·실패·미실행 상태, 선택 설정·source lineage·비용·paired 결과가 회수되고 실행 정상 여부와 논문 주장 지지 여부가 별도로 판정된다.

**최종 test와 논문 주장 범위를 판단한다 — 외부 작업 대기**
- 대상: 후속 validation 결과 및 held-out provenance.
- 새 비교의 반례·시간 손실·비용까지 본 뒤 유지할 주장을 결정한다. 현재 단계에서 논문 채택 가능성이나 SOTA 달성을 확정하지 않는다. 최종 test는 별도 승인·데이터 적격성 확인 후 한정 평가한다.

이번 단계의 완료 조건은 검토 가능한 입력/평가/실행 수/예산/공정성 계약이다. 위 구현이나 GPU 실행까지 완료됐다는 뜻은 아니다.
