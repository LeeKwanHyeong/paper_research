# 검색 계약·mean 재사용·최소 비교 검토

검토일: 2026-10-12 Asia/Seoul. 대상: `/Users/igwanhyeong/PycharmProjects/paper_research`.
역할: retrieval_contract_explorer_1012, 읽기 전용 검토. 프로젝트 파일 수정0, SSH/API0, GPU/학습/추론/observer/automation/launch0. 이 파일만 임시 경로에 작성했다. TEST_SESSION_PROTOCOL.md 0·1·9·12 적용. Test/held-out 예측·성과와 Validation/Test 혼합 파일 열람0.

## 결론

- 현재 CNN+GRU54 이력 보정 중심 주장을 위해 검색을 실행할 필요는 없다는 parent의 권고와 일치한다. 검색 규칙을 별도 기여로 유지할 경우에만 새로운 비교·자원 계약이 필요하다.
- 기존 3dataset×3seed×4arm은 36개의 과학 비교 조건이다. 현재 확정 mean 재사용0. 안전한 기본안은36 fresh fit, RTX4에 대한 새 bridge가 통과하면 검색27+새mean5=32 new fit이다.
- 2dataset×3seed×4arm=24 조건은 범위를 축소한 새 개발 비교다. 현재 요구한 원래3dataset Q2 campaign 완료를 대신하지 않는다. 예를 들어 Taxi+Intermittent는 RTX Intermittent3 mean 재사용 bridge가 통과하면21 new fit이다. RAF+Intermittent는 RTX4 bridge 후20 new fit이나 대상 선택 이유와 노출된 Validation에 의한 사후 선택을 공개해야 한다. 이는 실행 권고·승인이 아니다.
- fixed/mean만2dataset×3seed로12 fit은 좁은 추가 효과 비교이며 weighted·generic512를 포함하는 Q2 귀납 편향/예산 대조를 충족하지 않는다.
- 전체 큐시간·비용 ETA 미확정. 기존 동시 epoch wall time을 구조 효율이나 다른 arm 속도로 전용하지 않는다.

## 기존 네 arm과 공통 과학 계약

`search_artifacts/titantpp_contribution_retrieval_20261008_v1/design_contract.json:2-19`와 frozen source를 확인했다.

| arm | 동작 | 추가 활성 query 파라미터 |
|---|---|---:|
| mean (`titantpp_cnn_gru54` alias) | 기존 hidden cosine으로 prototype top4 선택, 균등 평균 | 0 |
| weighted | 같은 hidden cosine/top4, 선택된 cosine score softmax 평균 | 0 |
| fixed | 시간4개·수량4개 요약의 두 query를 따로 unit 정규화, 0.5씩 cosine 합산, top4 softmax 평균 | 512 = 4×64×2 |
| generic512 | 동일8개 요약으로 hidden+0.5×φW 공동 query를 unit 정규화, top4 softmax 평균 | 512 = 8×64 |

공통: width64, prototype bank64×64, top4, cosine keys, 온도1, 최근 관측32개의 시간·수량 각각 last/mean/population SD/last difference, 기존 Train point statistics, target 미노출, zero query 초기화/RNG 소비0, CNN+GRU54 encoder와 common heads. full mean은 기존 균등 집계 경로와 계산이 같은 기준선이며 후속 검색 모델로 warm start하지 않는다.

Prototype은 학습된 정적 파라미터 bank다. 실제 관측 사건 저장소·online write·inference parameter update가 아니다. fixed two-view cosine에는 concatenation 표현과 대수적 동등성이 있으며 separate/joint normalization 귀납 편향 비교다. generic521/adaptive521은 이번512 대조에 포함하지 않는다.

근거(아래 경로는 project root 상대이며 모두 frozen):

- `source/models/TPPs/CountAwareTitanContextualRetrieval.py:56-93`: bank·context·normalization·params 메타데이터.
- 같은 파일 `:104-138`: padding skip, withheld segment reset, 최근32 관측 요약.
- 같은 파일 `:159-190`: weighted/fixed query와 top4/softmax.
- 같은 파일 `:193-205`, `:231-243`: frozen CNN+GRU54 공통 encoder와 mean delegation.
- `source/models/TPPs/CountAwareTitanFixedRetrievalControls.py:91-115`: generic512 공동 query, tau1, 추가512.
- `source/paper/scripts/run_titantpp_fixed_retrieval_controls.py:121-141`: 공통 초기 tensor·constructor RNG 동일 검증; 추가 score/identity buffer만 허용.
- `execution_contract.json:756-763`: B128/min40/max300/patience40, Validation 전체 raw수량RMSE 최초 strict finite 최소, warm_start false.

## Q2와 비용 gate

판정 구현 `paper/scripts/finalize_titantpp_contribution_campaigns.py:165-197`:

- fixed가 mean·weighted·generic512 모두에 대해 전체 raw수량RMSE strict 개선.
- 같은 두 개 이상 dataset 각각 모든42/52/62 seed가 이를 만족해야 accuracy development support.
- 해당 dataset fixed native 비용이 mean 대비 inference median≤1.20, training step median≤1.30, peak allocated≤1.20이어야 joint support.
- 선택 epoch의 전체/큰수량 MAE·RMSE·TimeNLL 여섯 지표, 3seed 평균·표본 SD와 불리한 paired delta 모두 보존. timing·tail 개선은 별도 근거.
- 개발 일관성 근거이며 통계 유의성·참신성·외부 일반 우월성·독립 평가의 증명이 아니다.

`execution_contract.json:820-827`의 warmup5/measure20와 frozen `operation/campaign.py:200-236`은 synthetic B128 단독 arm 측정이다. cost gate는 candidate eligibility이며 existing training launch gate가 아니다. 3 fit 동시 wall time은 이 구조 비용비를 대체할 수 없다.

## 비용길이 불일치와 RAF256 후속 확인

**확인 사실:** 문서의 T64/T256 요구와 달리 frozen `operation/campaign.py:200`은 `sorted({min(64,max_seq_len),max_seq_len})`를 사용한다. Taxi/Intermittent는64/256, RAF loader max_seq_len84여서64/84만 측정한다. RAF84 결과를256으로 표시하면 안 된다. 현재 검색 qualification 자체 미실행이다.

**RAF84 모델에256 synthetic tensor를 입력하는 것이 코드상 불가능한가? — 아니다. 기존 frozen forward에는 positional interpolation 경로가 있다.**

- `source/paper/scripts/run_titantpp_fixed_retrieval_controls.py:104-111` / `run_titantpp_cnn_gru.py:107-114`는 data loader max_seq_len84로 모델을 구성한다.
- `source/models/TPPs/CountAwareTPP.py:1039-1051`은 learned position table 길이를 max_seq_len로 전달한다.
- `source/models/Titan/backbone.py:145-148`은 learned positional parameter `[1,84,64]`를 만든다.
- 같은 파일 `:201-211`의 `_get_pos(L)`은 L이84를 넘으면 **기존 position table을 linear interpolate하여 길이L의 tensor를 반환**한다. 새 Parameter 생성0, 새로운 RNG 소비0. tensor를 반환할 뿐 `self.pos_emb`를 교체·확장하지 않는다.
- `source/models/TPPs/CountAwareTitanCNNGRU.py:264-284`는 입력의 실제 길이에 `_get_pos(encoded.size(1),...)`를 사용한다. 여기에는 loader max_seq_len 초과를 거부하는 cap이 없다.
- `source/paper/scripts/count_aware_tpp_backbone/core.py:60-101`의 target_outputs는 현재 synthetic tensor를 pad/reorder하고 target 위치를 가리며 encode_task_states에 전달한다.84로 자르는 경로는 없다.

**코드에 근거한 추론:** build_model의 data/science/max_seq_len84와 초기화를 그대로 유지하면서 operation의 synthetic length 목록에256을 추가하면, 기존 interpolation 경로를 포함한 동일한84-position-parameter 모델의256 stress 비용을 측정할 수 있다. scientific trainable parameter budget·initialization·dataset window를 바꿀 필요는 없어 보인다. 다만 실제 native forward/backward/finite/peak와 비용 결과는 실행0이므로 미확정이다. 새 synthetic length는 운영 계약·operation SHA·qualification 및 비용기록에 명시해야 한다.

**금지할 혼동:** data loader max_seq_len 자체를256으로 바꾸거나 build_model에256을 주면 position table이84×64에서256×64로 늘어 parameter가11008개 증가하고, constructor RNG/이후 초기 tensor의 순서가 달라진다. 이는 같은 scientific model/mean 초기화 비교를 깨므로 단순 비용길이 수정으로 취급할 수 없다. RAF256은 실제 RAF 학습/Validation window 길이가 아니라 synthetic stress length다.64/84 실제 contract 비용과 구분해 보고한다.

기존 gate를 조용히64/84로 완화하는 것은 권고하지 않는다. T64/T256 gate를 유지할 경우 새 운영 계약에 모델은84를 유지한 synthetic256 측정을 추가하고, 승인·native 검증까지 launch 사전차단으로 두는 것이 일관된다.

## frozen source/data 계약 일치와 현재 코드 구분

두 bundle `execution_contract.json`의 source.files maps는 각각129개다. stdlib SHA로 각129 frozen source 파일을 읽어 maps와 대조해 전부일치 확인했다. source maps 공통126개는 전부동일 SHA다.

- retrieval 전용3개: `CountAwareTitanFixedRetrievalControls.py`, `run_titantpp_fixed_retrieval_controls.py`, `test_titantpp_fixed_retrieval_controls.py`.
- history 전용3개: `CountAwareTitanCNNHistoryControls.py`, `run_titantpp_cnn_history_controls.py`, `test_titantpp_cnn_history_controls.py`.
- datasets 객체 전체·input_files maps·training 계약은동일.
- claim contract rawSHA binding은둘다 `568ef19ab7347ff063ddc0ca1492a4469f4a81fe06dfcf8cd533116bd5639205`. 새로운 claim 승인·runtime 존재를 뜻하지 않는다.
- frozen science revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`.
- retrieval closure `4a0007f06d887127ecaedb21f7a94abf8b146b7c8115f7d2f4214ad19cb4089b`, history closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`는 전용3파일 때문에다름.
- 현재 작업트리의 공통126 중 `paper/scripts/evaluate_titantpp_gru_controls.py`는 frozen과다르다. 현재SHA `1deaedae266b5e54293d7a8ee5577a8535eb482a4b569b90ac713a08ca7234a5`, frozen `ae174e9712e5302a04450cedbf72b7ca8bd6cfaf14738e71c299bab14beee27e`. 현재 작업트리의 과학코드가 frozen bundle과 동일하다고 포괄 주장하지 않는다.

## mean 재사용 후보와 미확정 조건

최신 history selected_conditions.json에는 full9가 A1005 + RTX4로보존된다.

| dataset·seed | mean 원본 host | 새RTX 검색의 기준선 후보 |
|---|---|---|
| Taxi42/52/62 | 부모 A100 | 재사용불가, RTX 새mean 필요 |
| RAF42 | 부모·복구 physical A100 이관 | 재사용불가, RTX 새mean 필요 |
| RAF52 | 복구 A100 | 재사용불가, RTX 새mean 필요 |
| RAF62 | RTX5080 | bridge 검증 전 미확정 후보 |
| Intermittent42/52/62 | RTX5090 | bridge 검증 전 미확정 후보 |

RTX qualification actual runtime:

- 5080: GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, driver595.99.02, Python3.12.13, torch2.11.0+cu130, CUDA13.0, cuDNN91900.
- 5090: UUID `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, 같은driver/Python/torch/CUDA, cuDNN92000.
- 각float32/deterministic algorithms/cudnn deterministic true, TF32 off,4threads/interop1; frozen 검색 host 기대/runtime/python/GPU와 대응한다. 현재 서버의 live runtime이나 idle 확인은 하지 않았다.

기존 full initial canonical SHA:

- RAF62: `7464570b37726f9d958162f7aff023eba6a229a6f423238529bf0a1d3c6816af`.
- Intermittent42: `3a7ee878463c546049f08bbf3c016dc65509e8632a63c116682242e0e2321da3`.
- Intermittent52: `23707941a04d85065f4a58dcc19c434128655bc7eb968cbd63b5649652e48848`.
- Intermittent62: `3c5659e6769940bf2d8dc1f4e357d80a85b9ed79c64919929bb5bd517852bc16`.

기존 originals의 mean input_receipt와 노출 기록을 확인했다. 데이터·split·Train statistics/target identity는 frozen 계약과 일치하고 held_out_materialized false다. Train exposure가 RAF62 60epoch, Intermittent42 55/52 51/62 53epoch로 존재한다. 이는 후속 검색의 동일 epoch-prefix와 대조할 수 있는 원본이지 후속 대조 검증 완료가 아니다.

재사용 구현 `paper/scripts/finalize_titantpp_contribution_campaigns.py:117-139`의 기존 요구:

1. 같은 host/dataset/seed와 common Train epoch-prefix·Validation order SHA.
2. prior originals/report/dependency 검증과 prior canonical 연결.
3. dataset/input/training/parent source126 동일.
4. 양 stage qualification runtime JSON 완전동일, Python 경로·GPU UUID 동일.
5. qualified mean initial canonical SHA가 기존mean initial과 동일; 새 retrieval arm initialization에도 별도 binding.
6. resolved_inputs path mapping만 제외한 input receipt 전체동일.

현재 retrieval bundle은 prepared; qualification·fit·completion·retrieval claim 원본이없고 control/integrity_receipt만존재한다. 새 검색의 실제 initial/exposure·native costs가없으므로 현재 확정 재사용0. 재사용하려면 새 계약에 completed RTX4 immutable 원본·원관측시각과 bridge를 명시하고 검증해야 한다. old retrieval prior_stage는 `history_20261008_v1`이어서 최신 completed RTXhandoff와 직접맞지 않는다.

## 실행계약과 최대3 동시 실행

`retrieval/current.json:10-15`: prepared, old history prior_stage 포인터. `retrieval/execution_contract.json:803-811`: workers_per_host1. 기존계약은 새 parallel3 launch 계약이 아니다. 새 자원·동시 실행·mean bridge 계약이 필요하다. 승인 전 실행0.

같은 dataset/seed의 네arm은 하나의 GPU·Runtime에 배치한다. max3이면 4arm을 한 번에 실행할 수 없다. 계약에 배치를 미리 정하면 다음 중 하나가 가능하다.

- fresh mean+weighted+generic512 세fit, 자연종료 후 fixed1.
- 검색세fit 후 fresh mean1.
- mean을 먼저단독 또는 bridge 재사용 검증 후 검색세fit.

다음 seed로 넘어가기 전 해당군 종료 gate·원본을확인한다. GPU간에 **다른 dataset/seed군**을 나누는 병렬 작업은가능하지만 같은군의 네arm을5080/5090에나누지 않는다. 슬롯수·batch배치·host scheduling이동일 구조효율 또는 통계 효율을 뜻하지 않는다. 실제비용은유휴/단독GPU에서별도측정한다. source/input/model/selector 변경0, automatic retry/resume0, old failure/claim 원본보존이후속계약기본이다.

## SHA 증적

각 rawSHA는 stdlib hashlib.sha256(file bytes)로 계산했다. canonical_json SHA와 rawSHA를혼용하지 않는다.

| 파일 | rawSHA |
|---|---|
| retrieval execution_contract.json | acce81d0f2dd6dda9ac300d6997a207ddb3f5194f1b7cfc5c08363490e475b96 |
| retrieval current.json | 1930909b61c63eca3fa659277864f262f8899f1b58d1437d7dce22cb396a4a2f |
| retrieval design_contract.json | 12e57cad0622b95345e1cae34eace05e5a9761c3382e23d030816f3cf99c6d99 |
| retrieval operation/campaign.py | e47bea24d796c5eeeff4f4fcc3c702308b461ac4c446cca1aab354d951d4af09 |
| history RTX execution_contract.json | ef97d038c9b9cf90a6e6c00cc3f3637a14a42f5fd00f6aa0b1dfcec1db4a2058 |
| history RTX analysis/completion_receipt.json | 656eef38d3192d57f89e86aaa2ed65e69ae5c256288668c1819db7e838c48362 |
| history RTX analysis/selected_conditions.json | 41ddf10b5379b2ba647e1c8500b7e02e2214bfa127e797d0ee293c1050668742 |
| history RTX originals/5080/verified/originals/qualification/receipt.json | d57a13f9869d4bb8b26140f12a4c154da53bd6e0637e187cc19c63bcb98e5faa |
| history RTX originals/5090/verified/originals/qualification/receipt.json | dfd90bd99bf9c0f774d0d221298c31c8ce69c1cb0e9077897b4bc10bfa954178 |
| retrieval source/models/Titan/backbone.py | f3531234235a9f50de05cf36c80b0a742a45709145d49555440249c814a20e98 |
| retrieval source/models/TPPs/CountAwareTitanCNNGRU.py | 9d1bcf6c754cbf6216e4929f832d08af60ccf02573edd3b6c1c7a2088115dfc3 |
| retrieval source/models/TPPs/CountAwareTitanFixedRetrievalControls.py | 68590f0b39dbfccd6aa3bc467e5f21d5f8ccc6961119319bb82ea13965856c9f |
| retrieval source/paper/scripts/count_aware_tpp_backbone/core.py | c5cc72a265a05199868332c9beaa32ce9568afaf5f4cfa0930b836215ab36084 |

다음 확인은 planning 검토를완료한다. 실행과새native측정은다음명시적승인전미실행으로유지한다.
