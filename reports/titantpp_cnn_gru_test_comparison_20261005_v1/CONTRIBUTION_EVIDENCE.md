# CNN·GRU 후속 후보의 학술 근거와 기여 경계

**구조·원본 문헌 검토 — 완료 / 성능 승자 확정 — 본 문서 범위 밖**

2026-10-05. 질문: 완료된 seed42 CNN+MLP16·GRU54·CNN+GRU54 가운데 무엇을 논문 후속 후보로 삼을 수 있으며, 어떤 주장은 아직 성립하지 않는가? 이 문서는 학술·구조 해석만 담당한다. 신규 Test 결과를 예상하거나 별도 평가하지 않았다. 실제 성능 순위는 같은 폴더의 통합 비교 결과를 따른다. 사용자는 이미 Test를 열람한 상태에서 후속 개발을 진행하므로 이후 결과를 독립 미접근 Test라고 표현하지 않는다.

## 1. 주요 판단

**CNN+GRU 결합 자체와 Q/K/V convolution 자체를 새로운 기여로 내세우기는 어렵다.** CTPP는 이미 continuous-time convolution과 GRU로 local/global 문맥을 결합했고, Titans는 Q/K/V projection 이후 depthwise convolution을 사용한다. 따라서 후속 후보의 기여는 관측 사건 기준의 causal 처리, 기존 64차원 표현을 보존하는 잔차 보정, 근접 파라미터 예산의 통제 대조, 그리고 수량 꼬리·시간 적합도의 상충관계를 실증하는 범위로 한정하는 것이 타당하다. 이는 검증할 연구 방향이며 현재 성능 우월성이나 독창성의 확정 판정이 아니다. [CTPP 원문 Fig.1·§4.3](https://arxiv.org/pdf/2306.14072), [Titans 원문 §4.4](https://arxiv.org/html/2501.00663v1)

## 2. 학술 plane: 원본 4개로 확인한 선행 구조

검색 경로: `web` 원본 문헌 확인 + `local` Serena 구조 검색. 아래의 근거 등급은 학술 원본 여부이며, 도메인 신뢰 점수와 논문 주장 검증 여부는 별개다. oma-search의 고정 도메인 레지스트리는 arxiv.org·kdd.org·proceedings.mlr.press에 명시 점수를 주지 않으므로 도메인 라벨은 **unknown(—)** 으로 보존한다. 이를 저품질 논문이라는 의미로 사용하지 않는다.

| 원본·근거 등급 | 직접 확인한 사실 | 이번 후보 해석 |
|---|---|---|
| [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663v1), academic/저자 원문, §3·§4.4 | 테스트 시점 associative-memory 학습이 핵심이며, Q/K/V projection 후 depthwise convolution도 명시한다. | QKV CNN의 존재만으로 새로운 Titans 원리나 test-time 학습을 주장하지 않는다. 현재 GRU의 hidden-state 갱신은 gradient 기반 online parameter update와 다르다. |
| [CTPP: Intensity-free Convolutional Temporal Point Process](https://arxiv.org/pdf/2306.14072), academic/저자 원문, Fig.1·§4.2–4.4 | continuous-time CNN local encoder, residual connection, GRU global encoder, intensity-free 시간 분포 decoder를 결합한다. | CNN+GRU 또는 local/global 융합의 포괄적 최초 주장을 직접 배제하는 선행 사례다. 이번 사건 순서 k=3 QKV CNN은 CTPP의 연속 시간 convolution과 다른 연산이다. |
| [RMTPP: Recurrent Marked Temporal Point Processes](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process), academic/공식 학회 초록 | 사건 이력의 영향을 RNN 표현으로 학습하고 intensity를 이력의 비선형 함수로 모델링한다. | recurrent history 요약은 확립된 선행 접근이다. 이번 GRU는 전체 backbone 교체가 아닌 블록 사이 잔차 보정이라는 연결 위치가 비교 대상이다. |
| [Transformer Hawkes Process](https://proceedings.mlr.press/v119/zuo20a.html), academic/공식 학회 논문 | self-attention을 이용한 사건 의존성 모델링을 제안하고 recurrent 방법과 비교한다. | attention·recurrence의 상대 장점은 데이터와 평가 계약으로 판단해야 한다. 논문의 기존 벤치마크 성능이 이번 count-aware 비교 우월성을 보증하지 않는다. |

위 4개 원본은 구조 주장에 사용했다. RMTPP는 공식 초록까지만 확인했고, 나머지 세 문서는 원문 본문/PDF를 확인했다. CTPP 출판사 페이지와 잘못 지정된 v2 HTML은 접근 실패하여 arXiv PDF로 확인했다. 무관한 arXiv 검색 결과는 배제했다. 유료 검색·커뮤니티 조사·포괄적 최신 문헌조사는 수행하지 않았다. 이 범위는 특정 선행 사례를 확인하기에 충분하지만 모든 관련 연구에 대한 독창성 증명은 아니다.

## 3. 코드·문서 plane: 이번 구현이 실제로 비교하는 것

근거: [동결 설계](../titantpp_cnn_gru_width16_design_20261004_v1/README.md), [구현 계약](../titantpp_cnn_gru_implementation_20261004_v1/implementation.md), [동결 manifest](../../paper/reproducibility/snapshots/cnn_gru_a100_seed42_prelaunch_20261004_v2/manifest.json), [CNN·GRU 구현](../../models/TPPs/CountAwareTitanCNNGRU.py). 설계와 구현 문서 및 Serena symbol 검색을 함께 사용했다. 현재 두 소스 파일의 SHA가 prelaunch 동결 manifest와 일치함을 확인했다. 과거 실행 결과를 현재 소스로 다시 생성한 것은 아니다.

- `CountAwareTitanCNNGRU.py`: `9d1bcf6c754cbf6216e4929f832d08af60ccf02573edd3b6c1c7a2088115dfc3`
- `CountAwareTitanCausalQKV.py`: `402f48187658401371ccadf0bf6ba2dc3413e1442a94d837562dac6b5d5bba68`

**확인된 구조:** CNN은 첫 encoder의 event Q/K/V projection 뒤에 붙는 k=3 채널별 잔차 연산이다. persistent K/V를 convolution하지 않는다. lag는 물리적 padding 위치가 아니라 현재와 직전 두 관측 사건이다. 유효 미관측 사건은 새 CNN/GRU 구간을 끊는다. 이 reset이 기존 attention에 이미 포함된 과거 문맥 전체를 제거한다는 뜻은 아니다. 따라서 continuous-time kernel이나 causal inference 모델이라고 부르지 않는다.

**확인된 구조:** GRU는 64→54→GRU54→64이며 기존 hidden에 더하는 보정이다. 첫 관측은 상태만 갱신하고 두 번째부터 보정한다. padding은 상태를 유지하고 미관측 valid 위치는 reset하며, 호출·표본·창 사이 상태는 전달하지 않는다. ‘작은 순환 보정’은 64차원 주 경로에 비해 제한된 별도 상태라는 뜻으로만 사용할 수 있다. 16차원 GRU, 임의 길이의 장기 기억, 테스트 중 가중치 학습으로 표현하지 않는다.

**비교 한계:** 보정 파라미터는 MLP16 24,576개, GRU54 24,732개(약 0.635% 증가), CNN 추가 576개다. 근접 예산은 동일 계산량·처리 지연·표현력의 증명이 아니다. MLP→GRU 교체는 입력 문맥, 활성 문턱, 합산 계수와 비선형 연산도 바꾸므로 그 결과를 recurrence 하나의 인과 효과로 해석하지 않는다. [설계의 변경 경계](../titantpp_cnn_gru_width16_design_20261004_v1/README.md)

## 4. 성능 winner와 Contribution 후보의 분리

아래는 본 검토의 **조건부 권고**다. 수치의 확정 순위는 통합 비교표에 연결한다.

| 관측 결과 | 후속 판단 |
|---|---|
| CNN+MLP16가 같은 seed MLP16보다 개선하고 GRU 추가가 더 낫지 않음 | CNN+MLP16를 단순한 후속 성능 후보로 선택하고 GRU를 핵심 기여에서 제외한다. 국소 처리의 효과를 좁혀 검증한다. |
| CNN+GRU54가 CNN+MLP16 및 GRU54를 모두 개선 | 결합 후보로 남기되 두 구성의 독립 효과와 interaction을 검증한다. 한 dataset·한 seed 개선으로 보편적 synergy를 주장하지 않는다. |
| GRU가 시간 NLL을 개선하지만 수량 RMSE를 악화 | 시간 적합도와 수량 예측의 상충 사례로 기록한다. 임의 종합점수로 수량 선택 계약의 패배를 감추지 않는다. |
| 결합 모델이 큰 수량 지표만 개선 | tail 특화 연구 가설로 남긴다. 전체 수량 오차·시간 손실·tail 표본 수를 함께 공개하고 전체 승자와 구분한다. |
| MLP 폭 확대나 기존 공통-head backbone이 우세 | CNN·GRU의 부정적 대조도 보존한다. 단순성·용량의 효과가 더 설명력이 있으면 구조 복잡화 주장을 내려놓는다. |

참고로 학습 종료 문서의 **선택 checkpoint Validation / seed42 / 세 신규 후보 내부 비교**에서 CNN+MLP16는 세 데이터 모두 전체 RMSE가 가장 낮다. Intermittent 큰 수량 RMSE는 CNN+GRU54가 세 신규 후보 중 가장 낮고, 시간 NLL은 세 데이터 모두 GRU54가 가장 낮다. 따라서 현재 구조 검토만으로 CNN+GRU54를 기본 후속 모델로 미리 고정할 이유는 없다. 이는 기존 MLP4/8/12/16·TPP baseline까지 포함한 승자 판정이 아니며 Test 결과를 반영한 문장도 아니다. [Validation 원본 요약과 수치](../titantpp_cnn_gru_training_terminal_20261005_v1/README.md)

공통-head 대조는 backbone·보정 효과를 비교하는 데 사용하고 native-head 대조는 전체 모델 체계 성능으로 구분한다. 서로 다른 시간 likelihood/support/reduction/clamp의 NLL은 숫자만 합쳐 순위를 매기지 않는다. 평가 계약이 일치하는 수량 지표와 각 모델 자체의 시간 지표를 구별한다. [프로젝트 평가 원칙 §0.2·9·12](../../TEST_SESSION_PROTOCOL.md)

## 5. 후속 기여를 검증하는 순서

**기존 성능과 비교 계약을 확정한다 — 다음 작업**
- 통합 세션에서 동일 split·선택 규칙·seed의 기존 baseline과 신규 결과를 연결한다. 선택은 사전 계약의 Validation 기준을 유지하고 Test는 해당 선택의 관측 결과로 표시한다.
- 이미 본 Test가 후속 구조 선택에 영향을 주었다면 그 사실을 명시한다. 같은 Test 재평가나 새로운 seed만으로 독립 최종 검증이 생기지 않는다.

**동일 seed의 2×2 대조로 효과를 분리한다 — 다음 작업**
- MLP16, CNN+MLP16, GRU54, CNN+GRU54를 paired 비교한다. 각 metric에서 CNN 효과 두 개, 보정 모듈 교체 효과 두 개, `결합−CNN단독−GRU단독+MLP16` interaction을 기록한다.
- 기존 폭4/8/12/16으로 단순 용량 증가 효과를 함께 해석한다. 순환 자체를 주장하려면 입력 범위·활성 조건·계수·예산이 맞는 비순환 대조를 추가로 설계한다.

**재현성과 비용을 확인한다 — 다음 작업, 학습 실행은 본 검토에 포함하지 않음**
- seed42·52·62에서 paired 변화 방향, 평균·표본 표준편차·범위를 공개한다. 3seed만으로 통계적 강건성을 단정하지 않는다.
- 실제 전체·trainable 파라미터, 같은 장비·batch·길이의 지연·메모리를 함께 기록한다. 현 seed42의 이종 GPU 실행 시간을 직접 구조 효율 순위로 사용하지 않는다.
- tail 표본 수, train 고정 경계, 전체 RMSE·MAE·tail RMSE·시간 NLL을 같은 선택 epoch에서 비교한다. tail 또는 시간 지표로 checkpoint를 사후 교체하지 않는다.

**최종 기여 문장을 확정한다 — 앞선 검증 뒤**
- 적절한 후보 문장: ‘관측 사건의 국소 패턴과 제한된 순환 보정을 기존 count-aware 표현에 결합하고, 근접 파라미터 예산 대조를 통해 수량·큰 수량·시간 적합도 사이의 조건별 효과를 분석한다.’
- 개선이 반복 확인될 때만 데이터·metric 범위를 명시하여 ‘개선한다’를 추가한다. ‘최초 CNN+GRU TPP’, ‘새로운 Titans test-time memory’, ‘모든 데이터에서 우월’은 현 근거로 허용되지 않는다.

## 6. 신뢰도와 미확인 범위

- **높음:** CTPP에 CNN→GRU 선행 구조가 존재하고 Titans에 QKV convolution이 명시되어 있다는 원문 사실; 동결 소스 SHA 일치; 현재 GRU 폭·reset·잔차 계약.
- **중간:** 이 구현의 논문 기여를 통제된 응용·실증 분석으로 좁히는 것이 타당하다는 연구 판단. 채택 가능성을 보장하지 않는다.
- **미확인:** 전체 baseline 포함 Test 승자, 다중 seed 개선, recurrence 단독 효과, 실제 지연 우위, tail 메커니즘의 인과 설명, 독립 미접근 외부 평가.
- **미수행 plane:** community/market는 이 구조 기여 질문에 필요하지 않아 조사하지 않았다. 문헌과 현 프로젝트 근거의 교차 검토만 수행했다. `.agents/` 보호 경로는 수정하지 않았고 지정된 프로젝트 실행 프로토콜 경로는 현재 존재하지 않았다. 본 보고서 외 코드·원격·학습·평가·설정 변경은 없다.
