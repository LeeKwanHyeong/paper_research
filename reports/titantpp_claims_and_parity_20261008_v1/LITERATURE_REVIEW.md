# 핵심 주장과 검증 계약을 위한 표적 문헌 검토

- 상태: 완료 — 2026-10-08, 7편의 관련 방법 절 검토. 체계적 문헌 조사나 신규성 확정은 아님.
- 질문 1: 인코더가 prefix 이력을 이미 사용해도, 현재·직전 관측 hidden state를 별도 보정 경로에 연결하면 추가 이득이 있는가?
- 질문 2: 학습된 정적 prototype 64개 중 top-4를 검색할 때 시간·수량 점수를 고정 비중으로 결합하는 제약이 일반 학습 검색보다 도움이 되는가?
- 산출물 성격: 내부 연구 설계용 기술 메모. 학회 제출용 본문 초안이 아님. 실험 실행·코드 변경 없음.

## 판정

**핵심 1은 검증할 가치가 있는 중심 가설이지만 방법의 신규성은 미확정이다.** 이력을 신경망으로 표현하는 것, attention으로 과거 상태를 조회하는 것, residual 경로를 사용하는 것은 이미 선행 방법에 있다. 따라서 신규성 후보는 수량 예측에서 특정 현재·직전 상태 연결과 제한된 보정 구조가 제공하는 추가 효과로 좁혀야 한다. 기존 MLP8 구조 대조를 CNN+GRU54의 검증으로 옮겨 쓰지 않는다.

**핵심 2는 조건부 보조 가설이다.** 정적 학습 기억, top-k 선택, 유사도 가중합 각각은 기존 구성요소다. 시간·수량의 분리된 점수 설계가 동일한 입력·활성 파라미터·검색 예산의 일반 검색보다 유리한지가 남은 질문이다. prototype 내용이 좋아졌다는 주장이나 실제 과거 사건 bank를 검색한다는 설명은 현재 과학 계약과 다르다.

위 판정은 아래 원문 사실과 프로젝트에서 제공된 설계 설명을 대조한 **해석**이다. 동일한 전체 방법이 문헌에 없다는 증명은 아니다.

## 출처별 사실과 읽은 범위

모든 항목의 증거 종류는 **academic / primary**다. 방법 설명의 신뢰도는 높지만, 전체 문헌 대비 최초성 판단의 신뢰도는 낮다. 초록만으로 신규성 결론을 내린 항목은 없다.

| 원문 | 확인한 방법 사실 | 읽기 깊이와 비교 의미 |
|---|---|---|
| [Neural Hawkes Process](https://papers.nips.cc/paper/2017/file/6463c88460bd63bbe256e495c63aa40b-Paper.pdf) | continuous-time LSTM hidden state가 과거 사건의 순서·시간을 반영하고 intensity를 결정한다. | §3.2.2, 식 4–7. 이미 이력 요약이 존재하므로 '기존 모델에는 이력이 없다'는 전제는 부적절하다. |
| [Transformer Hawkes Process](https://proceedings.mlr.press/v119/zuo20a/zuo20a.pdf) | causal self-attention이 과거 사건을 가중해 표현하며 미래 위치를 mask한다. | §3.1, 식 4–5와 masking 설명. 최근 상태 연결의 이득은 이력 attention 자체와 구분해야 한다. |
| [Self-Attentive Hawkes Process](https://proceedings.mlr.press/v119/zhang20q/zhang20q.pdf) | 시간 encoding을 포함한 사건 표현으로 과거를 가중 요약하고 intensity의 파라미터를 만든다. | §4, history hidden vector와 intensity 식. 시간 정보를 검색·표현에 넣는다는 넓은 주장은 차별점이 아니다. |
| [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://www.cs.jhu.edu/~jason/papers/yang%2Bal.iclr22.pdf) | A-NHP는 continuous-time query와 과거 사건 attention을 쓰고, layer embedding 식에 residual 연결이 있다. | §2–3, 식 1–6. residual과 이력 연결의 존재만으로 신규성을 주장하면 가장 직접적인 반론이 된다. layer residual과 현재·직전 상태 별도 경로는 구별해야 한다. |
| [Titans: Learning to Memorize at Test Time, v1](https://arxiv.org/html/2501.00663v1) | context에 따라 갱신되는 neural memory와 input-independent persistent parameters를 구분하며, context·gate·layer 결합을 제시한다. | §3.1, §3.3, §4.1–4.3. 현재의 정적 prototype 검색을 Titans의 test-time memory update와 동일시하면 안 된다. 이 검토는 v1 기준이다. |
| [Augmenting Self-attention with Persistent Memory](https://arxiv.org/pdf/1907.01470) | 입력과 독립적인 학습 key/value 벡터를 attention pool에 추가한다. | §4.1–4.2, 식 9–14. 정적 학습 prototype과 task-level memory의 넓은 아이디어는 선행 영역이다. |
| [Large Memory Layers with Product Keys](https://papers.neurips.cc/paper/9061-large-memory-layers-with-product-keys.pdf) | 학습 query로 top-k key를 고르고 softmax 점수로 value를 가중합한다. key/value는 입력 token이 아닌 학습 embedding이다. | §3.1, 식 1–3 및 product-key 설명. learned top-k weighted retrieval 자체는 새로운 기여로 삼기 어렵다. 이 논문의 대용량 product-key 구성과 우리의 64개 동일 bank는 다르다. |

## 계약에 반영할 비교 설계

### 1. 명시적 직전 상태 연결의 추가 효과 — 다음 작업

- 같은 encoder와 출력 계약을 유지하고, 보정 없음·현재 상태만 받는 활성 용량 대조·현재와 직전 상태를 받는 보정을 비교한다. '현재 상태만'도 이미 prefix 이력을 포함하므로 무이력 대조라고 부르지 않는다.
- 동일 파라미터 수만 맞추는 데 그치지 않고 입력 차원, 활성 연산, 초기 residual 출력 분포를 기록한다. 사용하지 않는 dummy parameter로 예산을 맞추지 않는다.
- 단순한 lag-1 concatenate/project 경로 또는 작은 두 상태 MLP를 가까운 내부 대조 후보로 둔다. 새 보정이 이 단순 경로보다 나아야 복잡한 메커니즘의 필요성을 설명할 수 있다. 이는 문헌에서 도출한 **설계 추천**이며 현재 실행 승인을 추가하지 않는다.
- 직전 상태의 인덱스, series 경계, padding, target 전에 관측된 정보만 사용한다는 조건을 계약으로 적는다. 관측 수에 따른 분기 활성화를 서로 다른 lag나 다양한 시간 해상도로 해석하지 않는다.
- 전체·큰 수량 지표와 같은 선택 epoch의 TimeNLL을 함께 보고한다. 수량 개선을 '모든 예측이 개선된다'로 확대하지 않는다.

### 2. 시간·수량 점수 분리의 추가 효과 — 기준선 정합 이후

- mean → weighted 비교는 가중합 효과, weighted → fixed 비교는 query/score 설계 변경 효과다. 여러 변경이 함께 들어가면 점수 분리 하나에 원인을 귀속하지 않는다.
- fixed와 일반 검색에 같은 과거 정보, prototype bank 크기 64, top-4, value 경로, 학습·선택 규칙을 제공한다. fixed의 추가 512개와 기존 generic의 521개는 동일 예산으로 표현하지 않는다.
- 시간 query만·수량 query만·고정 혼합은 후보 기전 대조다. 고정 비중 0.5/0.5의 선정 경위와 score normalization·temperature·tie handling을 미리 적는다.
- **수학적 표현 동등성 점검:** 두 점수가 각 view의 내적이면 가중합은 스케일된 view들을 concatenate한 단일 내적으로 표현 가능하다. 각 view를 먼저 정규화한 cosine도 그 정규화된 표현 수준에서는 같은 점검을 할 수 있다. 실제 구현의 normalization 순서와 score 변환을 확인하기 전에는 모델 전체가 동등하다고 단정하지 않는다. 이 경우 기여 후보는 완전히 새로운 검색 연산보다는 입력 분리·제약이 주는 inductive bias다. 이는 원문 주장 아닌 대수적 설계 해석이다.
- 선택된 top-4를 고정하고 결합만 바꾸는 대조와, 결합 규칙을 고정하고 선택만 바꾸는 대조가 검색과 집계의 효과를 구분한다. 필요한 실험 수는 본 계약에서 우선순위를 정한다.
- 참조 bank와 query를 따로 평가하지 않고 end-to-end 학습하면 bank도 달라질 수 있다. 이를 검색 함수만의 순수 인과 효과로 부르지 않는다. prototype 의미가 개선됐다는 해석에는 별도 진단이 필요하다.

### 3. 두 기여를 독립적으로 주장하기 위한 조건 — 후속 검증

- 이력 보정 유무 × 평균/고정 검색 2×2에서 같은 지표의 주효과와 상호작용을 구분한다. 결합 성능만으로 두 독립 기여나 시너지를 선언하지 않는다.
- seed·dataset 반복과 미노출 평가가 필요하다. Test를 보고 고른 fixed는 후속 가설이며, 이미 열람한 Taxi Test를 다시 평가해도 새로운 독립 평가가 되지 않는다.
- 공통 출력부 외부 비교는 전체 모델의 실용적 성능 근거이고, 내부 예산 대조는 변경 부분의 기여 근거다. 원 논문의 native 설정 성능과 common-head 구현 결과를 구분한다.

## 조사 범위와 미확인

- 학술 평면: 목표 7편 중 7편의 관련 방법 절에 접근. 모든 본문·부록을 정독한 것은 아니다. 표에 적힌 절만 근거 범위다.
- Web/docs/code 평면: 문헌을 원 출판처·저자 페이지에서 취득하는 데 사용. 이번 하위 작업은 로컬 구현을 새로 감사하지 않았으며, 프로젝트 구조 설명은 부모 작업의 동결 설계 정보를 전제로 한다.
- Community/market 평면: 이 질문의 신규성 판정 근거로 적합하지 않아 제외. 검색에 노출된 블로그·커뮤니티·비공식 구현은 근거로 채택하지 않았다.
- `oma` CLI 없음, Knows 검색 endpoint는 web 도구에서 오류. 원문을 직접 사용했다. OpenReview A-NHP PDF는 challenge로 차단되어 저자 JHU 공개 PDF로 회복했다. persistent-memory HTML 오류는 arXiv PDF로 회복했다. 유료 API 호출 0회.
- 명시된 shared Codex execution protocol 파일은 해당 로컬 skill 경로와 프로젝트 `.agents` 검색에서 찾지 못했다. `.agents`에는 쓰지 않았고 승인된 두 보고서 경로만 사용했다.
- 최신 2026년까지의 포괄적 인용망 조사, count/quantity-aware marked TPP의 가장 가까운 방법, S2P2의 정확한 방법 절 대조, task-specific multi-view retrieval 전반은 이번 7편 범위 밖이다. **발견하지 못함은 존재하지 않음을 뜻하지 않는다.**
- 따라서 PAKDD 채택 가능성, 최초성, 통계 유의성, 현재 구현의 누출 부재를 이 메모가 확정하지 않는다. 다음 문헌 작업은 S2P2와 count-aware TPP, lag/local residual 보정의 최근 선행방법을 직접 대조하는 것이다.

기계 판독 기록: [literature_review.json](evidence/literature_review.json).
