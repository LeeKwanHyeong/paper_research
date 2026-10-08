# S2P2와 관련 방법의 기여 대조

2026-10-08에 확인한 연구 검토 자료다. 근거 유형은 모두 **academic / primary**다. 아래 여섯 문헌의 관련 방법 절에 접근했다. 이는 표적 문헌 대조이며 전체 인용망을 포괄하는 조사나 최초성 판정은 아니다. 프로젝트 Test 성과를 열람하거나 외부 모델을 실행하지 않았다.

## 1. 무엇이 이미 알려져 있고 무엇을 검증할 것인가

| 선행연구와 확인 위치 | 원문에서 확인한 사실 | 이번 원고에 반영할 차이·한계 |
|---|---|---|
| **S2P2**, Chang et al. (2025), §2.1, §3.1–3.4, 식 8–9·12·14. [출판처](https://papers.nips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html), [방법 원문 v2](https://arxiv.org/html/2412.19634v2) | 유한 categorical mark 공간에서 연속시간 상태 전이와 사건 impulse를 모델링한다. 식 14에는 GELU·residual·LayerNorm이 있고, 연속 hidden state에서 intensity를 구성한다. | 이력·residual의 존재 자체는 차별점이 아니다. 공통 수량 회귀·lognormal duration head를 붙인 S2P2는 **backbone adaptation**이다. native likelihood·연속시간 decoding 성과를 재현했다고 표현하지 않는다. |
| **Deep renewal**, Türkmen et al. (2021), §3.1, §4.3–4.4, 식 3–4, Table 3. [원문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764) | 이전 수요 크기와 수요 간격을 recurrent representation에 입력하고 조건부 간격·크기 분포를 모수화한다. 연속시간 확장도 설명한다. | 수량과 도착 간격을 이력으로 함께 예측한다는 넓은 주장은 선행 영역이다. 범주형 사건 유형을 생략해도 수량은 수학적으로 mark가 될 수 있다. “최초 mark-free 수량 TPP”라는 표현을 채택하지 않는다. |
| **RMTPP**, Du et al. (2016), §5.1, Fig. 2, 식 8. [저자 PDF](https://people.mpi-sws.org/~manuelgr/pubs/rmtpp.pdf), [DOI](https://doi.org/10.1145/2939672.2939875) | 시간·mark와 이전 hidden state의 영향을 recurrent vector로 요약한다. | 질문은 이력 유무가 아니라, 이력을 학습하는 구조 안에 추가한 명시적 보정 경로의 효과다. |
| **A-NHP**, Yang et al. (2022), §2 식 1–3, §3. [저자 PDF](https://www.cs.jhu.edu/~jason/papers/yang%2Bal.iclr22.pdf) | 시간 query/key/value attention과 layer residual을 사용하며 기본 generative 모델을 A-NHP로 설명한다. | 시간 attention·residual 일반 개념과 CNN 국소 Q/K/V 보정 및 recurrent residual의 구체적 설계를 구분한다. |
| **Persistent memory**, Sukhbaatar et al. (2019), §4.1–4.2, 식 9–14. [원문](https://arxiv.org/pdf/1907.01470) | 입력에 조건화되지 않은 학습 key/value를 attention pool에 추가한다. | 정적 학습 prototype 자체는 새로운 발명이 아니다. 이번 bank는 실제 관측 사건을 저장하거나 평가 시 갱신하는 메모리가 아니다. |
| **Product-key memory**, Lample et al. (2019), §3.1 식 1–3. [출판처 PDF](https://papers.neurips.cc/paper/9061-large-memory-layers-with-product-keys.pdf) | 학습 query로 top-k key를 선택하고 softmax로 value를 결합한다. | top-k와 가중합 자체를 기여로 세지 않는다. 논문의 대형 product-key bank와 이번 64-slot bank를 동일 구조로 간주하지 않는다. |

## 2. 이번 설계에 대한 해석

**주질문 Q1:** 공통 입력·출력 계약과 초기화 아래, 명시적 국소·순환 보정 경로가 수량 예측에 추가로 도움이 되는가? 기존 MLP8의 직전 상태 대조는 동기 근거다. 대표 CNN+GRU54는 최근 3개 위치의 Q/K/V 보정과 recurrent residual을 함께 포함한다. 따라서 full 대조가 통과해도 “lag-1 하나만의 인과 효과” 또는 “CNN과 GRU 각각의 독립 효과”가 입증되는 것은 아니다.

**보조질문 Q2:** 시간·수량 summary의 분리 정규화와 고정 점수 결합이 같은 관측 정보·활성 파라미터 예산의 일반 검색보다 유리한가? 두 view의 cosine 평균은 정규화된 표현을 연결한 내적으로 쓸 수 있다. 기여 후보는 검색 연산의 발명보다 제약과 정규화가 주는 귀납 편향의 실증적 효과다. 두 모델의 학습 과정 전체가 동등하다는 결론은 아니다.

새 fixed 후보는 이전 평가 관찰 뒤에 형성된 후속 가설이다. 현재 Validation 실험은 개발용 재검증이며 독립 확인이 아니다. 이 문서에서 과거 Test 수치나 예측을 읽거나 재계산하지 않았다.

## 3. 외부 비교의 서술 규칙

본문에는 RMTPP·NHP·THP·SAHP·A-NHP·S2P2의 **공통 출력부 adaptation 비교**와 원 논문의 native 모델을 구분한다. 각 실행의 source revision·입력 population·time/quantity head·예산·선택 규칙·seed를 묶은 표가 필요하다. 이번 63조건은 외부 모델의 새 짝 비교를 포함하지 않는다. 과거의 특정 seed·버전 결과로 최종 대표 구조의 외부 우월성을 확정하지 않는다.

**동결 S2P2 adapter 확인:** categorical mark ID가 아니라 관측 gap·quantity의 log1p 연속 특징을 Linear(2,64)로 투영해 event impulse 입력에 전달한다. 관측 사건 직후의 right state를 target 직전 위치에서 읽어 공통 출력부에 연결한다. `simple_mark=True`를 단일 categorical mark 설정으로 해석하지 않는다. 로컬 scan은 O(N log N) element work와 O(log N) stages로 설명돼 있어 원저자의 work-efficient 복잡도를 그대로 주장할 수 없다. 과거 wrapper·vendor kernel·공통 caller의 SHA는 현재와 같지만 공통 model/factory는 달라, 전체 과거 학습 동등성은 미확정이다. [동결 adapter](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/source/models/TPPs/CountAwareAdditionalTPP.py:58).

Deep renewal은 수량 예측과 직접 관련된 필수 선행연구다. native 확률예측과 현재 log-quantity 회귀의 목적함수 차이를 설명하며, 이번 문헌 추가를 새 baseline 실행으로 해석하지 않는다.

## 4. 남은 검증과 문헌 공백

- **성능 검증 대기:** 대표 구조 Q1 27조건, Q2 27조건 및 새 mean9 재사용, 불리한 데이터·시간·큰 수량 지표와 비용을 모두 집계한다.
- **현재 대조의 한계:** full의 복합 경로를 lag-1만·CNN만·GRU만으로 분해하거나 단순 두 상태 MLP와 비교하는 조건은 현재 63조건에 없다. 특정 연산의 필수성을 주장하려면 별도 설계가 필요하다.
- **외부 비교 계약:** 적응된 S2P2와 native S2P2를 구분하고 기존 외부 실행의 provenance를 최종 대표 모델에 맞춰 확인한다. 현재 코드 관찰은 과거 checkpoint 실행 동일성의 증명이 아니다.
- **최초성 미확정:** 가장 가까운 time–quantity multi-view query/정규화 방식에 대한 넓은 인용망 조사가 남아 있다. 이번 여섯 논문으로 동일한 방법이 문헌에 없다고 주장하지 않는다.
- **독립 평가 대기:** 미노출 자료의 접근 계보를 확인하고 모델·selector를 동결한 뒤 별도 평가한다.

## 5. 서지 정보

1. Yuxin Chang, Alex Boyd, Cao Xiao, Taha Kass-Hout, Parminder Bhatia, Padhraic Smyth, Andrew Warrington. *Deep Continuous-Time State-Space Models for Marked Event Sequences*. NeurIPS 2025, Advances in Neural Information Processing Systems 38. DOI: 10.52202/085713-5431.
2. Ali Caner Türkmen, Tim Januschowski, Yuyang Wang, Ali Taylan Cemgil. *Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes*. PLOS ONE 16(11), e0259764, 2021. DOI: 10.1371/journal.pone.0259764.
3. Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, Le Song. *Recurrent Marked Temporal Point Processes: Embedding Event History to Vector*. KDD 2016, pp. 1555–1564. DOI: 10.1145/2939672.2939875.
4. Chenghao Yang, Hongyuan Mei, Jason Eisner. *Transformer Embeddings of Irregularly Spaced Events and Their Participants*. ICLR 2022.
5. Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Hervé Jégou, Armand Joulin. *Augmenting Self-attention with Persistent Memory*. arXiv:1907.01470, 2019. 이 항목은 확인된 arXiv 서지를 사용한다.
6. Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. *Large Memory Layers with Product Keys*. NeurIPS 2019.

기존 7편 검토는 [이전 문헌 메모](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_claims_and_parity_20261008_v1/LITERATURE_REVIEW.md)에 보존돼 있다. 중복 방법은 위 대조에 통합했으며 기존 메모와 동결 과학 계약은 수정하지 않았다.
