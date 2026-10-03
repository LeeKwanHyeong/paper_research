# 이력 보완 TitanTPP 논문 기획: 출처와 확인 범위

검토일: 2026-09-27. 연결 문서: [연구 결정과 Contribution 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_scope_20260927_v1/research_decision.md).

## 조사 방법과 신뢰 표시

현재 코드와 완료된 validation 증거를 먼저 확인한 뒤, 국소 이력·분해·연속 시간 상태공간·혼합 속성 사건 예측·간헐 수요를 중심으로 원논문과 저자 연결 구현을 탐색했다. `oma-search`의 원출처 우선 원칙을 적용했다. 아래에서 **출판·저자 원출처**, **공개 구현**, **미확인**을 구분한다. 웹에 공개된 구현이라는 사실만으로 재현 충실도나 실행 호환성을 보증하지 않는다.

- `A`: 학회·저널·기관 또는 저자가 제공한 논문/서지 원출처. 이는 출처의 직접성을 뜻하며 연구 주장의 무오류를 뜻하지 않는다.
- `B`: 원논문·저자가 연결한 공개 코드. 읽은 범위까지만 근거로 삼는다. 별도의 실행 재현은 하지 않았다.
- `미확인`: 접근 제한, 설치·학습 미수행, 구현 연결의 불명확성 등 남은 검증 사항.
- 검색 엔진 결과나 논문 목록은 발견 경로로만 사용했다. 비교 모델 선정과 기술적 설명은 아래 원출처에 연결했다.
- 도메인 신뢰 휴리스틱과 원출처 여부는 별개다. GitHub 전체에 대한 일반 평점으로 저자의 코드 연결을 대신 판단하지 않았다. 학회·저널별로 확인되지 않은 수치형 신뢰 점수를 임의 생성하지 않았다.
- 전체 문헌의 체계적 전수조사나 동일 아이디어 부재 증명은 아니다. “최초”라는 단정의 근거로 사용하지 않는다.

## S01. CTPP: 가까운 국소·전체 이력 선행 연구

**Wang-Tao Zhou, Zhao Kang, Ling Tian, Yi Su. _Intensity-free Convolutional Temporal Point Process: Incorporating Local and Global Event Contexts_. Information Sciences, 646, 119318, 2023.**

- A: [출판사 서지](https://www.sciencedirect.com/science/article/abs/pii/S0020025523009039), [arXiv](https://arxiv.org/abs/2306.14072), [본문 HTML](https://arxiv.org/html/2306.14072).
- B: [논문이 연결한 CTPP 저장소](https://github.com/AnthonyChouGit/Convolutional-Temporal-Point-Process).
- 확인: 연속 시간 convolution을 통한 국소 문맥과 RNN 기반 전체 문맥 결합 설명, 논문 내 코드 연결, 저장소 README·파일 목록.
- 이번 판단: 국소 문맥을 TPP에 추가했다는 사실만으로 신규성을 주장할 수 없다. 현재의 인접 은닉 상태 잔차와 구조를 직접 대비해야 한다.
- 미확인: 전체 코드의 모델·학습 구현 충실도, 현 데이터 adapter 및 Runtime 호환성. 모델을 실행하지 않았다.

## S02. TimeMixer: 수량 예측과 분해 구조의 핵심 비교

**Shiyu Wang et al. _TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting_. ICLR 2024.**

- A: [논문](https://arxiv.org/abs/2405.14616), [본문 HTML](https://arxiv.org/html/2405.14616), [OpenReview](https://openreview.net/forum?id=7oLshfEIC2).
- B: [저자 저장소](https://github.com/kwuking/TimeMixer), [모델 코드](https://raw.githubusercontent.com/kwuking/TimeMixer/main/models/TimeMixer.py).
- 확인: 본문 방법 절의 과거 분해·mixing과 예측 결합, 코드의 forecast 경로 및 미래 시간 feature 선택 항목.
- 이번 판단: 분해 기반의 강한 수량 예측 후보이자 구조적으로 인접한 비교다. 사건 순서에 맞춘 입력은 원 논문의 규칙적 시간 예측과 다른 적응형임을 명시한다.
- 미확인: 짧은 prefix·padding·관측 간격 feature의 adapter와 현 데이터 성능. OpenReview 접근 제한 시 arXiv 본문과 저자 자료를 대조했다. 설치나 학습은 하지 않았다.

## S03. TimeMixer++: 최신 후속 연구를 포함하되 구현을 혼동하지 않음

**Shiyu Wang et al. _TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis_. ICLR 2025.**

- A: [학회 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/2b187165e28fdfdc0ffb34d1bfff2b0c-Paper-Conference.pdf), [arXiv](https://arxiv.org/abs/2410.16032), [본문 HTML](https://arxiv.org/html/2410.16032).
- B: 논문은 [TimeMixer 저장소](https://github.com/kwuking/TimeMixer)를 연결한다.
- 확인: 여러 해상도의 패턴 분석·분해·mixing이라는 연구 방향과 코드 공개 안내.
- 이번 판단: 관련 연구에 포함한다. TimeMixer와 동일 모델이라고 처리하지 않는다.
- 미확인: ++에 해당하는 정확한 실행 파일·버전·재현 환경. 이번 검토에서 확인하지 못했으며 코드 부재를 단정하지 않는다.

## S04. Deep Renewal Processes: 간헐 수요의 직접적인 선행 연구

**Ali Caner Türkmen, Tim Januschowski, Yuyang Wang, Ali Taylan Cemgil. _Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes_. PLOS ONE 16(11): e0259764, 2021.**

- A: [저널 본문과 DOI](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764).
- B: 논문이 연결한 GluonTS의 [현재 renewal estimator 경로](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/renewal/_estimator.py).
- 확인: 수요 크기와 사건 간격을 모델링하는 renewal 관점, 평가 문제 설정, 공개 코드 경로. 현재 파일은 MXNet 계열 아래에 있다.
- 이번 판단: 간헐 수요의 시간·수량 공동 모델링은 새 문제가 아니다. 이 분야의 경쟁력을 주장한다면 해당 계열의 비교가 필요하다.
- 미확인: 현재 저장소 버전의 설치 가능성과 다음 사건별 수량·시간 출력 경로. 원 논문의 달력 기간별 예측 점수를 본 과제의 다음 사건 수량과 직접 비교하지 않는다.

## S05. S2P2: 최근 TPP 인코더 비교로 선정

**Yuxin Chang, Alex Boyd, Cao Xiao, Taha Kass-Hout, Parminder Bhatia, Padhraic Smyth, Andrew Warrington. _Deep Continuous-Time State-Space Models for Marked Event Sequences_. NeurIPS 2025.**

- A: [공식 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2412.19634), [본문 HTML](https://arxiv.org/html/2412.19634).
- B: [저자 저장소](https://github.com/UCIDataLab/state_space_point_process), [torch_s2p2.py](https://raw.githubusercontent.com/UCIDataLab/state_space_point_process/main/easy_tpp/model/torch_model/torch_s2p2.py).
- 확인: 연속 시간 latent dynamics·jump와 비선형 결합, 병렬 scan 설명, 이산 mark 입력 및 사건 전후 상태를 반환하는 코드 인터페이스. 저장소의 Python 3.12 이상 요구를 확인했다.
- 이번 판단: 과거 수량 입력과 공통 head를 붙이는 **S2P2-matched**의 비교 가치를 선정했다. 이는 원래 S2P2 intensity 모델의 재현과 구분한다.
- 미확인: adapter, 초기화·학습 규칙, 설치·GPU 성능. 원 논문의 likelihood 개선율을 우리 수량 예측의 예상 개선율로 사용하지 않았다.

## S06. FlexTPP: 혼합 속성 사건 예측 비교로 선정

**Felix Draxler, Yang Meng, Kai Nelson, Lukas Laskowski, Yibo Yang, Theofanis Karaletsos, Stephan Mandt. _Transformers for Mixed-type Event Sequences_. NeurIPS 2025.**

- A: [공식 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html), [논문 PDF](https://papers.nips.cc/paper_files/paper/2025/file/a6c7515ac435277dc92b75a07bb2257c-Paper-Conference.pdf), [OpenReview](https://openreview.net/forum?id=MtwsRjPZhf).
- B: proceedings가 직접 연결한 [공개 저장소](https://github.com/czi-ai/FlexTPP), [pyproject.toml](https://raw.githubusercontent.com/czi-ai/FlexTPP/main/pyproject.toml).
- 확인: 공식 초록의 혼합 속성·연속 분포 head, 검색에 노출된 논문 4.1절의 속성별 autoregressive 분해, 저장소 README와 의존성. Python 3.12 이상 요구를 확인했다.
- 출처 주의: 학회 페이지는 위 저장소를 공개 코드로 연결하지만 README는 “unofficial implementation”이라고 적는다. **논문 연결 구현**으로 기록하며, 검증된 공식 재현이라고 단정하지 않는다.
- 접근 한계: 전체 PDF는 웹 도구의 파일 크기 제한, OpenReview는 접근 제한이 있었다. 전체 수식·부록을 완독한 수준의 구현 감사는 아니다.
- 이번 판단: 연속 수량·시간 외부 방식 비교 후보로 선정한다. 미래 시각을 주지 않고 수량을 예측하는 경로와 정수 관측 변환은 후속 과제다. 연속 속성 head를 사용했다는 이유로 이산 수량 확률질량까지 검증됐다고 표현하지 않는다.

## S07. Titans: 프로젝트 이름과 실제 메모리의 차이를 명시

**Ali Behrouz, Peilin Zhong, Vahab Mirrokni. _Titans: Learning to Memorize at Test Time_. NeurIPS 2025.**

- A: [공식 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2501.00663).
- 확인: 학습·추론 시 neural memory를 갱신하는 원 논문의 핵심 설명.
- 이번 판단: 현재 static Hard-LMM과 온라인 갱신 메모리를 구분해야 한다. 현재 논문에 Titans의 기여를 그대로 이전하지 않는다.
- 미확인: Titans 전체 구현 재현. 이번 선정 대상은 원래 Titans와의 전체 언어 모델 벤치마크가 아니다.

## S08. Intensity-Free TPP: 시간 head의 선행 연구

**Oleksandr Shchur, Marin Biloš, Stephan Günnemann. _Intensity-Free Learning of Temporal Point Processes_. ICLR 2020.**

- A: [저자 기관 페이지](https://www.cs.cit.tum.de/en/daml/intensity-free-tpp/), [arXiv](https://arxiv.org/abs/1909.12127), [논문 PDF](https://openreview.net/pdf?id=HygOjhEYDH).
- 확인: 조건부 사건 간격 분포를 직접 다루는 접근과 lognormal mixture 설명.
- 이번 판단: lognormal 기반 시간 예측 자체를 현재 모델의 신규 기여로 삼지 않는다. 현재 단일 조건부 lognormal 관측 head와 mixture 모델을 동일시하지 않는다.
- 미확인: 새로운 native baseline 구현·실행. 최신 비교 역할은 S2P2와 FlexTPP가 우선이다.

## S09. EdiTPP: 2026년 연구까지 검토하되 과제 차이로 후순위

**David Lüdke, Marten Lienen, Marcel Kollovieh, Stephan Günnemann. _Edit-Based Flow Matching for Temporal Point Processes_. ICLR 2026.**

- A: [저자 기관 페이지](https://www.cs.cit.tum.de/daml/editpp/), [OpenReview](https://openreview.net/forum?id=FNf9IV1P2L), [arXiv](https://arxiv.org/abs/2510.06050).
- 코드 발견 경로: [저자 연결 저장소](https://github.com/martenlienen/editpp). 이번에는 구현 내부를 검토하지 않았다.
- 확인: 출판 정보와 삽입·삭제·치환을 이용한 사건열 생성이라는 문제 설정.
- 이번 판단: 최신 연도만으로 핵심 비교에 넣지 않는다. 사건열 생성·기간 내 사건 수와 다음 사건의 수량은 다른 예측 대상이다.

## S10. DLinear: 낮은 복잡도의 수량 비교 후보

**Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. _Are Transformers Effective for Time Series Forecasting?_. AAAI 2023.**

- A: [arXiv](https://arxiv.org/abs/2205.13504), [학회 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089).
- B: [저자 저장소 LTSF-Linear](https://github.com/cure-lab/LTSF-Linear).
- 확인: 선형 기반 시계열 예측 비교와 코드 출처. 유사 이름의 외부 fork를 공식 출처로 쓰지 않았다.
- 이번 판단: 추가 저비용 비교 후보. 동일 용량 이력 MLP라는 기전 대조군을 대체하지는 않는다.
- 미확인: 사건 순서 입력에 대한 adapter와 현재 데이터 성능.

## S11. PatchTST: 예비 수량 비교 후보

**Yuqi Nie et al. _A Time Series is Worth 64 Words: Long-term Forecasting with Transformers_. ICLR 2023.**

- A: [논문](https://arxiv.org/abs/2211.14730).
- B: [저자 저장소](https://github.com/yuqinie98/PatchTST).
- 확인: patch 기반·channel-independent 시계열 예측이라는 방향과 공개 코드 연결.
- 이번 판단: 우선 TimeMixer를 선정하고 PatchTST는 예비 후보로 남긴다. 성능 결과를 본 후 유리한 비교 모델만 남기는 용도로 사용하지 않는다.
- 미확인: 전체 구현 감사 및 Runtime 검증.

## S12. EasyTPP: 구현·평가 비교를 위한 참고

- B: [EasyTemporalPointProcess 저장소](https://github.com/ant-research/EasyTemporalPointProcess), [구현 차이 문서](https://ant-research.github.io/EasyTemporalPointProcess/advanced/implementation.html).
- 확인: TPP 비교 구현과 구현별 차이에 대한 문서 경로, S2P2 연계 여부.
- 이번 판단: 비교 구현의 출발점으로 유용하지만 공통 head로 변경한 모델을 원 알고리즘의 native 성능이라고 보고하면 안 된다.
- 미확인: 이 프로젝트에 도입할 정확한 commit과 동결 Runtime. 현재 공유 환경에 설치하지 않았다.

## 로컬 증거와 검토 범위

| 파일 | 이 문서에서의 용도 |
| --- | --- |
| [CountAwareTitanMultiLagDetail.py](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanMultiLagDetail.py) | 실제 수준/변화 연산, 분기별 참조 거리·활성 규칙, 0 초기화, 삽입 위치. |
| [CountAwareTPP.py](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTPP.py) | 입력·공통 head·수량 목적함수와 기반 모델 확인. |
| [memory.py](/Users/igwanhyeong/PycharmProjects/paper_research/models/Titan/common/memory.py) | static Hard-LMM과 메모리 동작 구분. |
| [replication 계약](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/local_detail_benchmark_replication_v1.json), [복구 계약](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/local_detail_instacart_remaining_recovery_v1.json) | 완료 실행의 architecture 및 모델 파일 SHA 연결. |
| [final_report.md](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/final_report.md) | 완료 학습, validation 결과, 실패 이력과 한계. |
| [three_seed_summary.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/three_seed_summary.json) | 평균·표본 SD·구간별 표본수·비용 재확인. |
| [terminal_audit.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/terminal_audit.json) | 54조건/108재평가 및 실행 성공과 성능 판정 구분. |

실제 데이터나 checkpoint를 로드하지 않았고, held-out 예측·성능 및 validation/test 혼합 결과를 열람하지 않았다. 새 장기 학습, 원격 실행, 패키지 설치, 코드·계약 변경도 하지 않았다. 공개 저장소 링크는 움직이는 branch일 수 있으므로 실행 계약 단계에서 commit을 고정해야 한다. 이번 수치 검증과 파일 식별 결과는 [verification_receipt.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_scope_20260927_v1/verification_receipt.json)에 기록한다.
