# Related Work reference register

## 3.0 Classical temporal point processes

### Spectra of Some Self-Exciting and Mutually Exciting Point Processes

- **Reference:** Hawkes, A. G. Biometrika, 58(1), 83-90, 1971.
- **Source:** [Oxford Academic](https://doi.org/10.1093/biomet/58.1.83)
- **Role:** classical temporal point process와 self-exciting process의 기본 배경을 제공한다.
- **Boundary:** neural encoder, continuous quantity, demand forecasting을 직접 다루는 문헌은 아니다.

### An Introduction to the Theory of Point Processes

- **Reference:** Daley, D. J. and Vere-Jones, D. An Introduction to the Theory of Point
  Processes, Volume I: Elementary Theory and Methods, 2nd ed. Springer, 2003.
- **Source:** [Springer](https://link.springer.com/book/10.1007/b97277)
- **Role:** conditional intensity와 point-process history formulation의 표준 이론 배경이다.
- **Boundary:** neural TPP나 intermittent demand forecasting을 직접 다루는 문헌은 아니다.

## 3.1 Recurrent neural temporal point processes

### Recurrent Marked Temporal Point Processes: Embedding Event History to Vector

- **Reference:** Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M., and Song, L.
  KDD, 2016.
- **Source:** [ACM DOI](https://doi.org/10.1145/2939672.2939875),
  [KDD PDF](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)
- **Role:** recurrent history vector와 joint time/mark prediction의 출발점이다.
- **Boundary:** 긴 시퀀스에서의 성능 저하를 직접 입증하는 문헌은 아니다.

### The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process

- **Reference:** Mei, H. and Eisner, J. NeurIPS, 2017.
- **Source:** [NeurIPS](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
- **Role:** continuous-time LSTM을 이용한 recurrent neural TPP 계열을 설명한다.

### Neural Temporal Point Processes: A Review

- **Reference:** Shchur, O., Türkmen, A. C., Januschowski, T., and Günnemann, S. IJCAI, 2021.
- **Source:** [IJCAI](https://www.ijcai.org/proceedings/2021/623)
- **Role:** neural TPP의 history encoder, intensity/decoder, mark space를 정리하는 일반 근거이다.

## 3.2 Attention- and memory-based history encoders

### Self-Attentive Hawkes Process

- **Reference:** Zhang, Q., Lipani, A., Kirnap, O., and Yilmaz, E. ICML, 2020.
- **Source:** [PMLR](https://proceedings.mlr.press/v119/zhang20q.html)
- **Role:** event history에 self-attention과 time-shifted positional encoding을 적용한 선행연구이다.

### Transformer Hawkes Process

- **Reference:** Zuo, S., Jiang, H., Li, Z., Zhao, T., and Zha, H. ICML, 2020.
- **Source:** [PMLR](https://proceedings.mlr.press/v119/zuo20a.html)
- **Role:** attention 기반 TPP의 핵심 비교 문헌이며 THP-matched encoder의 출처이다.
- **Boundary:** THP-matched는 원 논문의 encoder를 프로젝트 quantity interface에 맞춘 adapter이다.

### Titans: Learning to Memorize at Test Time

- **Reference:** Behrouz, A., Zhong, P., and Mirrokni, V. NeurIPS, 2025.
- **Source:** [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)
- **Role:** attention과 장기 메모리를 결합하는 설계 배경이다.
- **Boundary:** TitanTPP는 원 논문의 test-time learning을 구현하지 않는다. 현재 frozen model은
  causal memory attention, learnable persistent memory, static LMM을 사용한다.

## 3.3 Event marks and continuous quantity

### An Analysis of Transformations

- **Reference:** Box, G. E. P. and Cox, D. R. Journal of the Royal Statistical Society: Series B,
  26(2), 211-243, 1964.
- **Source:** [Oxford Academic](https://academic.oup.com/jrsssb/article/26/2/211/7028064)
- **Role:** skewed positive-valued targets에 power/log 계열 변환을 적용하는 통계적 배경이다.
- **Boundary:** TitanTPP의 magnitude-mark/residual factorization 자체를 제안한 문헌은 아니다.

### Decoupled Learning for Factorial Marked Temporal Point Processes

- **Reference:** Wu, W., Yan, J., Yang, X., and Zha, H. KDD, 2018.
- **Source:** [KDD](https://www.kdd.org/kdd2018/accepted-papers/view/decoupled-learning-for-factorial-marked-temporal-point-processes)
- **Role:** 하나의 discrete marker만 사용하지 않고 사건 속성을 여러 marker로 분해하는 흐름을
  설명한다.
- **Boundary:** continuous quantity regression을 직접 다루는 문헌은 아니다.

### Transformers for Mixed-type Event Sequences

- **Reference:** Draxler, F., Meng, Y., Nelson, K., Laskowski, L., Yang, Y., Karaletsos, T., and
  Mandt, S. NeurIPS, 2025.
- **Source:** [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html)
- **Role:** discrete event attribute와 continuous event attribute를 서로 다른 head로 모델링하는
  직접 근거가 될 수 있다.
- **Boundary:** TitanTPP의 log-magnitude/residual factorization 자체를 제안한 문헌은 아니다.
- **Current manuscript status:** 4-page draft에서는 reference risk와 분량 제약을 줄이기 위해 제외했다.

## 3.4 Intermittent-demand forecasting as event prediction

### Forecasting and Stock Control for Intermittent Demands

- **Reference:** Croston, J. D. Operational Research Quarterly, 23(3), 289-303, 1972.
- **Source:** [Publisher DOI](https://doi.org/10.1057/jors.1972.50)
- **Role:** 비영 수요의 크기와 발생 간격을 분리해 추정하는 고전적 출발점이다.

### Forecasting Intermittent and Sparse Time Series: A Unified Probabilistic Framework via Deep Renewal Processes

- **Reference:** Türkmen, A. C., Januschowski, T., Wang, Y., and Cemgil, A. T. PLOS ONE,
  16(11):e0259764, 2021.
- **Source:** [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)
- **Role:** 비영 수요의 도착 간격과 크기를 함께 모델링하고 renewal/point-process 관점과 연결한다.

## Ablation background: joint-objective optimization

### Gradient Surgery for Multi-Task Learning

- **Reference:** Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. NeurIPS, 2020.
- **Source:** [NeurIPS](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- **Role:** 여러 목적을 공동 학습할 때 gradient interference가 생길 수 있다는 일반 근거이다.
- **Boundary:** TitanTPP의 mark-quantity conflict는 Taxi V2/V3b ablation으로 직접 확인한다.

### Neural Marked Temporal Point Processes for Probabilistic Predictive Modeling of Continuous-Time Event Data

- **Reference:** Bosser. PhD dissertation, 2024.
- **Source:** [ORBi](https://orbi.umons.ac.be/handle/20.500.12907/50787)
- **Role:** marked TPP의 time/mark 공동 학습과 최적화 문제를 연결하는 보조 근거이다.
