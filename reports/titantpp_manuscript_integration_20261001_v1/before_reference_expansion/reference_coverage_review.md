# TitanTPP 참고문헌 범위 검토 — 2026-10-01

**판단:** 현재 9편은 방법·관련 연구·실험을 포함한 이 원고의 범위에 비해 얇다. 단순한 개수보다 설계의 선행 연구와 연구 위치를 설명하는 연결이 부족하다. 이 판단은 편집적 검토이며 학회의 최소 인용 수 규정이나 채택 확률을 뜻하지 않는다.

**현재 기준선 — 완료**

- 실제 원고 인용 목록 9편: 외부 비교 모델 6편(RMTPP/NHP/THP/SAHP/AttNHP/S2P2), Titans, Deep Renewal Processes, FlexTPP.
- TPP encoder 계보, 온라인 메모리, 수치형 mark/간헐 수요의 최근 직접 관련 연구는 포함되어 있다.
- 이번에는 아래 후보의 출처와 관련성을 확인했다. 본문 인용·references.json·BibTeX는 아직9편을 유지한다. 후보를 인용하는 것은 해당 모델의 새 비교 실험을 자동 승인하거나 요구하는 것이 아니다.

**설계의 차이를 선행 연구와 연결한다 — 다음 작업 / 우선**

| 후보 | 확인한 원문 내용 | TitanTPP에서 설명할 지점 | 연결 위치 |
|---|---|---|---|
| [Houlsby et al., 2019, Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html) | 원문2.1절은 down-project→nonlinearity→up-project의 병목 adapter와 내부 skip connection, near-identity 초기화를 설명한다. 사전학습 backbone을 고정하고 adapter를 학습한다. | 병목 잔차 자체를 새로운 원리로 제시하지 않는다. TitanTPP의 두 인접 contextual state 결합·이력 가용성·encoder 사이 삽입과 전체 모델 공동학습을 구체적으로 비교한다. | 관련 연구2.3 또는 compact history correction 단락 |
| [Bachlechner et al., 2021, ReZero is all you need: fast convergence at large depth](https://proceedings.mlr.press/v161/bachlechner21a.html) | 공식 UAI2021 논문은 잔차 연결의 scalar gate를 영 초기화하는 방식을 설명한다. | TitanTPP는 scalar learned gate 대신 출력 행렬 V_b를 영 초기화한다. 공통 초기 함수 보존과 학습 중 보완이라는 관계를 설명하고, ReZero의 수렴 결과를 TitanTPP의 실험 근거로 전용하지 않는다. | 초기화 설계의 관련 연구와방법3.3 |

**시간분포와 수요 예측의 연구 위치를 설명한다 — 다음 작업**

| 후보 | 확인한 원문 내용 | TitanTPP에서 설명할 지점 | 연결 위치 |
|---|---|---|---|
| [Shchur et al., 2020, Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127) | 저자 원고는 intensity 대신 사건 간 시간의 조건부분포를 직접 모델링한다. ICLR2020 발표는 [저자 연구실](https://www.cs.cit.tum.de/daml/intensity-free-tpp/)에서도 확인했다. | 수량 점 회귀와 기록 시간 확률질량의 조합이 어느 계보에 위치하는지 설명한다. 이 논문의 mixture/flow와 TitanTPP의 단일 조건부 lognormal 및 정수구간 관측 모델은 구분한다. | 관련 연구2.1, 시간 head3.5 |
| [Shchur et al., 2021, Neural Temporal Point Processes: A Review](https://www.ijcai.org/proceedings/2021/623) | IJCAI Survey Track, pp.4585–4593. Neural TPP의 설계 선택·원리·응용·미해결 과제를 정리한다. | TPP의 공통 문제 정의와 표현/출력 설계의 위치를 소개하는 용도로 쓴다. 최초성이나 성능 우위를 survey 인용으로 입증하지 않는다. | 관련 연구 도입 |
| [Croston, 1972, Forecasting and Stock Control for Intermittent Demands](https://link.springer.com/article/10.1057/jors.1972.50) | 출판사 초록은 수요 크기와 발생 빈도를 분리 추정하는 접근을 설명한다. | 간헐 수요를 사건 간격과 크기로 보는 동기에 연결한다. 기존 재고/기간별 수요 목적과 현재 next-event 목적이 동일하다고 쓰지 않는다. | 서론·관련 연구2.2 |

Houlsby는 원문 PDF2.1절, 나머지는 공식 proceedings/출판사 초록 또는 저자 원고 초록·발표 정보를 확인했다. 모든 후보를 전체 논문 정독·포괄적 최신성 조사까지 완료한 것으로 표시하지 않는다. OpenReview HTML 조회의 브라우저 검증 제한은 저자 공식 페이지와 arXiv 원고로 보완했다.

**데이터·구현·평가의 출처를 보완한다 — 이후 작업**

- Transformer, residual learning, 사용 optimizer 등 실제 의존 요소의 원문 인용이 필요한 문장에만 추가한다. 구현에서 사용하지 않은 기법을 인용 수 확보 목적으로 넣지 않는다.
- Taxi·Instacart·RAF의 원출처와 사용 데이터 버전/전처리 경로를 점검한다. Intermittent의 upstream provenance 미확인은 현재 한계로 유지한다.
- RMSE 선택과 MAE·시간 NLL 병기는 현재 평가 목적에서 설명하고, 문헌 인용을 임의의 선택 기준 정당화에 사용하지 않는다.
- 최신 TPP/수치형 mark/간헐 수요 연구의 직접 관련성을 추가로 확인해 누락을 점검한다. 이번5편을 exhaustive novelty review로 취급하지 않는다.

**완료 기준:** 관련 연구에 각 후보의 역할과 TitanTPP의 차이를 문장으로 연결하고, 실제 본문에서 쓰는 항목만 bibliography에 포함한다. 목표 숫자를 채우기 위한 무관한 인용은 추가하지 않는다. 문헌 보강은 기존5090 종료 확인·감사와 병행 가능하며 GPU 실행을 포함하지 않는다.
