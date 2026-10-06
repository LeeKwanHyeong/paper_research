# CNN+GRU54 검색부 5방식 Taxi seed42 Validation 대조

**검색부의 추가 효과를 구분한다 — 구현 완료 / native 검증 다음 작업**
- 공통 CNN+GRU54, learned static prototype64개와 top4, 동일 원본 Train/Validation을 사용한다. 관측된 과거32사건의 시간·수량 요약으로 검색 query만 바꾼다.
- 기존 평균 검색, cosine softmax 가중합, 시간·수량 query 고정0.5 혼합, 이력에 따른 가변 혼합, 같은 문맥·추가521개 파라미터의 일반 query/temperature 검색을 비교한다.
- 추가 파라미터는 0/0/512/521/521개, 총115167/115167/115679/115688/115688개다. 고정/가변 비교에는 gate9개 차이가 포함되며, 일반 검색은 query/temperature 패키지 대조다.
- prototype는 실제 과거 사건을 저장한 표가 아니다. 원본 평균 검색과 공통 텐서·난수 흐름을 유지하고, 가중합 네 방식은 같은 초기 query/출력에서 시작한다.

**같은 GPU에서 다섯 방식을 비교한다 — 5090 먼저 / 5080 기존54 원본 회수 대기**
- 5090에서 다섯 fresh fit을 먼저 진행한다. 5080도 다섯 방식을 같은 순서로 진행하며, 기존54조건 실제 종료·원본 SHA 회수·최종 Validation 보고가 검증되기 전에는 새 GPU 검증/학습을 시작하지 않는다.
- 과학적 구성5개를 각 GPU에서 반복하는 실행10개다. 두 seed나 독립 평가로 합산하지 않는다. 5090을 탐색의 주 비교, 5080을 같은 seed의 GPU 반복으로 보고한다.
- 같은 GPU/seed/데이터 블록 내 비교이며 이종 GPU 학습 시간을 구조 효율로 혼합하지 않는다. 기존 MLP8 54조건 결과는 별도 보존한다.

**학습과 후보 판정 기준을 고정한다 — 계약 확정**
- Taxi seed42, batch128, min40/max300/patience40, 최초 strict finite 최소 fullValidation raw수량RMSE epoch를 선택한다. 공통 손실·optimizer·시간 분포·단위를 유지한다. 큰 수량 경계는 Train의 >3449, Validation8268개 중 큰 수량79개다.
- 선택한 같은 epoch의 전체·큰 수량 RMSE/MAE/TimeNLL을 모두 보고한다. 수량이 주목표이고 시간 손해는 필수 보고한다. 새 시간성능 기준을 사후 선택에 쓰지 않는다.
- 가변 방식이 평균·고정·일반 검색보다 전체와 큰 수량 RMSE 모두 엄격하게 낮고 비용 기준도 만족해야 5090 pilot 성공으로 기록한다. 유사도 가중합 대비 변화도 함께 보고한다.
- native GPU별 B128/T64 및 T256, warm5/measure20으로 추론 중앙값 +20%, 학습 step 중앙값 +30%, peak allocation +20% 이하를 미리 고정한다. 비용 기준 초과는 후보 탈락 근거이며 기술 검증과 구분하고 불리한 fit을 누락하지 않는다.
- 서버별 worker1, 조건7200초/호스트전체28800초, qualification1800초, CPU16GiB/GPU80%/root8GiB. 호스트별 시작 허가에서 8시간 기한을 고정하며 마감을 ETA로 쓰지 않는다. 자동 retry/resume 없다.

**검증과 원본을 보존한다 — 종료 후 다음 작업**
- 로컬 CPU 합성 모델13·운영20·배포5 검사를 통과했다. 실제 RTX qualification과 학습은 별도 영수증으로 확인한다. 실제 Train optimizer update 없이 synthetic native 동작·인과성·gradient·프로파일과 원본 Train/Validation population/time identity를 검증한 뒤 허가한다.
- 관측은 live host당 읽기전용 SSH1회, verified terminal cache는 원 관측시각을 보존해 SSH0이다. 조회 실패와 학습 실패를 구분한다. 실제 저장/best epoch 및 fit 자체 최근10완료 epoch 중앙값으로 조건부 ETA만 보고한다.
- 양호스트10terminal·source/operation/원본 SHA·소유PID 부재·supervisor exit0 확인 후 selected/last binary와 작은 원본/이력을 manifest/archiveSHA로 한번 회수한다. lock/claim/failure를 보존하며 자동 retry하지 않는다.
- Test/held-out 성과·예측 행·혼합성과를 열지 않는다. Validation 개발 대조이며 독립 Test/독립 평가/통계적 유의성/외부 우월성 증명이 아니다. 확대 실험은 pilot 후 별도 판단한다.

실행 계약과 증적: search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/current.json 및 execution_contract.json. 현재 상태는 SCHEDULER.md와 hourly 보고에서 확인한다.
