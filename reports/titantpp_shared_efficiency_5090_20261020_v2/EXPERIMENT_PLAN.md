# 최종 공유형과 Titans-MAC의 5090 동일 입력 비용 측정

## 현재 비교 조건을 고정한다 — 완료

- 대상은 최종 TitanTPP Shared15와 기존 Titans-MAC event adapter, 원본 RAF·Taxi·Intermittent다. 내부 MLP16은 이번 비교에 포함하지 않는다.
- 원본 raw9의 source983개를 그대로 보존하고 기존 MAC 구성 helper2개만 더한 source985개를 사용한다. 공통 수량·시간 head, 입력·통계·손실·optimizer를 유지한다.
- 모델 구조의 비용을 측정한다. 매번 seed42로 새 모델을 만들고 scratch 학습 step을 수행하며, 과학적 학습 checkpoint는 읽거나 저장하지 않는다. Validation/Test 정확도는 계산하지 않는다.

## 5090에서 동일 batch로 직렬 측정한다 — 진행 중

- 기존 5090 Runtime과 GPU UUID를 유지한다. 배포 직전과 각 worker 실행 전 실제 CUDA 유휴를 확인한다. 5080 추가2seed 학습은 변경하지 않는다.
- 세 데이터의 Train만 먼저 필터링한 뒤 데이터별 seed42 순열에서4,736개 target을 고정한다. 같은37개 CPU batch를 두 모델·학습/평가·모든 반복이 재사용한다.
- batch128, RAF 길이84, Taxi/Intermittent 길이256. 예열5batch 뒤32batch를 측정한다. 모델·head 초기값과 각 batch tensor SHA를 결속한다.
- 두 모델×세 데이터×train/eval×3회=36개 측정 worker를 하나씩 실행한다. 반복2의 모델 순서는 반대로 한다. 추가 CPU/CUDA 합성 검증4개와 prepare1개를 포함한 직접 자식은41개다.
- 전체1,800초·자식600초 상한이다. 실패·기존claim을 보존하며 자동재시도하지 않는다. 새 비용 측정 child의 process group만 정리한다.

## 종료 원본을 회수하고 비용을 집계한다 — 다음 작업

- 전체36개 결과와41개 실제 자식exit0, 외부 supervisor가 관측한 controller exit0·PID 부재, 종료 GPU compute PID 부재와 모든 manifest SHA를 확인한다.
- 데이터별 학습step·평가batch의 반복별 중앙값,3반복 평균±표본SD, MAC/Shared15 가속비, 처리량, peak allocated/reserved와 warm baseline, 파라미터 수를 보고한다. 평가 memory 악화도 제외하지 않는다.
- GPU 동기화 경계를 포함하고 전송 시간은 별도로 기록한다. 평가 경로는 head/loss를 포함하므로 순수 inference로 부르지 않는다. epoch·수렴비용이나5seed 정확도와 섞지 않는다.

## 사용자 Notion 페이지에 증거를 연결한다 — 다음 작업

- 사용자가 지정한 ‘논문 중심’ 페이지에 이번5090 새 측정의 시작·완료·조건·수치·한계를 추가한다. 기존5080 결과와 새5090 결과를 구분한다.
- 원고 본문은 수정하지 않는다. 기존3seed 및 진행 중 추가2seed와 checkpoint·실패 기록·Runtime도 보존한다.

## 배포 포장 수정과 원본 보존 — 완료

v1은 첫 CPU 합성 검증에서 source/sample_data 빈 경로가 없어 비용 측정 전 종료됐다. controller 실제exit1, 측정worker0, 종료GPU부재와원본13개SHA를 별도 FAILURE_RECEIPT로 보존했다. v2는 같은985과학소스와 benchmark를 사용하고 빈프로젝트표시디렉터리만 포함한다. 기존v1실행이나claim을 재사용하지 않는다. 포장회귀검사와독립검토를통과한 뒤 새경로에배포했다.
