# 관측 μ·σ CNN 준비 검증 결과

**계약·구현·로컬 검증 — 완료**
- 대상: paper_research 저장소의 별도 moment-conditioning 준비 bundle. 기존 source134와 신규 모델·준비 adapter를 묶은 source136의 SHA를 확인했다.
- 기존 CNN, 블록 내부 RevIN CNN, 관측 μ·σ 조건 CNN, 같은 통계·파라미터 수의 용량 대조군을 구분했다. 조건 CNN과 용량 대조군은 각각 195개 파라미터를 추가한다.
- 기존 입력·손실·출력 head·선택 기준을 유지했다. 기존 모델 source, 연구 결과, 실패 기록, 원고, 진행 중인 자동화는 수정하지 않았다.

**합성 CPU 검증 — 완료**
- 동결 source만 불러오는 별도 실행에서 30개 검사 통과, 실패·오류·생략 0개. 총 실행시간 10.53초다.
- 통계 수계산, 미래·target·padding 차단, batch 불변, 공통 초기 tensor·RNG·head 동일성, 비영 커널에서의 attention 연산 동일성, 조절 범위, μ와 σ 각각의 영향, 신규 파라미터 gradient, checkpoint 구조 혼동 거부, 무결성 실패 시 출력 차단과 덮어쓰기 거부를 검증했다.
- 로컬 환경: Python 3.12.10, torch 2.14.0, CPU. 실제 학습 서버의 GPU·Runtime 검증을 대신하지 않는다.
- 실제 데이터 행과 선택 checkpoint binary를 읽지 않았고 optimizer step, 새 학습, Test, 원격 호출은 수행하지 않았다.

**해석해야 할 한계 — 확인 완료**
- 이번 대조군은 affine=False인 인과적 블록 내부 RevIN이다. 표준 모델 입출력 RevIN을 그대로 적용한 실험이 아니다.
- 현재 선형 CNN에서 같은 query 통계로 정규화·복원하면 잠재 특징의 표준편차가 상쇄된다. 따라서 일반 블록 RevIN은 평균 중심화 필터와 같고, 이를 표준편차 정규화의 독립 효과라고 해석할 수 없다.
- 제안 CNN은 별도로 관측 log1p 수량의 μ·σ를 사용해 현재와 과거 두 관측의 필터 비중을 조절한다. 수량 σ의 영향은 합성 검증으로 확인했지만 학습 후 성능 이득은 미확인이다.
- 용량 대조군은 같은 통계와 195개 추가 파라미터를 사용한다. 함수 형태와 최적화가 완전히 같다는 뜻은 아니다.

**학습 범위 결정 — 승인 필요**
- 현재 준비 계약의 데이터·기준선 식별자는 Taxi에 결속돼 있다. Taxi 첫 비교를 선택하면 신규 세 구조 × seed42·52·62 = 9 fit이고 기존 CNN3은 재사용한다.
- RAF·Intermittent 실행 계약, 실제 입력·binary 재검증, 원격 실행 구성은 아직 준비 완료로 표시하지 않는다.

**승인 범위의 native 검증과 학습 — 다음 작업**
- 승인 이후 같은 host의 입력·checkpoint·source SHA와 GPU·Runtime을 확인하고 native 합성 검증을 수행한다.
- 통과한 고정 조건으로 서버당 최대3 병렬 학습을 구성한다. 이번 준비 단계에서는 학습 경로를 활성화하지 않았다.
- 종료·원본 증적을 검증한 뒤 Validation RMSE·MAE 및 여섯 지표의 3seed 평균·표본SD와 seed별 차이를 보고한다. Test와 다른 데이터셋 확장은 별도 결정 사항이다.

기계 검증 증적: `VERIFICATION.json`, `synthetic_frozen_v1.xml`. 과학 조건과 비교식: `EXPERIMENT_CONTRACT.md`, `ANALYSIS_PLAN.md`.
