# 시간·수량 손실의 균형 확인

## 외부 비교 실험을 보존한 채 멈춘다 — 완료

- 대상: Cap 외부27 유한 실행 큐, 5080과 5090. 사용자 최신 중단 요청을 적용했다.
- 2026-10-18 10:09:33/34 KST에 서버별 dispatcher와 소유 학습 worker3개를 SIGSTOP했다. 새 학습 admission을 먼저 막고 실제 PID·시작시각·명령·작업 경로를 확인했다.
- 완료11개(5080=2, 5090=9), 정지6개, 대기10개다. 이는 학습 단계 수이며 Validation/Test 완료 수가 아니다.
- 원본 source·checkpoint·로그를 수정하지 않았다. optimizer/RNG 등 메모리 상태와 CUDA context/VRAM도 유지된다. 프로세스 종료나 새 checkpoint 저장과는 다르다.
- 자동 재개하지 않는다. 기존 wall-clock 제한은 정지 중에도 흐르며 임의 연장하지 않는다. 재개 전에 실제 신원·남은 제한시간을 확인해야 한다.
- 증적: PAUSE_RECEIPT_5080/5090.json과 transport 원문. 최초 로컬 import 오류는 원격 호출 전 발생했으며 별도 보존했다.

## 같은 모델 상태에서 세 수량 손실의 영향을 비교한다 — 진행 중

- 대상은 Cap 기준 MLP16, Taxi·RAF·Intermittent × seed42·52·62다. 기존 CNN은 수정·학습·추론하지 않는다.
- 각 seed의 fresh 초기 상태와 기존 Validation으로 이미 선택한 MLP16 checkpoint를 별도 조건으로 사용한다. checkpoint를 재선택하거나 덮어쓰지 않는다.
- 초기 수량 head의 weight=0 때문에 초기 수량 손실의 backbone gradient는 0일 수 있다. 이를 시간 손실의 학습 전반 지배 증거로 해석하지 않는다.
- 순수 Train의 seed별 고정 shuffled 첫3배치(batch128, 원래 이력 범위)에서 같은 forward를 공유한다. Validation/Test 추론, optimizer update, 새 fit은 없다.
- 시간 손실은 기존 recorded positive-integer duration NLL, 수량 손실은 기존 log-MSE / ((p−q)/C)^2 / |p−q|/C다. C는 Train Q3 상한 Taxi65·RAF10·Intermittent20이며 λ=1을 고정한다.
- 공통 파라미터의 gradient norm 비율·방향 cosine, 시간/수량 전용 파라미터 norm, 전체 결합 gradient norm과 clip1 조건을 구분한다. 그래프 참여와 실제 nonzero 기울기도 구분한다.
- 공통 norm을 맞추는 λ는 해당 모델·배치의 기술통계로만 기록한다. 최적 λ나 학습 전체에서 통용되는 상수로 채택하지 않는다.
- parameter/buffer/context 불변과 RNG 복원, checkpoint/input/source SHA를 확인한다. 합성검사와 독립검토 뒤 별도 경로에서 실행한다.

## 실제 진단과 결과 기록을 마친다 — 다음 작업

- 정지된 CUDA context를 해제하지 않으므로 기존 서버 Python/torch2.11 환경의 CPU에서 제한된 진단을 실행한다. GPU 학습의 수치·속도·최적화 궤적과 같다고 주장하지 않는다.
- 데이터셋·seed·배치별 결과와 예외를 보존하고, 손실 scalar 크기와 gradient 영향력을 혼동하지 않는다. 초기/선택 상태의 국소 민감도이며 AdamW 실제 update나 성능 개선을 증명하지 않는다.
- 진단 후 해석과 한계를 로컬 보고서 및 기존 Notion 연구 이력에 기록한다. 이 결과만으로 외부27을 재개하거나 신규 손실18개 학습을 시작하지 않는다.
