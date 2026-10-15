# 고정 checkpoint 평가 준비 — 구현·합성검사 완료, 독립검토 대기

## 현재 기준선 — 완료
- 새 학습 모델·runner는 기존 봉인 바이트를 유지한다. 과학 source147은 frozen source145와 새 모델/runner2로 구성한다. 추가 평가·집계 도구는 학습 source와 별개다.
- Current와 Temporal 각각 RAF seed42/52/62, 총6개가 대상이다. 원형 Q/K/V 지역 CNN576과 FFN의 W1/W2를 유지하며 두 arm 모두 FFN 계수1152를 추가한다. RAF max_seq_len84에서 등록105311·활성80579를 직접 검사한다.
- 기존 CNN3 및 외부 모델은 검증된 저장 결과만 재사용한다. 신규 기준선 학습·추론·평가는 없다.

## 종료 원본과 선택을 결속한다 — 다음 작업
- 학습 helper의 실제 종료·회수 complete 및 전체 SHA/원본 archive/선택·종료증거가 확인된 후만 prepare를 수행한다. 현재 이 준비 문서는 학습 종료나 평가 실행을 증명하지 않는다.
- selected epoch는 전체 Validation raw 수량 RMSE의 최초 strict finite 최소값이다. binary·tensor·원본 metadata·source·입력·초기상태·선택 이력을 결속하고 종료 checkpoint는 그대로 둔다.
- 준비 진입점은 `paper.scripts.prepare_titantpp_post_attention_ffn_raf6_test --training-integrity-sha <실제 학습 봉인> --registry-sha <고정 외부 등록부 SHA>`다. 실제 반환 bootstrap SHA를 이어지는 `paper.scripts.advance_titantpp_post_attention_ffn_raf6_test --mode advance --integrity-sha <실제 반환 SHA>`에 사용한다. 명령 예시만으로 실행하지 않았으며, 이번 작업에서는 실제 prepare/advance/SSH를 실행하지 않았다.

## 동일 모델의 Validation을 재현한 뒤 Test로 넘어간다 — 학습 종료 대기
- 같은5090 Runtime에서6개 native Validation 전체의 동일 대상과 여섯 지표 및 실제 worker exit0를 검증한 gate가 있어야 Test6을 시작한다. 평가 pipeline은6개 순차이고 학습 최대3병렬과 구분한다.
- 학습/평가의 미완료 attempt·claim·failure를 반복하지 않는다. 조회 실패와 과학 실패, pipeline child exit와 wrapper 자신의 OS exit를 구분한다. 원본은 exclusive 경로에 보존하며 완료 cache도 누락·빈 inventory를 거부하고 archive/종료/결과를 다시 검산한다.

## 성능 통과와 시간 내용의 효과를 따로 판단한다 — 구현 완료
- 전체/큰수량 RMSE·MAE 및 전체/큰수량 recorded positive-integer TimeNLL의3seed 평균·표본SD(ddof1), 모든 seed의 Current−CNN·Temporal−CNN·Temporal−Current 절대/상대 차이와 악화를 보존한다. 단순 수량 기준선 TimeNLL은 N/A다.
- 각 후보는 Validation/Test의 RMSE·MAE3seed 평균 모두 기존 CNN과 외부 지표별 최우수보다 엄격하게 낮아야 통과한다. 통과후보의 확장 선택은 Validation RMSE→MAE→정확 arm 문자순이다. Test는 통과자격에 사용되지만 checkpoint 선택에는 사용되지 않는다.
- Temporal−Current의 두 지표·두 split 개선 여부를 별도로 기록한다. Current만 통과해도 명시적 과거 내용의 효과라고 부르지 않는다. Current도 attention이 반영한 이력 정보를 포함하며, 동일 용량이 동일 함수공간·최적화를 뜻하지 않는다. downstream 메모리 영향 및 공통 가중치 공동학습이 포함된다.
- 기존 CNN/외부5080 cuDNN91900과 새5090 cuDNN92000 차이를 명시한다. 반복 Test 참고 탐색이며 독립 일반화·통계 유의성·원인 확정을 주장하지 않는다.

## 합성검사와 실제 실행의 경계를 확인한다 — 완료
- 53개 합성검사를 frozen147 overlay에서 통과했다. 여섯 지표 집계와 paired delta50회, 외부 네 기준·선택 순위·signed 차이 통계50회를 독립 수계산으로 대조했다.
- 테스트 실행 중 실제 parquet 읽기·torch checkpoint load·socket connect·실제 child process를 차단했다. pipeline exit 출력은 mock worker 합성 증적이며 native 실행을 뜻하지 않는다.
- 첫50개 검사에서 새 signed 차이를 비음수 오류 통계에 넣은 결함과 테스트 guard의 matplotlib 폰트 초기화 차단을 발견했다. 원문을 보존하고 signed 차이 전용 통계와 guard 적용 순서를 수정했다. 두 번째51개 및 최종53개는 actual exit0다.
- 로컬 Python3.12.10/torch2.14.0 CPU 결과다. 실제5090 Python3.12.13/torch2.11.0+cu130/cuDNN92000의 native Validation/Test는 미실행이다. 부모 source145는 검사 전후 바이트가 같으며, 작업트리의 기존 두 파일 드리프트를 복원하지 않고 동결 source를 사용했다.

## 남은 작업 — 다음 작업
1. 별도 독립 구현 검토와 합성검산 결과를 결속한다.
2. 부모가 평가 코드만 별도 커밋하고 준비 증적을 봉인한다. 이 문서 작성 시 EVALUATION_READY는 만들지 않았다.
3. 실제 학습6 종료·원본 회수 검증 후 prepare→native Validation6 gate→Test6→원본/SHA/집계 검증 순서로 진행한다.
