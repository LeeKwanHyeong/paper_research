# 공유 이력 보정과 단일 MLP 비교

## 현재 기준선을 보존한다 — 완료
- 대상 paper_research / codex/shared-history-mlp-control-20261022. 기존 A–D36조건·외부5seed·MAC 결과/실패/checkpoint를 보존한다. 기존 frozen987소스는 수정하지 않고 새 모델·실행 경로만 추가한다.

## 같은 정보와 총 파라미터를 가진 대조를 고정한다 — 완료
- B의 현재·직전 관측 및8개 가용 이력 조건을 단일MLP에도 제공한다. 136→48→64, biasFalse·exactGELU·출력0초기화·잔차/8로9600parameters를 맞춘다. active-prefix와함수클래스차이는 명시한다.
- B9/D9는 원래payload/선택/실제exit/Validation원문을 검증해 재사용하고 MLP9만 새학습한다. Test를 이용한 구조선택과 새Test평가는0이다.

## Validation 진단과 구현을 검증한다 — 진행 중
- 기존Validation저장예측으로시간NLL꼬리와seed별교환관계를분석한다. 소스·학습손실·선택조건을임의로바꾸지않는다.
- 모델/runner/profiler와manager/controller를독립파일로병렬구현하고공통계약뒤단위/계약/독립QA와실제양hostsynthetic/capacitygate를직렬통과한다.

## 같은 GPU에서 학습·Validation을 진행한다 — 다음 작업
- seed42는5080,52/62는5090. GPU당과학worker1개. 새9fit과기존18reuse,선택본Validation27재현을모두검증한다. 기존Runtime/네트워크/인증·정지작업은보존한다.

## Train 공통 입력의 속도·메모리를 비교한다 — 다음 작업
- 전체Validation27gate뒤동일Train37batch(예열5·측정32)에서27조건×학습/평가×3반복=162freshprocess측정. 원본checkpoint는저장하지않고전후SHA동일성검증. GPU/seed별절대값과짝비교를보존한다.

## 회수·독립 비교 검토를 마친다 — 다음 작업
- 단계별실제exit0/신원부재/GPUguard뒤최초회수1회. archive/manifest/member/source/input/legacy/선택SHA검증,Validation-only순수집계/불리한모든seed/비용보고를작성한다. 최종대표모델이나논문기여를자동확정하지않는다. MAC은계속보류한다.

## 학습 전 검증 결과를 고정한다 — 진행 중 (2026-10-22 23:52 KST)
- 기존 A/B/D의 Validation 예측 27개·911,187행에서 원 receipt의 수량 RMSE·MAE·시간 NLL을 재현했다. Taxi에서는 B가 D보다 수량 오차를 줄이지만 세 seed 모두 시간 NLL이 악화된다. 시간 점수 정의의 오류를 입증하지 못했으며, 저장 예측에 분포 파라미터가 없어 원인은 미확인이다.
- 기존 B/D checkpoint 18개의 선택 epoch·binary/tensor SHA·학습과 Validation의 실제 종료 및 원 예측 재현을 로컬에서 확인했다. 이를 새 모델로 이름만 바꾸지 않고 원 payload 그대로 재사용한다.
- 모델·실행·제어·비교·최초 배포 경로의 통합 로컬 테스트와 전체 Ruff가 실제 종료 코드 0을 반환했다. 합성 NaN 손실 누락과 native dispatcher 종료 증거 누락은 학습 전에 수정했고, 실패 원문도 보존했다. 정확한 테스트 수와 원문 SHA는 local_checks_20261022T145220371808/receipt.json에 연결한다.
- 양 서버의 최초 읽기 전용 preflight는 성공했다. 당시 5080의 기존 정지 작업 4개와 5090의 유휴 상태를 확인했다. 이 과거 관측을 현재 상태로 간주하지 않으며, 새 학습은 아직 시작하지 않았다.
- 다음 순서는 독립 QA 봉인 → 현재 브랜치의 신규 소스 커밋 → 계약·archive 봉인 → 양 서버 합성/용량 검증 → MLP 9개 학습과 전체 27개 Validation이다. Test와 MAC은 계속 보류한다.
