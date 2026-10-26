# 전체 과거조건 지원 범위와 PHEN 관측 기록 확보

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-26
- Verification Status: 진행 중

## 원본과 실행 범위를 고정한다 — 완료
- Intermittent Train·Validation만 사용한다. seed42 best1/last41, seed52 best5/last45의 기존 checkpoint와 frozen source를 보존한다.
- 사용자는 원자료 지원 범위 계산과 5080/5090의 기존 작업을 보존한 전체 추론을 승인했다. 재학습·head 학습·Test 열람·기존 Runtime 변경·원고 변경은 범위 밖이다.
- source/checkpoint/target identity는 일치해야 한다. 실제 checkpoint 실행 전에 PRECISION_DECISION.json에서 장치 간 q 비교는 atol/rtol=1e-5/1e-5, NLL은 1e-4/1e-4(기존 전체 GPU probe 기준), log-MSE·cosine은 1e-7/1e-5로 구분해 고정했다. native float64 시간 산술·원본 신원 검증과 장치 간 비교를 구분한다. 실패 뒤 허용차를 늘리지 않는다.

## 전체 원자료의 비교 가능한 과거 조건을 계산한다 — 완료
- 지원 범위의 exact·history-binned·coarse 기준을 결과 전에 support/support_spec.json에 고정한다.
- 전체와 기존2048 표본의 사건 수·시계열 수·공통조건 비율을 비교한다. 원자료만으로 검색 조합의 다중성이나 추가 정보를 보장하지 않는다.
- 독립 QA PASS. exact 조건의 비교 가능 최소 기준을 충족한 집단은 532개, Validation 29,720건(34.44%)이다. Train에 exact 조건 자체가 없는 Validation은 45,792건(53.07%)이다.

## 누락된 전체 관측을 기존 checkpoint에서 확보한다 — 진행 중
- 기존 best 전체 Validation을 신원·정밀도·필드별 검증 후 재사용한다. 이미 저장된 float32 μσ와 원래 float64 NLL의 정밀도를 구분한다.
- 새 관측은 top4 IDs·유사도, 수량 예측·손실, float64 μσ·recorded-time NLL, target identity를 compact chunk로 저장한다.
- 필요한 best Train의 실제 final head input64만 float32로 보관한다. 미래 head 학습은 실행하지 않는다.
- host의 GPU 작업·보호된 정지 작업을 확인한 뒤 새 전용 artifact 경로에서만 실행한다. 기존경로 덮어쓰기·작업kill/resume·재배포는 하지 않는다.
- 기존 best Validation 172,570건은 원본29필드 그대로 compact 변환했고 독립 QA PASS다. 원래 없는 native log-MSE와 hidden은 미확보로 표시했다.
- CPU qualification v1은 모델 적재·forward 전에 input sidecar 경로와 외부 helper import 문제로 actual exit1이었다. 원문·partial·계약을 보존했다. split manifest는 원본 SHA 그대로 staging에 추가했으며, collector_v2에서 import 격리만 수정해 별도 새 qualification을 수행한다. 과학 설정과 허용차는 변경하지 않는다.

## 실제 종료·원본·수치·누락 여부를 독립 검증한다 — 다음 작업
- 짧은 테스트/qualification 뒤 전체 추론, 실제 child 종료와 state 불변, chunk row count/SHA/target coverage를 검증한다.
- 재사용과 신규 자료의 범위를 분리하고, 후속 Prototype·시간 분석이 가능한 최종 inventory를 남긴다.

## 병렬 책임
- support worker: 원자료 support/ 산출물. collector worker: collector.py와 관련 합성검사. 본진: 계약·서버조회·실행·회수. QA: 독립검산. 공통계약과 실행은 직렬이다.
