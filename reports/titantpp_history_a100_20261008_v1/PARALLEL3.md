# 이력 보정27 — A100 동시3개 변경

사용자의 2026-10-08 “병렬로하자. 기존에3개씩병렬로돌렸으니까맞춰서진행해” 승인으로 단일 worker 운영을 동시3개로 변경했다.

## 완료한 변경

- 기존 Pod `qq2uk2dpfi2rum`과 A100-SXM4-80GB 한 장을 그대로 사용한다. 추가 Pod와 충전은 없다.
- 신규root `/workspace/titantpp_contribution_history_a100_parallel3_20261008_v1_a100`에 별도 운영 계약을 봉인했다. 부모root/실패/claim은 변경하지 않았다.
- 첫 serial 시도는 native qualification exit0 후 첫 worker의 `Native timeout parent required` 검사에서 종료됐다. 절대 deadline의 정수/실수 문자열 차이가 원인이며, run 디렉터리/학습 checkpoint 생성 전이었다. 실패 원본은 부모bundle `preserved_attempt1/`에 회수했다.
- 제한시간 검사는 `/proc`의 정확 argv·PID·PPID·PGID 및 숫자 deadline 비교로 수정했다. 세 worker는 각자의 lease·progress·claim·원본 경로를 사용한다.
- 한 batch는 같은 dataset·seed의 full/보정없음/현재상태전용 3개다. 세 개의 실제 종료·fullValidation endpoint·원본 SHA 및 동일 batch 순서를 확인한 뒤 다음 batch로 이동한다.
- 순서는 Taxi→RAF→Intermittent, 각 seed42/52/62. 총27 fresh fit, 중복 실행0, 자동 retry/resume0.
- 학습 전 단일 native qualification와 별도 3프로세스 합성 동시 실행 검사를 수행한다. B128/T256, warm5/measure20, 동시에 준비된 세 GPU PID, 업데이트 시간 겹침, 각 exit0와 GPU idle을 확인해야 학습허가를 만든다.
- 소스129·입력7·모델/초기화·B128/min40/max300/patience40·loss/optimizer·Validation 선택 규칙은 동일하다. 18개 집중 로컬 테스트가 통과했다.

## 비용과 관측

기존 Pod 생성 시각부터 총비용을 합산한다. GPU $1.59/h, 저장 포함 추정 $1.596944/h. 동시3개는 한 장을 공유하므로 시간당 비용을3배로 계산하지 않는다. $45 준비·학습 기한과 $5 회수·정리 예산, 총 $50 상한을 유지한다. 절대기한을 병렬화 시각부터 다시 계산하지 않는다.

독립 local manager가 비용 종료를 관리하고 provider stopAfter도 요청했다. 요청 자체를 종료 보장으로 보고하지 않는다. 별도 terminal service는10분 간격 관측·미검증 live backup을 수행한다. 실제 소유 CPU/GPU 프로세스 종료 후에만 immutable archive/manifest를 회수하고, SHA 검증과 Validation 집계를 마치면 전용 Pod를 정리한다. 회수 실패는 보존하며 자동 retry하지 않는다. 미검증 live backup은 최종 원본으로 취급하지 않는다.

fit별 ETA는 해당 fit의 최근 최대10개 완료 epoch 중앙값만 사용한다. 동시에 실행한 wall time은 공유 GPU 처리량이며 독립 실행 구조 효율로 해석하지 않는다. 전체27 완료 시간과 $50 내 완료 여부는 실측 전 미확정이다.

## 현재 봉인

- canonical: `4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a`
- source129 closure: `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`
- operation closure: `bec6a925503b43e083dcc02bc03b2fa25dc43b98be0803d9e135f97990a97a33`
- control integrity rawSHA: `6d3274569d9eb542825dc683034f16201140e683deb36eca373959b9815307f7`
- GPU UUID: `GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de`

## 주제별 상태와 남은 순서

1. **Instacart — 진행 중:** 기존 RTX5080/5090 실행과 승인된 관측을 유지한다.
2. **CNN+GRU54 이력 보정27 — 진행 중:** 동시3개 운영 배포와 native 검증 후 실제 학습 PID·checkpoint를 확인한다. 최신 상태는 신규bundle `hourly_monitor/summary.json`에 기록한다.
3. **고정 검색27 — 대기:** 보정 결과의 full9가 A100 기준선이므로 후속 검색도 공통 GPU·Runtime 조건을 맞춘 운영 계약이 필요하다. 추가 A100 과금을 자동으로 승인한 것으로 해석하지 않는다.

모든 결과는 Validation 개발 증거다. Test·CPU binary 재추론 감사·새 독립 평가는 별도 미완료다.
