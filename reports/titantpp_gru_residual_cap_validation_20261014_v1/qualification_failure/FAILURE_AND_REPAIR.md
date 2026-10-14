# 후보9 실행 상태와 검사 실패 수정안

## 실제 실행 결과 — 차단됨

승인된 9개 후보는 아직 **학습 0/9개**다. 5080·5090 새 격리 경로에 배포한 뒤 실제 검사 단계에서 모두 exit1로 중단됐다. 모델의 수치 학습 실패가 아니라, 새 운영 계약에서 기존 연결부가 요구하는 메타데이터를 빠뜨린 구현 오류다. 원본 CNN·FULL 결과와 기존 학습·claim은 변경하지 않았다.

| 서버 | 예정 후보 | 검사 종료(KST) | 실제 검사 exit | 후보 학습 | 마지막 관측 GPU compute PID |
|---|---:|---|---:|---:|---|
| 5080 | 7 | 2026-10-14 16:46:18 | 1 | 0 | 없음(16:46:58 관측) |
| 5090 | 2 | 2026-10-14 16:46:17 | 1 | 0 | 없음(16:46:57 관측) |

위 시각은 저장된 원관측이며 현재 GPU의 전역 유휴 상태를 뜻하지 않는다. qualification outer process와 timeout 하위 프로세스의 실제 exit1을 보존했다. pipeline wrapper 자체 OS 종료는 직접 관측하지 않았다. GPU UUID·Runtime·입력 SHA 사전 관측은 통과했지만, 실제 cap GPU 합성 검사는 이번 누락으로 완료되지 않았다. 검사 통과나 학습 시작으로 세지 않는다.

## 원인과 로컬 수정 검증 — 완료

고정 과학 준비 계약은 cap 모델 정보를 갖고 있으나, base.time_interface가 임시로 읽는 부모의 `design_sha256`와 `architecture`가 운영 계약에 없었다. 먼저 `KeyError: 'design_sha256'`가 발생했고, 그 값만 보완하면 `architecture`에서도 같은 오류가 발생한다. 두 값은 부모 계약 그대로 보완해야 한다. cap adapter는 이 임시 backbone 정보를 cap 전용 정보로 교체하므로 손실·head·rho·예산·source·모델 경로를 바꾸는 수정이 아니다.

- `local_repair_v2/metadata_only_contract.proposed.json`: 기존 모든 계약 값을 보존하고 위 두 항목만 추가한 진단용 수정안이다. 실행 허가나 배포용 계약이 아니다.
- `local_repair_v2/receipt.json`: 세 데이터셋의 실제 허용된 Train·Validation 입력에서 누락 오류 두 가지를 재현한 뒤 수정된 time_interface와 training_args가 모두 통과했다. Train 시간 통계·Validation 대상 지문 검증, 원래 입력 SHA 검증을 거쳤다. 모델 생성·학습·Test·원격 호출은 없었다.
- 최초 로컬 검사인 `local_repair_v1`은 native와 다른 작업 경로 때문에 상대 입력 경로를 찾지 못했다. 증적은 보존하고, native와 동일한 입력 작업 경로를 사용한 별도 v2 검사로 검증했다.
- 회수·집계·수정 경계 테스트21개와 공개 replay 통합 테스트1개, 총22개가 통과했다. 공개 replay는 합성 checkpoint 9경로 및 잘못된 arm·seed·epoch·initial SHA·source 45개 거부를 포함한다. native GPU 검사를 대신하지 않는다.
- 집계 도구는 여섯 지표의 평균·표본SD, seed별 후보−FULL·후보−CNN 차이, 악화·동률을 JSON과 Markdown 모두에 남긴다. 현재 실제 후보 결과는 없으며 회수·집계 실행도 없었다.

## 이 실험을 하는 이유 — 유지

기존 CNN+GRU의 Validation 우세가 Test에서 이어지지 않은 뒤, **GRU 보정 출력의 과도한 크기가 불안정성에 기여하는지**를 탐색하기 위한 실험이다. rho=0.25 하나와 기존 손실·head·학습 예산·3seed를 고정한다. 기존 FULL과의 비교 및 CNN과의 비교를 구분하고, 불리한 결과도 보존한다. FULL을 반드시 우승시키기 위한 탐색으로 바꾸지 않는다. 이번 Validation 비교만으로 Test 일반화 개선이나 순위 반전의 인과 원인을 입증하지 않는다. 상세 목적은 `../PURPOSE_AND_DECISIONS.md`에 있다.

## 다음 실행 순서

**새 운영 경로의 수동 검사 재실행 — 승인 필요**
- `MANUAL_REQUALIFICATION_PROPOSAL.json`에 두 메타데이터 보완, 서버별 새 격리 root, 기존 host 절대기한 유지를 명시했다. 실패한 v1 경로·claim·failure·봉인은 보존한다.
- 고정 계약의 `limits.automatic_retry=false` 및 실행 승인 기록의 `automatic_retry=false` 때문에 이번 실패 뒤 자동으로 검사나 학습을 다시 시작하지 않았다. 새 운영 launcher revision과 별도 수동 승인 결속이 필요하다. 이 진단용 수정안을 그대로 배포하지 않는다.

**동일한 후보9 학습 — 다음 작업**
- 수동 검사가 승인되고 두 서버의 실제 GPU·Runtime·입력·합성 검증이 각각 exit0으로 끝나면 기존에 승인된 같은9개만 실행한다. 5080 7개와 5090 2개는 독립적으로 진행할 수 있다. 새 seed·rho 후보·Test는 추가하지 않는다.

**원본 회수와 3seed Validation 비교 — 다음 작업**
- 실제9 worker 종료, 선택 checkpoint·history·endpoint replay·manifest SHA를 확인한 뒤 기존 CNN/FULL18개와 같은 Validation 대상의 여섯 지표를 비교한다.
