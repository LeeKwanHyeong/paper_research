# 고정 GRU 보정량 제한 후보9 실행 검증

## 목적과 승인 — 완료

GRU 보정 출력의 크기 제한이 기존 FULL 대비 Validation 성능의 불안정성을 줄일 수 있는지 탐색한다. 기존 Test 순위 반전 뒤에 제안된 후속이며, 이번 Validation 비교만으로 Test 일반화 개선이나 반전 원인을 입증하지 않는다. 고정 rho=0.25, 3dataset×3seed=9 fresh fits와 완료 후 기존 CNN/FULL18개와의 Validation 비교에 대한 사용자 실행 승인을 적용한다. 상세 판단은 PURPOSE_AND_DECISIONS.md에 보존한다.

## 현재 기준선 — 완료

- 과학 준비 계약 rawSHA `fae92b1695483ee4c978a6f72ddff7628881fc9bb2af42b246508835bddf06c9`, canonical `6b5564173eb6f9feed35bc89142bfdde50944c285c4a402d2a4f5b311e2983f0`.
- 준비된 source134와 참조 증적241개의 SHA를 재확인했다. 기존 CNN/FULL18개 결과와 준비 당시의 실행 전 승인 상태는 그대로 보존한다. 현재 승인은 별도 execution 계약에 기록한다.
- 5080/5090 읽기 전용 사전 관측에서 GPU UUID·Python3.12.13·Torch2.11+cu130·cuDNN91900/92000·입력7개 SHA·새 root 부재·GPU compute PID 부재가 일치했다. 이 과거 관측을 이후 유휴 상태의 증명으로 재사용하지 않는다. 원문과 SHA는 native_preflight에 있다.

## 실행 도구 검증 — 완료

- 학습 루프는 봉인된 frozen_runtime.py를 사용하고, cap 후보 adapter만 별도 운영 연결부에 결합했다. strict seed, 선택/종료 checkpoint 재현, 저장 history, 실제 학습 대상, 예산 검사를 유지한다.
- host별 단일 worker로 순차 실행한다. 5080은 Taxi3·RAF3·Intermittent62 1개, 5090은 Intermittent42/52 2개다. 조건36시간 및 host168시간 기한을 연장하지 않는다.
- CPU 합성 검증을 포함한 실행·관측·native 검사 연결부35개 테스트가 통과했다. 실제 native GPU 합성 검증은 배포 후 별도 phase이며 아직 완료로 간주하지 않는다.
- qualification은 독립 timeout 하위 프로세스, 학습은 각 조건의 독립 timeout 하위 프로세스로 실행한다. 실제 하위 프로세스 종료와 pipeline wrapper 자체 종료를 구분한다. wrapper 자체 OS 종료는 직접 관측했다고 주장하지 않는다.
- 실패·claim·시작 시도 기록을 보존하고 자동 재시도하지 않는다. Test·기존 학습·기존 claim·Runtime·인증·원고를 변경하지 않는다.

## 남은 순서

**고정 실행 payload 배포와 실제 GPU 검사 — 다음 작업**
- 같은 두 서버의 새 격리 root에만 배포하고 합성 검사와 실제 입력 지문을 검증한다. host별 실제 검사 exit0 이후만 해당 host 학습을 허용한다.

**후보9 학습 및 종료 관측 — 다음 작업**
- 고유9개만 집계하고 완료·진행·대기·미확정을 분리한다. 같은 fit의 저장 timing에서만 조건부 ETA를 계산한다.

**원본 회수와 3seed Validation 비교 — 다음 작업**
- 모든 worker 종료·terminal binary SHA를 검증한 뒤 원본을 별도 로컬 경로로 회수한다. 같은 선택 checkpoint의 여섯 지표와 seed별 차이·평균·표본SD를 후보−FULL, 후보−CNN으로 각각 정리한다.
