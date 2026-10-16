# 시간·수량 결합 pre-check의 seed52·62 확장

사용자 승인: “그러면 지금 RAF, Intermittent, Taxi 관련해서 52, 62 모두 실험에 추가하자.”

기존 seed42의 두 구조를 각 데이터셋에서 seed52·62로 추가 학습한다. 신규 12개와 기존 6개를 합쳐 총 18개이며 원수량 입력을 사용한다. Q3 Cap 실험과 Test 평가는 포함하지 않는다. 기존 CNN·이력 MLP16은 학습하거나 추론하지 않고 저장된 MLP16 Validation 9개만 비교에 재사용한다.

| 서버 | 추가 학습 | 기존 작업과 합친 동시 상한 |
|---|---|---:|
| 5080 | Taxi 두 구조 × seed52·62, Intermittent 시점별 MLP 대조군 × seed52·62 | 3 |
| 5090 | RAF 두 구조 × seed52·62, Intermittent Cross Attention × seed52·62 | 3 |

기존 seed42의 종료를 기다리는 작업은 별도 유한 큐에 남는다. 기존 작업을 중단하지 않으며 빈자리마다 다음 작업이 시작된다. fit당 36시간, 추가 큐 72시간을 넘기지 않는다. Codex의 반복 개선 자동화는 생성하지 않는다.

## 현재 기준선 — 완료

- 모델·runner·원본 입력은 부모 source933개와 동일한 바이트다. 변경은 seed와 별도 운영 코드 두 파일이다.
- 로컬 독립 검사 22개를 통과했다. 이후 실제 배포에서 발견한 admission CLI의 빈 stdout 연결 오류는 `PRELAUNCH_FAILURE.json`에 보존했다.
- 이 오류는 학습 시작 전 결과 읽기에서 발생했다. 기존 runtime·계약·실패 기록을 변경하지 않고 로컬 launcher v2가 서버에 저장된 qualification을 읽도록 했다. v2 검사 증적은 별도로 보존한다.

## 시작과 관측

첫 시작에는 `paper.scripts.launch_titantpp_cross_fusion_seed_extension_v2 --host <5080|5090>`를 사용했다. 이미 시작한 서버에 다시 호출하지 않는다. 기존 manager의 `deploy-start`는 admission stdout 연결 오류가 있어 사용하지 않는다.

현황은 `paper.scripts.manage_titantpp_cross_fusion_seed_extension --mode status --host <5080|5090>`의 관측 시각·worker·실제 프로세스·GPU를 함께 확인한다. 저장된 마지막 관측을 현재 시각으로 바꾸지 않는다. 실제 시작 현황은 `START_VERIFIED.json`을 따른다.

## 학습 종료 후 비교 — 다음 작업

신규 worker12개의 성공 종료와 선택 checkpoint를 확인한 뒤 원본을 별도 경로로 회수한다. 기존 seed42와 합쳐 데이터셋별 각 구조의 3seed Validation RMSE·MAE 평균과 표본표준편차, seed별 Cross−시점별 MLP 및 각 후보−저장 MLP16 차이를 비교한다. 종료 전 중간 best를 최종 결과로 표현하지 않는다.

Intermittent는 기존과 동일하게 대조군5080/cuDNN91900, Cross5090/cuDNN92000이므로 구조 비교에 서버 환경 차이가 포함된다. RAF와 Taxi의 두 구조는 각각 같은 서버에서 비교한다.

저장소는 `paper_research`, 브랜치는 `codex/contextual-retrieval-pilot-20261007`이다. 학습 과학 코드의 기준 revision은 `743691743837c0b1b66460923565c64f09659982`, 추가 운영 코드의 봉인 revision은 `597b0c338187767685b2a8d97cc34d681a78dd1a`다.
