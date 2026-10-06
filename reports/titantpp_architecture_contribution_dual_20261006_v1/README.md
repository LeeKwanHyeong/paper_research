# TitanTPP 구조 기여 대조: 세 데이터 × 3seed × 여섯 구조

## 목적과 승인

수량 우위를 주요 목표로 두고 시간 성능 손해를 반드시 함께 보고한다. 사용자는
2026-10-06 여섯 구조 구현, 두 RTX 환경·예산 검증, 신규54조건 학습을 직접 승인했다.
기존 설계 계약의 당시 학습 미승인 기록은 보존하고, 새 실행 계약과 승인 기록에
이번 승인을 별도로 연결한다. 이번 작업은 Train/Validation 범위다.

현재 기준선은 Validation에서 선정한 이력 보정 폭8이다. 이미 노출된 과거 Test를
새 독립 평가로 표현하지 않는다. CNN+GRU54 결과를 이 대조의 기준 구조로 바꾸지 않는다.

## 여섯 구조와 비교 목적

| 구조 | 마지막 prototype 검색 | 블록 사이 수량 이력 보정 | 개입 파라미터 |
|---|---|---|---:|
| P0_H0 | 없음 | 없음 | 0 |
| P1_H0 | 기존 prototype | 없음 | 4,096 |
| P0_H1 | 없음 | 기존 폭8 | 12,288 |
| P1_H1 | 기존 prototype | 기존 폭8 | 16,384 |
| PF_H1 | 동일 문맥 일반 신경망 | 기존 폭8 | 16,384 |
| P1_HC | 기존 prototype | 동일 예산 현재 상태 신경망 | 16,384 |

앞의4구조는 구성요소와 추가 용량의 결합 효과를 비교한다. 뒤의2구조는 동일 예산의
구조 대조다. PF의 초기 잔차 분포가 prototype과 다르므로 초기 잔차 크기와 실제
gradient 참여를 기록한다. current 상태에도 인과적 과거 정보가 들어 있으므로 HC를
모든 과거 정보가 제거된 모델이라고 설명하지 않는다. RAF의128번째 이력 분기는
기존 길이84 제한으로 비활성이고, 양쪽 대조에 같은 조건을 적용한다.

## 실행 및 검증

대상 저장소 `/Users/igwanhyeong/PycharmProjects/paper_research`, 브랜치 `master`.
새 전용 artifact root에서만 실행하며 공용 Runtime과 인증은 유지한다.
5080은 Taxi·RAF36조건, 5090은 Intermittent18조건이며 서버당 worker1이다.
같은 데이터·seed의 여섯 구조는 같은 GPU·Runtime에서 동결 순서대로 실행한다.

기존123개 과학 소스를 SHA로 복사한 뒤 신규 구현만 추가한다. 기존 checkpoint를
재사용하지 않고54조건 모두 처음부터 학습한다. 공통 초기 tensor·RNG·dropout,
인과성·padding·reset, 실제 파라미터 수, 기준 폭8과의 정확한 출력 일치를 검증한다.
양쪽 native GPU qualification과 전체 Train/Validation target SHA를 확인한 이후에만
training permit을 발급한다.

손실·optimizer·batch·수량/시간 단위·최초 최소 finite 전체Validation수량RMSE 선택을
유지한다. min40/max300/patience40, 조건36시간/전체168시간, 자동 retry 없음이다.
RMSE·MAE·TimeNLL은 동일 선택 epoch에서 읽고 전체와 큰 수량을 나눈다.
완료는 terminal scientific_success, selected/last fullValidation replay, 소형 원본SHA,
실제 소유 프로세스 종료까지 검증한 뒤 집계한다. Binary 회수와 CPU 감사는 별도다.

실행 bundle: `search_artifacts/titantpp_architecture_contribution_dual_20261006_v1`.
부모 설계SHA: `e94093fd3b94bae1ee241dfc33d02c80176fd75d2f3bc999f104bc5d41202f9d`.
실제 qualification·시작시각·계약SHA·PID·checkpoint 증거는 실행 후 추가한다.

**여섯 구조 구현·동작 검증 — 완료**
- 로컬 구조·계약·운영63개 테스트와6개 하위 테스트 통과.
- Native GPU qualification은 실제 서버에서 확인해야 한다.

**5080·5090 환경과 실행 예산 검증 — 진행 중**
- 15:10/15:11 KST에 두 GPU compute PID 부재와 기존 Runtime·충분한 디스크를 확인했다.
- 동일 데이터 SHA·실제 GPU 결정성·메모리·자체 timing을 검증한다.

**신규54조건 학습 — 다음 작업**
- 두 서버 qualification 통과 후 서버별 단독 worker를 시작하고 실제 학습 진입을 확인한다.
- 이후 시간별 읽기 전용 관측으로 진행·실패·완료와 조건부 ETA를 보고한다.

**기여 판단과 원본 보존 — 학습 종료 이후**
- 동일 epoch의3seed Validation을 비교하고 불리한 TimeNLL 결과도 보고한다.
- 이 결과만으로 독립 Test 우월성이나 구성요소의 순수 독립 효과를 확정하지 않는다.
