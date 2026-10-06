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
현재 실행 계약SHA는 `19b2670327f1654923476a34c70c6eae5a065d4399098877148fafddb9bf2966`.
동결 과학 소스127개 closure는 `2b1d69786e5988ae7c302d90c2485438c6472e845b16a129821acd92d99e9ec0`,
운영 closure는 `f300da390b8072374705eea433106812a3cc78a5913a50028eba4fcf6c2feede`.
현재 두 서버 root는 각각 실행 bundle 이름의 `_5080_attempt2`, `_5090_attempt2` 폴더다.
Native 검증 후 과학/운영 소스를 변경하지 않았다. 실제 최초 lease 마감
2026-10-13 15:20:32 KST를 유지하며 이를 완료 예상으로 해석하지 않는다.

첫 준비 attempt는 동결 패키지의 빈 `sample_data` 경로가 누락돼 실제 학습·optimizer
업데이트0에서 종료됐다. 원래 증거와 remote root를 보존한 뒤 패키지 표식만 수정하고,
새 attempt2의 양쪽 검증을 다시 통과한 후 학습을 시작했다. 수치적 학습 실패가 아니다.

**여섯 구조 구현·동작 검증 — 완료**
- 로컬 구조·계약·운영63개 테스트와6개 하위 테스트 통과.
- 배치·실행 절차22개 오프라인 테스트와 독립 패키지 경로 회귀 검증 통과.
- 여섯 구조의 파라미터·초기 상태·동작 및 기준 폭8과의 일치를 확인했다.

**5080·5090 환경과 실행 예산 검증 — 완료**
- 양쪽 실제 GPU의 여섯 구조·3seed 초기화, 인과성·결정성·메모리 검증 통과.
- 전체 Train/Validation 모집단과 target SHA, runtime 및 소스 SHA를 확인했다.
- 합성 입력 GPU 예약 메모리 최대 비율은5080 12.35%, 5090 6.09%였다.
- 합성 학습 시간 시나리오는 Validation·로딩·replay·회수를 제외하므로 큐 ETA가 아니다.

**신규54조건 학습 — 진행 중**
- 2026-10-06 15:28 KST에 양쪽 서버 단독 worker를 시작했다.
- 실제 supervisor/timeout/worker PID와 GPU PID 연결, 양쪽 첫 checkpoint 저장을 확인했다.
- 아래 수치는 서버별 실제 관측시각을 보존한 시작 증거다. 전체0완료·2진행·52대기,
  실패0·미확정0이며 전체 큐 종료 예상은 미확정이다.

| 서버 | 실제 KST 관측시각 | 완료/실패/진행/대기/미확정 | 현재 조건 | 저장/best epoch |
|---|---|---|---|---|
| 5080 | 10/6 15:28:53 | 0/0/1/35/0 | Taxi · PF_H1 · seed42 | 2/1 |
| 5090 | 10/6 15:33:51 | 0/0/1/17/0 | Intermittent · PF_H1 · seed42 | 5/1 |

- 해당 fit의 초기 실측 기준으로 best 미갱신·속도 유지 시 조건부 잔여 시간은5080 약6분27초,
  5090 약38분35초다. 최대300epoch 시나리오는 각각 약49분17초·5시간16분10초다.
  이후 best가 갱신되면 바뀌며 다음 fit이나 전체 큐의 완료 시간을 뜻하지 않는다.
- 시작 증거는 `launch_confirmation.json`, 추가5090 첫 저장 검증은
  `first_checkpoint_confirmation.json`, 실제 관측은 `hourly_monitor/latest_*.json`에 연결했다.
- 매시간 읽기 전용 관측 자동화 `titantpp-54`를 활성화했다. 기존 PAUSED 자동화는 변경하지 않았다.

**종료와 원본 보존 — 다음 작업 / 학습 종료 이후**
- 두 서버54조건의 terminal, selected/last Validation과 실제 프로세스 종료를 검증한다.
- 승인된 원본 manifest/archive SHA를 회수한 뒤54조건을 같은 선택 epoch로 집계한다.
- 후속 control35개 오프라인 테스트와 독립 검토를 통과했다. 두 코드 SHA를 별도 integrity
  receipt에 고정하고 최초 호출에서 `pending_verified_54terminals`·원격호출0을 확인했다.

**구조 기여와 시간 손해 비교 — 다음 작업 / 원본 검증 이후**
- 전체·큰 수량 RMSE/MAE/TimeNLL의3seed 평균·표본SD와 고정된 구조 대조를 보고한다.
- 동일 예산 두 대조에서 같은2개 이상 데이터의 모든3seed 수량 RMSE 개선 여부를 확인한다.
- 이 결과만으로 독립 Test 우월성이나 구성요소의 순수 독립 효과를 확정하지 않는다.
