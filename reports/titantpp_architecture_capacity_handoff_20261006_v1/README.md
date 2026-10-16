# TitanTPP 구조 대조: Intermittent seed62의 5080 이관

**분배 계약과 중복 실행 방지 — 완료**
- 사용자가 “지금 5090에 남은 작업에 대해 분배해서 5080에서 병렬로 진행하자 작업사항에서 승인사항은 승인한다”라고 직접 승인했다.
- 5090의 실행 중 조건은 그대로 두고, 미시작 Intermittent seed62 여섯 구조만 새 5080 전용 폴더로 옮긴다. 같은 dataset·seed의 여섯 구조는 같은 GPU·Runtime에서 비교한다.
- 원래 5080의 Taxi·RAF 36조건은 검증된 완료 기준선이다. 원 관측시각과 terminal cache를 보존하며 새 작업 진척으로 표시하지 않는다.

| 소유 서버 | 조건 | canonical 조건 수 |
|---|---|---:|
| 원래 5080 root | Taxi·RAF × seed42/52/62 × 여섯 구조, 완료 보존 | 36 |
| 원래 5090 root | Intermittent × seed42/52 × 여섯 구조 | 12 |
| 새 5080 root | Intermittent × seed62 × 여섯 구조 | 6 |

새 5080 순서는 부모 계약의 `PF_H1 → P0_H1 → P0_H0 → P1_H0 → P1_H1 → P1_HC`다. 전체는 여전히 54개이며 이관 조건을 두 번 세지 않는다. 새 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_architecture_capacity_handoff_20261006_v1_5080`이다.

부모 실행 계약 canonical SHA는 `19b2670327f1654923476a34c70c6eae5a065d4399098877148fafddb9bf2966`, 과학 소스127개 closure는 `2b1d69786e5988ae7c302d90c2485438c6472e845b16a129821acd92d99e9ec0`, 과학 revision은 `a699448f2991465b4fc17c75fafcc1b85845d1d6`이다. 부모 계약·과학·운영 원본을 보존하고 이관 adapter와 새 운영 계약을 별도로 봉인한다.

**5080의 새 데이터와 실행 환경 검증 — 완료**
- 기존 5080 `ai_env`, GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`를 사용한다. 원래 5090 전용 환경과 공용 Runtime·인증은 변경하지 않는다.
- 새 root의 원본 데이터·split·과학/운영 SHA, 전체 Train/Validation 모집단, 여섯 구조의 초기화·결정성·인과성·메모리와 파라미터 수를 실제 5080에서 검증한다.
- 원래 5090의 초기화·interface 증거와도 대조한 뒤, 여섯 source run/claim 부재를 확인하고 배타 claim을 예약한다. 부분 예약·경합·실패는 보존하며 자동 retry/rollback/claim 삭제를 하지 않는다.

**5080·5090 병렬 학습과 관측 — 진행 중**
- 학습·loss·optimizer·selector·min40/max300/patience40, 서버당 worker1, 조건36시간·전체168시간을 유지한다. 기존 절대 마감 `2026-10-13 15:20:32 KST`를 연장하지 않는다.
- 새 조건은 fresh fit이다. 완료된 조건이나 현재 5090 학습을 재시작하지 않는다.
- seed62는 다른 GPU에서 실행되므로 이종 GPU의 3seed 개발 실험으로 보고한다. 같은 seed의 구조 차이를 비교하되, GPU가 다른 실행 시간을 구조 효율 근거로 섞지 않는다.
- 학습 관측에서는 Test·예측 행·Validation/Test 혼합 성과를 열지 않는다. 과거 A100·CNNGRU·시간 head·중단 GRUcontrol 작업은 조회하지 않는다.

**54조건 원본 회수와 기여 판정 — 다음 작업 / 종료 이후**
- 원래 5080 완료36, 5090 소유12, 새 5080 소유6의 scientific terminal·selected/last full Validation·실제 종료·SHA를 확인한다.
- 원래 5090이 첫 seed62 예약 claim에서 `FileExistsError`로 종료하면, 정확한 claimpath·12개 terminal·소유 PID 부재가 모두 검증된 경우에만 의도한 이관 경계로 분류한다. 기존 failure/status는 보존하며 새 학습 실패로 중복 집계하지 않는다.
- 세 소유 root의 원본 manifest/archive SHA를 회수한 후 54조건을 한 번씩 집계한다. 원본 회수와 학습 완료는 별도다.
- 기존 고정 기준인 `P1_H1`의 `PF_H1`·`P1_HC` 대비 RMSE 개선이 같은 2개 이상 데이터의 모든 3seed에서 나타나는지 판정한다. 시간 손해도 함께 보고한다. Test·CPU binary 재추론 감사·독립 평가와 외부 우월성 검증은 별도 후속이다.

운영 계약 canonical SHA: `af89908d34de77344b6cef9da80c82e7d2705df16433515e4edfbb1f57e9271b`. 새 운영 closure: `20445d1e3a23447759f8e80fad05a46da4acbebc6d6d40cbbeee384de542c7fb`. 신규 운영 38개·하위 8개 테스트 및 독립 검토 PASS. 배치 원본155개와 패키지 SHA를 실제 5080에서 확인했고 GPU 유휴·기존 Runtime 일치를 검증했다. 실제 학습 진입은 별도 확인한다.

2026-10-07 00:10:13 KST 새 5080 native qualification을 시작해 정상 종료(exit0)·PASS를 확인했다. 원래 5090의 초기화/과학 interface와 같고, source seed62 여섯 claim을 예약한 뒤 새 dispatcher를 00:13:31 KST에 한 번 시작했다. 시작 요청과 실제 checkpoint 저장은 구분한다. 기존 `titantpp-54` 자동화만 이관 관측기로 갱신했고 매시간 보고를 유지한다.

실제 최초 학습/저장 관측은 [INITIAL_STATUS.md](INITIAL_STATUS.md), 원본 증거는 새 bundle의 `launch_confirmation.json`과 `hourly_monitor/latest_5080.json`·`latest_5090.json`에 연결했다. 전체41완료·2진행·11대기, 실패0·미확정0이고 모든54canonical ID가 고유하다. 최종 회수 호출은 로컬 pending·원격호출0으로 확인했다.
