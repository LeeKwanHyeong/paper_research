# 완료판정 보정과 seed62 병렬 학습 적용

**완료판정·종료 캐시 오류를 수정하고 검증한다 — 완료**

- 기존 v5가 최종 terminal manifest 완료를 이전 exposure checkpoint 검사로 덮던 오류를 별도 verifier/hourly_v2에서 수정했다. 원본 source131/input7/operation4/local40/과거 관측은 보존했다.
- focused53개와 원 launcher18개 검증을 통과했고 독립 QA에서 원문24/3 완료, 부정 조건, 원시각·leaseexpiry 뒤 cache 재검증 SSH0를 확인했다.
- paper_research / codex/contextual-retrieval-pilot-20261007 코드commit: 59414b18d40846429316f23bcd17d0fb90584e7e. 기존 staged draft.tex를 포함하지 않았다. Push는 수행하지 않았다.
- 원5080 Taxi·RAF24개는 원관측08:27:20 KST의 종료 캐시로 처리한다. 원dispatcher 종료06:50:48 KST를 새 시각으로 바꾸지 않았다. 해당cache가현재GPU전역idle을보증하지않는다.

**미시작 seed62 네 조건의 생성 중복을 차단하고 새5080에서 시작한다 — 완료**

- 원5090의정확GPU/PID/argv/봉인·62의전체미시작을확인한뒤firstNONE62 행정claim을원자게시했다. 생성 배제 proof rawSHA c260a70d8c8d1c1bec6139db60134261e18e01895b305ddfc107d82d2da214eb, claim rawSHA15cfb9cd9cf6c1ce9e6ff61549cb002cbd3902d1dc8bc9d1f2edceba258343b9이다. 부모전체owner해제는아니다.
- 새5080 contract canonical81972d7b774c3b924d4bf6517252a0d0bed8291da9165100ab5b8932128ef1c1. native qualification 실제child/phaseexit0와3capacity admission actualexit0를확인했다. 원host절대deadline연장0이다.
- 2026-10-13 **09:22:39 KST**에새5080 train pipeline을시작했다. 새root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_seed62_handoff_20261013_v1_5080`이다.
- **09:23:36 KST** 실제동시GPU PID84546/84548/84550, guard84545/84547/84549, dispatcher84530의계약·job·정확argv·PPID·GPU UUID를확인했다. Intermittent62 NONE/CNN/GRU3개가첫저장전이며, 세조건자연종료후FULL이실행된다. admission을학습으로집계하지않았다.
- 원5090 seed42 FULL/seed52는자연진행한다. 52종료뒤62 첫claim의예정된FileExistsError·actualadministrativeexit1·phasefailure·claim을보존한다. 이종료는아직관측되지않았고원12완료또는수치학습실패로바꾸지않는다.

**현재 checkpoint와 한 시간 모니터링을 확인한다 — 진행 중**

실제원관측: 원5090 **2026-10-13 09:23:35 KST**, 새5080 **09:23:36 KST**. 고유36 ID: **완료27·진행4·대기5·미확정0**.

| 대상 | 상태 | 저장/best | 조건부 잔여 |
|---|---|---|---|
| 원5080 Taxi·RAF24 | 종료 cache·SSH0 | 모든최종값원관측보고보존 | 종료 |
| 새5080 Intermittent62 NONE/CNN/GRU | 실제3 GPU worker 동시진행 | 첫저장전·best미확정 | 해당fit timing없어미확정 |
| 새5080 Intermittent62 FULL | 대기1 | 저장전·best미확정 | 미확정 |
| 원5090 Intermittent42 대조3 | 완료3 | NONE70/30·CNN68/28·GRU54/14 | 종료 |
| 원5090 Intermittent42 FULL | 진행1 | **45/15** | best미갱신시약1시간21분·300epoch시약34시간22분 |
| 원5090 Intermittent52 네구조 | 대기4 | 저장전·best미확정 | 미확정 |

- 5090 FULL ETA는해당fit자신의최근10epoch 중앙값485.253초다. 남은조건기한약29시간53분보다300epoch대안이약4시간29분길다. 현기한내300epoch는보장되지않으며deadline을연장하지않는다. 전체큐종료시각은미확정이다.
- 기존자동화 `cnn-gru36-checkpoint-eta`를동일채팅/ACTIVE/1시간으로갱신했다. 새helper는원5080 종료캐시0회·live원5090/new5080 각status1회·검증terminalcache0회이며root별queryfailure와identity미확정을별도로보존한다.
- 이번통합관측round rawSHA04a3a8a35d5be1413bb88686a8c2e4f200eb7c3402049bfdd5c82954c423e677, latest rawSHAe0463f8ebf1200dc5ff080d50d6c5e62184d9a898aed30effeba40ad9a67857d. 별도EXECUTION_RECEIPT rawSHA4a2d0dbc11a7dc7167d76c54a9e55a8181b1819a8617b6141244faa7d52b59f0이다.

**종료와 원본·논문 근거를 확정한다 — 다음 작업**

- 한 시간 관측으로36개과학worker 실제종료와원5090계획된행정종료를확인한다. 원/new5080은실dispatcherexit0·train_complete, 원5090은실exit1·failure/claim·8과학workerexit0를구분한다.
- 이후로컬immutable원본회수계약 → archive/manifest/선택·최종binary 로컬SHA → 모든seed/대조여섯Validation지표 → Results/Methods/주장정합순서다. 원본회수·CPU감사·Test·검색·원고변경은수행하지않았다.
- 같은seed네arm의동일GPU/Runtime비교를유지했다. Intermittent42/52는5090/cuDNN92000,62는5080/cuDNN91900이므로seed간환경차이를평균·SD에명시하고시간은host별로분리한다. pipeline wrapper 자체OSexit는직접관측하지않았다.
