# CNN·GRU 구성요소 대조: 5080 먼저 실행

**프로세스 판정 오류를 검증하고 실제 시작한다 — 진행 중**

- 최신 사용자가 수정·검증 후5080 학습 시작을 승인했다. 대상은 Taxi·RAF 각각4구조×3seed, 총24개 fresh fit이다. 과학 source131·입력7·원래 손실·optimizer·selector를 유지한다.
- v2 metadata 누락과 v3/v4 동시 수용 판정 실패 원본을 보존한다. v3 native 검증은 실제exit0지만 전체 동시 수용 PASS는 아니다. 당시 실패 proc argv가 없어 종료 경합은 추론으로 기록했다.
- v5는 정상 종료한 소유 PID가 NVML에 잠깐 남은 경우만 최대5초·25회·0.2초 간격 내에서 소멸을 재확인한다. 외부 PID/명령을 허용하거나 미확인을 성공으로 바꾸지 않는다. 실제 native·동시 수용 종료0과GPU idle 후만 학습한다.
- 5080의 원래 시작1791794258.15949와deadline1792399058.15949를 계승하고 연장하지 않는다. 실패root에서 qualification·fit을 재실행하지 않는다.

**5080에서 같은 seed의 네 구조를 비교한다 — 진행 중**

- Taxi42→52→62 뒤RAF42→52→62를 실행한다. 각seed는NONE/CNN/GRU3개 동시 실행·모두 종료 뒤FULL1개, 이후 다음seed다. 최대3작업이다.
- Train/Validation만 사용한다. B128/min40/max300/patience40와 최초strict finite 최소 전체Validation raw수량RMSE 선택을 유지한다. resume/warm start/자동fit retry/Test/검색은0이다.
- 초기실행확인을마쳤으며최신사용자의별도승인으로1시간단위관측을추가했다. 매회livehost status SSH1회만하며학습코드/운영봉인은불변이다. [시간별스케줄](hourly_monitor/SCHEDULER.md).

**5090의 Intermittent12개를 진행한다 — 진행 중**

- 최신사용자 “5090도 진행하자” 승인후5090 ownnative/admission 실제exit0와GPU유휴를확인하고Intermittent12 fresh학습을시작했다. 5080과독립승인·nativegate를사용하고기존5080 permit과실행을유지한다.

**모든 종료·원본을 확인한 뒤 논문 근거를 판정한다 — 다음 작업**

- 실제worker/dispatcher 종료·마지막phase complete·GPU/PID 부재·selected/last binary·manifest/archive와 로컬SHA를 확인한 뒤36개ID를 집계한다.
- 여섯Validation 지표·모든seed/대조·평균/표본SD를 보고한다. 활성 파라미터 NONE89859/CNN90435/GRU114591/FULL115167의 차이와 FULL단독/나머지3개동시 부하 차이를 공개한다. 순수용량통제·인과시너지·통계유의성·새독립일반화의 증명으로 확대하지 않는다.
- 열린main.tex와기존결과는 이번운영수정에서 편집하지 않는다.

실제5080 native/admission/qualification campaign exit0 및초기3GPU·세조건checkpoint저장을확인했다. [실제시작관측](START_OBSERVATION.md)과[마지막초기확인](FINAL_LAUNCH_CHECK.json)을기준으로한다. 학습전체24개종료는미완료다.

5090 실제시작과최초저장전상태는 [5090 관측](START_OBSERVATION_5090.md), [최종초기확인](FINAL_LAUNCH_CHECK_5090.json)에원관측시각으로기록했다.

**checkpoint·ETA를매시간관측한다 — 적용완료 / 학습진행중**

- 사용자승인으로1시간heartbeat `cnn-gru36-checkpoint-eta` ACTIVE를등록했고각fit의실제저장/best와조건부ETA를보고한다. [등록증적](hourly_monitor/SCHEDULER_APPLIED.json), [관측규칙](hourly_monitor/SCHEDULER.md). 기존초기확인뒤정기조회0문구는이번명시적관측승인이대체한다.
