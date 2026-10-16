# CNN·GRU36 시간별 모니터링

**1시간 단위 관측을 적용한다 — 완료**

- 최신 사용자가 매시간 checkpoint·best epoch·남은 시간 보고를 승인했다. 자동화 `cnn-gru36-checkpoint-eta`는 현재채팅의 heartbeat이며 ACTIVE이고1시간간격이다. 기존8개PAUSED자동화는변경하지않았다.
- 모니터 코드/테스트 commit38af5c30c39bab9f97c3d828e762849dddb4c70e, 별도 monitor_integrity rawSHA a2b1095b7cbdd9fb0a65442378b5e7817b4dbfe448c3667d3506fca0efee35ec를고정했다. ETA17·관측제어5개검증과실제host별SSH1회통합관측을통과했다. 기존과학source131/운영40봉인불변이다.
- 매회 프로젝트root에서 PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.observe_titantpp_components_hourly --bundle /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_components_20261012_v5 --monitor-integrity-sha a2b1095b7cbdd9fb0a65442378b5e7817b4dbfe448c3667d3506fca0efee35ec 를정확히한번실행한다. livehost마다읽기전용status SSH1회, 검증terminal cache는SSH0/원관측시각유지다. 별도SSH·반복poll·학습·재실행·retry/resume·claim삭제·Runtime변경·Test·원본회수0이다.
- helper가모니터승인/코드봉인·기존bundle/current/계약/source/input/op/control과양host승인을SHA로검증한다. mismatch이면원격0으로알리고원본을보존한다. 조회실패를학습실패로바꾸지않고미확정저장/PID/계약을완료로바꾸지않는다.

**실제 저장 epoch와 조건부 ETA를 보고한다 — 진행 중**

- 실제KST시각·host별완료/진행/대기·dataset/seed/arm·저장checkpoint epoch·history 기준첫strictfinite 최소 fullValidation 수량RMSE best를매시간한국어표로보고한다. final selected binary는fit 자연종료후확인한다.
- 각fit자신의최근최대10완료epoch finite양수elapsed_seconds 중앙값을사용한다. best 미갱신/속도유지조건의min(300,max(40,best+40))까지잔여와300epoch까지잔여를구분하며표본수도보고한다. 未시작/다른fit/GPU속도대입0, 전체36개큐ETA미확정, 종료후endpoint재추론·원본회수·CPU감사는별도다.
- 조건36시간/host168시간 deadline을ETA로사용하거나연장하지않는다. 원문snapshot claims의남은기한보다300시나리오가길면현기한내보장되지않는다고표시한다. 첫실관측19:12:44KST에5090GRU의300시나리오는52시간12분이며남은조건기한약35시간27분보다길다. 이는관측속도유지시나리오이고실제실패나기한연장승인으로해석하지않는다.

**모든36개 종료 뒤 모니터링을 마치고 원본·분석을 확정한다 — 다음 작업**

- host별24/12 고유ID·실제worker/superviseddispatcher exit0·마지막phasecomplete·GPU/PID부재·모든원격binary manifest SHA가검증되어helper all_training_terminal_verified=true일때만최종관측을보고하고이자동화하나만삭제한다. partial/failed/미확정이면유지하고필요한사용자조치를알린다.
- 원격학습종료와localimmutable 원본archive/manifest/최종Validation 집계를구분한다. 모니터는자동회수나새평가를하지않는다. 이후원본회수·여섯Validation지표·모든seed/대조·평균/표본SD 확인→CNN+GRU54 Results/Methods/주장정합순서다. pipeline wrapper 자체OS exit는직접관측하지않았다는한계를유지한다.
