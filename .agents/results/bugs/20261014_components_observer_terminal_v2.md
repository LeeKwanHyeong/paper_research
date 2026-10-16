# CNN·GRU 구성요소 관측의 종료 상태 오류

상태: 수정·합성98회귀·독립QA·실제읽기전용연결 검증완료.

원인: v5 status는dict active_jobs만가정하여행정failed+[]에.values()를호출한다. 기존one_call은CalledProcessError stdout/stderr/returncode를보존하지않는다. failed status에서ownerPID가빠져emptyprocesses만으로부재를확정하는경로도강화가필요했다. 과거stderr가없어과거traceback은미확인으로유지한다.

수정: 원본봉인은보존하고새titantpp_components_status_transport_v2.py 및titantpp_components_parallel_monitor_v2.py 운영버전을추가했다. status전용 guard,형태/PID/argv/계약검증,오류원문SHA보존,실제transportexit/timeout구분,dispatcher/wrapper/worker PID명시조회,초기부재/중간소멸/재사용/queryerror구분,안정된live·명시적absent필수gate,새종료cache와같은시간별자동화연결을적용했다.

검증: 98tests PASS. gate를테스트프로세스에서만제거하면음성회귀실패를확인했다. 기존382증적·source131/input/op/local40/monitor봉인전후SHA동일. source/runtime/claim/failure/학습변경0,실제Test평가0.

실제원관측2026-10-14 08:03:03 KST: 원5090 유지8scientificworker exit0/104terminalfiles SHA검증. dispatcher는05:28:27 KST에seed62exclusiveclaim의계획된FileExistsError로실OSexit1. claim/failure원문보존,owner10 PID부재/GPU0. pipelinewrapper OSexit는직접관측하지않았다. 전체36은35완료1진행. 원5090원래12전체완료나수치학습실패로해석하지않는다.

증적: reports/titantpp_components_seed62_handoff_20261013_v1/hourly_monitor_v2/README.md 및FINAL_QA.json. 기존자동화cnn-gru36-checkpoint-eta만새버전에연결하고한시간주기/ACTIVE/같은채팅을검증했다. 원본회수·최종Validation집계는별도다.
