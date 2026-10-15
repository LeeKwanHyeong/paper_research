# CNN backbone 반복 운영 — RAF V 관측 차분 루프3

프로젝트 TEST_SESSION_PROTOCOL.md 0·1·9·12와 SCIENCE_SPEC.json·EXPERIMENT_PLAN.md·LOOP_STATE.json·현재 receipt를 먼저 읽는다. 사용자의2026-10-15 반복개선 사전승인, 기존CNN보존 지시를 적용한다. 완료된 루프1·2 학습/평가 helper와 다른 Taxi 자동화를 재실행하지 않는다.

신규 후보 titantpp_cnn_value_difference_v1 ×RAF seed42/52/62 총3개를5090에서최대3병렬로 학습한다. 기존CNN은 모델/checkpoint/원본을 변경·재학습·재평가하지 않고 기존CNN3·외부모델 저장결과만 재사용한다. Q/K 기존3tap구조·공통투영은 정상학습한다. V만 존재하는인접관측차분2개로보정하며상수잠재V는짧은이력에서도보존한다. CNN계수576→512이며용량감소·경계·차분parameterization효과가함께있다. 과거LevelHistoryQKV와아이디어중복을명시하고새원리·raw수량편향원인확정으로주장하지않는다.

공통입력/log1p/손실/head/메모리/예산/선택을유지한다. 신규GRU/RevIN/gate/QKbound/confidence0. 현재CNNcheckpoint warmstart0. 첫strictfinite fullValidation rawRMSE 최소epoch, 동률첫epoch로고정하며Test재선택0. 기존host절대deadline을연장하지않는다.

현재 준비중이면 신규모델/runner/운영도구/합성계약검사를완료하고독립검토·커밋·source/input/기존CNN3/외부등록부/운영/승인SHA봉인을순서대로진행한다. LOCAL_READY나실제prepare반환의정확한integrity를확인한검증된진입점만사용하며명령·SHA를추측하지않는다. 이미시작한root재설치/재시작0이다. 새root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1_5090, GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, 기존ai_env Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads다. 실제유휴/소유상태와GPU/Runtime/입력/후보3동시합성native검증뒤exclusive intent·permit으로한번만시작한다.

live상태관측은회차당1회,검증종료cache는SSH0·원관측시각유지다. 실제PID/argv/start_ticks/lease/GPU/qualification/permit/workerexit·dispatcher/pipelineexit·terminalbinarySHA와소유PID/GPU명시적부재를구분한다. 조회실패를학습실패로바꾸지않는다. 불명확하면중복학습하거나유효ETA로바꾸지않는다. stdout/stderr원문·actualtransportexit를보존하고미확인원격재시도/claim삭제/kill/resume하지않는다. 승인된안전한국소도구수정은새버전에서검증하며기존원본·실패는보존한다. supervisor wrapper자체OSexit는직접관측여부를구분한다.

실제3scientificworkerexit0·dispatcher/pipeline실제exit0·train_complete·모든terminalbinarySHA·명시적owner/GPU부재뒤만exclusivearchive/manifest로원본을회수한다. 선택checkpoint3 nativeValidation의같은대상·여섯지표·actualexit0가모두통과한뒤Test3을한다. 기존CNN/외부모델재평가0. 평가도구는별도새파일로합성·독립검토·정확한integrity봉인뒤사용한다. 실제Test로도구디버깅0.

Validation/Test각각RMSE·MAE의3seed평균이기존CNN및같은대상외부지표별최우수보다모두엄격히낮아야사용자기준통과다. 평균±표본SD(ddof1),모든seed별절대/상대차이,악화,큰수량q>200·시간지표·단순기준선을보존한다. 기존CNN5080/cuDNN91900대신규5090/cuDNN92000차이,현재Test를참고한탐색적반복이라는한계를명시한다. 독립평가/보편적일반화/원인확정/유의성으로확대하지않는다.

RAF통과후같은구조만Taxi5090와실제유휴·기존Taxi평가/회수완료가확인된5080Intermittent에각최대3병렬로확장한다. 과거종료cache는현재유휴증명이아니다. 미달이면원본/실패보존→오류분석→다음단일backbone가설/대조→새격리계약·구현·검증→5090RAF를계속한다. 무근거무작위후보추가0,서로다른구조의데이터셋별성공을공통모델로합치지않는다.

Notion3fabbe40-5613-80e5-91e9-e66ff02bb190에루프시작·가설/구조·증적·결과·판정·실패·한계·다음작업을한국어로누적한다. 원고main.tex/main.reviewed.tex,공용Runtime/driver/auth,기존원본/source/claim/failure,다른자동화,삭제된A100은변경·접속하지않는다. 정상진행/대기는조용히유지하고시작/주요전환/판정/완료/실패/사용자조치만알린다. 자동화cnn-backbone-raf는첫RAF완료로삭제하지않고동일구조3데이터셋검증과최종기록완료또는사용자중단때종료한다.
