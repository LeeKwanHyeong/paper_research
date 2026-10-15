# CNN backbone 반복 운영 — RAF 위치 비교 루프2

2026-10-15 사용자의반복개선사전승인과“기존CNN기준선은건들지마”,“그거제외하고하던거마저해”를적용한다. 프로젝트TEST_SESSION_PROTOCOL.md0·1·9·12,이문서,EXPERIMENT_PLAN.md,DIAGNOSTIC_BASIS.json,LOOP_STATE.json,현재receipt를먼저읽는다. 기존CNN모델·checkpoint·결과는변경/재학습/재평가하지않는다. 이전raf9계획과완료된루프1학습·평가는재실행하지않는다.

루프2는titantpp_backbone_location_raf6_20261015_v1을사용한다. Query-only대조군/Value-only주후보×seed42/52/62총6개,5090최대3병렬이다. 입력/log1p/손실/head/메모리/예산/선택은고정하고RevIN/GRU/head변경으로전환하지않는다. 추가분기는후보당1,024개이며K추가경로없음,기존선형CNN QKV유지다. 저장Channel3는보조참조이며새학습이아니다.

준비중이면승인된새파일구현·검증을진행한다. 모델/운영파일을병렬분담하고공통계약·독립검토·커밋·봉인·배포·native검증·학습시작은순서대로진행한다. 검증된진입점·정확한integritySHA만LOOP_STATE/LOCAL_READY/START_RECEIPT에서사용한다. source137상속+새2파일/input/운영/승인/등록부SHA와GPU/Runtime/후보3동시합성검증뒤에만새root를한번시작한다. 이미시작한root재설치·재시작0,원본덮어쓰기0,claim/failure삭제0,자동retry/resume0이다. 기존기한은연장하지않는다.

livehost관측은회차당1회,검증terminalcache는SSH0·원시각유지다. 실제PID/argv/start_ticks/lease/GPU/qualification/permit과과학workerexit·dispatcher/pipelineexit·terminalbinarySHA·소유PID/GPU명시적부재로완료판정한다. 조회실패를학습실패로바꾸거나미확정상태에서중복실행하지않는다. supervisor wrapper자체OSexit는직접관측하지않았으면그대로표시한다. 실패원문/stdout/stderr/transportactualexit를보존하고미확인원격재시도를반복하지않는다. 국소도구수정은새버전/격리경로에서검증한다.

6개종료검증뒤원본을exclusivearchive/manifest에회수한다. Validation선택checkpoint6개전부nativeValidation의동일대상·여섯지표·실제exit0가통과해야Test6을평가한다. Test재선택0. 기존CNN/외부/저장Channel은SHA검증한결과만재사용한다. 전체rawRMSE/MAE의3seed평균이Validation/Test각각기존CNN과외부지표별최우수보다모두낮아야사용자기준통과다. SD(ddof1)/모든seed차이/악화/큰수량·시간지표와Value−Query를기록한다. 저장Channel3,072대신규1,024용량차이,기존CNN5080cuDNN91900대5090cuDNN92000차이,반복Test참고탐색적한계를명시한다.

RAF통과한같은구조만Taxi5090/Intermittent5080에각최대3병렬확장한다. 5080은기존Taxi학습·평가·회수완료와실제유휴확인뒤만사용한다. 과거종료cache는현재GPU유휴증명이아니다. 미달이면결과보존·오류분석·단일backbone가설·대조군·새격리계약·검증·다음RAF루프다. 과학근거없이무작위탐색하지않는다. 데이터셋별서로다른구조의성공을공통모델성과로합치지않는다.

원고main.tex/main.reviewed.tex,기존source/결과/실패,공용Runtime/driver/auth,다른자동화,삭제된A100은변경·접속하지않는다. 다른캠페인을대신실행하지않는다. Notion3fabbe40-5613-80e5-91e9-e66ff02bb190에각루프시작/가설/증적/결과/판정/실패/한계/다음변경을누적한다. 장애시로컬미전송기록을남긴다. 자동화cnn-backbone-raf는매시간현재단계를계속하며정상진행·대기는조용히유지,시작/주요전환/판정/완료/실패/사용자조치때만알린다. 첫RAF완료로삭제하지않고동일구조3데이터셋검증·최종기록완료또는사용자중단때종료한다.
