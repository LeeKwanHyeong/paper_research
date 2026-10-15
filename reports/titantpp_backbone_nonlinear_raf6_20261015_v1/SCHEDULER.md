# CNN backbone 반복 운영 — RAF 신규후보6개

## 승인과 범위
2026-10-15 사용자의 반복개선 사전승인과 최신 “기존 CNN기준선은 건들지마”, “ㅇㅇ그거 제외하고는 하던거 마저해”를 함께 적용한다. 기존 CNN 모델·checkpoint·결과·실패기록은 수정/교체/재학습/재평가하지 않는다. 이미 검증된 기존결과를 비교에만 사용한다. 과거RAF9계획의신규CNN3은취소됐으며9개용도구를실행하지않는다.

프로젝트 TEST_SESSION_PROTOCOL.md0·1·9·12, 이파일, EXPERIMENT_PLAN.md, LOOP_STATE.json과 실제receipt를 먼저 읽는다. 현재캠페인은 titantpp_backbone_nonlinear_raf6_20261015_v1. 신규학습은 titantpp_cnn_channel_nonlinear_v1/titantpp_cnn_joint_nonlinear_v1 ×42/52/62 총6개,5090최대3동시다. 새root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_nonlinear_raf6_20261015_v1_5090이다.

## 진행
로컬source/input/contract/approval/등록부SHA검증→nativeGPU/Runtime/입력/후보3동시합성검사→후보6학습→종료검증→원본회수→Validation6재현→Test6→비교→Notion기록 순서다. 별도승인을반복질문하지않는다. 실제검증된진입점과integritySHA는LOOP_STATE/LOCAL_READY/START_RECEIPT를따르며미준비명령을추측하지않는다. 준비중이면승인된구현·검증을계속한다.

공통입력/log1p/손실/head/메모리/seed/예산/선택을유지한다. RevIN/GRU/head변경으로방향을바꾸지않는다. 조건당36h/host168h를계약에고정하고연장하지않는다. 새루프는새경로·계약·source봉인으로만진행하며기존fit재시작/원본덮어쓰기/claim삭제는하지않는다. 공용Runtime/driver/auth,원고,다른자동화,삭제된A100은변경·접속하지않는다.

기존5080Taxi학습·평가·회수를건드리지않는다. 현재GPU점유가있으면대기하며kill하지않는다. 과거종료cache는현재유휴증명이아니다. 학습중관측은회차당livehost1회,검증terminalcache는SSH0로원시각유지한다. PID/argv/start_ticks/lease/GPU/승인·qualification·permit·worker/dispatcher/pipeline종료·binarySHA를확인하고누락·변조를완료로처리하지않는다. wrapper자체OSexit는직접본경우만직접관측으로표시한다. 조회실패와학습실패를구분한다. 실패원문과stdout/stderr를보존하며미확인실패에서원격재시도·중복학습하지않는다. 필요한국소수정은새버전에서검증한다.

선택은전체fullValidation raw수량RMSE의최초strictfinite최소epoch(동률첫epoch)다. 종료원본과selected를보존하고전6개Validation의동일대상·여섯지표·실제exit0통과뒤만Test6한다. Test로재선택/재학습하지않는다. 기존CNN3와외부6모델×3seed는검증된고정결과만재사용한다.

## 판정과 확장
사용자성능기준은Validation과Test각각RMSE와MAE모두3seed평균이기존CNN및외부지표별최저평균보다엄격히낮은경우다. 평균은반올림전값,SD는ddof1로계산한다. seed별절대·상대차이/악화와전체·큰수량·TimeNLL여섯지표도보존한다. 외부RMSE/MAE최우수가다르면지표별기준으로표시한다. 누락·대상/지표/선택불일치는미확정이며통과가아니다.

Joint는주후보,Channel은비선형·용량대조군이다. Joint−Channel과기존CNN대비는분리보고한다. 대조군만좋으면Joint기여성공으로표시하지않는다. 기존CNN5080/cuDNN91900과신규5090/cuDNN92000차이는한계로명시하고기준선재학습은하지않는다. 반복Test참고이므로탐색적평가이며독립일반화·원인확정·보편우월성·통계유의성으로확대하지않는다.

RAF기준통과시동일방법론·공통조건을고정해Taxi5090와Intermittent5080를각최대3동시로진행한다. 5080은기존Taxi학습뿐아니라승인된Validation/Test·회수완료와실제유휴확인후만사용한다. 기존CNN은모든단계에서재사용한다. 미달시결과보존→오류분석→하나의backbone가설과필요대조군→새5090RAF루프다. 서로다른데이터셋의서로다른구조성공을공통모델성과로합치지않는다. 근거없는무작위탐색은하지않고근거가없으면보고한다.

## Notion과자동화
Notion page3fabbe40-5613-80e5-91e9-e66ff02bb190에루프시작·가설·구조·고정조건·코드/계약/source/inputSHA·서버/Runtime/seed·실제KST·결과/SD/seed차이·외부비교·실패/한계·다음변경을누적한다. 원본은로컬immutablearchive/manifest에보존한다. Notion장애시로컬미전송기록을남긴다.

자동화cnn-backbone-raf는매시간승인된현재루프를계속한다. 정상진행/대기/변화없음은조용히유지하고시작/주요전환/판정/완료/실패/사용자조치때알린다. 기존taxi-history-cnn6-validation자동화는변경하거나대신실행하지않는다. 첫RAF완료만으로이반복자동화를삭제하지않고동일구조3데이터셋검증·증적·Notion기록완료또는사용자중단때종료한다.
