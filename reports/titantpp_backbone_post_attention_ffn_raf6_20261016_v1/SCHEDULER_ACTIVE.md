# 현재 단계: 최종 FFN의 이력 결합 설계 완료, 구현·검증 다음 작업

TEST_SESSION_PROTOCOL.md0·1·9·12와 이 경로 SCIENCE_SPEC.json, ACCEPTANCE.json, RESEARCH_AND_DESIGN.md, DUPLICATE_AUDIT.md, DESIGN_REVIEW.json, LOOP_STATE.json을 먼저 읽는다. 기존 KeyContent/루프1~5 학습·평가·진단은 모두 완료다. 해당 helper를 다시 실행하지 않는다.

기존CNN 모델/checkpoint/결과 변경·학습·추론·평가0, 저장결과만 비교한다. 입력/log1p·전처리·공통손실/head·메모리 규칙·예산·seed·선택은 고정한다. 원고/옛source/결과/실패/claim/공용Runtime·driver·auth/다른자동화/삭제A100을 변경·접속하지 않는다. RevIN·GRU·head로 전환하지 않는다.

설계는 두 새 후보 titantpp_cnn_ffn_current_v1 대조군과 titantpp_cnn_ffn_temporal_v1 각RAF42/52/62=6개다. 최종 FFN 첫 선형층 뒤128차원에 채널별 비선형 관측3tap 잔차를 추가한다. 원래FFN 전체 교체가 아니다. 수식/가용성/초기화/귀속 한계는 SCIENCE_SPEC를 따른다. 독립 설계·중복검토는 통과했지만 구현·합성·native 검증은 아직 없다. 명령·integritySHA를 추측하지 않는다.

다음은 새 모델/runner 구현, 합성 및 독립 구현 검증이다. 해당 작업은 기존 사용자 사전승인 범위다. 원CNN 구조에서 fresh 초기화하며 기존checkpoint warmstart0이다. 신규무작위draw0·공통tensor/RNG·dropout경로를 보존한다. c0시 a/b 첫 task-gradient0, weight decay로 값은 움직일 수 있음을 검증한다. 비선형 대조의 비축약·일반상태 gradient·mask/withheld·미래차단·checkpoint 식별을 확인한다. 같은 파일/계약/커밋/봉인/실행은 직렬, 독립 파일 구현·리뷰는 역할과 소유를 분리해 병렬 가능하다. 완료 전 실제학습 준비 완료로 표현하지 않는다.

그 뒤 신규운영도구·실험실행계약을 새 경로에서 구현/합성/독립검토/커밋/봉인한다. native진입점과 integrity는 실제prepare 반환만 쓴다. 기존5090 Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, condition36h, 상속absolute deadline1792679647.915156 및 원permit b3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a를 유지한다. 기한연장0. 모든로컬SHA/계약/승인뒤 fresh유휴/소유·nativeRuntime/GPU/입력/후보3동시합성검증을 하고 단1회 시작한다. 학습은Current3가모두끝난뒤Temporal3,최대3병렬이다. 현재원격root·학습시작은없다.

실행후 livehost 회차당관측1회·완료cache SSH0/원시각 유지. PID/argv/startticks/lease/GPU 신원을 구분한다. 종료는6worker/dispatcher/pipeline 실제exit0·terminalbinarySHA·소유PID/GPU명시부재 뒤 원본회수/전체SHA검증한다. wrapper자체OSexit는 직접관측 여부를 구분한다. 실패·미완료attempt/claim·timeout·신원미확정은 stdout/stderr·byte/SHA·actualexit를 보존하고 미확인원격retry/resume/kill/claim삭제0. 국소수정은 새버전에서 검증하며 옛봉인은 보존한다.

고정fullValidation rawRMSE 최초strictfinite최소epoch/동률첫epoch checkpoint로 Validation6 여섯지표·대상·실제exit0를 모두 재현한 뒤 Test6. 기존CNN/외부재평가0, Test재선택0, 실제Test로도구디버깅0. 외부등록부57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49를 재사용한다. 각후보Validation/Test RMSE/MAE3seed평균 네기준이CNN및같은대상외부지표별최우수보다모두엄격히낮아야사용자통과다. 평균±SDddof1·모든seed절대/상대차이·큰수량q>200/TimeNLL·악화·단순기준선TimeNLL N/A를 보존한다. Temporal−Current는 별도 귀속 기준이다. Current가 통과/선정되면 시점별FFN개선으로 보고하며 temporal 성공으로 부르지 않는다.

통과후보는Validation RMSE→MAE→arm문자순으로선택하되 Test통과자격사용을명시한다. RAF통과 동일구조만Taxi5090와기존Taxi완료+fresh유휴5080Intermittent에각최대3병렬확장한다. 기존CNN/외부5080cuDNN91900차이와공동최적화·memory조회변경가능성한계를기록한다. 반복Test참고탐색이며독립일반화/인과/유의성주장0. 미달이면원본보존후근거있는다음가설만검토하며무작위탐색/같은삭제진단반복0.

Notion3fabbe40-5613-80e5-91e9-e66ff02bb190에 단계·가설·증적·결과·한계·다음작업을누적하고읽어확인한다. 정상진행/대기/변화없음은조용히유지하며시작/주요전환/판정/완료/실패/사용자조치때만알린다. 첫RAF완료/미달로자동화삭제0,동일구조3데이터셋검증·최종기록완료또는사용자중단때종료한다. 승인부족으로 중단하거나 이미승인한범위를재질문하지않는다.
