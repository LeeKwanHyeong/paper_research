# RAF CNN 비선형 경로 위치 비교 — 루프 2

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15 Asia/Seoul
- Verification Status: 계약 고정·구현 및 검증 진행 중, 신규 학습 미시작
- Version Label: nonlinear_location_raf6_v1

## 이전 결과를 기준선으로 보존한다 — 완료
루프1 Channel·Joint 두 후보는 Validation/Test RMSE·MAE 사용자 기준에 모두 미달했다. 기존 CNN 모델·checkpoint·결과와 이번 후보 원본은 변경·재학습·재평가하지 않는다. 기존 CNN3와 외부 지표별 최우수 결과는 검증된 등록부를 그대로 사용한다. 같은5090의 Channel(Q/K/V 추가분기)3개는 별도 보조 비교다.

## 오류와 연결되는 단일 가설을 검증한다 — 다음 작업
전체 Validation 6,690개 중 큰수량(q>200)50개에서는 오차가 줄었으나, 나머지6,640개의 제곱오차 증가가 이를 상쇄했다. 고정 후보의128 Train 및128 Validation 표본에서 Q 추가분기의 상대적 변화가 K보다 컸다. Channel의 Q/K norm만 복원한 attention 산술 진단에서는 같은 모델에서 추가 Q/K항을 제외했을 때와의 차이가 세 seed 모두 줄지 않아, Channel 후속 후보로 norm보존을 이번 루프에서 우선하지 않는다. Joint에서는 차이가 줄었으므로 norm효과가 일반적으로 없다고 결론내리지 않는다. 이 진단은 로컬CPU/torch2.14와 학습5090/torch2.11 환경이 다르고 작은 고정표본(큰수량 대상 Train0/Validation1개)이므로, Q변화가 전체 또는 큰수량 성능악화 원인이라고 확정하지 않는다. 이 산술 대조는 과거 CNN의 실제 attention이나 수정 모델의 예측성능 평가가 아니다.

가설은 동일한 추가 비선형 CNN 용량을 attention의 가중치 결정(Q)보다 전달 내용(V)에 배치하는 편이 유리한가다. Q/K/V 공통 투영과 기존 선형CNN은 정상 학습하며 Q/K를 동결하지 않는다.

| 구조 | 역할 | 추가 경로 | 추가 학습 가중치 |
|---|---|---|---|
| titantpp_cnn_query_nonlinear_v1 | 위치 대조군 | 첫 block의 event Q만 | 1,024 |
| titantpp_cnn_value_nonlinear_v1 | 주후보 | 첫 block의 event V만 | 1,024 |

두 구조 모두 기존 depthwise CNN Q/K/V를 유지한다. 추가분기는 관측된 최근3개, 채널별64→256 grouped Conv(groups64,k3), GELU, 256→64 grouped Conv(k1), bias없음이다. temporal768+pointwise256이며 pointwise는0으로 초기화한다. 같은seed에서 두 후보 추가분기의 초기tensor가 같고 공통원형의 tensor/RNG/초기출력도 일치해야 한다. 미사용분기·dummy가중치는 두지 않는다. K에는 추가 비선형 경로가 없다. memory token을 CNN에 넣거나 memory연산을 바꾸지 않는다. padding/누락경계/미래·target차단은 기존관측규칙을 유지한다.

## 공통 과학 조건을 고정한다 — 현재 기준선
입력/log1p/손실/수량·시간head/메모리/optimizer/seed/학습예산/선택은 기존계약 그대로다. hidden64,batch128,max_seq_len84,AdamW lr0.001/weight_decay0.01,grad_clip1,max300/min40/patience40. 선택은 전체 fullValidation rawRMSE 최초 strict finite 최소epoch(동률첫epoch)다. RAF Train25,779/Validation6,690과 기존입력SHA를 유지한다. 기존조건/host기한을 연장하지 않는다.
신규fit은 두후보×42/52/62=6개다. 5090 기존Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f에서 최대3개 동시, Query3 다음 Value3으로 실행한다. 새root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_location_raf6_20261015_v1_5090 이다. 기존학습/결과와격리하고 Runtime·driver·인증은 변경하지 않는다.

## 구현과 검증을 통과한 뒤 학습한다 — 진행 중
모델·runner·합성검사와 운영도구를 새파일로 병렬 구현한다. 독립검토 뒤 paper_research의현재작업브랜치에서 해당파일만커밋하고 source139(부모137+모델/runner2)·입력·운영·승인·등록부SHA를봉인한다. 순서대로 nativeGPU/Runtime/입력/후보3개동시합성검증을통과해야학습한다. 실제신규fit6개만허용하며 중복시작·retry/resume는하지않는다.
검사는 초기동치,위치격리,실제pointwise/temporalgradient,인과성,target차단,padding·누락·비유한값처리,batch동치,모든활성가중치수,checkpoint교차로딩거부를포함한다. 이전CNN재평가를추가하지않는다.

## 고정 모델을 평가하고 판정한다 — 다음 작업
6개 실제worker/dispatcher/pipeline exit0,terminalbinarySHA,소유PID/GPU부재를확인후원본을새archive/manifest로회수한다. 최초Validation선택모델6개를nativeValidation으로재현하고전6개의대상·여섯지표·실제exit0가통과한후동일checkpoint Test6을평가한다. 기존CNN·외부·Channel결과는재실행하지않는다.
사용자기준은 Validation/Test 각각 전체RMSE·MAE의3seed평균이기존CNN과외부지표별최저평균보다모두엄격히낮아야통과다. 평균±표본SD(ddof1),모든seed별절대·상대차이,큰수량RMSE/MAE·전체/큰수량TimeNLL의악화도보존한다. 주대조 Value−Query는위치효과,저장Channel대비는용량3,072대1,024차이를포함한보조비교다. 기존CNN5080/cuDNN91900과신규5090/cuDNN92000차이는한계로명시한다.
Value의방법론이득은Value−Query와기존CNN대비를분리판정한다. Query만기준통과하면Value기여성공으로부르지않는다. 두후보모두사용자기준통과시확장후보는Validation RMSE,그다음Validation MAE,그다음고정arm문자순으로정한다. Test를보고checkpoint를재선택하지않는다.
통과한동일구조만Taxi5090/유휴확인된Intermittent5080에확장한다. 미달시결과와실패를보존하고오류분석뒤다음단일CNN가설을새계약으로진행한다. 반복Test참고에따른탐색적후속검증이며독립일반화·원인확정·통계유의성을주장하지않는다.
