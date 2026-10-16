# 신규 Taxi·RAF24 평가와 캠페인별 결과 정리

## Material Passport

사용자 승인: 2026-10-14 이 채팅의 “이거를 그대로 진행하자. 승인할게”. 범위는 원source131 구성요소36 중 original5080 Taxi·RAF24 선택모델의 원본회수, native5080 입력·tensor 감사, Validation24 재현 gate 뒤 Test24, 이미 완료된 Intermittent12 재사용 및 캠페인별 최종 결과표다. 새 학습, Test 재선택, 기존 학습·claim 변경, 원고 편집은 없다. Side Chat에서 이전 Test를 요청했다는 사용자 진술과 기존 Test 접근 이력이 있으므로 새 독립 평가로 표현하지 않는다. 이전 Side Chat 결과의 존재 여부 검색은 사용자 요청에 따라 종료했다.

**선택 모델·입력·Runtime을 고정한다 — 완료**

- original5080 실제24 worker 종료0와 dispatcher 종료0/train_complete 증적에서 원본360개를 읽기 전용 회수했다. 실제 byte SHA와 선택 tensor SHA24, saved history의 최초 strict finite 전체 Validation raw 수량 RMSE 최소 epoch가 일치했다. 종료 모델은 감사용으로 함께 보존했다.
- source131 고정, Taxi full/train bytes동일, RAF 전체 부모의 Train·Validation이 학습 export와 정확히 같다. Train 통계·time 관측·Validation 대상 원본지문도 일치했다. native audit는 모델예측0/실제 종료0이다.
- native5080 ai_env Python3.12.13/torch2.11.0+cu130/CUDA13/cuDNN91900/GPU UUID GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6 고정. source/Runtime을 변경하지 않았다. 과거 종료캐시는 현재GPU idle 증명이 아니며 새 실행마다 고정GPU 소유 여부를 확인한다.
- 첫 회수 연결부는 job_ids 필드 누락으로 KeyError/transport1이었다. 당시 empty archive/오류 원문을 보존했고 GPU·모델·Test실행0이다. 로컬 fixture검증 후 schema를 맞춘 별도 명명 회수에서 archive/360SHA검증을 완료했다. 과학 평가 retry나 claim삭제는 없었다.

**Validation24 재현 뒤 같은 선택 모델의 Test24를 평가한다 — 진행 중 / 승인 완료**

- 전체 raw 수량 RMSE, 전체 MAE, 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL을 같은 checkpoint에서 계산한다. 큰 수량은 Train 기준 Taxi q>3449/RAF q>200을 유지한다. 같은 대상의 직전수량·관측이력 평균 기준선 TimeNLL은 N/A다.
- 정확24개의 Validation 지표·대상·parameter불변·native Runtime·actual workerexit0가 확인된 gate24만 Test를 허용한다. 평가를 통해 재학습/보정/선택은 하지 않는다.
- 합성 구조식별·target차단·batch일치·파라미터불변·지표손계산 PASS, 집중 software36개와 운영fixture4개 및 독립QA PASS. 실제 Test를 도구디버깅에 사용하지 않았다.
- 새 격리 root에서 48native worker를 하나씩 감독한다. 12시간permit/유한SSH시간/실제supervised childexit·stdout/stderr bytes/SHA를 보존한다. 실패시 자동retry0이며 다른 프로세스를 종료하지 않는다.

**신규 구성요소36과 기존 캠페인 결과를 별도로 정리한다 — 다음 작업**

- 신규 Taxi·RAF24 + 이미 완료된 Inter12(source131)만 고유36ID로 집계한다. dataset/arm의 3seed 평균·표본SD(ddof1), seed별 FULL−NONE/CNN/GRU 차이, Validation→Test 반전과 예외를 모두 보고한다. active parameter수 차이가 있으므로 동일용량실험으로 부르지 않는다.
- Inter42/52=5090/cuDNN92000, Inter62=5080/cuDNN91900의 차이를 명시한다. 시간은host별분리한다.
- 기존 대표9(source117/123), 보정27(source129), Instacart9(source126), 신규구성36(source131)은 각 표의 provenance/split을 유지한다. 서로 다른 FULL을 섞거나 좋은 값으로 교체하지 않는다. 외부 비교는 common-head/adapted encoder 평가이며 일반적 SOTA 우월성으로 확대하지 않는다.
- n=3 결과는 기술통계이며 통계적 유의성·독립 Test·모든데이터셋의 보편우월성으로 해석하지 않는다. 교수님 보고용 main.reviewed.tex 및 PAKDD12페이지 원고 반영은 별도 후속 작업이다.

