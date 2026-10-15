# 루프3 실제 시작 후 운영 인계

**5090 RAF 신규 후보 3seed — 진행 중**
- 2026-10-16 01:03:22 KST 실제 관측에서 seed42/52/62 진행3, 저장3, best3/1/3이다. 이 시각을 새 관측 시각으로 바꾸지 않는다.
- 모델·운영·평가 코드 commit f2452c7c96ca83f58d9150be8e65da508dc8dcd4, 기존 CNN 학습·재평가0이다. 실제 native qualification·3동시 합성검증·worker/lease/GPU가 확인됐으므로 prepare/seal/deploy/start를 다시 하지 않는다.
- source139 원본 상속+신규2=source141, closure 4fce860e41ee716592abf168af8c3985859b74ea6d080357690124b19a88d361. 계약 raw c40a2921595bf60ac13d34d915f5a5b3fabe57db00b0eabf394d085c3c00ad99, canonical1309c9b3b418040bac2f72fa93616690ce60f66f1221ea461063135c16b7ec2e.

**다음 관측·종료 후 회수 — 다음 작업**
학습 retrieval/final_receipt.json complete가 아니고 ACTION_REQUIRED/미완료 retrieval이 없을 때만 프로젝트 root에서 회차당 한 번 실행한다. 별도 SSH·observer·반복 polling은 하지 않는다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_value_difference_raf3 --integrity-sha a9dfc9a6c1ed11784c6cbf144bd81b3fa13cb827c6e634b5f37e7de6f9baf3d6
```

latest의 실제 round/summary 경로와 rawSHA를 확인한다. 3개 worker·dispatcher·pipeline 실제exit0, 모든 terminalbinarySHA, 명시적 owner/GPU부재 검증 전 회수0이다. wrapper 자체OSexit는 직접 관측하지 않았다. 기한1792679647.915156을 상속하며 연장하지 않는다.

**고정 선택 checkpoint Validation3/Test3 — 학습 종료 대기**
EVALUATION_READY.json rawSHA8bd41bc56df8b41702391d146a7a02a6d90e43ba7c2a1fa627c54bc1617f3391와 내부11파일SHA를 먼저 확인한다. 학습 final_receipt complete와 모든 원본·종료SHA를 검증한 뒤에만 준비한다. 이미 유효한 evaluation_v1/bootstrap_integrity.json이 있으면 준비를 생략한다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_value_difference_raf3_test --mode prepare --training-integrity-sha a9dfc9a6c1ed11784c6cbf144bd81b3fa13cb827c6e634b5f37e7de6f9baf3d6 --registry-sha 57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49
```

반환된 실제 bootstrap SHA를 보존한 뒤 같은 모듈의 --mode advance --integrity-sha 인자에 전달해 한 번 실행한다. SHA를 추측하지 않는다. 전3 nativeValidation의 동일대상·여섯지표·exit0 gate를 통과한 뒤만 Test3이다. 기존CNN3·외부결과 재평가0, Test재선택0이다. 실패·기존attempt/claim/미완료bundle이 있으면 원문을 보존하고 자동 반복하지 않는다.

**판정·기록 — 평가 종료 후**
후보3+기존CNN3의 3seed 평균±표본SD·seed별 절대/상대차이 및 여섯 지표를 보존한다. Validation/Test 각각 RMSE·MAE 모두가 기존CNN과 같은 대상의 외부지표별최우수보다 엄격히 낮아야 통과다. 새 Test3 receipt의 candidate3/baseline3/comparison6/seed3/Validation3/nativeworkerexit0count6/pipelineexit0count2와 모든 로컬SHA를 확인한다. 통과한 같은구조만 Taxi5090와 실제유휴확인5080Intermittent로 확장한다. 미달이면 결과보존·진단 후 다음 단일 backbone 가설로 이어간다.

Notion의 Backbone Enhancement History에 시작까지 기록했고 읽어 재확인했다. 자동화 cnn-backbone-raf를 새 진입점·정확한SHA·상속기한으로 갱신했다. 현재 단계에서는 자동화를 유지한다. 기존원고·기존source/결과/claim/failure·다른자동화·공용Runtime/인증·삭제A100을 변경하지 않는다. 과거LevelHistory와 차분 아이디어 중복, 자유계수576→512·경계·optimizer 효과의 혼재, 기존CNN5080/신규5090 Runtime차이 및 반복Test 탐색 한계를 유지한다.
