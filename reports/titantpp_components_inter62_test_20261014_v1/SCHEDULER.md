# seed62 학습 종료 후 Test4와 Intermittent3seed 집계

## Material Passport

사용자 원문: “그럼 seed62 FULL은 지금 어디까지 진행됐고 얼마나 남았어? 끝나면 네 구조 Test까지 진행해서 3seed 결과로 다시 정리하자.” 승인 범위는 기존 원36 조건의 Intermittent62 네 구조에 대한 고정 선택 모델의 Validation 재현·Test 평가와, 이미 완료한 Intermittent42/52 Test8을 재사용한 원12개 3seed 집계다. 새 학습·checkpoint 재선택·기존 원본/claim 변경·main.tex 변경은 하지 않는다. 기존 Test 접근 이력을 보존한다.

**FULL62의 실제 종료를 확인한다 — 외부 작업 대기**

- 기존 봉인 observer v2를 매시간 회차당 한 번 실행한다. 추가 status SSH나 polling은 하지 않는다.
- 학습 종료가 미확정이면 아래 후속 helper는 원격0·회수0·Test0로 대기한다. 현재 관측2026-10-14 08:55:39 KST는 저장45/best13, 최근10 epoch 중앙값539.725초이며 best 미갱신 조건53epoch까지1시간11분58초,10:07:37 KST다. 300epoch 시나리오는38시간13분50초로 현재 조건기한 안에 보장되지 않는다.

**종료 원본과 native5080을 결속한다 — 다음 작업**

- all_training_terminal_verified=true 및 handoff5080의 실제4worker/dispatcher exit0·train_complete·owner/GPU부재·qualification/permit bridge·모든terminal binarySHA가 확인돼야 한다. 원래 pipeline wrapper 자체 OS exit를 직접 관측하지 않았다는 사실을 유지한다.
- 새helper는 바뀌지 않은 source131·input·운영 코드와 기존 Test8 증적395파일을 먼저 stdlib SHA로 확인한다. seal mismatch이면 원격0·덮어쓰기0다.
- 원 training root에서 읽기 전용으로 정확60원본을 한 번 회수하고 archive/manifest/로컬모든SHA를 확인한다. native5080의 실제 tensor4·Runtime·고정 Train 통계·Validation/Test 입력지문을 새 격리 root에서 감사한다.

**Validation4 후 Test4를 실행한다 — 다음 작업 / 실행 승인 완료**

- 주 지표 전체 raw 수량RMSE의 최초strictfinite 최소 Validation epoch를 사용한다. 동일binary에서 전체MAE, q>187 큰 수량RMSE/MAE, 전체/큰 수량 recorded positive-integer TimeNLL까지 여섯 지표를 계산한다.
- 실제4개 native Validation 재현·대상지문·worker exit0이 모두 확인된 뒤만 gate4를 생성하고 Test4를 순차 평가한다. 오류·timeout·소유권 미확정이면 중단하고 claim/failure를 보존한다. 실제 Test로 도구를 디버깅하거나 자동 retry/resume하지 않는다. Timeout stdout/stderr 부분원문과 종료 미관측을 보존한다.
- 고정5080 Python/Runtime/cuDNN91900/GPU UUID를 유지한다. pipeline은 자기 소유 평가 자식만 정리하며 다른 작업에 신호를 보내지 않는다.

**원12개 3seed 결과를 정리한다 — 다음 작업**

- 기존5090 seed42/52 Test8은 재실행하지 않는다. 원12고유ID를 중복 없이 결속해 3seed 평균·표본SD와 seed별차이·같은대상의직전수량/이력평균 기준선을 보고한다. 비학습 기준선 TimeNLL은 N/A다.
- 42/52=5090/cuDNN92000,62=5080/cuDNN91900의 cross-seed GPU/Runtime 차이를 명시하고 시간을 혼합하지 않는다. 불리한수량/큰수량/시간 결과를 모두 보존한다. 새독립평가·통계유의성·모든데이터셋/외부신경망우월성으로 확대하지 않는다.

## 회차 실행

프로젝트 root에서 기존 observer를 회차당 정확히 한 번 실행하고 그 회차 JSON 경로/stdlib rawSHA를 확인한다. 이후 다음 helper를 회차당 한 번 실행한다(각 `<...>` 값은 같은 회차의 실제 출력에서 가져온다).

```text
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_components_inter62_test --round <same_round_absolute_path> --round-sha <same_round_rawSHA> --integrity-sha bc45e4e95fdeedc451d76e4a21416a72ff24bcd5f6091e919dfd4f5b0e275b23
```

후속 bootstrap rawSHA는 bc45e4e95fdeedc451d76e4a21416a72ff24bcd5f6091e919dfd4f5b0e275b23, 승인 rawSHA는397d8d89ac9bb3f032a311b219c7ad4fa18e39b246f60f4b9083ccb972274b0b다. 새 local bundle은 search_artifacts/titantpp_components_inter62_test_20261014_v1이고 native 격리 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter62_test_20261014_v1_5080이다.

기존 학습 관측 봉인은 4f2ab78970d862cf146365de6a75999950964c2d809c27183d1a970f37fedb40이며 원36/handoff/source131/운영/claim 모두 유지한다. 이전 관측 지침의 Test·원본회수0은 최신 사용자의 이exact4 승인 범위에서만 대체한다.

기존 cnn-gru36-checkpoint-eta 자동화는 원36 학습 완료만으로 삭제하지 않는다. 새final_receipt.status=complete·원12/3seed·선택/지표/actualexit/SHA 및 THREE_SEED_ANALYSIS.json/md를 확인하고 최종 보고한 뒤 이자동화 하나만 삭제한다. partial/failed/미확정이면 유지하고 필요한 사용자 조치를 보고한다. 다른 자동화·Pod·서버 작업·Runtime·인증은 변경하지 않는다.
