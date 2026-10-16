# CNN·GRU36 최종 관측과 Intermittent3seed 완료 보고

기록시각 2026-10-14T10:25:39.282637+09:00. 원36 학습 완료36·진행0·대기0·미확정0. seed62 후속 평가 완료 2026-10-14T10:22:46.396777+09:00.

**원36 학습과 마지막 선택 모델을 확인한다 — 완료**

- 새5080 원관측2026-10-14 10:18:32 KST: Intermittent62 NONE 저장60/best20, CNN50/10, GRU103/63, FULL53/13. 최종 선택 binary4와 실제worker/dispatcher exit0, train_complete, owner/GPU부재 및 모든 terminal binarySHA 확인. 학습 잔여 ETA 없음.
- 원5080 Taxi·RAF24 종료 캐시는 원관측2026-10-13 08:27:20 KST, dispatcher 종료06:50:48 KST를 보존한다. 원5090 Intermittent42/52 8은 원관측2026-10-14 08:03:03 KST, 과학worker8 actualexit0 및 예정된 행정 dispatcher exit1(05:28:27 KST)을 구분한다. parent_original12_complete=false와 기존 claim/failure를 보존했다.
- 원 training pipeline wrapper 자체 OS exit는 직접 관측하지 않았다. 과거 terminal cache는 현재 GPU 전체의 idle 증명이 아니다. 고유36 ID로 집계한다. 각 조건 저장/best 상세표: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_components_seed62_handoff_20261013_v1/hourly_monitor_v2/rounds/9580ed7a0719430b89287230f5cbc9a9.md

**원본 회수와 native5080 평가를 검증한다 — 완료**

- seed62 정확60 원본/archive/manifest/모든 로컬SHA, 네 선택binary·tensor·source131·입력지문·Runtime 결속 통과. Validation4의 동일대상·여섯지표 재현과 실제exit0 뒤 Test4를 실행했으며 Test4 및 평가 pipeline actualexit0를 확인했다. 신규 결과46파일 archive/manifest/SHA 통과.
- 기존5090 Test8을 재실행 없이 재사용해 원12고유ID·3seed를 집계했다. 봉인395파일과 이전 결과86파일/신규46파일의 SHA 및 여섯지표 평균·표본SD 독립 로컬 검증 PASS. 새로운 학습·checkpoint 재선택·자동retry·원본 덮어쓰기0.

**최종 3seed의 성능과 주장 범위를 정리한다 — 완료**

| 구조 | Validation RMSE 평균 ± 표본SD | Test RMSE 평균 ± 표본SD | Test MAE 평균 ± 표본SD |
|---|---:|---:|---:|
| NONE | 1.742402 ± 0.028427 | 2.765756 ± 0.148489 | 1.152105 ± 0.073608 |
| CNN | 1.722965 ± 0.073001 | 2.616125 ± 0.078209 | 1.061696 ± 0.036434 |
| GRU | 1.754427 ± 0.043557 | 3.225446 ± 0.258528 | 1.363685 ± 0.170751 |
| FULL | 1.700536 ± 0.050552 | 2.914264 ± 0.238046 | 1.160457 ± 0.093891 |

Test 전체 raw 수량RMSE에서 FULL은 NONE 대비 +5.370%, CNN 대비 +11.396% 변화로 악화했다(각1/3seed만 개선). GRU 대비 -9.648%로 개선했으며3/3seed에서 개선했다. 직전 수량 반복RMSE 3.206282 대비 FULL 개선 9.108%. Validation에서 FULL의 평균RMSE 최저가 Test의 CNN 대비 우세로 이어지지 않았다.

큰 수량(q>187) 및 시간 점수를 포함한 여섯 지표와 seed별차이 모두 THREE_SEED_ANALYSIS.json/md에 보존했다. 42/52=5090/cuDNN92000,62=5080/cuDNN91900의 cross-seed GPU/Runtime 차이를 명시하고 시간은 host별로 분리했다. 기존 접근 이력이 있는 chronological Test이며 새 독립평가·통계유의성·모든 데이터셋/외부 신경망 우월성으로 해석하지 않는다.

**자동화 종료와 논문 통합 범위를 구분한다 — 다음 작업**

- 원36 학습 및 이번 Intermittent12 Test·3seed 결과의 최종 완료 조건을 충족했다. 이 보고를 남긴 뒤 cnn-gru36-checkpoint-eta 자동화 하나만 삭제한다. 다른 자동화·학습·claim·서버작업·Runtime·인증·A100은 변경하거나 조회하지 않는다.
- 기존 CNN+GRU54 본실험의 Taxi·RAF3seed Test는 이미 완료된 기준선이다. 이번 source131 구성요소 실험의 Taxi·RAF24 Test 완료와 혼동하지 않는다. 기존 결과를 재실행하지 않았다. 이번Intermittent12 완료를 원36전체 Test 완료로 표현하지 않는다. 다음 작업은 완료 결과의 실험군·source·평가split을 구분하여 논문 주장 범위를 맞추는 것이다. 현재 main.tex는 변경하지 않았다.

최종 결과: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_components_inter62_test_20261014_v1/THREE_SEED_ANALYSIS.md

주요 증적 SHA:

- analysis_sha256: `de3236b990fa65fbffcf44176604b085f17beec2b2d65d3fc7bb9e9bdf7d60cd`
- report_sha256: `9eed8f11b7788991dde060cb8c12092da2cebd9d31e742cd142e68149072c2b0`
- seed62_archive_sha256: `018830141078e71e2dfb252193341d06cd58d1e841c30e6b858065234e6570b3`
- seed62_manifest_sha256: `856fa4e6646bf9ccb8350970e4e6f55321949027544793799b3e950e98ef550b`
