# B1 prior-prefix 출력 read 구현·실행 감사

작성일: 2026-09-05 KST. 작업 브랜치: `paper_research/codex/b1-prior-prefix-read`.

**현재 상태**

후보 계약·구현·로컬 검사와 수정된 source의 5090 CUDA 정확성·비용 검증을 완료했다. 새 source의 전체 데이터 e1 6개가 모두 정상 감사를 통과했고 seed42 screening을 실행 중이다. Taxi B1은 51 epoch에서 실행 감사를 통과했으며 현재 Taxi 후보를 학습 중이다. 공통 성능 개선은 아직 미판정이며, 추가 seed·held-out 평가는 시작하지 않았다.

**완료 / 로컬 — 한 후보와 서로 다른 판정 계약**

- [설명 계약](../contracts/titans_mac_prior_prefix_v1.md), [기계 판정 JSON](../contracts/titans_mac_prior_prefix_v1.json)을 후보 성능 학습 전에 동결했다. JSON SHA256은 `81f874eaf7568c0daf262d2cf49371c425a43352789ba227124d6d0b2152ebf7`이며 수정 과정에서도 유지됐다.
- B1의 segment16 pre-attention read·causal attention·write 입력과 메모리/모멘텀 궤적을 유지한다. 출력 read만 현재 관측 i의 write 전 상태 `M_(i-1)`를 읽어 다음 사건 i+1을 예측한다.
- 추가 파라미터·gate·head·loss·dataset별 모델 분기는 없다. 독립 window reset과 padding/target write 금지를 유지한다.
- 동일 초기 파라미터·RNG와 기존 B1 경로의 bitwise replay가 계약이다. 활성 후보 전체의 초기 출력 동일성은 요구하지 않는다. H1에서는 이전 write가 없어 기존 출력과 같고 H2부터 이전 write의 gradient 경로가 열린다.
- B1 추가 효과, T0의 body MAE 5% 개선 및 guardrail, 세 데이터셋 모두의 strict raw RMSE 개선, RMTPP/THP 대비 RMSE 우위를 각각 판정한다. 하나의 통과를 다른 주장으로 바꾸지 않는다.
- 기존 T0/RMTPP/THP의 effective cap300에 맞춰 새 B1/후보를 모두 cap300으로 fresh 학습한다. 과거 cap30 B1과 합치지 않는다. 고정된 27개 역사적 reference는 fresh paired rerun으로 표현하지 않는다.

**완료 / 로컬 — 구현과 검사**

- 실행 source: `c21aea563aeaa7f82a8814552594dbc77b8da458`. 최초 후보 구현은 `5206e46`이며 이후 모델 수학은 유지하고 검사·비용 측정·입력 범위 통제를 수정했다.
- 최초 통합 207 passed / CUDA 4 skipped, 최종 입력 범위 관련 통합 163 passed / CUDA 3 skipped. 수정된 실제 CLI/main 진입점 9개 검사는 세 데이터셋의 B1 단독·혼합·후보 경로에서 test 사전 제외와 동일 train/validation 입력·통계를 검증했다.
- Model/AdamW/RNG 복원, actual quantity/time loss gradient, 인과성·미래 target·padding, state/series reset, checkpoint route 격리와 잘못된 실험 구성 거부를 검사했다.
- 후속 실행 제어는 frozen source 밖의 독립 스크립트다. 현재 source·manifest·contract·launcher hash에 고정됐으며 실제 archive 결속을 포함한 22개 테스트를 통과했다.

**완료 / 5090 — 현재 source의 정확성과 비용**

- 서버: `RTX5090-server` / SSH alias `5090`. Python3.12.13, torch2.11.0+cu130, CUDA13.0, Polars1.39.3.
- 격리 source: `/home/leekwanhyeong/workspace/paper_research_prior_prefix_c21aea5_5090`.
- 커밋 archive의 390개 파일과 세 데이터·split manifest checksum을 확인했다. 기존 Runtime·서비스·데이터는 수정하지 않았다.
- **필수 검사 98 passed / 0 failed / 0 skipped**: 기존 계약 89 + 실제 입력 진입점 9. Eager/compiled forward·state·gradient, AdamW step, CUDA 저장/복원 후 다음 step 검사를 포함한다.
- 3 models × H16/64/255의 9개 측정을 각각 새 subprocess에서 수행했다. Cold compile+첫 step, warmup3, 측정10을 분리했고 각 모델의 cold/warmup/측정 peak 중 최댓값을 사용했다.

| 이력 H | 후보/B1 step | 후보/B1 peak | 후보 peak GiB | 후보/T0 step |
|---:|---:|---:|---:|---:|
| 16 | 1.4130 | 1.1256 | 1.0738 | 6.5038 |
| 64 | 1.2833 | 1.0550 | 2.4670 | 15.0537 |
| 255 | 1.2765 | 1.0211 | 7.6096 | 21.3170 |

원수치 기준 시간≤2배 B1, peak≤1.5배 B1 및≤28GiB를 모두 통과했다. 다만 T0 대비 step 시간이 6.50–21.32배다. 이 연구 비용 gate 통과를 T0 수준 효율이나 성능 개선으로 표현하지 않는다. [현재 CUDA proof와 원수치](../results/titans_mac_prior_prefix_20260905/cuda_c21aea5/status.json).

**완료 / 5090 — full e1 정상 실행 검증**

- 새 source·seed42의 B1/후보를 Taxi → Intermittent → Instacart 순서로 실행했으며 6개 모두 정상 감사를 통과했다. Taxi의 실제 launch에는 train38,524 / validation8,268만 포함됐고, 각 실행은 train target38,393 / validation target8,268을 처리했다.
- 각 dataset의 처리 대상 train targets는 38,393 / 393,824 / 1,991,192, validation targets는 8,268 / 86,285 / 503,733이다. 원본 행 수와 history가 필요한 target 수는 다르다.
- 6 runs 모두 full coverage, finite 계산, 실제 writer optimizer 활동, 동일 인터페이스, checkpoint·optimizer 복원 감사를 통과했다. e1 수치로 성능을 채택하지 않는다. 완료 시점은 2026-09-05T13:07:25 UTC이며 완료 proof SHA256 `a395f22dfd6b88ef25d569a2747c06aba39e83f72915b24c5fa63d54f98eb7c2`가 controller의 다음 단계 입력과 일치한다. 30개 증적 파일의 원격 hash를 다시 확인했다.

**진행 중 / 5090 — 고정 후보 seed42 screening**

- e1 통과 후 seed42 최대e300/min40/pat40의 6 runs를 시작했다. Taxi B1은 51 epoch에서 정상 종료·실행 감사 PASS, Taxi 후보를 학습 중이다. 수집 시점에 완료 1/6이며 데이터셋 간 공통 성능 판정은 미완료다. 완료된 Taxi B1의 5개 증적 파일 hash도 다시 확인했다.
- B1 추가 효과와 T0의 body/RMSE 조건이 세 데이터셋 모두에서 통과해야 seeds52/62의 12 runs로 확장한다. Controller hash와 이 진입·중단 조건을 읽기 전용으로 재확인했다. 기존 프로세스를 유지하며 중복 실행은 시작하지 않았다.
- 3-seed 산술평균·최소2/3 RMSE 개선 방향·개별 seed time/tail 보존을 확인한다. RMTPP/THP 우위는 별도 판정이다.
- 제어 스크립트는 각 phase를 명시 호출하고 선행 proof·모든 evidence hash를 재검증한다. 실행 오류와 성능 미달을 구분하며 실패 시 중단한다. 재시도·checkpoint resume·덮어쓰기는 하지 않는다.
- tmux: `prior_prefix_training_c21aea5`. 최신 상태는 `/home/leekwanhyeong/workspace/paper_research_prior_prefix_c21aea5_5090/outputs/full_validation/status.json` 및 각 phase의 `status.json`으로 확인한다.

**보존한 실패와 실제 데이터 접근 범위**

1. 첫 CUDA 시도의 CPU permutation bitwise 검사는 기존 B1에도 있는 반올림을 실패로 처리했다. 이미 동결된 CPU tolerance를 해당 검사에 적용했고 모델 코드·성능 문턱은 바꾸지 않았다.
2. 다음 비용 측정은 cuBLAS workspace가 cleanup 후 남아 중단됐다. 각 행을 독립 subprocess로 격리해 해결했으며 원본 실패와 합성 원인 진단을 보존했다.
3. `add02ba`의 첫 Taxi B1 e1에서는 기존 main loader 분기에 두 MAC 모델이 빠져 test8,327행도 원본/파생 frame과 series list에 적재됐다. 해당 실행은 감사에서 FAIL로 기각했고 controller가 중단됐다. 정상 e1·대조군·성능 채택 증거로 재사용하지 않는다.

코드와 고정 chronological split 계약상 test target의 학습·평가·checkpoint 선택 경로는 확인되지 않았다. 실제 test 행을 다시 조회하거나 평가하지 않았으며, test 행의 적재·전처리 위반 자체는 명확히 기록한다. [독립 데이터 범위 감사](../results/titans_mac_prior_prefix_20260905/e1_add02ba_rejected/data_scope_audit.md). `c21aea5`에서 두 모델을 적재 전 제외 분기에 넣고 실제 main 회귀 검사와 5090 검증을 완료했다.

현재 e1은 6/6 완료, seed42 screening은 1/6 완료이며 나머지 실행과 성능 판정은 미완료다. Held-out 평가, v0.7 최종 T0 교체, `paper_research/master` 병합·push는 수행하지 않았다. 모든 로컬 상태 복사본은 수집 시점 snapshot이며 최신 원격 상태와 구분한다.

최신 로컬 상태 수집 시점: 2026-09-05T14:06:26.809356+00:00 UTC. [상태와 검증 증적](../results/titans_mac_prior_prefix_20260905/training_c21aea5_screening_snapshot_20260905T140626Z/snapshot_audit.json)에 파일 hash와 추가 증적 수집 시각을 기록했다. 이전 `training_c21aea5_snapshot/`은 e1 2/6 당시의 이력으로 보존한다. 원격 작업은 계속 진행되며, 제어 스크립트 커밋은 `fbea9ce`, 학습 source는 `c21aea5`로 구분한다.
