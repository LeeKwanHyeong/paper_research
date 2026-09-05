# B1 prior-prefix 출력 read 구현·실행 감사

작성일: 2026-09-05 KST. 작업 브랜치: `paper_research/codex/b1-prior-prefix-read`.

**현재 판단**

후보 계약과 구현, 로컬 검증, 5090 CUDA 정확성·비용 검증을 완료했다. 첫 실제 데이터 e1은 held-out 사전 제외 계약을 어겨 감사에서 중단됐으며, 통과한 e1은 없다. 공통 성능 개선은 아직 미판정이다. 이 문서의 CUDA 통과를 성능 채택으로 해석하지 않는다.

**완료 / 로컬 — 하나의 개입과 판정 계약**

- [계약 설명](../contracts/titans_mac_prior_prefix_v1.md)과 [기계 판정 JSON](../contracts/titans_mac_prior_prefix_v1.json)을 후보 학습 전에 동결했다. JSON SHA256은 `81f874eaf7568c0daf262d2cf49371c425a43352789ba227124d6d0b2152ebf7`이다.
- B1의 segment16 pre-attention read, causal attention, write 입력·상태 궤적·모멘텀과 기존 결합은 유지한다. 출력 read만 현재 관측 i를 쓰기 전 상태 `M_(i-1)`를 읽어 다음 사건 i+1을 예측한다.
- 추가 파라미터·gate·head·loss·dataset별 모델 분기는 없다. 독립 window reset과 padding/target write 금지를 유지한다.
- 동일 초기 파라미터·RNG와 기존 B1 경로의 bitwise replay를 요구한다. 활성 후보 전체의 초기 출력 동일성은 요구하지 않는다. H1에서는 읽을 이전 write가 없어 기존 출력과 일치하고 H2부터 이전 write의 gradient 경로가 열린다.
- 같은 학습 규칙을 세 데이터셋에 적용한다. Full e1은 6 runs, screening은 seed42의 6 runs, 조건부 확인은 seeds52/62의 12 runs다.
- Frozen T0/RMTPP/THP의 cap300에 맞춰 새 B1과 후보 모두 cap300으로 fresh 학습한다. 과거 cap30 B1을 paired 대조군으로 합치지 않는다.

**완료 / 로컬 — 구현과 검사**

- 후보 구현: `5206e46ea21a07aa49e2b6b773154fee49cb5aec`.
- CPU batch permutation 검사 수정: `a2fda0efe23e8c77e0c9333f64da3b9853e09bc5`.
- 독립 프로세스 비용 측정: 실행 source `add02baae27c25da02008873c578f11926f8d125`.
- 최초 로컬 통합 207 passed / CUDA 4 skipped. 이후 변경된 검사 suite 86 passed / CUDA 3 skipped, 비용 측정기·launcher 66 passed / 0 skipped로 검증했다. 원본 결과와 source hash는 [증적 디렉터리](../results/titans_mac_prior_prefix_20260905/README.md)에 보존했다.
- Model/AdamW/RNG 복원, actual quantity/time loss gradient, 인과성·미래 target·padding, state/series reset, checkpoint route 격리와 잘못된 실험 구성 거부를 검사했다.

**완료 / 5090 — 정확성과 비용 gate**

- 실행 서버: `RTX5090-server` / SSH alias `5090`. Python3.12.13, torch2.11.0+cu130, CUDA13.0, Polars1.39.3.
- 격리 source: `/home/leekwanhyeong/workspace/paper_research_prior_prefix_add02ba_5090`.
- 커밋 archive의 389개 파일 hash와 기존 세 데이터·split manifest checksum을 확인했다. 실제 데이터는 기존 프로젝트 경로에서 읽는다.
- 최종 CUDA 필수 검사는 **89 passed / 0 failed / 0 skipped**다. Eager/compiled forward·state·gradient, AdamW step, 실제 CUDA 저장/복원 후 다음 step 일치를 포함한다.
- 3 models × H16/64/255의 9개 측정을 각각 새 subprocess에서 수행했다. Cold compile+첫 step, warmup3, 측정10을 분리했고, 각 모델의 처음·warmup·측정 peak 중 최댓값을 사용했다.

| 이력 H | 후보/B1 step 시간 | 후보/B1 peak 메모리 | 후보 peak GiB | 후보/T0 step 시간 |
|---:|---:|---:|---:|---:|
|16|1.5501|1.2092|1.1704|6.7610|
|64|1.3932|1.0550|2.4670|17.0950|
|255|1.2737|1.0341|7.7058|22.1338|

원수치 기준으로 시간≤2배, 메모리≤1.5배 및≤28GiB를 모두 통과했다. T0 대비 6.76–22.13배 step 비용은 여전히 크다. 이전 B1의 T0 대비 비용 실패를 뒤집거나 효율이 개선됐다고 주장하지 않는다.

첫 CUDA 시도에서는 CPU permutation bitwise 검사가 실패했고, 기존 B1에도 있는 반올림임을 확인해 이미 동결된 CPU tolerance로 검사만 수정했다. 두 번째 시도에서는 cuBLAS workspace가 남아 독립 비용 측정 시작 검사가 실패했다. 측정기를 subprocess로 분리했으며, 세 시도 동안 모델 코드·JSON 계약·성능/비용 문턱은 바꾸지 않았다. 실패 증적도 보존했다.

**차단됨 / 5090 — full e1 정상 실행 검증; 로컬 로딩 분기 수정 중**

- 같은 source·seed42의 B1/후보를 Taxi → Intermittent → Instacart 순서로 전체 train/validation에서 실행한다.
- 데이터별 train targets는 38,393 / 393,824 / 1,991,192, validation targets는 8,268 / 86,285 / 503,733이다.
- 6 runs 모두 full coverage, finite 계산, 실제 writer optimizer 활동, 동일 인터페이스, checkpoint·optimizer 저장/복원 감사를 통과해야 screening으로 진행한다.
- e1 수치는 성능 채택에 사용하지 않는다.

**후속 작업 / 5090 — 공통 성능 판정**

- 고정 후보 seed42의 최대e300/min40/pat40을 fresh B1과 짝지어 실행한다.
- B1 추가 효과, T0 기존 body MAE 5% 기준, 세 데이터셋 모두의 strict raw RMSE 개선, RMTPP/THP 대비 RMSE 우위를 각각 판정한다.
- B1 추가 효과와 T0의 두 기준이 세 데이터셋 모두에서 통과할 때만 seeds52/62로 확장한다. 3-seed 평균·최소2/3 RMSE 개선 방향·개별 seed time/tail 보존을 확인한다.
- 모든 단계는 검증된 선행 proof와 hash를 요구한다. 실행 오류나 성능 gate 실패에서는 해당 증거를 보존하고 중단한다. 자동 재시도·checkpoint resume는 하지 않는다.
- Held-out 평가, v0.7 최종 T0 교체, `paper_research/master` 병합·push는 수행하지 않았다.

**실행 중 작업의 확인 경로**

- 시작 확인 시점: 2026-09-05T11:37:37.319541+00:00 (UTC). Taxi B1 seed42 e1의 실행을 확인했다.
- 학습 source는 `add02ba`, 제어 스크립트의 커밋은 `2842ecc`다. 제어 스크립트는 frozen source 밖에서 각 phase launcher를 명시 호출한다.
- tmux: `prior_prefix_training_add02ba`. 전체 상태: `/home/leekwanhyeong/workspace/paper_research_prior_prefix_add02ba_5090/outputs/full_validation/status.json`.
- `training_launch_snapshot/`은 시작 시점 복사본이다. 최신 상태로 취급하지 말고 5090의 원본 status와 비교한다.
- 실행은 유한 job으로 이어지며 e1/screening/confirm 완료 증거에 따라 통과·미달·실행 오류를 구분한다. 현재 snapshot에서 e1은 진행 중, screening/confirm/held-out은 미실행이다.

**최신 상태 정정 — 첫 실제 e1 감사 실패**

`add02ba`의 첫 Taxi B1 학습은 1 epoch를 완료했지만 test 8,327행이 원본 frame에 적재되어 e1 감사가 FAIL했다. 두 모델이 기존 main의 사전 제외 분기에 누락된 것이 원인이다. 실제 평가 artifact는 validation-only이나, 사전 제외 계약 실패를 통과 처리하지 않는다. Controller는 중단했고 candidate e1·screening·confirm은 미실행이다. `e1_add02ba_rejected/`에 원본을 보존했다. 로딩 분기와 실제 진입점 테스트를 수정한 뒤 새 source로 다시 CUDA/full e1을 검증한다.
