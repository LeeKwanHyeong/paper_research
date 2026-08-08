# TitanTPP 최종 비교 Fast Track

## 연구 방향 판단 — 완료

- Intermittent를 가장 먼저 검증하는 방향이 맞다. 논문의 핵심 문제는 간헐 수요를 marked temporal event sequence로 재정의하고 RMTPP의 장기 이력 및 수량 mark 모델링 한계를 개선하는 것이다.
- 다만 정확한 표현은 “validation dataset”보다는 “primary task dataset” 또는 “핵심 실험 데이터셋”이다. validation split과 혼동하지 않도록 한다.
- Taxi는 long-sequence stress test, Instacart는 대규모 series와 일반화 검증으로 배치한다.

## 현재 기준선 — 완료

- 최종 계약은 RMTPP original, RMTPP matched, THP matched, TitanTPP의 primary 36 run과 Taxi V2 control 3 run이다.
- 현재 runner는 같은 epochs 경로 안에서는 resume하지만 e300과 e800 경로 사이의 continuation은 지원하지 않는다.
- 5080 단일 GPU 예상 시간은 Intermittent 약 9시간, Taxi 약 9시간, Instacart 약 106시간이다.

## 속도 우선 운영 원칙 — 확정

- 개인 연구 서버이므로 보안 점검, 전체 회귀 suite와 과도한 단위 테스트는 생략한다.
- 최소 검증은 39-run dry-run, continuation 1건, stale config 거부 1건, RMTPP/TitanTPP/THP CUDA e1 smoke만 수행한다.
- dataset hash, fixed split, run identity와 resume checkpoint 검증은 논문 결과의 유효성과 직접 연결되므로 유지한다.
- 오류가 나면 해당 run에서 멈추고 원인을 수정한 뒤 이어간다. 모델 또는 데이터 계산을 바꾼 수정이면 같은 조건의 앞선 run도 다시 실행하고, launcher/log/report만 고친 경우 기존 checkpoint를 유지한다.

## 실행 준비 — 다음 작업

- `ssh 5080`에서 프로젝트 경로, Git full SHA, `conda activate ai_env`, CUDA/GPU, 디스크와 세 fixed-split dataset hash를 확인한다.
- 39개 run을 하나의 frozen manifest로 만들고 queue 순서를 `Intermittent -> Taxi -> Instacart`로 고정한다.
- 각 dataset 안에서는 `RMTPP original -> RMTPP matched -> TitanTPP -> THP matched` 순서로 실행한다. Taxi V2 control은 네 primary 비교군 뒤에 둔다.

## 최소 구현과 smoke — 다음 작업

- `--resume-from-epochs 300`을 추가해 e800 run이 e300 checkpoint를 검증하고 epoch 301부터 시작하도록 한다.
- 완료 run skip, 실패 run resume와 ordered queue만 담당하는 launcher를 만든다.
- 로컬 최소 검증 후 5080의 `titan_final_fair` tmux에서 세 model class의 e1 CUDA smoke를 수행한다.

## Intermittent e300 — 다음 작업

- RMTPP original, RMTPP matched, TitanTPP 세 seed를 먼저 완료하면 약 6시간 안에 최소 논문 결과를 확보할 수 있다.
- THP matched 세 seed를 이어서 수행하면 약 9시간에 Intermittent 네 비교군 표가 완성된다.
- 핵심 판정은 original 대 matched RMTPP의 quantity formulation 효과와 matched RMTPP 대 TitanTPP의 encoder 효과다.

## Taxi e300 — 다음 작업

- 같은 core 순서로 네 primary 비교군을 실행하고 마지막에 Taxi V2 control을 수행한다.
- Taxi는 긴 sequence에서 recurrent RMTPP, Transformer THP, Titan encoder의 차이를 보여주는 stress test로 사용한다.
- V2 대 V3b는 main comparison과 분리해 supporting ablation으로 보고한다.

## Instacart e300 — 다음 작업

- 약 106시간이 필요한 병목이므로 RMTPP original, RMTPP matched, TitanTPP의 core 9 run을 먼저 실행한다.
- THP 3 run은 그 다음에 실행한다. 따라서 마감 직전에 중단되더라도 RMTPP/TitanTPP core 결과가 먼저 남는다.
- Instacart가 August 14까지 완주하지 못하면 초안에는 진행 중으로 표시하고, Intermittent와 Taxi 결과로 본문 구조를 완성한다.

## e800 연장 및 최종 평가 — 다음 작업

- 각 dataset e300 완료 직후 CPU에서 validation 수렴 판정을 생성하되 GPU queue는 다음 dataset으로 계속 진행한다.
- 전체 e300 뒤 연장 queue도 `Intermittent -> Taxi -> Instacart` 순서로 수행한다.
- 설정과 stopping policy가 고정된 뒤에만 held-out test를 한 번 열고 3-seed mean/std 최종 표를 작성한다.

## 논문 구성 — 확정

- 서사의 중심은 `RMTPP의 한계 -> quantity-aware problem formulation -> TitanTPP`로 둔다. THP는 Method의 주인공이 아니라 Transformer 비교군이다.
- 최소 실험표는 RMTPP original, RMTPP matched, TitanTPP 세 행이어야 한다. RMTPP와 TitanTPP만 두 행으로 비교하면 quantity formulation과 encoder 변화가 섞인다.
- THP matched를 추가하면 일반 Transformer 효과와 Titan 효과를 구분할 수 있으므로 최종본에는 강하게 권장한다.
- August 14 초안은 RMTPP/TitanTPP 결과를 먼저 채우고 THP 열과 관련 subsection은 미리 남겨 둘 수 있다.

