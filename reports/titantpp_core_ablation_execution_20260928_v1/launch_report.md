# TitanTPP 핵심 비교 36조건 — 2026-09-28

## 승인과 실행 범위
사용자의 “오케이 그렇게 진행하자 승인할게”, “5번까지 진행하자”에 따라 로컬 취합 → 계약 → 구현 → 검증 → 두 서버 학습 시작까지 진행한다. 기존 100회/480 GPU시간 제안은 이번 실행 범위가 아니다.

| 역할 | 조건 | 처리 |
|---|---|---|
| B·Full | 3 데이터 × 3 seed × 2 모델 = 18조건 | 완료된 학습과 selected/last 36개 체크포인트 재사용 |
| 이력 MLP·수준만·변화만·정적 검색 제거 | 3 데이터 × 3 seed × 4 모델 = 36조건 | 새로운 학습과 selected/last 72회 validation replay |
| 5080 | Taxi → Intermittent, 각 seed42 →52 →62 | 24조건, 조건별 독립 프로세스 |
| 5090 | Instacart, seed42 →52 →62 | 12조건, 조건별 독립 프로세스 |

각 dataset/seed 안에서는 MLP → 수준만 → 변화만 → 정적 검색 제거 순서다. B·Full의 추가 학습은 없다. RMTPP·THP·NHP·SAHP의 이전 결과와 이번 네 가지 구조 비교를 혼동하지 않는다.

## 고정한 비교 질문
- Full 대비 동일 추가 파라미터(6144개)의 이력 MLP: 구조의 효과와 단순 용량 증가를 구분한다. 연산량까지 동일하다는 뜻은 아니다.
- 수준만 / 변화만: 각 정보 제거의 영향. 각각 추가 4096개로 용량도 줄어드는 한계를 보고한다.
- 정적 검색 제거: Full의 이력 보완을 유지하고 마지막 64×64 정적 prototype 검색만 제거한다. encoder의 persistent memory16은 유지한다.
- 메모리와 이력 보완의 상호작용은 이번 조건만으로 확정하지 않는다. 불리한 seed와 지표를 모두 남긴다.

## 학습과 평가 규칙
max300 / min40 / patience40, batch128, AdamW(lr0.001, wd0.01), clip1. 원본과 동일 데이터·loader·관측 시간 likelihood·수량 log-MSE·train-only 초기화. 모든 selected 지표는 동일한 earliest strict validation raw-RMSE 최소 체크포인트에서 계산한다. 같은 seed의 실제 배치 순서를 B/Full과 겹치는 epoch 구간 전체에서 비교한다. 마지막 epoch도 별도로 재평가한다.

held-out 평가·추가 seed·외부 비교 모델·튜닝은 포함하지 않는다. raw data는 기존 경로에서 checksum 검증 후 train/validation 필터를 적용한다. test 예측이나 성능은 열람하지 않는다. CPU 테스트에는 합성 데이터만 사용했다.

## 새 자원 상한
- 두 서버 각각 공통 시작시각부터 최대120시간, 합산 최대240 GPU시간. 1서버 1작업.
- 종료 상한: **2026-10-03T00:18:06.667266+09:00 (KST)**. 이 시각은 완료 예상이 아니라 강제 종료 한도다.
- 조건당 학습과 replay 합계36시간. 서버당 qualification1800초/60 synthetic optimizer updates.
- 서버별 output16GiB 이하, 파일64MiB 이하, 잔여 디스크20GiB 이상.
- 시작 시 free VRAM2GiB 이상. 다른 GPU 프로세스는 관찰만 한다.
- 5분 무진척 stack 진단, 30분 무진척이면 해당 소유 worker 중단. 실패 시 후속 조건도 중단하고 자동 retry/resume하지 않는다.
- 기존 Full 실측 epoch를 그대로 가정한 조건부 추정은 5080 약48.3시간, 5090 약55.1시간(준비/replay 제외). 새 모델의 속도와 조기 종료 시점에 따라 달라진다. 전 조건300epoch 완료를 보장하는 예산이 아니다.

## 검증 증적과 현재 상태
- `baseline_audit.json`: 회수한18조건·36체크포인트의 byte SHA/텐서 SHA·선택 epoch·노출·기록된 초기화·strict state loading 검증.
- `verification_receipt.json`: 핵심·기존 회귀140개 통과, interface 정리 후 핵심11개 추가 통과.
- `native_cpu_preflight_*.json`: 서버의 기존 Runtime에서 초기화 해시 재구성 대조. 로컬 Torch2.14와 서버2.11의 초기화 해시는 같다고 가정하지 않는다.
- `frozen_execution/execution_contract.json`: canonical SHA `eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d`.
- `execution_receipt.json`은 준비 시점 기록이며 실제 시작 여부는 `launch_receipt_*.json` 및 `monitor/UTC/host/`에서 확인한다.
- 코드 대상: paper_research / codex/hard-lmm-causal-qkv. 기존 수정사항 보존. 커밋·Push 없음.
- ai_env와 공용 프로젝트 소스는 변경하지 않는다. 각 서버 `paper_research_experiment_artifacts/titantpp_core_ablation_HOST_20260928_v1/source`에 동결 사본을 둔다.

## 남은 작업 순서
1. **승인 범위의 학습과 endpoint replay 완료 — 진행 상태는 monitor 증적 기준**: 각 서버의 해당36조건만 수행한다. 오류 조건은 기록하고 자동 재시도하지 않는다.
2. **결과 회수와 동일 비교 감사 — 다음 작업**: 원본을 보존하고 소형 이력·노출·paired 증적을 로컬에 취합한다. 초기화·배치 prefix·조기 종료·same-checkpoint 지표·실측 비용을 감사한다.
3. **논문 주장 판단 — 다음 작업**: B/Full18+신규36의 총54조건을 중복 없이 구성하고 3seed 평균·표본표준편차·수량/시간/구간별 반례를 제시한다. 실행 성공과 성능 기준 통과는 별개다.
4. **외부 비교 및 최종 test — 별도 범위 확정·승인 필요**: 이번 승인에서 외부 모델과 held-out 평가로 자동 확대하지 않는다.

## 1~5단계 완료 확인 (2026-09-28 00:22 KST)
- 1~4단계 완료. 두 서버 native 초기화 대조와 CUDA60updates qualification 통과.
- 5단계 학습 시작 완료. 5080: Taxi seed42 MLP 2epoch/600steps 완료. 5090: Instacart seed42 MLP 첫 epoch 학습 중(완료epoch0, 실제 batch progress 및 소유 GPU PID 확인).
- 이번 요청은 학습 시작까지 완료됐고, 총36조건의 학습·replay 완료와 결과 해석은 진행 이후의 작업이다.
- 확인 파일: `completion_of_launch_steps.json`, `monitor/20260927T152153Z/5080/`, `monitor/20260927T152200Z/5090/`.
- 최초 배포 사전 CPU import 점검에서 빈 `source/sample_data` 경로 표식 누락을 발견해, GPU qualification 시작 전에 빈 디렉터리만 보완했다. 동결 소스·계약·마감은 변경하지 않았다. 전후 기록은 `preparation/pre_layout_native_cpu_*`와 `deployment_layout_receipt.json`에 보존했다.
- 기존 정지 RMTPP 복구 이력은 별도 이전 실험 기록이며, 이번 학습을 재시작한 이력은 없다.
