# AICA 2026 TitanTPP 최종 공정 비교 작업 순서

## 현재 기준선 — 완료

- `paper_research`의 long-epoch runner는 각 epoch마다 `last_epoch_state.pt`를 저장하고 같은 run path에서는 model, optimizer, RNG, loader generator, history와 best checkpoint 상태를 재개할 수 있다.
- 현재 run path에는 `epochs_300` 또는 `epochs_800`이 포함되므로 e300 명령을 e800 명령으로 바꾸는 것만으로는 이어서 학습되지 않는다.
- 최종 비교 계약은 RMTPP original, RMTPP matched, THP matched, TitanTPP의 네 primary 비교군을 요구한다. Taxi는 V3b를 primary로 사용하되 V2를 encoder-only control로 보존한다.
- 기존 THP e800 artifact는 seed 42 중심이고 현재 최종 계약과 loss/input 조건이 달라 본 논문의 최종 표로 그대로 사용하지 않는다.

## 5080 실행 기준선 확인 — 다음 작업

- 작업 대상은 `ssh 5080`으로 접속하는 GPU 서버이며, 예상 프로젝트 경로는 기존 artifact 기준 `/home/leekwanhyeong/workspace/paper_research`이지만 실제 경로를 먼저 확인한다.
- `conda activate ai_env` 후 Python, PyTorch, CUDA, GPU 종류와 여유 메모리, 디스크 공간을 기록한다.
- 원격 Git branch, full commit SHA, 작업 트리와 기존 tmux/학습 프로세스를 확인한다. 세 fixed-split dataset 파일과 hash도 함께 고정한다.
- 완료 조건은 source revision, runtime identity, dataset identity가 한 기록에 남고 기존 작업과 artifact 경로가 충돌하지 않는 것이다.

## 비교 계약과 실행 행렬 고정 — 다음 작업

- primary는 4 arms x 3 datasets x 3 seeds로 36 run이다. Taxi V2 control 3 run을 더해 e300 총 39 run으로 고정한다.
- 공통 조건은 fixed split, seeds `42,52,62`, learning rate `1e-3`, batch size `128`, `target_only`, identity value head, best validation total NLL checkpoint다.
- Intermittent/Instacart TitanTPP는 V2 `small_lmm`, Taxi primary는 V3b `mid_lmm`, Taxi control은 V2 `mid_lmm`을 사용한다.
- 기존 CLI의 loss/value/head 옵션은 한 명령에 포함된 모든 모델에 공통 적용되므로, arm별 명령을 매니페스트와 launcher로 묶어 수동 설정 오류를 막는다.

## e300→e800 continuation 구현 — 다음 작업

- 현재 `epochs_300`과 `epochs_800` 경로를 유지하면서 e800 실행이 같은 identity의 e300 checkpoint를 부모로 읽는 `--resume-from-epochs 300` 형태의 명시적 기능을 추가한다.
- e800은 별도 경로에 기록해 e300 초안 artifact를 보존한다. e800 자체 checkpoint가 있으면 그것을 먼저 사용하고, 없을 때만 e300 부모에서 epoch 301로 시작한다.
- epochs 외 frozen config, source revision, dataset hash가 다르면 실행을 중단한다. e800 manifest에는 부모 checkpoint 경로, SHA256, parent epoch와 config digest를 기록한다.
- 완료 조건은 CPU e2→e5 테스트에서 history가 1~5로 이어지고, optimizer/RNG/best state가 복원되며, e2 artifact가 변하지 않는 것이다.

## 로컬 테스트와 5080 CUDA smoke — 다음 작업

- `paper_research`에서 continuation, stale checkpoint 거부, 39-run inventory와 dataset별 설정 snapshot 테스트를 수행한다.
- 검증된 source revision만 5080에 동기화한다. tmux 세션은 `titan_final_fair`로 만들고 `control`, `smoke`, `e300`, `monitor`, `report` window를 사용한다.
- `smoke` window에서 RMTPP, THP, TitanTPP 대표 run을 e1로 확인한다. 본 학습과 smoke base-dir은 분리한다.
- 5080이 단일 GPU이면 본 학습 writer는 하나만 실행한다. 보고서 코드 준비는 GPU 학습과 병렬 가능하지만 artifact 집계는 완료된 run만 읽는다.

## e300 초안 비교 실행 — 다음 작업

- `e300` window에서 validation-only 39 run을 순차 실행한다. SSH가 끊겨도 tmux에서 유지되고, 프로세스가 중단되면 같은 명령으로 각 run의 마지막 epoch부터 재개한다.
- `monitor` window에서는 GPU 사용량, 최근 epoch, 실패 sentinel과 디스크만 확인한다. 연장 결정 전에는 held-out test metric을 집계하거나 읽지 않는다.
- 완료 조건은 39개 run이 epoch 300에 도달하고, 각 run에 manifest, history, best validation NLL checkpoint, last-epoch checkpoint와 summary가 존재하는 것이다.

## e300 검증과 초안 표 고정 — 다음 작업

- 누락/중복, seed 완전성, 설정 hash, finite metric과 checkpoint epoch를 자동 검증한다.
- e300 snapshot을 별도 보존하고 total/mark/time NLL, quantity MAE, value MAE, mark accuracy, selected epoch의 3-seed mean/std 표를 만든다.
- 이 결과는 August 14 초안의 비교표와 실험 진행 상태에 사용한다. 최종 우위 문구는 아직 provisional로 표시한다.

## e800 연장 대상 결정 — 승인 필요

- validation만 사용해 연장 규칙을 적용한다. 한 primary 모델이라도 2/3 seed에서 best epoch가 `241~300`이거나 후반부 best-so-far validation NLL이 사전 임계치 `0.5%` 이상 개선되면 해당 dataset 블록 전체를 e800으로 연장한다.
- 개별 모델만 연장하지 않는다. 해당 dataset의 RMTPP original, RMTPP matched, THP matched, TitanTPP를 모두 같은 최대 epoch로 맞추고 Taxi가 대상이면 V2 control도 함께 연장한다.
- 이 선택형 정책을 쓰면 논문에는 `epochs=800` 고정이 아니라 “predeclared validation stopping, maximum 800 epochs”로 계약을 수정해야 한다.
- 기존 Decision 2의 “모든 run e800” 문구를 그대로 유지하려면 이 단계에서 선택 판정을 쓰지 않고 39개 run 모두를 e800으로 재개한다.

## e800 continuation과 최종 평가 — 다음 작업

- 승인된 dataset 블록을 epoch 301부터 800까지 이어서 실행하고 parent checkpoint hash, 첫 resume epoch, history 연속성과 best checkpoint 보존을 검증한다.
- 설정과 최대 epoch가 완전히 고정된 뒤에만 best validation NLL checkpoint를 held-out test에 한 번 평가한다.
- test를 본 뒤 모델 설정, checkpoint, 연장 대상 또는 epoch 정책을 변경하지 않는다.
- 최종 산출물은 seed 원자료, mean/std main table, scale breakdown, convergence figure, RMTPP/THP 대비 claim audit다.

## 변경사항 정리 — 외부 작업 대기

- 구현 시 대상 저장소는 `/Users/igwanhyeong/PycharmProjects/paper_research`, 제안 branch는 `codex/final-fair-e300-e800`이다.
- 로컬 테스트와 5080 smoke가 통과한 변경만 독립 commit으로 정리한다. origin push 또는 5080 checkout/sync는 실행 승인을 받은 뒤 정확한 branch와 full SHA를 기준으로 수행한다.
- 실험 중에는 코드 revision을 바꾸지 않는다. 수정이 필요하면 현재 run을 보존하고 새 revision/base-dir로 별도 실험을 시작한다.

