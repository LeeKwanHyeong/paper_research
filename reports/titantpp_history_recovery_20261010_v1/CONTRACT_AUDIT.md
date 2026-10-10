# 남은 학습 복구 계약 독립 감사

상태: 로컬 계약·원본 감사 완료. 원격·Runtime 재현·새 GPU admission은 미실행.
작성시각: 2026-10-10T11:05:20.987521+09:00
역할: 프로젝트 `.codex/agents/architecture-reviewer.toml`의 gpt-6-astra/high 설정 확인. 이 작업은 architecture-reviewer 역할로 배정됐다. 지정 execution-protocol 경로는 로컬에 없으며, `.agents/` 수정 금지에 따라 승인된 reports 경로에만 작성했다.

## 문제 정의

운영 감독 프로세스의 오류로 중단한 과학 실행을 완료 원본과 중복 없이 이어가야 한다. A100 Pod가 삭제되어 GPU UUID가 바뀌며, 기존 계약은 fresh-only/resume=false와 물리 GPU 소유권을 봉인했다. 따라서 부모 계약을 고쳐 재사용하는 것이 아니라, 새 운영 복구 계약이 부모 원본·실행 ID·새 host·비용 원장을 연결해야 한다. 이 감사는 모델 설계 변경이나 성능 선택 작업이 아니다.

사용자의 2026-10-10 “문제 확인해서 수정하고 남은 학습 진행하자”는 운영 복구·남은 학습 진행 승인으로 해석한다. 새 실행은 총 $50 한도, 기존 과학 설정, 완료 원본 보존을 유지한다. 삭제된 Pod를 다시 접속하지 않는다.

## 확인된 원본과 범위

- 부모 canonical `4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a` 일치.
- source129+input7 모든 raw SHA, partial archive SHA, manifest SHA, 원본393파일 SHA를 Python stdlib로 재검증했다.
- archive `83d0d3f4fabd89554a0cc5acc86dc447a9919b22d5247e27a1aaea65ecc01235`; manifest `d25ca3393c1c9a295ebf6b37d0695f8b700adf8af68b777a4e09b57c245f2091`.
- A100 완료10: Taxi 3seed×3arm=9, RAF seed42 disabled=1. 결과가 좋은 원본만 고르는 기준이 아니라 terminal 완료 여부로 재사용한다.
- 부분2: RAF seed42 full E23/best9, current_only E46/best9. 두 run의 server_checkpoint_receipt가 지정한 다섯 파일 모두 raw SHA 일치, history/timing 길이 각각23/46, exposure train/validation 각각23/23,46/46이다.
- 미시작15: RAF seed52/62×3arm=6, Intermittent seed42/52/62×3arm=9.
- 따라서 남은 logical execution은17개이며, 정상 재개 시 과학조건 총27개를 유지한다. 부모 full/current의 중단과 복구를 실패·완료 수로 두 번 세지 않는다.
- Instacart는 별도9조건이며 마지막 CNNGRU54 seed62의 현재 상태를 parent가 확인 중이다. 마지막 알려진 E37/best1은 새 완료 증거가 아니다. 실제 terminal이 있으면 재학습하지 않고 원본을 회수한다.

## 복구 방식 비교와 권고

| 선택 | 구현 비용 | 운영 비용 | 관리 복잡성 | 향후 변경 비용·한계 |
|---|---|---|---|---|
| A. 별도 복구 계약 + 검증된 epoch state 재개 | 중간: native resume 어댑터·prefix 검증·원본 매핑 필요 | 이미 완료한 epoch를 재사용하므로 낮음 | 원본 attempt와 recovery attempt를 연결하는 원장 필요 | GPU 이동 사실을 영구 기록하며, 동일 물리 GPU·bitwise uninterrupted 재현을 주장할 수 없음 |
| B. 부분 원본 보존 후 두 조건을 fresh 대체 | 초기 구현은 작음, 종료·원본 집계 수정 필요 | full23/current46epoch를 다시 수행 | logical ID와 새 attempt를 분리하고 대체 규칙 고정 | optimizer/RNG 연속성 없음. 학습 노출 증가와 재시도임을 공개해야 함 |
| C. 같은 새 물리 GPU에서 RAF42 세 arm 전체 fresh 대조 | 구현 중간 | 완료disabled도 중복 학습하므로 가장 큼 | 보존 원본과 새 확인 대조를 따로 집계 | 같은GPU 대조 장점이 있지만 완료 원본 재학습을 포함하고 남은17 범위를 넘음 |

권고는 A다. 단 아래 admission을 모두 통과할 때만 E24/E47부터 실행한다. B는 resume state가 손상·누락됐을 때의 대안이며 자동 fallback으로 실행하지 않는다. C는 기존 완료 원본을 다시 학습하는 추가 실험이므로 현재 남은 학습 복구의 기본 경로로 삼지 않는다.

## 재개 admission과 완료 조건

1. 부모 owner 실제 종료·GPU PID 부재·Pod 삭제 receipt를 새 계약에 raw SHA로 연결한다. 부모 failure, claim, completion(status=partial)을 그대로 보존한다.
2. 새 root와 attempt ID를 만들되 27개의 원래 scientific execution ID를 유지한다. completed10→reuse, partial2→resume, untouched15→fresh의 상호배타적인 매핑을 봉인한다. 서로 다른 attempt의 성능 중 좋은 쪽을 선택하지 않는다.
3. frozen source129와 입력7, 데이터/Train 통계/분할, B128/min40/max300/patience40, loss/optimizer/initialization/selector를 유지한다. 새 GPU는 A100-SXM4-80GB와 기존 Python/torch2.11.0+cu130/CUDA13.0/cuDNN91900/numpy2.4.4/polars1.39.3 및 결정론 설정을 검증한다. source digest와 Runtime 정보만으로 bitwise 동일성을 보장하지 않는다.
4. native `_validate_resume_payload`를 실제 target Runtime에서 수행한다. schema2, epoch/history 연속성, earliest strict finite fullValidation 수량RMSE best, initial/current/best tensor SHA, optimizer 구조·step, CPU/CUDA RNG, loader generator state를 검사한다. 이 감사는 .pt를 unpickle하거나 재추론하지 않았으므로 해당 검증은 아직 미완료다.
5. partial run에는 별도 selected .pt가 없으나 frozen trainer는 last_epoch_state.pt에 `best_state_dict`, `best_epoch`, `best_state_sha256`를 저장한다. 외부 selected 파일 부재만으로 resume 불가로 판정하지 않는다. embedded best 검증이 필수다.
6. parent history/exposure/timing prefix를 그대로 복사·검증하고 optimizer steps가 완료 train exposure와 일치하는지 확인한다. 저장 epoch 이후의 미완료 배치 업데이트는 버려지며 완전 무중단 경로와 동일하다고 주장하지 않는다. 기존 elapsed를 초기화해 총시간을 축소하지 않는다.
7. partial RAF42는 남은 두 arm만 실행한다. 완료disabled를 동시3개 숫자를 채우려고 다시 실행하지 않는다. 미시작 dataset/seed batch는 full/disabled/current_only3개를 같은 새GPU에서 동시에 시작하며 최대3개를 유지한다.
8. GPU 소유권 검증은 실제 살아 있는 child/worker의 정확 argv·PID·PPID·process group·lease를 확인한다. child 종료와 GPU process 목록 사이의 경합을 회귀검증한다. foreign GPU를 무조건 무시하도록 완화하면 안 된다.
9. selected/last fullValidation의 동일 epoch RMSE/MAE/TimeNLL 및 large-quantity 지표, source/operation SHA, native exit0, owner/GPU idle, manifest/archive 원본 SHA까지 검증해야 terminal 완료로 집계한다.

RAF42 disabled의 마지막 server_checkpoint_receipt에 있는 exposure SHA는 현재 terminal exposure와 다르다. terminal 후 endpoint Validation replay가 추가되어 train49/validation50으로 기록된 결과다. 두 partial run의 receipt는 모두 일치한다. 완료 run 재사용은 epoch 중간 receipt가 아니라 최종 terminal_manifest/archive를 기준으로 검증해야 한다.

## GPU·Runtime 비교 경계

삭제된 GPU UUID `GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de`와 새 UUID는 같지 않다. RAF42 완료disabled와 재개full/current는 동일 물리GPU 대조가 아니며, 같은 A100 모델·동일 Runtime으로 이동한 recovery 대조라고 적는다. 새 계약은 이 변경과 한계를 명시해야 한다. 다른 RTX의 시간을 혼합하거나 새로운GPU 관측으로 이전 관측시각을 덮어쓰지 않는다. 동시 실행 wall time은 단독 구조효율의 근거가 아니다.

## 비용과 조건부 시간

- 마지막 manager 추정누적 $30.124240458, 실제 삭제확인12:23:55KST. provider 최종 청구액은 이 감사에서 확인하지 않았다.
- 추정 남은 총액 $19.875759542. 회수·정리 예비 $5를 유지하면 새 설치+검증+학습 배정은 최대 약 $14.875759542다.
- 이전 저장포함요율 $1.596944/h가 유지된다는 가정에서 설치+학습 약9.315시간. 최종청구·새가격·저장요율 확인 후 수치를 보수적으로 다시 고정해야 한다. 기존비용을0으로 재설정하거나 $50을 새Pod별 한도로 적용하지 않는다.
- 새 절대 종료시각은 잔여 금액과 실제 생성시각·신규요율에 연동한 recovery permit에 봉인한다. 종료한 부모의 deadline을 수정하지 않는다. provider stopAfter 요청은 실제 종료 증거와 별개이며 독립 로컬 guard와 원본 회수·정리 여유를 유지한다.
- 부분RAF 자체 최근10epoch 중앙값: full43.591초, current22.595초. best9 유지 시 E49까지 각각26epoch≈18분53초,3epoch≈1분8초; E300 시나리오는3시간21분15초/1시간35분39초. 이는 이전 동시실행 관측에 따른 조건부 시나리오이며 새GPU 속도 또는 전체큐 ETA가 아니다.
- RAF 미시작seed와 Intermittent에는 실측을 복사하지 않는다. 남은17개가 총$50 내 모두 완료될지는 미확정이다. 비용초과를 허용하거나 selector/patience를 줄여 완주를 만들지 않는다.

## 고정 검색 후속 범위

기존 승인된 과학범위는 weighted/fixed/generic512×3dataset×3seed=27이며, history full9를 mean으로 한 번씩 재사용하는 것이다. generic521은 대체 대조군이 아니다. fixed/generic은 추가active512, prototype64/top4/과거32요약/temperature1을 유지한다.

A100 history full9를 RTX 검색의 동일GPU 기준선으로 표시할 수 없다. 원래물리GPU가 삭제됐으므로 전체Q2에서 동일물리GPU 조건을 유지하려면 mean 재학습이 추가되고, 추가A100과금도 기존history $50에 자동 포함되지 않는다. 따라서 고정검색은 새 자원·기준선 계약 확정 전 대기다. 동일hardware class/runtime 비교로 완화하려면 그 변경을 명시하며 같은GPU 비용검사는 targetGPU에서 별도 mean native benchmark로 수행한다.

Q1: full이 두 대조 모두보다 같은2개 이상dataset 각3seed 전체RMSE strict개선. 현재 완료Taxi만으로 충족하지 않는다.
Q2: fixed가 mean/weighted/generic512 모두보다 같은2개 이상dataset 각3seed 전체RMSE strict개선 및 해당GPU T64/T256 추론+20%,학습step+30%,peak+20% 비용 한도 충족. 큰수량·MAE·TimeNLL 손해도 공개한다. Test·CPU binary 재추론 감사·새 독립평가는 별도 미완료다.

## 근거 파일과 남은 순서

- `search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/execution_contract.json`
- 동일 bundle `originals/partial/receipt.json`, `originals/partial/verified/manifest.json`, `control/cleanup_receipt.json`, `control/manager_heartbeat.json`
- 동일 bundle frozen `source/paper/scripts/count_aware_tpp_backbone/training.py`: resume identity164~, validation288~, state restore790~, save904~
- `paper/scripts/prepare_titantpp_history_a100.py`, `prepare_titantpp_history_a100_parallel3.py`, `titantpp_contribution_runtime.py`
- 참고 운영선례 `paper/scripts/titantpp_contextual_retrieval_repair_runtime.py`의 copy_resume/prefix검증. 그대로 재사용하면 seed42/고정arm/부모root 바인딩이 달라 실패하므로 해당실험에 맞춘 복구 어댑터가 필요하다.
- `reports/titantpp_history_a100_20261008_v1/PARALLEL3.md`, `ownership_reservation.json`
- `reports/titantpp_contribution_execution_20261008_v1/README.md`

1. 원본·잔여범위 확인 — 완료(본 감사).
2. 5080 실제 종료·GPU오류 확인 및 A100 감독오류 수정·회귀검증 — parent 진행 중.
3. 복구 계약·누적비용·새GPU Runtime·checkpoint admission 봉인 — 다음 작업.
4. 승인된 미완료 실행 시작, 첫 저장 checkpoint 확인, 시간별 관측 — 위 검증 후 진행.
5. 전체Validation 집계·Q1판정·고정검색 자원계약 — 후속.

## 부모 실시간 조회 전달에 따른 추가 감사

부모가 live MCP 조회 결과로 누적청구 `$30.124985997681506`, 현재 Pod 없음, A100-SXM4-80GB community CUDA>=13 availability LOW와 시간당 $1.39(secure $1.79)를 전달했다. 이 감사자는 외부 조회를 수행하지 않았다. 새 계약은 부모가 보관한 billing/catalog 원본 응답 및 시각을 연결해야 한다.

이를 적용하면 남은총액은 `$19.875014002318494`, 회수예비$5를 유지한 준비·학습 배정은 `$14.875014002318494`다. GPU비용만 $1.39/h로 나누면 약10.70시간이지만, storage·설치·qualification·회수비용을 포함하지 않은 상한이므로 실제 학습시간은 더 짧다. LOW재고는 생성가능 또는 Runtime일치 보장이 아니다.

사용자의 최신 복구·남은학습 실행 승인에는 삭제된 Pod 대신 신규 운영 환경으로 미완료 작업을 이어가기 위한 migration이 포함된다고 판단한다. 동일모델 A100과 일치하는 Runtime, 이전비용을 포함한 $50 상한, 같은 과학설정 및 부모 원본 보존을 전제로 하며, 재차 과학 설정 변경이나 완료10 재학습을 승인한 것으로 확대하지 않는다. new recovery contract에는 이 승인문구와 부모실행 종료, hardware UUID 변경을 명시한다.

Q1 결과표는 Taxi9(기존같은GPU), RAF42(기존disabled+새GPU재개full/current), RAF52/62·Intermittent(새GPU별3arm batch)를 구분해야 한다. 정해진 strict 수량RMSE 기준 계산은 가능하지만 RAF42에는 hardware migration이 섞인 개발 대조라는 표식을 붙인다. 표준 same-physical-GPU 대조와 같다고 표현하거나 소수점 동일성으로 이동의 무영향을 증명했다고 쓰지 않는다. RAF42를 결과에 따라 임의로 제외하거나 선택기준을 바꾸지 않는다. 비용표는 host/실행구간·동시성별로 따로 제시하며 native 단독비용과 shared wall time을 구분한다.
