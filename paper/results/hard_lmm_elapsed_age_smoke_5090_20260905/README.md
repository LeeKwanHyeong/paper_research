# Hard-LMM 경과시간 encoder — 5090 CUDA·전체 e1 검증

**완료: CUDA 56개 테스트와 Taxi·Instacart 전체 e1이 정상 실행·계약 검증을 통과했다.** 두 데이터셋 모두 시간 보정 계수와 optimizer 상태가 실제로 갱신됐고, 전체 처리 건수·유한성·저장/복원이 일치했다. E1 성능 채택은 평가하지 않았으며 원본 Hard-LMM을 일반 기준선으로 유지한다.

## 구현과 사전 계약

- 저장소/브랜치: `paper_research/master`.
- 구현 커밋: `77e5f4a05751b412f38e1ed148bc4bef10e70a75`.
- 최종 실행 커밋: `ad81a1704340245172252b1004483dd980503e51`. 차이는 optimizer 복원 테스트의 scalar step counter 비교뿐이며 모델·optimizer·geometry·수치 허용오차·성능 기준은 동일하다.
- 후보: `titantpp_elapsed_age_static_memory`, 역할 `t0_elapsed_age_static_retrieval`. Separate-key encoder attention에 causal prefix의 정규화된 경과시간과 사건 순서 위치의 차이를 전달한다. `encoder.elapsed_age_beta[2,4]` 8개 계수를 0에서 학습한다.
- 기존 log-gap·log-quantity 입력, 사건 위치 embedding, K/V, top-4, temperature, 결합, head, loss, optimizer와 checkpoint 선택을 유지했다. 첫 gap·target·padding·future·persistent key 처리와 초기 동등성은 [후보 계약](../../contracts/hard_lmm_elapsed_age_v1.md)에 고정했다.
- 최초 구현의 [로컬 검증](../hard_lmm_elapsed_age_local_20260905/README.md)은 122개 통과, CPU float16 검사 1개 제외였다. CUDA에서 fp16/bf16 검사를 모두 통과해 해당 범위를 검증했다.

## 실행 경로와 무결성

| 항목 | 확인값 |
|---|---|
| 서버 / GPU | 5090 / NVIDIA GeForce RTX 5090 |
| Runtime | Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0, Polars 1.39.3 |
| Snapshot | `/home/leekwanhyeong/workspace/paper_research_elapsed_age_smoke_ad81a17_5090` |
| Artifact | `/home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_elapsed_age_smoke_5090_20260905_r2` |
| tmux | `elapsed_age_taxi_insta_e1_0905_ad81a17` |
| 시작 / 완료 | 2026-09-05 09:51:36 / 09:54:34 KST |
| 학습 조건 | Seed42, e1/min1/patience1, batch128, hidden64, lr0.001, grad clip1 |
| 실행 범위 | Train/validation 전체, fresh 후보 학습, held-out 없음 |

커밋된 소스와 감사용 원본 참조를 묶은 패키지의 SHA256은 `2a7ae1f328064b75d8b3eed0a4c64922e201e55f6f6809dd6a3244101ecde243`이다. [Source manifest](source_manifest.json)의 **524개 파일**을 전송 후·실행 중·완료 후 검증했다. Manifest SHA256은 `c17da878f6bc0d283702447b46510be5aa66d199c01568351c67cefa1b0503fa`이다.

원본 참조 checkpoint는 hash·초기화 계약 감사에만 사용했으며 fresh 후보의 초기 파라미터로 로드하지 않았다. 두 데이터셋 및 split manifest의 checksum도 고정 registry와 대조했다. 기존 공용 checkout·서비스·Runtime을 변경하지 않았고, 기존 미추적 `scripts/`는 패키지에 포함하지 않았다. 종료 후에도 GPU compute process와 최근 kernel error가 없었고 32,107MiB가 비어 있었다.

## CUDA와 실제 데이터 결과

CUDA 테스트는 **56 passed, 0 failed, 0 skipped**다. Zero-beta 출력 동등성, prefix 인과성·padding·persistent 처리, beta gradient, AMP fp16/bf16 유한성, 모델·optimizer 저장/복원과 다음 step, batch128/h64/길이64·256 학습 형태를 포함한다.

| 데이터셋 | Train target | Validation target | Train batch / beta step | 갱신된 beta | 최대 할당 / 예약 GPU 메모리 | 실행·감사 시간 |
|---|---:|---:|---:|---:|---:|---:|
| Taxi | 38,393 | 8,268 | 300 / 300 | 8 / 8 | 1.81 / 2.01 GiB | 11.96초 |
| Instacart | 1,991,192 | 503,733 | 15,557 / 15,557 | 8 / 8 | 0.36 / 0.43 GiB | 162.29초 |

메모리는 PyTorch allocator의 peak이며 GiB는 2³⁰ bytes다. 시간은 각 데이터셋의 실행과 직후 감사를 포함하고 전체 CUDA 준비 시간은 제외한다.

- Train event·batch 수, train 시간 통계의 target 수, validation의 수량/이력 구간별 건수 합계를 모두 대조했다. 부분 데이터 실행이 아니며 e1 history는 `[1]`이다.
- 두 dataset의 beta 텐서와 AdamW `exp_avg`, `exp_avg_sq`는 유한하고 0이 아니었다. Step counter는 전체 train batch 수와 같다.
- Best checkpoint, last-best state, last-current state의 canonical hash가 e1 기준으로 일치했다. 불규칙 관측 H=3의 복원 출력도 CPU에서 정확히 일치했다.
- 최종 checkpoint와 optimizer 상태, summary·history·구간별 집계는 유한했다. Held-out artifact는 생성되지 않았다.
- 내려받은 결과에 `--audit-only`를 적용한 [로컬 재감사](local_audit.json)도 통과했다. [독립 감사](independent_audit.json)는 저장된 증적을 별도로 확인한다.

## CUDA 검사에서 수정한 부분

첫 snapshot `77e5f4a`의 CUDA 검사에서는 55개가 통과하고 optimizer 복원 검사 1개가 실패했다. 기존 AdamW의 scalar step counter는 CPU에 있었지만 `torch.load(map_location=cuda)`로 복원한 counter는 CUDA에 있어, 값이 같아도 장치까지 동일하다는 assertion이 실패했다. 이 시점에 실제 데이터 학습은 시작되지 않았다.

5090의 PyTorch 소스에서 non-capturable AdamW가 로드된 step tensor를 그대로 유지하는 동작을 확인했다. Scalar step만 양쪽을 CPU로 옮겨 값·dtype을 exact 비교하도록 수정했다. Moments의 장치·수치 검사, 실제 CUDA 로드 경로와 다음 optimizer step의 허용오차는 유지했다. 로컬 replay 단일 검사는 통과했다.

실패 snapshot과 원시 파일을 보존하고 테스트 수정만 독립 커밋한 뒤 새 snapshot·새 `_r2` 출력 경로에서 수행했다. 두 source manifest를 비교해 **테스트 1개 파일만 달라졌음**을 확인했다. 자동 resume·조건 완화는 없었다. 자세한 증적은 [복구 기록](../hard_lmm_elapsed_age_cuda_recovery_20260905/recovery_record.json)에 있다.

## 판단과 남은 작업

**현재 기준선 — 구현 후보의 정상 실행과 계약 준수 확인 완료**
- 초기 중립성과 학습 경로, 전체 e1 처리, 메모리 사용·저장/복원이 검증됐다. 이는 새로운 시간 표현을 성능 평가할 수 있는 실행 기반을 확보했다는 뜻이다.
- E1 수치를 원본·separate-key의 e300 성능과 비교해 채택하지 않았다. 시간 표현의 개선 효과와 Taxi의 기존 성능 보존은 아직 확인되지 않았다.

**다음 작업 · 승인 필요 — 고정 기준의 seed42 성능 screening / 5090**
- 같은 후보 하나로 Taxi·Instacart의 fresh e300/min40/patience40을 실행하고, 기존 validation joint selector로 선택한다. 실행 전에 이번 정상 종료·source·Runtime·데이터 기준선을 재확인한다.
- 원본 대비 body MAE 5% 이상 개선, RMSE·tail MAE 악화 각각 2% 이하, Time NLL 증가 0.01 이하를 유지한다.
- 추가로 Instacart는 separate-key 대비 body MAE 5% 이상 개선, Taxi는 body MAE 악화 1% 이하를 요구한다. 두 dataset 모두 separate-key 대비 RMSE·tail MAE 악화 1% 이하와 Time NLL 증가 0.01 이하를 동시에 적용한다.
- 기준은 이번 e1 이전에 고정됐다. E300·추가 seed·dataset·held-out·scheduler·push는 실행하지 않았다.

원시 checkpoint와 CSV는 ignored `search_artifacts/hard_lmm_elapsed_age_smoke_5090_20260905_r2/`에 보존한다. 실행 명령은 [launch_record.json](launch_record.json), 상세 결과는 [status.json](status.json), [Taxi 감사](yellow_trip_hourly/audit.json), [Instacart 감사](insta_market_basket/audit.json)에 기록했다.
