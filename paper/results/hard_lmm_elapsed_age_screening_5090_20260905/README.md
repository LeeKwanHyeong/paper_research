# Hard-LMM 경과시간 encoder — seed42 e300 screening

**진행 중: 5090에서 Taxi·Instacart fresh seed42 e300을 시작했다.**
2026-09-05 10:19:54 KST에 시작했으며, 10:20:25 KST 초기 확인에서 Taxi epoch 1–3과 GPU 학습 프로세스를 확인했다. Instacart는 Taxi 실행·감사 이후 순서대로 진행한다. 성능 판정은 아직 완료하지 않았다.

새 실행기와 기존 e1 실행기 대상 로컬 테스트 **31개 통과, 실패·제외 0개**를 확인했다. 실행 명령·원시 XML hash는 [로컬 검증](local_validation.json)에 기록했다.

## 현재 기준선

- 저장소/브랜치: `paper_research/master`, 준비 시작 기준 `e97a12a`.
- 후보 구현: `77e5f4a`, CUDA 실행 소스: `ad81a17`. 로컬 122개 및 CUDA 56개 테스트와 전체 e1 검증 완료.
- 후보: `titantpp_elapsed_age_static_memory`. 기존 separate-key encoder에 signed elapsed-age attention bias 8개만 추가하며 fresh beta0 초기화를 유지한다.
- 이번 승인으로 e300 실행 범위를 새 계약에 추가한다. 기존 모델 계약과 e1 증적, 사전 성능 기준은 변경하지 않는다.

## 실행·판정 계약

- Taxi 다음 Instacart를 직렬 실행한다. 두 데이터셋 모두 fresh seed42, 최대 300 epoch, 최소 40 epoch, patience40, batch128, AdamW lr0.001과 기존 validation joint selector를 유지한다. 정상 early stopping은 허용된다.
- Train/validation 전체를 사용한다. 선택된 validation checkpoint의 성능을 원본 Hard-LMM과 separate-key의 고정 참조와 비교한다.
- 원본 대비 두 데이터셋 모두 body MAE 5% 이상 개선, 전체 RMSE·p99 초과 MAE 악화 각각 2% 이하, Time NLL 증가 0.01 이하를 요구한다.
- Separate-key 대비 Instacart body MAE는 5% 이상 개선, Taxi body MAE는 악화 1% 이하를 요구한다. 두 데이터셋 모두 RMSE·p99 초과 MAE 악화 1% 이하, Time NLL 증가 0.01 이하를 동시에 적용한다.
- 모든 기준을 반올림 전 참조값으로 계산한다. 성능 gate 미달이어도 승인된 다음 데이터셋을 실행하며, 실행·무결성 감사 실패는 중단한다.
- 단일 seed의 validation screening이므로 일반화나 최종 모델 채택의 증거로 확대 해석하지 않는다.

## 실행과 초기 확인

- 실행 소스 커밋: `e6e59337a816d53e9d1d3afaf0f6cede48ef5e2f` (`paper_research/master`). 모델·학습 의존 파일 167개가 e1 소스와 동일하다.
- Snapshot: `/home/leekwanhyeong/workspace/paper_research_elapsed_age_screening_e6e5933_5090`.
- 서버 결과: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_elapsed_age_screening_5090_20260905`.
- tmux: `elapsed_age_taxi_insta_e300_0905_e6e5933`.
- 전송한 소스·참조 534개를 검증했고, 실행 직전 Runtime·data·split·참조와 기존 CUDA56개/full e1 감사를 다시 통과했다.
- 실제 Taxi 설정은 최대300/min40/patience40, seed42, batch128, lr0.001, validation-only, `partial_smoke=false`다. 초기 GPU process는 PID3930103/Python, 2684MiB를 사용했다.
- [실행 명령](launch_record.json), [전송 검증](deployment.json), [실행 직전 감사](launch_preflight.json), [초기 학습 확인](initial_training_check.json), [검증 요약](launch_verification.json)에 증적을 보존했다.
- 초기 epoch 지표는 학습 진입 확인에만 사용하며 성능 채택이나 조기 gate 판단에 사용하지 않는다.

## 실행 전 확인

[초기 preflight](initial_preflight.json): 5090 GPU compute process 없음, GDM inactive, 최근 kernel 오류 없음, 여유 메모리 32,107MiB. Runtime은 Python3.12.13/PyTorch2.11.0+cu130/CUDA13.0/Polars1.39.3이다. 기존 e1 소스 manifest와 두 데이터셋·split·원본 참조 checksum 검증을 통과했다.

## 남은 작업 순서

**현재 기준선 · 완료 — 실행 준비·초기 학습 확인 / 로컬·5090**
- 새 계약·실행기 검증과 독립 커밋, 전송 무결성, 기존 CUDA/e1 재감사, tmux 학습 시작 및 Taxi epoch1–3을 확인했다.

**진행 중 · 외부 작업 대기 — e300 실행·자동 감사 / 5090**
- 초기 설정·GPU process·첫 학습 진입까지만 확인한다. `TEST_SESSION_PROTOCOL.md`의 현재 세션 운영 지침에 따라 지속 polling이나 scheduler는 추가하지 않는다.
- 각 데이터셋 종료 시 runner가 처리 건수, 최적/마지막 checkpoint, optimizer step과 beta 학습, 사전 성능 기준을 검사해 저장한다.

**다음 작업 — 완료 결과 재검증 / 로컬**
- 결과 확인 요청 시 artifact를 내려받아 독립 감사를 수행하고 두 데이터셋의 기준 통과 여부와 다음 가설을 정리한다.
