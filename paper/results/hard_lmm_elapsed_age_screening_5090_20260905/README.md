# Hard-LMM 경과시간 encoder — seed42 e300 screening

**진행 중: 사용자가 Taxi·Instacart seed42 e300 실행과 사전 기준 평가를 승인했다.**
현재 문서는 실행 준비 기록이다. 학습 시작과 완료 상태는 실제 증적 확인 후 갱신한다.

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

## 실행 전 확인

[초기 preflight](initial_preflight.json): 5090 GPU compute process 없음, GDM inactive, 최근 kernel 오류 없음, 여유 메모리 32,107MiB. Runtime은 Python3.12.13/PyTorch2.11.0+cu130/CUDA13.0/Polars1.39.3이다. 기존 e1 소스 manifest와 두 데이터셋·split·원본 참조 checksum 검증을 통과했다.

## 남은 작업 순서

**진행 중 — 실행 준비·초기 학습 확인 / 로컬·5090**
- 새 실행 계약·runner를 검증하고 `paper_research/master`에 커밋한 소스만 별도 snapshot으로 전송한다. 소스 무결성을 확인한 뒤 tmux에서 실행한다.

**외부 작업 대기 — e300 실행·자동 감사 / 5090**
- 초기 설정·GPU process·첫 학습 진입까지만 확인한다. `TEST_SESSION_PROTOCOL.md`의 현재 세션 운영 지침에 따라 지속 polling이나 scheduler는 추가하지 않는다.
- 각 데이터셋 종료 시 runner가 처리 건수, 최적/마지막 checkpoint, optimizer step과 beta 학습, 사전 성능 기준을 검사해 저장한다.

**다음 작업 — 완료 결과 재검증 / 로컬**
- 결과 확인 요청 시 artifact를 내려받아 독립 감사를 수행하고 두 데이터셋의 기준 통과 여부와 다음 가설을 정리한다.
