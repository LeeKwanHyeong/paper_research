# PHEN 추출 검증 기록

검증일: 2026-10-27. 작업 브랜치: `phen`.
기존 source 비교 기준은 `3685adf9f6d8f758d900d9764672660ce4bd7193`이다.
파일별 hash와 실제 preflight 출력은 [verification.json](verification.json)에 남겼다.

## 완료

`bash scripts/phen/test.sh --tb=short`: **43 passed**, 22.90초.

- PHEN과 비교군 6종의 seeded 초기 가중치 및 caller RNG 일치.
- nonzero history correction을 포함한 forward·gradient 일치.
- PHEN의 dropout 활성 상태와 AdamW 1-step 후 가중치 일치.
- 기존 loader와 신규 window·target population hash 일치.
- 합성 데이터로 7개 모델의 1-epoch 학습 및 선택 checkpoint replay 완료.
- 같은 seed의 합성 데이터 재실행에서 history·모델 tensor 일치.
- target 값 변경이 예측 입력에 노출되지 않음.
- held-out 접근 gate 및 validation-selected checkpoint 제한 확인.
- 모델별 shell dry-run, 기본 3-seed 구성, bash 문법 확인.
- 기존 실험 모듈을 import하지 않는 독립 runtime 확인.
- source snapshot만 풀어 Git·기존 `models/` 없이 PHEN 학습 완료.

실제 데이터는 **모델을 생성하지 않는 preflight만** 수행했다.

| 데이터 | train target 수 | validation target 수 | 계약 확인 |
| --- | ---: | ---: | --- |
| RAF | 25,779 | 6,690 | 통과 |
| Taxi | 38,393 | 8,268 | 통과 |
| Intermittent | 393,824 | 86,285 | 통과 |
| Instacart | 1,991,192 | 503,733 | 통과 |

각 데이터에서 파일 SHA256, canonical target 식별자·수량 hash, train log-quantity
통계, train-target 시간 초기화 통계를 검증했다. 수량 보고 구간도 기존 고정 경계를
유지한다. 모든 실제 preflight에서 `held_out_materialized=false`였다.

## 미실행

- 실제 데이터 학습·모델 추론, 실제 held-out 성능 계산.
- CUDA 환경 qualification 및 원격 GPU 실행.
- 과거 checkpoint와 결과표의 학습 trajectory 재현.
- 원격 저장소 push.

따라서 이번 검증은 source 분리와 실행 경로의 준비 상태를 뒷받침한다.
논문 성능의 재현 완료를 뜻하지 않는다.
