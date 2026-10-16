# RAF 국소 비선형 CNN 대조 실험 — 루프 1

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15 Asia/Seoul
- Verification Status: 구현·로컬 검증 중; native 학습 결과 미확인
- Version Label: nonlinear_local_raf9_plan_v1

## 연구 질문과 변경
공통 입력·손실·head·메모리를 유지할 때, CNN 국소 경로에서 최근 관측 시점과 특징 채널을 함께 비선형 처리하는 방식이 원형 CNN 및 채널별 비선형 대조보다 RAF의 수량 오류를 줄이는가? 기존 오류 분석은 이 개선의 효과나 원인을 확정하지 않는다. 이 실험은 해당 가설의 대조 검증이다.

세 구조 모두 첫 encoder block의 사건 Q/K/V projection 뒤에 같은 최근 3개의 관측을 사용한다. padding은 관측으로 세지 않고 예측 대상·미관측 이후 정보는 사용하지 않는다. 메모리 토큰은 CNN 경로에 넣지 않는다. 원형 CNN의 인과적 depthwise 세 tap 잔차는 그대로 유지한다.

- 기존 CNN: 기존 동결 `titantpp_cnn_gru54_cnn_only`와 같은 모델을 신규 캠페인의 `titantpp_cnn`으로 식별한다. 과거 결과를 바꾸지 않는다.
- 채널별 대조군: 각 64개 채널에서 관측3개 → hidden4 → GELU → 해당 채널1개의 추가 잔차를 계산한다. bias 없는 가중치 64×(3×4+4)=1,024개/QKV.
- 시점·채널 결합 후보: 관측3개×채널64 → hidden4 → GELU → 채널64의 추가 잔차를 계산한다. bias 없는 가중치 3×64×4+4×64=1,024개/QKV.
- 두 후보는 각각 추가 3,072개 가중치를 사용한다. 출력 projection은 0으로 초기화하고 앞 projection은 비영으로 초기화하여 최초 출력과 공통 RNG/가중치를 원형 CNN과 맞춘다. 두 경로의 초기·후속 gradient를 검증한다.

같은 파라미터 수가 같은 함수 공간·중간 폭·계산량·학습 동역학을 뜻하지는 않는다. 대조군 대비 효과를 보고할 때 이 차이를 명시한다. 이 구조 자체의 학술적 신규성은 이번 구현만으로 확정하지 않는다.

## 공통 조건
- dataset: raf_spare_parts, seed42/52/62, 5090에서 9개 fresh fits, 최대3동시.
- 입력: 기존 Train/Validation 전용 준비 데이터와 split SHA 그대로. Train 대상25,779, Validation 대상6,690; 실제 입력·target identity는 실행 전 재검증한다.
- hidden64, batch128, max_seq_len84, train shuffle/Validation 비셔플, loader worker0.
- AdamW lr0.001, weight_decay0.01, grad_clip1.0. 기존 log 수량 손실과 heteroscedastic lognormal duration head 및 recorded positive-integer 시간 관측을 유지한다.
- max300/min40/patience40, 전체 fullValidation raw quantity RMSE 첫 strict finite 최소 epoch 선택. warm start/resume 없음. 기존 조건당36h와 host168h 한도를 새 계약 시작시 고정하며 실행 도중 늘리지 않는다.
- 기존5090 Python3.12.13/torch2.11.0+cu130/cuDNN92000, GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, threads4/interop1. 공용 환경을 변경하지 않는다.
- 기존 CNN 기준선도 같은5090에서 학습하여 Runtime 차이를 통제한다. 과거5080CNN은 원래 사용자 비교 기준으로 별도 보존한다.

## 실행 순서와 완료 조건
1. **계약과 입력·외부 비교 기준 결속 — 진행 중**: 모델 차이, source/input, baseline registry, selector, 예산, 승인 및 새 root를 SHA로 고정한다. 외부 기준은 후보 평가 전에 고정한다.
2. **구현 및 로컬 합성 검사 — 진행 중**: 원형 동치, 원형 공통 RNG, 파라미터 예산, 초기화·gradient, 인과성/padding/target 차단, batch 일치, checkpoint 구조 식별과 여섯 지표를 검증한다.
3. **5090 native 검증 후 학습 — 승인 완료**: GPU/Runtime/입력SHA와 실제3동시 synthetic qualification 통과 후 새 tmux에서 9개를 최대3개씩 실행한다. 기존 다른 작업이 점유하면 대기한다.
4. **종료·원본 회수와 Validation/Test — 승인 완료**: 실제 scientific worker/dispatcher/pipeline 종료와 SHA를 검증하고 원본을 별도 archive/manifest로 회수한다. 선택 checkpoint 전9개 native Validation 재현 gate 후 Test9를 수행한다.
5. **비교·판정·Notion 기록 — 승인 완료**: split별 RMSE·MAE 평균±표본SD 및 seed별 차이, 전체/큰수량/시간 결과, 원형·대조군·외부 비교를 공개한다. 사용자 통과 기준은 SCHEDULER.md를 따른다.
6. **확장 또는 다음 가설 — 조건부 승인 완료**: RAF 통과 시 같은 구조를 Taxi5090와 실제 유휴5080 Intermittent에 적용한다. 미달 시 오류를 분석하고 다음 단일 backbone 변경을 새 버전으로 검증한다. Test를 참고한 반복이라는 한계를 유지한다.

모델 구현, 운영 도구 구현, 외부 결과 증적 검증은 파일을 나눠 병렬 진행한다. 계약 봉인·통합 검사·커밋·native 검증·학습 시작은 의존 순서대로 진행한다. 기존 사용자 변경이 있는 paper_research/codex/contextual-retrieval-pilot-20261007에서 이번 신규 파일만 커밋하며 원고나 다른 변경을 포함하지 않는다. Push/MR는 이번 실행에 필요하지 않다.
