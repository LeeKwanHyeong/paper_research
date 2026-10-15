# RAF CNN backbone 두 후보 비교 — 루프 1 실행 범위

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15 Asia/Seoul
- Verification Status: 구현·검증 후 native 확인 대기
- Version Label: nonlinear_local_raf6_v1

## 현재 기준선 — 완료
기존 TitanTPP CNN의 고정 checkpoint와 Validation/Test 3seed 결과를 읽기 전용으로 재사용한다. 기존 CNN 재학습·재평가·결과 교체는 하지 않는다. EXTERNAL_BASELINES.json은 앞서 후보 평가 전에 검증한 등록부의 동일 바이트 사본이다. 과거 9개 계획은 보존하되 실행하지 않는다.

## 두 후보의 차이와 공통 조건 — 확정
첫 encoder block의 Q/K/V projection 뒤, 원 CNN의 인과적 depthwise 세 tap 잔차를 유지하고 최근3개 관측으로 추가 비선형 잔차를 만든다. padding은 관측에서 제외하고 예측대상과 미래정보를 차단한다. 메모리 토큰은 이 CNN 경로에서 제외한다.

- 채널별 비선형 대조군: 각64채널의 관측3개 → hidden4 → GELU → 해당채널1개. bias없이 Q/K/V당1,024개 가중치.
- 시점·채널 결합 후보: 관측3개×채널64 → hidden4 → GELU →64채널. bias없이 Q/K/V당1,024개 가중치.
- 둘 모두 추가3,072개, RAF 실제문맥84에서 학습가능82,499개. 출력projection은0으로 시작하며 공통 초기 가중치·RNG·초기출력을 유지한다.
- 같은 가중치 수·관측범위·적용위치를 맞추지만 중간폭·함수공간·계산량·최적화동역학은 같지 않다.
- 입력/log1p/손실/head/메모리/optimizer/학습예산을 유지한다. hidden64,batch128,max_seq_len84,AdamW lr0.001/weight_decay0.01,grad_clip1.0,max300/min40/patience40. 선택은 전체 fullValidation rawRMSE의 첫 strict finite 최소epoch다.
- RAF Train대상25,779, Validation대상6,690과 기존입력SHA를 유지한다. 학습은 Train/Validation만 사용한다.
- 신규 학습은 두 후보×seed42/52/62=6개, 5090 최대3동시. 기존5090 Python3.12.13/torch2.11.0+cu130/cuDNN92000/threads4/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f. Runtime/driver/auth 변경 없음.

## 구현·검증·학습 순서
1. **계약·구현·로컬 검증 — 진행 중**: 신규6개만 생성하며 기존CNN학습이 포함되면 차단한다. 원형동치·인과성·target차단·동일예산·초기화·gradient·checkpoint식별과 운영증적 누락/변조를 검사한다.
2. **5090 native 검사 후 후보 학습 — 승인 완료**: 새격리root,source/input/SHA/Runtime/GPU와 후보만으로 실제3동시 합성검사를 통과한 뒤6개를 최대3개씩 학습한다. 합성검사는 연구fit이 아니다. 현재 진행 중인root를 다시 시작하지 않는다.
3. **종료·회수·고정모델 평가 — 승인 완료**:6개 실제실행증적과 종료·binarySHA·소유PID/GPU부재 뒤 원본을 회수한다. 선택모델 전6개의 nativeValidation 재현 통과 후 같은checkpoint Test6을 평가한다. 기존CNN은 재평가하지 않는다.
4. **판정·기록·다음 루프 — 승인 완료**: 기존CNN/외부모델 대비 split별3seed RMSE/MAE 평균±표본SD와 seed별차이,여섯지표,불리한결과를 모두 기록한다. 성공 시 동일방법론 Taxi5090와 유휴5080 Intermittent로 확장하고 미달 시 하나의 backbone가설을 새계약으로 검증한다.

모델·운영·평가 작업은 신규 파일별로 병렬 분담한다. 봉인·커밋·배포·native검사·학습시작은 순서대로 진행한다. 기존CNN과 신규후보의host/Runtime 차이는 비교의 한계로 명시하며 새CNN학습으로 범위를 늘리지 않는다. 기존Test를 참고한 탐색적 후속 실험으로, 독립적 일반화·원인확정·통계유의성을 주장하지 않는다.
