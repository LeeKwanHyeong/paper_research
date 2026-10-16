# CNN과 GRU 결합 이유를 검증하는 두 서버 실행 준비

**현재 상태 — 로컬 준비 완료 / 원격 실행은 외부 작업 대기**

2026-10-12 사용자는 기존 결과의 GPU·Runtime 혼합을 피하기 위해 네 구조를 새로 비교하는 최대36개 학습을 선택했다. 이후 5080 단독 대신 **5080·5090 병렬 배치를 먼저 준비하고, 현재 다른 작업이 있는 5090은 다음 “진행하자” 지시 뒤 확인하여 실행**하도록 변경했다. 이 최신 경계가 앞선 단독 서버 실행 승인보다 우선한다. 이번 준비에서 SSH·GPU·실제 학습·Test 평가를 실행하지 않는다.

**이미 수행한 비교와 이번에 빠진 질문을 구분한다 — 완료 / 현재 기준선**

- 과거 MLP16·CNN+MLP16·GRU54·CNN+GRU54 비교는 존재한다. CNN+MLP16에는 MLP 보정이 있어 진짜 CNN-only 대조가 아니다. 이력 보정27의 full·보정 제거·current-only도 CNN-only/GRU-only를 모두 포함하지 않는다.
- 기존 결과와 원본은 보존한다. 과거 A100·RTX 학습을 이번 동일 계약의 새 학습으로 재명명하거나 checkpoint 초기값으로 사용하지 않는다. H1/H2 진단은 현재 기준선이며 재실행하지 않는다.
- 새 질문은 같은 입력·초기화·손실·최적화·선택 조건에서 **보정 없음, CNN만, GRU만, CNN+GRU** 중 결합 경로가 두 단일 경로보다 나은지다. 경로 제거 대조이며 파라미터 용량을 같게 맞춘 원인 분리 실험은 아니다.

**같은 데이터셋의 대조를 한 GPU에 모아 서버를 나눈다 — 준비 완료**

| 서버 | 데이터셋 | seed | 네 구조 학습 수 | Runtime 기준 |
|---|---|---|---:|---|
| 5080 | Taxi | 42·52·62 | 12 | 기존 ai_env, cuDNN 91900 |
| 5080 | RAF | 42·52·62 | 12 | 위와 동일 |
| 5090 | Intermittent | 42·52·62 | 12 | 기존 전용 resume runtime_env, cuDNN 92000 |
| 합계 | 3개 데이터셋 | 각각 3seed | **36** | 기존 Runtime 보존 |

- 5080은 Taxi42→52→62→RAF42→52→62, 5090은 Intermittent42→52→62 순서다. 두 큐는 병렬로 진행한다.
- 각 dataset·seed 묶음은 **보정 없음·CNN만·GRU만 세 학습 동시 실행 → 세 학습 모두 종료 → CNN+GRU 한 학습 → 다음 seed** 순서다. 서버당 최대3개, 두 서버 합계 최대6개다. 네 구조 전체가 같은 GPU·Runtime을 사용한다.
- 데이터셋별 세 seed도 같은 서버에 유지한다. 이 배치는 서로 다른 Runtime의 seed를 같은 데이터셋 평균에 섞지 않는다. 데이터셋 사이에는 GPU·Runtime 차이가 남으므로 서버를 넘나드는 속도·성능 차이를 구조 효과로 해석하지 않는다.
- Intermittent는 epoch당3077배치, Taxi300·RAF202배치다. 학습 개수24:12로 종료 시각이 같아진다는 보장은 없다. 새 구조의 완료 epoch 속도가 없으므로 전체 큐 ETA는 미확정이다. 빈 GPU를 이유로 묶음을 자동 이동하거나 검색을 실행하지 않는다.

**네 구조의 과학 조건을 고정한다 — 구현·로컬 검증 완료**

- 전체 등록 파라미터는 네 구조 모두115,167개다. 활성 학습 파라미터는 보정 없음89,859, CNN만90,435, GRU만114,591, CNN+GRU115,167개다. CNN 추가576개와 GRU 추가24,732개의 용량 차이를 결과와 함께 공개한다.
- 같은 seed에서 원래 FULL 모델을 새로 생성한 공통 tensor와 난수 상태를 유지한다. 비활성 경로는 동결·실행 우회·optimizer 제외한다. 새 MLP나 학습 가능한 대체 경로는 추가하지 않는다.
- 기존129개 과학 파일과7개 입력 파일의 SHA를 유지하고 신규 모델·runner 두 파일만 추가한다. 신규 파일은 원래 commit에 있었다고 기록하지 않고 확장 파일 SHA로 고정한다.
- B128, AdamW lr0.001/wd0.01, clip1, 기존 수량·시간 손실 및 Train-only 정규화, 최소40/최대300epoch·patience40을 유지한다. 동일한 전체 Validation raw 수량 RMSE의 최초 strict finite 최솟값으로 선택한다. fresh fit이며 warm start·resume·자동 retry는0이다.
- Train·Validation만 사용한다. 이미 접근한 기존 Test를 다시 평가하거나 독립 확인으로 부르지 않는다. RAF cap84와 기존 시간 head·손실·prototype mean readout은 유지한다.
- 총56개 로컬 검증(모델14·campaign16·준비5·launcher21)이 통과했다. CPU 검증은 공통 초기값·FULL/NONE 원본 일치, 활성 gradient·비활성 호출0, 인과성·padding·상태 초기화와 checkpoint 오표기 거부를 확인한다. native GPU 검증은 아직 실행하지 않는다.

**사용자 시작 지시 뒤 두 서버의 실행 가능 상태를 확인한다 — 외부 작업 대기**

- 5090의 기존 작업을 중지하거나 기존 tmux·GPU 프로세스를 삭제하지 않는다. 현재 점유 여부를 추정으로 확정하지 않고 다음 시작 지시 시 실제로 확인한다.
- 그때 두 서버의 GPU UUID·기존 Python/Torch/CUDA/cuDNN, GPU 유휴, 재부팅 요구·driver/NVML 일치·업데이트와 패키지 잠금을 확인한다. 하나라도 점유·불일치·실패이면 배포·학습을 시작하지 않고 이유를 보고한다. 단독 서버로 임의 전환하지 않는다.
- 새 격리 root에 봉인 payload를 한 번 배포하고 host별 native qualification과 합성3worker 동시 수용 검증을 한 번 수행한다. **두 서버가 모두 실제 exit0로 통과한 뒤에만** 학습 permit을 발급한다. native 미실행을 로컬 PASS로 대신하지 않는다.
- 시작 intent·exclusive claim·실패를 보존한다. 조건36시간·host168시간 lease와 소유 작업에만 적용되는 timeout을 유지한다. 기존 공용 Runtime·driver·인증·다른 원본은 변경하지 않는다.
- 초기 실제 학습 진입·PID/PPID·고정 GPU·checkpoint 저장을 확인한 뒤 지속 원격 polling을 하지 않는다. 이후 결과 확인은 사용자 요청 시 수행한다. 자동화를 새로 만들지 않는다.

**모든 대조와 불리한 결과를 같은 선택 기준으로 정리한다 — 다음 작업 / 학습 종료 후**

- 실제 worker·supervised dispatcher 종료, selected/last checkpoint·history·timing·원본 manifest SHA와 모든36개 ID를 확인한다. 원본 회수·분석은 종료 확인 뒤 진행한다.
- 전체/큰 수량 RMSE·MAE·TimeNLL 여섯 지표를 같은 선택 epoch에서 보고한다. seed별 paired 차이, 평균·표본 SD, FULL 대 세 대조와 단일 경로 조건부 효과를 모두 남긴다.
- 사전 개발 지지 기준은 같은 두 개 이상 데이터셋의 각각 모든3seed에서 FULL 전체 raw 수량 RMSE가 NONE·CNN-only·GRU-only보다 strict 개선되는 것이다. 통과하지 못한 데이터셋·seed와 시간·큰 수량 손해를 제외하지 않는다.
- 낮을수록 좋은 지표의 상호작용은 I=M11−M10−M01+M00다. 음수는 그 지표 척도에서 추가적인 결합 개선을 뜻할 수 있으며 RMSE와 MSE에서 불변하지 않다. 기계적 시너지·통계 유의성·보편적 필수성·참신성·새 독립 일반화 증명으로 확대하지 않는다.
- 세 작업 동시 실행과 FULL 단독 실행의 부하가 다르므로 arm별 wall time을 구조 효율 순위로 쓰지 않는다. 서버·GPU·동시 작업 구간을 따로 기록한다.

**핵심 근거가 확정된 뒤 원고와 후속 검증을 연결한다 — 다음 작업**

- 이번 결합 대조의 결과를 먼저 판단하고 CNN+GRU54 중심 Results·Methods·주장을 맞춘다. 현재 열린 main.tex는 이번 준비에서 수정하지 않는다.
- RAF 수량 목표나 Intermittent 시간 폭의 실제 개선이 필요하면 이번 구조 결과와 연결되는 최소 별도 계약을 검토한다. 진단 결과만으로 원인을 확정하거나 자동 학습하지 않는다.
- 최종 모델·대조를 동결한 뒤 미노출 확인과 CPU binary 감사의 별도 범위를 정한다. 검색은 핵심 주장에 필요한 경우만 검토하며 현재 대기다.

로컬 파일 구현·검증은 독립 작업으로 병렬 진행한다. 공통 실행 계약·원본 봉인·양 서버 qualification gate와 원고 편집은 의존관계 순서로 확정한다.


준비 계약 canonical SHA는 `593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3`다. source131·input7·operation4와 로컬12파일, 배포 archive143 member를 SHA로 검증했다. local-only launcher check의 실제 exit0와 원래 launch hold 보존, 새 authorization·release·permit 부재를 확인했다. 실제 native 검증·학습·Test는 미실행이다. [실행 순서](EXECUTION_RUNBOOK.md), [독립 QA](PRELAUNCH_QA.md)를 기준으로 다음 사용자 지시 뒤 진행한다.
