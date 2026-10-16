# CNN×GRU 36개 독립 QA 준비 기록

상태: 로컬 검토 완료 — LOCAL_READY_REMOTE_PENDING. 최신 사용자 지시는 5080/5090 분담 계획·코드 준비까지만이며 다음 진행 요청 전 SSH·GPU·학습0이다. 최종 목표 상태는 LOCAL_READY_REMOTE_PENDING이며 launch 승인이 아니다.
검토자는 SSH·GPU 실행·Test 성과 열람·소스 수정 없이 로컬 자료만 검토한다.
프로젝트 QA 역할 설정 `.codex/agents/qa-reviewer.toml`은 Astra/high다.
`.agents/skills/_shared/runtime/execution-protocols/codex.md`는 존재하지 않아 사용자 skill 경로의 oma-qa 및 resources/execution-protocol.md를 적용했다. `.agents`는 변경하지 않는다.

## 현재 확인한 설계 근거

- 사용자 최초 선택: 네 구조 fresh 최대36 fit. 최신 준비안: 5080 Taxi+RAF24개, 5090 Intermittent12개. 각 dataset의 모든 seed42/52/62와 네 구조는 같은 host/Runtime에 배치한다. 과거 checkpoint warm start 없음. 두 host의 Runtime 차이는 기록하며 dataset 간 wall-clock 비교는 host 차이와 분리되지 않는다.
- 기존 `models/TPPs/CountAwareTitanCNNGRU.py:53`과 `:76`에서 `titantpp_cnn_mlp16`은 MLP16 보정24,576개를 포함한다. 따라서 과거 CNN+MLP arm을 새 대조의 CNN-only로 이름만 바꿔 사용할 수 없다.
- 예정 active parameter 수는 없음89,859 / CNN90,435 / GRU114,591 / CNN+GRU115,167이다. CNN 추가576개와 GRU 추가24,732개로 용량이 다르므로 이 실험은 경로 제거 대조이며 동일 용량 원인 분리 대조가 아니다. 신규 코드와 독립 CPU 테스트에서 이 수를 재확인했다.
- 낮을수록 좋은 지표에서 상호작용은 I=M11−M10−M01+M00다. 음수는 해당 지표 척도에서 추가적인 결합 개선을 뜻할 수 있으나 RMSE/MSE 척도에 불변하지 않다. 기계적 시너지·보편적 필수성·참신성 증명으로 확대하지 않는다.
- wave3+1은 동시 작업 부하가 다르다. arm별 wall-clock 학습시간을 독립적인 구조 속도 우위로 해석하지 않는다.

## 사전 완료 조건

- [x] 기존 과학129파일 보존과 신규2파일을 포함한131파일 closure 확인
- [x] 네 경로, common tensor/RNG, 원 full 초기화 및 forward 동일성 검증
- [x] CNN-only에 MLP/GRU 미활성, GRU-only에 CNN 미활성, 없음 모두 미활성 및 optimizer 제외 확인
- [x] causal·padding·withheld·state reset·gradient CPU 테스트 확인
- [x] 동일 데이터/분할/손실/최적화/Validation selector/예산과 Test 잠금 확인
- [x] 실행 계약 및 입력·operation seal과 승인 범위 확인
- [ ] 5080 및5090 native qualification 실제 exit0 및 GPU/Runtime 식별자 확인 후 fit permit 발급
- [x] max3·같은 dataset/seed·wave3+1, claim/timeout/OS exit 및 재시도0 확인
- [x] 전체 paired contrast·mean/sampleSD·상호작용·여섯 지표·불리한 결과를 사전 명시

## 검토 결과

현재 미해결 CRITICAL/HIGH/MEDIUM/LOW 결함은 없다. 마지막 launcher 검토에서 발견한 MEDIUM 종료 증적 결합 누락도 수정·독립 회귀 통과했다. 아래 HIGH 두 건은 독립 재현·수정·회귀 통과 후 해결됐다. 최종 실행기 및 immutable bundle 검증은 아직 진행 중이며 native 실행은 미실행이다.

## 독립 CPU 초기 검증

- 신규 모델·runner AST 통과. `run_titantpp_cnn_gru_components.py --synthetic-check --seed42`는 초기값·활성 parameter·gradient 구간을 통과한 뒤 GRU padding의 `torch.equal`에서 실패했다.
- 별도 최소 재현에서 원 FULL과 새 GRU-only 모두 finite이며 padding 최대 절대차1.4901161193847656e-08, reset 최대 절대차1.1175870895385742e-08이었다. 서로 다른 projection shape의 미세 부동소수점 차이로 확인되며 실제 이력 누출로 판정하지 않는다.
- 해결한 HIGH(local gate 차단): `paper/scripts/run_titantpp_cnn_gru_components.py:240` 및 `:246`의 다른 shape 간 bitwise 조건. 권고: finite 확인 뒤 tight float32 tolerance의 `torch.allclose`로 해당 두 비교만 평가하고 관측 차이를 기록한다. 같은 shape의 future/withheld exact assertion은 유지한다. 모델 과학 소스 변경은 필요하지 않다.

- 수정 뒤 동일 CLI seed42 독립 재실행 PASS. 다른 shape 비교에 finite + rtol1e-5/atol1e-6만 적용했고 실제 관측 차이는 ≤1.49e-08이다. 같은 shape 인과성 검사는 exact를 유지한다.
- 독립 `python -m unittest paper.tests.test_titantpp_cnn_gru_components -v`:14개 PASS (0.547s). 전체3seed 초기값/gradient·inactive parameter optimizer 제외·FULL/NONE 원경로 동등성·checkpoint 오표기 거부·Validation 전용 replay를 확인했다.
- 현재 초안 계약/운영기 통합에서는 extension_provenance 문자열 불일치와 모든 host qualification 선행 정책의 코드 강제를 담당자에게 전달했다. 완성본에서 재검토하며 아직 최종 판정하지 않는다.

## 원격 준비 전 소유 프로세스 정리 검토

- 해결한 HIGH(수정 전 실행 차단): phase wrapper `paper/scripts/run_titantpp_cnn_gru_component_phase.py:108`은 campaign process group만 SIGTERM한다. Campaign의 native/fit child는 별도 `start_new_session=True`로 생성되므로 이 signal을 받지 않는다. Campaign의 기본 SIGTERM은 Python except cleanup을 실행하지 않아 손자 timeout/worker가 남을 수 있다. Qualification은 최대5400초까지 잔류할 수 있다.
- 권고 수정: campaign이 SIGTERM을 명시적인 예외로 변환하는 handler를 main scope에 설치해 기존 qualify/dispatch의 owned-child 정리를 실행하고, wrapper grace가 세 자식의 bounded cleanup보다 길도록 맞춘다. 다른 process group을 조회해 무차별 kill하지 않는다. 작은 실제 CPU process-tree 회귀검증으로 shutdown 후 손자 부재를 확인한다.

- SIGTERM scoped handler가 예외를 발생시키고 재신호는 정리 중 무시하도록 수정했다. Wrapper는75초 정리 유예를 준다. 독립 ops16+prepare5, 총21개 테스트 PASS(0.169s), 실제 CPU 별도 session 자식에 대해 controller SIGTERM 뒤 자식 부재까지 확인했다. 소유 외 프로세스 kill 없음.
- extension_provenance 계약/검증 문자열 일치 수정 완료. 모든 host qualification 실제 종료·각 raw receipt와 native source/runtime/GPU를 확인하는 paired bridge가 fit permit에서 강제된다.
- 사전 결합 지지 기준은 FULL 대 NONE·CNN-only·GRU-only의 모두를 같은2개 이상 dataset의 각각3seed에서 strict개선하는 조건으로 보완됐다.

## 로컬 실행기 검토

- 원본 launch hold를 유지하며 새 사용자 지시와 그 hold 원문 SHA에 결합한 별도 release가 없으면 모든 원격 모드를 SSH 전에 거절한다. `check`는 stdlib 및 로컬 파일 검증만 수행한다.
- 두 host 건강 확인은 배포·phase 이전이며 원격 tar의 모든 member를 추출 전에 검사한다. 링크·경로 이탈·중복·extra·SHA 변조를 거절하고 파일 생성은 exclusive다. SSH command와 tmux environment는 shlex quoting으로 구성한다.
- 원격 전역 자격 gate와 기존 intent/failure 보존을 확인했다. root가 발견한 건강 관측 재요청 및 학습 status 구분 수정·최종 회귀 테스트는 아직 기다리는 중이다.

## 종료 상태 증적 결합 보완 요청

- MEDIUM(최종 봉인 전 수정 필요): `paper/scripts/launch_titantpp_cnn_gru_components.py:542` 이후 train 완료 분기는 exit의 os_exit_code/contract만 확인하며 train phase·launch PID/PPID/argv 연결·worker 실제 exit 증적을 확인하지 않았다.
- 독립 fake snapshot 최소 재현에서 train_launch 없음, train_exit.phase=qualify·pid999·argv=[foreign], worker_exit 원문0이어도 training_complete_remote_verified가 반환됐다. SSH·GPU·실제 데이터 사용0이다.
- 권고: 종료 snapshot에 worker_exits 원문을 포함하고 state.completed의 worker_exit_sha256 및 manifest SHA와 결합한다. train launch/exit phase·정확 argv·PID/PPID, 각 job의 실제 child exit0를 검사하고 위 malformed snapshot을 거부하는 회귀 테스트를 추가한다. 현재 launcher20개 회귀는 독립 PASS이나 이 완료 분기는 기존 테스트가 없었다.

- 위 MEDIUM 수정 후 launcher21개 독립 회귀 PASS(1.395s). train phase/launch/exit/supervisor claim의 exact argv·PID/PPID, 모든 worker actual exit0·job·parent·timeout argv·claim deadline과 completed rawSHA 연결을 확인했다. malformed 완료 증적은 거부된다. 총 독립 로컬 회귀56개(모델14+campaign16+preparation5+launcher21)가 통과했다. native·실제 학습0은 그대로다.

## 최종 봉인 확인

2026-10-12T17:10:38+09:00 (KST): 신규 immutable bundle의 source131·input7·operation4·localtools12·tar143 member의 exact 집합과 stdlib SHA를 독립 확인해 PASS했다. 원본 hold=true와 실행 authorization/release/permit/control 부재, main.tex 보존을 확인했다. 최종 로컬 판정 PASS / LOCAL_READY_REMOTE_PENDING이며 원격/native/학습/Test0이다.
