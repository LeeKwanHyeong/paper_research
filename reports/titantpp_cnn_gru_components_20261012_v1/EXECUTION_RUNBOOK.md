# 사용자 시작 지시 뒤 적용할 실행 순서

**공통 계약과 실행 대기 상태를 확인한다 — 로컬 준비**

- 대상 저장소는 `/Users/igwanhyeong/PycharmProjects/paper_research`, 현재 브랜치는 `codex/contextual-retrieval-pilot-20261007`이다. 기존 사용자 변경·원고·결과를 보존한다. 이 준비에서는 commit·push·원격 동기화·실제 학습을 수행하지 않는다.
- 로컬 bundle은 `search_artifacts/titantpp_cnn_gru_components_20261012_v1`이다. `launch_hold.json`을 원본으로 보존하고 시작 authorization·release·permit은 생성하지 않는다. 준비 단계의 실행 가능 검사는 다음 local-only 명령이다.

```bash
/usr/local/bin/python3 paper/scripts/launch_titantpp_cnn_gru_components.py --bundle search_artifacts/titantpp_cnn_gru_components_20261012_v1 --mode check
```

- SHA·root·배정·계약 불일치이면 원격0으로 중단한다. 파일을 조용히 덮어써 통과시키지 않는다. 생성기의 재실행도 기존 bundle을 대체하지 않는다.

**새 사용자 시작 지시를 계약에 결합한다 — 사용자 지시 대기**

- 사용자의 다음 “진행하자” 지시를 확인한 후에만 `launch_authorization.json`과 별도 `launch_release.json`을 발급한다. 이는 원본 hold를 삭제하거나 바꾸지 않고, 새 지시와 계약·hold 원문 SHA를 연결하는 행정 기록이다.
- authorization에는 approved=true, explicit_new_user_instruction=true, 정확한 contract_sha256, hosts=[5080,5090], 실제 user_instruction과 KST 시각을 적는다. release에는 released=true와 같은 계약 SHA, original_hold_raw_sha256, authorization_raw_sha256을 적는다. 이런 기록이 없으면 check 이외 모든 launcher 모드는 SSH 전에 거절된다.
- 해당 시점 로컬 지침에 따라 이번 연구의 신규 source·운영 코드만 저장소의 위 브랜치에 독립 commit으로 확정하고 SHA를 다시 확인한다. 기존 staged 원고나 다른 사용자의 변경을 commit에 포함하지 않는다. push는 실행 준비에 필요하지 않다.

**두 서버가 모두 실행 가능할 때 새 root를 한 번 배포한다 — 다음 작업**

- 향후 `--mode deploy`는 두 서버의 읽기 전용 건강 확인을 먼저 수행한다. 같은 GPU UUID·기존 Runtime·유휴·driver/NVML·reboot-required·업데이트·실제 패키지 잠금을 확인한 뒤 두 서버가 모두 통과해야 배포한다.
- 5090이 여전히 다른 작업에 사용 중이면 두 서버 모두 새 배포·qualification·fit0으로 대기한다. 기존 작업을 kill하거나 자동으로 5080 단독 실행으로 전환하지 않는다. 건강 관측 원본은 보존하며 새 사용자 요청의 재확인과 자동 학습 재시도를 구분한다.
- 배포 root는 아래 두 곳뿐이다. 원래 프로젝트 checkout이나 다른 실험 root·공용 Runtime은 변경하지 않는다.
  - 5080: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_components_20261012_v1_5080`
  - 5090: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_components_20261012_v1_5090`
- 봉인 payload의 모든 member·raw SHA·source131·input7·operation4를 검증하고 새 root에만 추출한다. root나 intent가 이미 있거나 실패하면 보존하고 자동 재배포하지 않는다.

**두 서버의 native 검증을 먼저 끝낸다 — 다음 작업 / 실제 학습 전 필수**

- 배포 증적 뒤 향후 `--mode qualify`를 한 번 실행한다. 각 서버의 별도 tmux에서 qualification phase를 시작하며 실제 source·입력 모집단·공통 초기화·3seed 합성 검증과 합성3worker 동시 GPU 수용을 확인한다. 새 실험 fit은0이다.
- qualification phase는 campaign/native worker의 실제 OS exit·argv·PID·PPID와 완료·GPU 부재를 기록한다. wrapper 자신의 OS exit는 직접 관측하지 않았으면 미관측으로 유지한다.
- 사용자가 결과 확인을 요청한 시점에 `--mode status`로 host별 읽기 전용 snapshot을 한 번 확인한다. 연속 polling이나 새 자동화는 하지 않는다. 확인된 종료 cache는 원관측시각·SHA를 유지한다.
- 두 host의 실제 passed receipt·종료 증적·원문 SHA가 모두 있어야 다음 학습 단계가 가능하다. 하나만 성공하거나 pending/failed이면 새 fit0이다. 실제 native 호환성과 메모리 수용은 지금 로컬 PASS와 별개이며 아직 미확인이다.

**양 서버 학습을 시작하고 실제 첫 진입을 확인한다 — 다음 작업 / 전역 검증 뒤**

- 향후 `--mode train`은 양 qualification을 같은 permit에 결합한 뒤 각 host의 새 train phase를 한 번 시작한다. 모든 fit이 자체적으로 이 양쪽 증적을 다시 검증한다.
- 5080 Taxi→RAF, 5090 Intermittent의 독립 큐를 동시에 진행한다. 각 데이터셋의 seed42→52→62와 네 구조의3+1 barrier를 유지하며 다른 seed를 빈 slot에 미리 넣지 않는다.
- 첫 실제 fit의 명령·PID/PPID·고정 GPU·동시 worker와 저장 checkpoint/history/timing을 확인한다. 첫 epoch 전에는 저장 전이라고 적고 admission을 학습으로 세지 않는다. 전체 큐 ETA는 미확정이며 lease deadline을 ETA로 쓰지 않는다.
- 조건36시간·host168시간, exclusive claim·실제 child exit·failure와 소유 process group 정리를 유지한다. 과학 fit의 자동 retry·resume·checkpoint warm start는0이다. 상위 종료 시 해당 campaign이 자신의 별도 session 자식들을 정리할 시간을 확보한다.

**자연 종료를 확인한 뒤 원본과 분석을 확정한다 — 다음 작업 / 학습 종료 후**

- 완료가 확인된24/12개의 worker·supervised dispatcher exit0와 마지막 phase complete, 실제 소유 process/GPU 부재, 각 terminal manifest와 selected/last binary를 확인한다. live/pending/identity mismatch를 완료로 바꾸지 않는다.
- immutable 원본 회수와 로컬 manifest/archive·모든 파일 SHA를 확인한 후 같은 선택 epoch의 여섯 지표·seed별 대조·mean/sample SD와 사전 결합 지지 기준을 분석한다. 분석·회수의 실제 완료 전에는36개 완료라고 보고하지 않는다.
- 새 Test 추론·검색·목표/시간 head 변경·RunPod 작업은 이 실행 순서에 포함되지 않는다. 논문 후속 순서는 README와 갱신된 PAPER_WORK_ORDER를 따른다.

실행 CLI의 deploy·qualify·train·status 명령은 위 조건을 만족한 다음 단계용이다. 이 문서와 명령 예시 자체는 실행 승인이나 현재 서버 유휴 확인이 아니다.
