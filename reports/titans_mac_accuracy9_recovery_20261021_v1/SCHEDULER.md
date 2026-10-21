# 5090 장애 기록과 MAC9 재개

**원본과 승인 범위를 고정한다 — 완료**
- 2026-10-21 사용자가 다음 네트워크 단절의 상태·로그 수집과 현재 MAC 실험 재개를 명시적으로 승인했다. 기존 결과·checkpoint·실패·source를 보존한다.
- 원 MAC9은 RAF3·Taxi3 완료, Intermittent seed42/52/62만 저장 epoch76/74/67에서 중단됐다. `preflight/ORIGINAL_SNAPSHOT_v3.stdout.json`은 실제 CPU checkpoint 로드와 1,141개 원본 SHA를 담는다. 이전 두 읽기 도구의 오류와 stderr도 보존한다. 이는 학습 실패가 아니다.
- 원본 코드989개, 입력9개, scientific contract, GPU UUID, Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads, 원36시간 fit 예산과 절대기한1792888885.815062를 유지한다. 5080 추가2seed와 자동화2seed는 변경하지 않는다.

**다음 장애의 상태와 로그를 남긴다 — 진행 중**
- 별도 stdlib 수집기가10초 간격 LAN 링크·경로·gateway·SSH listen·서비스·Tailscale의 지정 peer만 기록한다. UTC/monotonic/bootID, 명령 exit/timeout과 수집 오류를 보존한다.
- 로그는 fsync하며10MiB×8개로 제한한다. journal은30초 간격, 256KiB·8초 상한으로 cursor 이후 완전한 행부터 이어간다. 무효 cursor와 누락 가능성을 표시한다. 자동 네트워크 복구·GPU 조회·실험 호출·외부 업로드는 없다.
- 사용자 서비스와 관리자 서비스는 같은 lock을 사용한다. 현재 사용자에게 시스템 journal 읽기 권한과 linger가 없고 `sudo -n true`는 비밀번호 필요(exit1)다. 사용자 수집과 root 시스템 로그·부팅 자동 시작을 구분한다. root 설치를 완료했다고 앞서 보고하지 않는다.

**원 엔진으로 저장 epoch 다음부터 이어간다 — 진행 중**
- 새 root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_accuracy9_recovery_20261021_v1_5090`.
- 원 checkpoint를 동일 SHA로 새 경로에 복사하고 model·AdamW·best·embedded history·Python/NumPy/Torch/CUDA RNG·loader generator를 native 엔진으로 복원한다. 원 source_revision·execution_role·interface를 유지한다.
- 새 joint time-head proof는 prefix의 optimizer step과 관측 수를 복원한다. 현재 state와 best state의 시간 head를 혼동하지 않는다. 이번 segment elapsed_seconds는 중단 전을 합친 총학습 시간이 아니다.
- 완료 RAF/Taxi6개는 무학습 reference copy다. reference worker의 새 exit와 원 fit의 과거 actualexit0를 구분한다. Intermittent3개만 최대3병렬로 재개한다. 새 launch/claim/failure를 삭제하거나 무한 재시작하지 않는다.

**Validation9 재현 뒤 고정 Test9를 평가한다 — 다음 작업**
- 재개3개 종료 후 fullValidation9 모두 원 개별 예측과 수량·시간 지표를 재현해야 Test9를 시작한다. 과거 완료6개의 선택 모델을 변경하지 않는다.
- 새 supervisor는5090 tmux 안에서 dispatcher의 실제 OS exit를 기록한다. SSH 단절이 학습을 종료시키지 않는다. 자신의 OS exit를 직접 봤다고 보고하지 않는다.
- 최종 회수는 모든 stage receipt·child exit·source/input·원본/manifest SHA와 owner PID/GPU 부재를 확인한 뒤 별도 경로에 수행한다. 기존 비교·실패·결과를 덮어쓰지 않는다.

**검증과 실제 배포를 구분한다 — 진행 중**
- 로컬 CPU 합성 exact-resume17검사, 운영 계약·queue25검사와 네트워크20검사를 통과했다. 연속4epoch와2+2epoch의 model/best/optimizer/RNG/loader/history/선택/proof가 정확히 일치한다. 이는 실제 재개 완료나 scientific Test 결과가 아니다.
- 실제 설치·서비스 상태·초기 새 epoch 저장 증적은 후속 receipt에 기록한다. 코드 commit·contract SHA·관측 시각은 해당 실제 파일을 기준으로 확인한다.
