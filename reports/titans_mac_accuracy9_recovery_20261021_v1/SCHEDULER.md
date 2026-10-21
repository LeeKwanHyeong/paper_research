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

## 실제 배포 의존성 보완 — execution_v2

- v1은 누락된 운영 helper import에서 멈췄다. 원격 run 디렉터리와 GPU worker가 없었음을 `preflight/FAILED_LAUNCH_v1.json`에 기록했다. 새 scientific fit/재개 step은0이며 v1 root·stderr·계약을 보존했다.
- 현재 작업 경로는 위 v1 이름 대신 `titans_mac_accuracy9_recovery_20261021_v2_5090`, 로컬 bundle은 `search_artifacts/titans_mac_accuracy9_recovery_20261021_v2`, 운영 보고는 `execution_v2/`다.
- 과학 소스989는 그대로 두고, 운영 파일은 recovery worker/runtime와 기존 bounded helper(SHA4edc46490bf67c4bb47f8b7fc9b217347afc0802c98aa4f1a6066010078de377)3개를 봉인했다. 기존 helper는 수정하지 않았다.
- 격리된 source989+운영3만으로 import하는 회귀 검사와 설치 직후 native --modecheck gate를 추가했다. 이 gate가 실패하면 tmux 시작을 요청하지 않는다. 운영26검사·worker17검사·네트워크20검사 모두 실제exit0, 독립 검토PASS다.

## 원래 프로젝트 시작 폴더까지 복원 — execution_v3

- v2는 reference6개의 복사를 완료했으나 train3 child는 training 모듈 import에서 실제exit1로 종료했다. 원래 prepare가 만든 빈 `source/sample_data` 폴더가 새 배포에 누락되어 project-root 탐색이 차단된 운영 오류다. 모델 복원·optimizer step 전 실패이며 모든 원문·v2 root·세 actualexit를 보존한다.
- 현재 manager는 `execution_v3/`, `search_artifacts/titans_mac_accuracy9_recovery_20261021_v3`, 원격 `titans_mac_accuracy9_recovery_20261021_v3_5090`만 대상으로 한다. v1/v2를 재실행하지 않는다.
- 원래 project bootstrap인 models/utils/sample_data 세 디렉터리를 고정하고 science989·입력·운영3과 함께 새root에 구성한다. native 시작 전 전체 training/recovery import를 검증하므로 폴더 부재가 scientific child 시작으로 이어지지 않는다. 과학 코드는 수정하지 않는다.

## 학습은 유지하고 후속 검증 연결만 분리 — execution_v4 준비

- v3은 실제 2026-10-21 14:17:07KST Intermittent3 worker를 시작했다. 14:18:32KST PID57080/57081/57082·start_ticks200976·정확한 command/cwd/pgid와 해당GPU3process 및 backward 경로를 확인했다. checkpoint 사본은 원76/74/67 SHA 그대로였다. 이 관측은 새 epoch 저장 완료와 구분한다.
- v3의 세 번째 Popen 직후 신원 조회 순간 경합으로 `Owned child identity changed`가 기록됐다. dispatcher는 이미 시작한3fit을 끝까지 drain하며, 학습은 중단·중복 실행하지 않는다. 원 failed 표식·claim·로그를 삭제하거나 성공으로 바꾸지 않는다.
- 새 v4 completion bridge는 같은5090에서 짧은 로컬 파일 확인으로 v3의 실제3fit exit0와 완료receipt를 기다린다. 새 학습은0이다. v3의 동일 운영 오류와 모든 성공한 scientific trainreceipt만 확인한 뒤9fit 결과를 새root로동일SHA복사하고 Validation9→고정Test9를 수행한다.
- seed62의 원 train_exit.start_ticks가None이면 그원문을보존하고, 이미확인한initial live PID/start_ticks/argv/cwd/pgid와원부모의actualexit0를연결한새사본에만보강한다. supervisor가 worker OSexit를직접봤다고추정하지않는다.
- v4의 새 검증 child는 actualPopen직후에만 최대2초의한정된 exec진입확인을 사용한다. 이미등록된 child의신원불일치는재시도하지않는다. 5080·원학습·checkpoint·과학source989/Runtime/기한은변경하지않는다.


## 실제 가동 확인 — 2026-10-21 14:32 KST

**네트워크 상태·관리자 로그를 자동 수집한다 — 완료**
- 사용자 설치 후 root 서비스 `paper-5090-network.service`가 14:27:24 KST부터 실제 PID82489로 실행 중이고 부팅 시작도 enabled임을 확인했다. 사용자 서비스는 inactive/disabled로 중복 실행하지 않는다.
- 14:32:04 KST 실제 관측에서 설치 collector SHA는 `efb930ec8045618117e1116e27343ad5837ff0708b59f0066fce7fe229e6c369`와 일치한다. `/var/log/paper-5090-network/network.jsonl`은 root 소유·그룹 읽기0640이며 uid0의 실제 journal 수집 returncode0과 10초 snapshot이 계속 기록된다. 로그는 10MiB×8개로 제한하고 원 사용자 로그도 보존했다.
- 증거: `network/v2/ROOT_VERIFIED.json`, 해당 ROOT_VERIFICATION 원문. 네트워크·SSH/Tailscale 설정과 인증정보는 변경하지 않았다. 이번 수집은 다음 장애의 원인을 구분하기 위한 것이며 이전 단절의 원인을 확정한 결과가 아니다.

**진행 중인 학습을 유지하며 후속 검증을 연결한다 — 진행 중**
- v4 bridge는 14:27:50 KST 설치·preflight·tmux launch actualexit0, 14:30:43 KST actual start와 `waiting_for_preserved_training`을 확인했다. dispatcher PID83345/start_ticks265346, contract SHA `191d2534245ab6c02efff817fe9e21bb43e6589415b1839e22f1a7dec40797c2`, 새 학습0이다. `execution_v4/ACTUAL_WAITING.json`과 실제 transport 원문을 기준으로 한다.
- v3 Intermittent3의 실제GPU PID57080/57081/57082는 14:31:22 KST에도 동일하게 실행 중이다. 당시 마지막 저장 epoch는76/74/67로 아직 새 epoch 저장 전이며 실행 중인77/75/68 완료를 추정하지 않는다. 원본1,141개 SHA는 모두 그대로다.
- 다음은 실제 새 epoch 저장 확인, 원래3fit 성공 종료, v4 Validation9 전체 재현 gate와 고정Test9, 한 번의 회수다. 회수 도구의 로컬 독립 검증이 끝나기 전에는 실제 회수를 요청하지 않는다.


## 실제 재개 후 새 checkpoint 저장 확인 — 2026-10-21 14:35:20 KST

**중단 전 상태부터 새 epoch를 저장한다 — 완료**
- Intermittent seed42는76→77, seed52는74→75, seed62는67→68 epoch를 새 경로에 저장했다. 각 fit의 누적 optimizer step은233852→236929, 227698→230775, 206159→209236으로3077씩 증가했다. 실제 GPU 사용률94%, 세 PID57080/57081/57082 연산 중이다.
- 새 checkpoint SHA는 각 job의 `execution_v3/RESUMED_NEW_EPOCH_VERIFIED.json`에 기록했다. 중단 전 원본 checkpoint·완료6fit·과거실패를 포함한1,141파일은 원 SHA와 모두 일치한다. 재개 진행을 확인한 것이며 전체 학습·Validation/Test 완료가 아니다.

**짧은 종료 확인 뒤 한 번 회수한다 — 진행 중**
- 회수 코드/테스트 독립30검사 actualexit0, 실제모듈 smoke와 CLI check-local remote0 통과. 코드 commit `a5cd89f5`, collector SHA `f9be7ab6a26aace6ede264b02b66022955954e7e7fc81510a983b32a279c4b8a`.
- 첫 실제 advance는14:34:28 KST `waiting`, SSH1/actualexit0/0.238초/회수0이다. C4 PID83345/start_ticks265346의 실제 신원 일치를 검증했고 원문과 SHA는 `execution_v4/collection_v1/rounds/20261021T053428_504b17561750`에 보존했다.
- 이후 확인은900초 간격의45초 상한 단일 SSH이며 기다리는 SSH·별도 finisher를 만들지 않는다. owner/GPU부재와27scientific actualexit0(원6+재개3+새Validation9+Test9), dispatcher actualexit0 및전체SHA를 확인한 뒤 최초900초 상한 전송을 수행한다. 회수 실패/partial/기존attempt는 보존 후 차단하며 자동재회수하지 않는다.


**회수 이후 기존 Shared3seed와 비교한다 — 다음 작업**
- 기존 비교 CLI는 원래 v1 회수 경로에 묶여 있어 변경하지 않고 별도 `compare_titans_mac_accuracy9_recovery` local-only wrapper가 기존 순수 비교 함수를 재사용한다. 회수 검증을 선행하고, 새 `execution_v4/comparison_v1/`에만 배타적으로 작성한다. partial은 보존하며 자동 덮어쓰지 않는다.
- 독립19검사 actualexit0과 실제모듈 미회수 check-only remote0 통과. 원본 scientific contract·Sharedmanifest·공통조건·전체원본SHA 검증 후3데이터×Validation/Test×3seed의36집계/108seed별 지표를 계산한다. 전체/큰수량 qty RMSE·MAE·recorded TimeNLL, 평균±표본SD(ddof1), 절대/상대차이와 불리한seed를 보존한다. 기존 Shared Test5080/cuDNN91900과MAC5090/cuDNN92000 평가환경 차이도 기존 함수로 명시한다.
- 지금은 미회수이므로 실제 Test값을 읽거나 비교보고를 생성하지 않았다. 완료 뒤 check-only→별도보고생성→check-only로 검증한다. 기존3seed와5080추가2seed 결과·자동화는 변경하지 않는다.


## 최종 연결 상태 — 2026-10-21

**MAC 완료 확인과 회수를 별도 자동화로 이어간다 — 완료**
- 별도 heartbeat `mac9`(MAC9 재개 검증·회수 마무리)을 현재 채팅에15분 간격 ACTIVE로 등록하고 실제 저장 설정을 다시 확인했다. `execution_v4/AUTOMATION.json`에 도구 생성·대상thread·promptSHA를 기록했다. 기존2seed와 다른 자동화는 변경하지 않았다.
- 매회 `CONTROL_BINDINGS.json` SHA `f255070c3d55089d6209a690d0a1c1791afd4ec33ed9760f831cb2d47ed7db01`과 모든 운영 소스SHA 검증 후 `collect_titans_mac_accuracy9_recovery --completion-bridge --mode advance`를 정확히 한 번 실행한다. 정상 waiting/busy는 조용히 유지한다. 통신/과학/partial실패 원문을 보존하고 차단 후 한 번 알리며 자동학습/재회수하지 않는다.

**남은 과학 실행과 보고를 기다린다 — 외부 작업 대기**
1. 현재5090 v3의 Intermittent3fit을 원래 예산으로 끝낸다. 새77/75/68epoch 저장까지는 확인 완료다.
2. 이미 실행 중인 v4 bridge가 실제3fit exit0를 확인한 뒤 Validation9 전체 재현 gate→고정Test9를 수행한다.
3. mac9가 실제 성공 종료·owner/GPU 부재와 전체SHA를 검증하고 한 번 회수한다.
4. local-only 비교 check-only→별도3seed보고생성→check-only 후 결과·불리한seed·평가환경 제한을 보고하고 mac9만 종료한다. 전체 실험 완료로 앞서 보고하지 않는다.
