# v2 독립 host 시작의 QA 기록

상태: 진행 중. 최신 사용자 지시는5080 먼저 실행하고5090은 별도 명령 전까지 모든 원격 작업0이다. QA 검토자는 소스·원격·GPU를 직접 실행/변경하지 않고 로컬 코드·봉인·전달된 native 증적만 확인한다.

## 현재 기준선 — 완료

- v1 local-only launcher check 실제 exit0을 재확인했다. canonical `593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3`, hold `e2441d121f4337b0f347891adfa285bb64030319a51e726b3cf75783a8e99983`, localtoolseal `c13a8fd652834b73ea66fa6998e14a26583ae9fa7494821262a16f618557c82f`가 보존됐다. source131·input7·operation4 일치, remote0이다.
- 과학 source131 `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`와36ID·동일 초기화·데이터·손실·optimizer·selector·4arm3+1을 유지하며 운영 승인·schema/root만 v2로 분리한다.
- TEST_SESSION_PROTOCOL0·1·9·12를 다시 확인했다. `.codex/agents/qa-reviewer.toml`의 프로젝트 설정은 Astra/high다. 이전 oma-qa 적용을 이어가며 `.agents`에 기록하지 않는다.

## 검토 완료 조건

- [x] v1 원본·소스·bundle·원고 보존
- [x] v2 host별 승인/hold release가 해당host와 계약에 결합
- [x] 5080 독립 local gate·own qualification으로만 진행
- [x] 5090 health/status/SSH/deploy/qualification/fit0 강제 negative tests
- [x] 동일 과학 source/input/구조·초기화·예산·선택 규칙 검증
- [x] v2 소스·운영·payload·localtool seal stdlib SHA 검증
- [ ] 5080 실제 native 결과의 식별자/exit0/CPU와GPU 구분 확인

## Findings

판정 미완료. 신규 파일 준비를 기다리며 미확인 사항을 결함이나 통과로 보고하지 않는다.

## 로컬 코드 검토 — 완료

- v2 campaign20, preparation3+launcher11 총34개 독립 CPU/fake-remote 회귀 PASS. singletonhost와 ownq bridge, 미승인5090의 모든 원격 모드 SSH0 및5080 동작의5090 호출0을 확인했다.
- 초기 운영기/실행기 정책 키 불일치는 host_independent_start로 통일해 해결했다. package import가 있는 prepare_v2는 -m 호출에서 --help exit0이며 직접 파일 실행은 ModuleNotFoundError를 발생시키므로 root에 module CLI 사용을 알렸다.
- v1 과학/구조·선택 규칙과 scheduler/fit/native 함수 보존, 큰 payload stdin JSON transport 안전성을 확인했다. 현재 미해결 CRITICAL/HIGH/MEDIUM/LOW0, 봉인·실native 증적은 대기다.

2026-10-12T17:24:17+09:00 (KST): v2 봉인 및5080 승인 scope 독립 stdlib PASS.19개 localtool/143tar member 포함, 과학source131 불변·5090 authorization/release 부재·원고 보존 확인. PRELAUNCH_PASS_5080_ONLY_NATIVE_PENDING.

## 실제 건강 조회의 차단 — 확인

5080 health exit1/진단 PermissionError를 로컬 보존 증적으로 확인했다. /var/lib/dpkg/lock O_RDONLY 권한 거절이며 학습 실패가 아니다. 잠금 미확인을 정상으로 우회하지 말고 읽기 전용 판별 보완을 권고했다. NATIVE_QA.md에 native 미실행으로 기록했다.

##5080 읽기 전용 건강 판별의 행정 보완 — QA 완료

최초 health는 root 소유 패키지 lock 파일을 open하는 단계의 PermissionError로 끝났으며 mutation·native·fit0이었다. 원래 v2 launcher/source/bundle·실패 증적을 바꾸지 않고 별도 `paper/scripts/launch_titantpp_components_5080_health_compat.py`를 검토했다.

- 독립5개 회귀 PASS(0.002s): root 소유 lock을 열지 않고 stat inode/device와 `/proc/locks`를 비교, 다른 inode/device 거부, lock inventory 읽기 실패는 fail-closed, 다른 host 호출/업로드 거부, 원격 health 단일 block만 변경.
-5080과 v2 canonical 및 sealed launcher SHA를 고정한다. 인메모리 health 문자열만 대체 후 finally 복원하며 배포·qualification·train·status 동작은 봉인 v2에 위임한다.
- bridge rawSHA: `fac25ea166692e1500169cce6ad7a1c8bef566a4ee9fabe3b4274c6ead6e31be`. 보완된 health script SHA: `13a08649761e9987063375214e07c3d72e950daf388bd65bf9dc9321d91c0784`. 행정 receipt로 원문 provenance를 결합하며 기존 source131·Runtime·권한·driver·인증은 변경하지 않는다.
- 이 PASS는 잠금 관측 구현의 검증이다. 현재 실제 GPU/Runtime/유휴/lock 및 native 통과 여부는 새 실제5080 증적에서 확인해야 한다.5090 작업0 경계는 유지된다.

## 상주 종료 대기 서비스의 엄격한 분류 보완 — 로컬 QA 완료

root는 실제 PID1514의 argv가 `/usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal`이며 종료 신호 대기 서비스임을 읽기 전용 확인했다고 전달했다. 실제 분류는 후속 건강 receipt에서 확인한다.

`launch_titantpp_components_5080_health_compat_v2.py`의 독립 코드 검토와 기존 lock5개+observer4개, 총9개 회귀 PASS(0.006s). argv exact, stat start_ticks 전후 일치, sleeping S/PPID1/rootUID, helper source SHA `8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0`, logind `PreparingForShutdown=b false` 모두 통과해야 idle_shutdown_observers로 분리한다. 실제 updater·모르는 argv·종료 진행·source변경·조회실패는 계속 차단한다.

- 새 bridge rawSHA: `eeec07a62f425b56c12d48537559f84bf097c964dfc70a2f31db8f66ae2e9f12`.
- 새 health script SHA: `6421df74575e446ea0cb7104d8f164dfe18f2dd6f60c1c6414480011818d51f3`.
- 봉인 v2·bridge1·최초 실패·이전 waiting 증적은 그대로 보존한다. 행정 receipt_v2로 별도 provenance를 결합한다. OS 프로세스 중지·권한/Runtime/driver 변경은 없다.5090 호출 금지 그대로다.

## HIGH — v2 실제 interface의 필수 계약 key 누락

`paper/scripts/prepare_titantpp_cnn_gru_components.py:85`의 parent 필드 복사에서 design_sha256를 제외했고, 봉인 `source/paper/scripts/run_multilag_detail_execution.py:319`는 이를 필수 접근한다. v2도 그 계약을 보존해 실제 qualification이 exit1로 끝났다. 기존 CPU/fake 검증의 공백이다. v2 재봉인/재시도를 하지 않고, 원 frozen design key와 provenance를 새 v3 계약에 명시하며 실제 Train/Validation interface 통합 회귀를 통과해야 한다. source131 변경은 필요하지 않다. 상세 원관측은 NATIVE_QA.md 참조.
