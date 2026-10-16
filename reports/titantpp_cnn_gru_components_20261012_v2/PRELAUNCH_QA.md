# v2 독립 host 실행 사전 QA

상태: **PRELAUNCH_PASS_5080_ONLY_NATIVE_PENDING**. 로컬 코드와 immutable v2 봉인·승인 대조 완료. 5080 시작은 사용자가 승인했으며5090은 별도 지시 대기다.

## Review Result: PASS (현재 로컬 코드 범위)

### CRITICAL

없음.

### HIGH

없음.

### MEDIUM

없음.

### LOW

없음.

## 확인 근거

- 독립 campaign20개 PASS(0.181s), preparation3개+launcher11개 PASS(1.166s): 합계34개다. GPU/SSH 직접 실행 없이 CPU/fake remote로 검증했다.
- 5080 singleton approval과 own qualification bridge만 허용하고 broad approval·다른 host bridge를 거절한다. 5090 승인 없이5090 deploy/qualify/train/status 모두 SSH 전에 거절한다.5080 동작은5090 건강 확인·status·업로드도 호출하지 않는다.
- v1 과학source131·input7·36ID·24/12배정·모델/초기화/손실/optimizer/selector·3+1 barrier는 그대로다. Scheduler·fit·qualification summary 함수 AST 불변 검증도 통과했다.
- Phase wrapper의 실제 exit·소유 자식 SIGTERM 정리 및 완료 증적 결합을 보존한다. v2 transport는 큰 permit을 stdin JSON으로 보내 shell interpolation과 Linux single-argv 한계를 피한다.
- 준비/실행기 정책 키의 초안 불일치는 host_independent_start로 통일돼 해결됐다. 준비기는 `python3 -m paper.scripts.prepare_titantpp_cnn_gru_components_v2`로 실행하는 package CLI를 확인했다.
- v1 local-only check는 source131/input7/op4/로컬 도구 봉인과 원본 hold 모두 일치했다. v1파일·bundle을 덮어쓰거나 그 승인을 재사용하지 않는다.

## 승인과 남은 확인

5080의 사용자 지시를 v2 계약과 원본 hold에 결합한 host별 authorization/release가 필요하다.5090 authorization/release를 만들거나 호출하지 않는다. 이후5080의 현재 건강·유휴·Runtime, 실제 native qualification/admission의 exit0와 증거를 확인해야 fit permit을 사용할 수 있다. CPU PASS는 native 통과나 실제 학습 완료가 아니다.

## QA 변경 파일

이 디렉터리의 REVIEW_NOTES.md·PRELAUNCH_QA.md·NATIVE_QA.md만 소유한다. 소스·bundle·원격·GPU·Test 성과·열린 원고 및 `.agents`를 변경하거나 열람 범위를 확대하지 않는다.

## 독립 최종 봉인·승인 검증 — 완료

관측 시각: 2026-10-12T17:24:17+09:00 (KST). stdlib로 source131·input7·operation4·localtools19 및 tar143 regular member의 exact 집합·내용 SHA를 모두 대조해 PASS했다. 원 v1과 source·입력·데이터·훈련·진단·구조·분석·병렬·36job/배정·arm/seed/policy 필드가 같다.

| 대상 | SHA256 |
|---|---|
| v2 canonical | `5d5abc55beb4c7543504f52ccf1fe43e8576bef9554226e40afe0397a5bbad0d` |
| source131 | `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53` |
| operation4 | `41f7c41cbe5d4f17fc912c94ad9a1aa2f075929a736c2b300a36cb2d18adab8b` |
| archive | `cf17b4814dc7ec76c48a991fe7126f25606c924a2b1150fbd6d1d00343f6ab4b` |
| localtools raw | `83069ece9fe0d369dfe0ceb1d9f96a6be8861d8a98c3b4282aa120e6e06c4e40` |
|5080 authorization raw | `d451a39afada46a10cf780fce8f6040cfb138c099920190cd05f6408843191e0` |
| original hold raw | `fdf650e493ae827915bef964abb123a54e1e10ca65f55f7ce062f83a3a7bd48b` |

5080 authorization의 사용자 원문은 “우선5080 먼저 진행하자.5090은 내가 진행하자 명령하면 그때 하고”라는 승인 범위이며 실제 기록의 공백을 포함한 원문 SHA를 위와 같이 검증했다. hosts=[5080], approved=true, explicit_new_user_instruction=true와 계약 결합, release의 authorization/hold rawSHA 결합이 일치했다.5090 authorization/release는 없다.

main.tex SHA `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354` 보존도 확인했다.5080 실제 native 결과는 아직 미확인이다. 다음 단계는 root가 승인된5080에서만 건강·배포·qualification을 수행하는 것이며 QA 직접 원격/GPU 호출0이다.

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

## 실제 native 후속 결과 — v2 실행 적합 판정 철회

사전 CPU 범위 PASS와 달리 실제5080 qualification은 필수 design_sha256 누락으로 exit1이었다. v2는 현재 실제 실행 gate 실패 상태이며 PRELAUNCH_PASS를 native 성공으로 사용하면 안 된다. 원봉인·실패를 보존하고 새로운 v3에서 필수 계약과 실제 interface 통합을 검증해야 한다. 상세 증거와 HIGH 발견은 NATIVE_QA.md·REVIEW_NOTES.md에 있다.
