# RTX history12 배포 전 검토

## Review Result: PASS — 코드·최신 배포 봉인 검증 완료

상태: 로컬 정적 검토·단위 검증 완료. 발견 사항은 담당자가 수정했고 회귀 검증을 통과했다. 원격/API 실행0, 소스 변경0. 이 보고서만 작성한다. 대상은 `deploy_titantpp_history_rtx_handoff.py`, `prepare_titantpp_history_rtx_control.py`, 두 호스트의 생성 pipeline/admission 및 관련 테스트다.

### CRITICAL

미해결 사항 없음.

### HIGH

미해결 사항 없음. 아래 초기 발견 사항은 수정·재검증 완료했다.

- 초기 `paper/scripts/deploy_titantpp_history_rtx_handoff.py:56`의 실행권 해제 bridge가 다른 GPU·누락된 원본 SHA를 가진 요약을 승인했다. Temporary bundle에서 `GPU-FOREIGN`와 요약만 전달했을 때 `released_to_rtx=True`가 생성되는 문제를 재현했다.
- 수정 후 `:62–96`은 봉인 parent canonical, 예약의 root/Pod/GPU, 이관 원본12 job 객체, 예약 barrier raw SHA, 정확15 terminal ID와64자리 SHA, 실제 supervisor/failure SHA, 실제 GPU 및 owner 부재를 검증한다. 출처 원문은 그대로 복사하고 원문·예약 SHA를 새 승인 파일에 연결한다.
- 새 deployment 테스트7개가 foreign GPU, 누락 SHA, 다른 barrier, 누락 terminal, 부분 완료, 살아 있는 owner, 예약만 존재하는 상태 및 기존 승인 덮어쓰기를 차단함을 확인했다.

### MEDIUM

미해결 사항 없음. `validate_local`의 봉인 project_files 검사 누락은 `:51–54` containment+rawSHA 검증으로 수정됐다. 수정된 코드가 아직 이전 봉인에 연결됐을 때 실제 `Project tool drift`로 배포가 차단되는 것도 확인했다.

### LOW

없음.

## 검증 결과

- control12 + bundle9 + deployment7 = **28/28 단위 테스트 PASS**.
- 각 호스트 tar143개 membership·regular file·상대경로·rawSHA 및 archiveSHA 전부 PASS.
- 초기 local_tools 봉인10개와 project_files 일치 확인. release bridge 수정 이후 최신 project SHA 재봉인 확인은 아래 최종 기록에 남긴다.
- pipeline은5080 완료3·5090 완료9를 각각 요구하고12를 오인하지 않는다.
- exact root/Python/environment/operation/source, native qualification 실제exit0+idle,3개 동시 CUDA admission 실제exit0+idle 후 training permit 발급을 유지한다.
- 3개 GPU PID 동시 목격, process group, 유한 loss/gradient/update, aggregate memory80%, per-process30% 검증이 있다. 이는 합성 admission이며 과학 fit 수로 세지 않는다.
- fresh root mkdir 및 tar 멤버별 exclusive 생성, 별도 start permit/intent의 exclusive 쓰기로 실패 시 자동 재시도를 막는다. launch 직전 NVML UUID와 compute PID 부재를 재확인한다.
- control 생성기의 기존 배포 감지 패턴에 실제 `deploy_intent`/`deploy_receipt`를 추가한 `prepare_titantpp_history_rtx_control.py:169–173` 수정 확인.

## 수락 조건

- [x] host별3/9 조건과 같은 dataset/seed의3arm 묶음
- [x] Runtime/source/loss/optimizer/selector 동결 검증 경로
- [x] native qualification/admission/actual exit/idle 순서
- [x] 새 root만 배포, 기존 파일과 원본 보존
- [x] 해제 증적의 parent identity·완전성 수정 재검증
- [ ] 실제 각 호스트 qualification·동시 admission 결과 및 학습 진입 — 원격 담당 단계

실제 GPU 동작은 이번 QA가 실행하지 않았으며 단위 검증의 phase runner와 GPU probe는 mock이다. UI/Node 변경이 없으므로 접근성·npm audit은 대상이 아니다.

## 실제 수신 증적 읽기 검증

- `ownership_release.json`과 `handoff/ownership_release_source.json`을 읽었다. A100 실제 관측 Unix1791602238.3791552(2026-10-10 12:17:18 KST), 정확한 GPU UUID, process/GPU process 없음,15 terminal 목록, planned operational exit1 보존을 확인했다.
- 수신 원문 raw SHA는 `e7d6fbaf59b1caa7763d2662ac1ed29d65f003f6846f8bc3aaff96fa6d3a8ea1`로 승인 파일에 연결돼 있다. 이 QA가 원격 관측을 새로 실행한 것은 아니다.
- 코드 검토 PASS와 실제 RTX 학습 시작 여부는 별개다.5080 GPU 상태 복구 및 각 호스트 native qualification/admission이 남아 있다.

## 최종 봉인 확인

`deploy_titantpp_history_rtx_handoff.validate_local('5080')` 및 `('5090')` 모두 PASS. 현재 source/input/operation/control/package/project_files raw SHA가 봉인과 일치한다. 갱신된 `control/local_tools_integrity.json` raw SHA: `9349554f61f5861324dd32f1d1ea6a447eb7f0fc5ea3c26e3dc02eaebc8f1c51`. 이전 봉인을 유지한 채 배포한 것으로 해석하지 않는다.

최종 재봉인 테스트3개 추가 후 control12+deployment7을 다시 실행해19/19 PASS를 확인했다. 이전 봉인 `c84d38d9b4ca5c176b9ae7843e4240ce787ad57097b3a78eaa855b4c19328979`의 보존 파일 raw SHA를 재검증했고, 이전/현재 seal의 `files`와 `hosts`가 같아 host control·package·operation 바이트가 바뀌지 않았음을 확인했다.
