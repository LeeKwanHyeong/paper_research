# 5090 재부팅과 백본 학습 중단

**원인 확인 — 완료**

- 2026-10-19 19:55:44 KST의 읽기 전용 조사에서 `uptime -s`, `/proc/stat`의 btime, `last -x`, journal의 boot 목록이 모두 **2026-10-19 17:24:09 KST 서버 재부팅**을 가리켰다. 현재 boot ID는 `ce4f7a36-ed56-4abe-8c14-1c4b6aeb48d6`다.
- 학습 dispatcher·실행 중 worker3개·정지 외부 비교군4개가 함께 사라진 것은 재부팅으로 설명된다. 과거 `status.json`의 running은 현재 프로세스가 아니다. 원래 중단 worker3개의 OS 종료 코드는 알 수 없으며 성공이나 학습 수치 실패로 바꾸지 않는다.
- 일반 계정은 system/kernel journal을 볼 수 없고, 비대화형 관리자 로그 조회도 암호가 필요해 거절됐다. 따라서 재부팅을 일으킨 전원·사용자 작업·커널 오류 등의 세부 원인은 미확정이다. 권한·인증 설정은 변경하지 않았다.
- 17:55:47 KST의 로컬 완료 대기 SSH `Operation timed out / Broken pipe`는 별도 원문으로 보존한다. 이것을 원격 학습의 정상 종료 코드로 사용하지 않는다.

**기존 원본과 재개 상태 보존 — 완료**

- 원격 원본을 변경하지 않고 전체1126파일을 로컬 `search_artifacts/titantpp_mlp16_late_path18_5090_recovery_20261019_v1/origin_archive`에 회수했다. archive·manifest와 추출한 모든 파일의 SHA를 확인했다.
- 원본 archive SHA256: `cd6dcfd4d81f99216bef9136daa7fc0b34e1e24019fac4941c8c565669db621f`.
- 원본 manifest SHA256: `ab6d142c5941dd029881ff3222c96f0b48c88636d6e086b50407e590c86902df`.
- RAF6은 실제 train worker exit0와 선택 checkpoint를 보존한다. 다시 학습하지 않는다.
- Intermittent52 관측15epoch, 52 용량대조13epoch, 62 관측13epoch의 `last_epoch_state.pt`를 실제 native Python에서 CPU로 읽었다. current/best 모델, AdamW 상태, Python/NumPy/Torch/CUDA 난수, loader generator, 전체 history, 원래 과학 조건과 시간 head 미학습 증적이 남아 있다. 이 세 개는 검증한 epoch 다음부터 재개한다. 62 용량대조 하나만 미시작이다.
- 원본 source963·선택 checkpoint·claim·실패 기록·원본 계약은 그대로 둔다. 새 복구 코드는 별도 경로에서 원래 과학 함수를 사용한다.

**근거 경로**

- `rootcause/20261019T105543989703Z/OBSERVATION.json`, stdout/stderr 및 transport SHA.
- 이전 직접 관측: `reports/titantpp_mlp16_late_path18_20261019_v2/diagnostics_readonly/20261019T104445866239Z/`.
- 과거 완료 대기 실패: `reports/titantpp_mlp16_late_path18_20261019_v2/followup_v1/failure.json`.
