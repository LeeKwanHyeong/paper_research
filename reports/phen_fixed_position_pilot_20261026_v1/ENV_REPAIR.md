# Native qualification 실행 환경 수정

최초 원격 시도는 RAF synthetic 검사에서 actual exit 1로 중단됐다. CUDA의
`libnvrtc-builtins.so.13.0` 탐색 오류이며, 제어 부모가 계약에 보존된
`LD_LIBRARY_PATH` 등 runtime 환경을 자식에게 전달하지 않은 구현 누락이다.
실제 RAF·Taxi 학습은 시작되지 않았다. 최초 source check는 exit 0이었다.

최초 원격 디렉터리, 실패 stdout/stderr와 종료 증적은 보존한다. 로컬
`NATIVE_FAILURE_DIAGNOSIS.stdout`에는 원문과 실제 exit·byte·SHA가 보존되어 있다.
수정은 `remote_parent.py`의 환경 전달 및 전용 cache 경로 설정뿐이다.
원래 학습 소스, adapter 모델 코드, 입력, loss, seed, 위치 인코딩 및 종료 조건은
동일하다. `attempt2/ENV_REPAIR_CHECK.json`은 계약의 15개 runtime 환경값이 두 단계
subprocess에도 그대로 전달됨을 actual exit 0으로 확인한다.

`attempt2`는 학습 재시도가 아니라 최초 두 학습에 앞선 실행 환경 복구다.
새 원격 디렉터리에서 native synthetic/capacity 네 검사가 모두 exit 0일 때만
RAF와 Taxi를 차례로 학습한다. 실패 시 자동 과학 재시도는 없다.
