# Cap36 v1 패키징 실패 독립 검토

## Review Result: FAIL

### CRITICAL
- 없음.

### HIGH
- `paper/scripts/manage_titantpp_mlp_cap36.py:99` — 배포 archive에 root 식별에 필요한 `source/sample_data`가 빠졌다. 동결된 `simple_lab_test/common/pathing.py:16,38`은 `models`, `utils`, `sample_data`가 모두 있어야 프로젝트 root로 인정한다. 실제 5090 qualification이 이 import 단계에서 실패했다. 별도 v2 archive에 데이터 없는 디렉터리/marker를 포함하고 격리 압축 해제 경로에서 import를 검사해야 한다.

  수정 예시(기존 v1에는 적용하지 않음):
  ```python
  marker = tarfile.TarInfo("source/sample_data")
  marker.type = tarfile.DIRTYPE
  marker.mode = 0o755
  archive.addfile(marker)
  ```

### MEDIUM
- 없음.

### LOW
- 없음.

## 확인한 사실

archive SHA는 PREPARED 증적과 같고 951개 항목 중 `source/sample_data`는 없다. 5090 qualification은 2026-10-17 00:53:23 KST에 실제 transport exit1로 종료했다. stdout은 0 bytes, stderr는 4,501 bytes이며 원문 SHA를 검증했다. `synthetic_check`의 core import에서 실패했으므로 해당 5090 Cap scientific fit과 Test는 시작되지 않았다.

5080의 마지막 저장 관측은 2026-10-17 00:54:37 KST다. 배정된 16개 job의 stage 기록이 모두 비었고 `waiting_for_idle_gpu`, `training_started=false`, `native_synthetic_started=false`였다. 이 과거 관측을 현재 상태나 현재 GPU 유휴 증명으로 사용하지 않는다. 당시 5080의 GPU PID 두 개는 그대로 기록하며 Cap worker라고 분류하지 않는다.

## 이전 QA의 누락과 후속 조건

이전 PASS는 로컬 데이터·모의 운영 검증에 한정됐으며, 배포 archive를 독립 경로에 풀어 import하는 검사가 빠졌다. 이번 오류를 놓친 검토 범위 누락이다. 기존 QA 두 파일과 실패 원문을 덮어쓰지 않고 이번 실패를 별도로 기록한다. 현재 v1 패키지는 실행 준비 완료로 볼 수 없다.

다음 작업은 소유 담당자가 새 v2 패키징을 수정하고 격리 archive import 검사 후 독립 검토를 받는 것이다. 기존 5080 대기 프로세스의 소유와 상태도 별도로 확인해야 한다. 이 검토에서는 원격 연결·학습·추론·소스 수정 없이 로컬 증적만 읽었다.
