# Native qualification and launch audit

## Review Result: WARNING

**양 서버의 qualification과 실행기 시작 증거는 검증됐지만, 첫 후속 확인에서 5080의 native 실패 기록이 발견됐다. 전체 학습 진행이나 완료로 보고할 수 없다.**

### CRITICAL
- 확인된 사항 없음.

### HIGH
- 확인된 사항 없음.

### MEDIUM
- `rounds/20261022T151900061201_9c521d019522/5080/check.stderr:6` — `fit_validation.failure.json`이 있다는 원격 응답으로 첫 확인이 실패했다. 현재 로컬 원문에는 실패한 과학 단계와 구체적 원인이 없다. 실패 원문을 보존하고, 별도로 승인된 읽기 전용 진단으로 원인을 확인해야 한다. 학습 재시도나 claim 삭제로 우회하지 않는다.

### LOW
- 확인된 사항 없음.

## 완료 증거

- CONTROL_BINDINGS의 **164개 파일 SHA 일치**, mismatch 0.
- 5080/5090 각 synthetic 9개 + capacity 3개, 총 **24개 child actual exit 0**.
- 양쪽 **dispatcher actual exit 0**, 총 2개. 부모 supervisor 자신의 OS exit를 직접 관측했다고 주장하지 않는다.
- 각 서버 46개, 총 92개 native snapshot 파일의 SHA·원문 전송 byte·base64 대응을 확인했다. child 원래 exit 및 stdout/stderr, dispatcher 원문, 고정 argv/cwd/boot 신원이 일치한다.
- capacity 6개의 전체 Runtime 기대값과 GPU UUID가 정확히 일치한다. 구조 검증 9개씩까지 모두 수치 설정을 수행했다고 확대해서 표현하지 않는다.
- 당시 원격 종료 확인 프로그램의 성공 원문을 통해 qualification child/dispatcher의 신원 부재 검증 통과를 확인했다. 이는 현재 GPU 유휴나 지금의 프로세스 상태를 새로 조회했다는 뜻이 아니다.

## 실제 시작과 첫 후속 확인

- 5080: fit_validation 실행기 시작 관측 2026-10-23T00:17:11.405174+09:00; PID 873390, start_ticks 107884317.
- 5090: fit_validation 실행기 시작 관측 2026-10-23T00:17:40.780600+09:00; PID 3434774, start_ticks 12444319.

이 기록은 **fit_validation 실행기가 시작된 증거**다. 첫 작업이 재사용 검증일 수 있으므로 MLP의 실제 epoch나 optimizer step이 시작됐다는 증거로 바꾸지 않는다.

첫 후속 확인은 5080에서 actual exit 1, timeout false, transport_error null, stderr 382 bytes였다. `fit_validation.failure.json`을 발견했으며 5090에는 추가 확인하지 않았다. 따라서 SSH 연결 실패로 단정하지 않고, native 단계 실패 표식을 관측한 것으로 기록한다. 원인 원문은 아직 이 회차에 회수되지 않았다.

## 남은 작업

**native 실패 원인을 보존하며 확인한다 — 차단됨**
- 기존 실패·출력·checkpoint를 보존한 채 별도 읽기 전용 진단으로 실제 실패 단계와 원인을 확인한다. 현재 실패한 제어에서 추가 원격 호출은 0이다.

**Validation과 비용 비교를 검증한다 — 다음 작업**
- 실패 원인과 승인 범위를 정리한 이후의 작업이다. Validation27, Train 비용162와 전체 실험 완료는 아직 확인되지 않았다. 모델 우열이나 논문 기여 결론도 내리지 않는다.

이번 검토는 로컬 파일만 읽었으며 원격·GPU·실데이터 추론·Test 열람은 모두 0회다. 소유 변경은 LAUNCH_QA.json과 이 Markdown뿐이다.
