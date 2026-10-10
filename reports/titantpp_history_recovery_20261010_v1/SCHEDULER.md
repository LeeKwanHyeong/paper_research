# 복구 실행의 관측과 승인 경계

자동화 `titantpp-instacart-63`만 새 복구 bundle로 변경했다. 사용자 요청에 따라 매시간 한국어로 주제별 완료 상태와 실제 관측 시각을 보고한다.

## 현재 기준선

- Instacart9 완료 원본은 `search_artifacts/titantpp_instacart_completion_recovery_20261010_v1`의 로컬 SHA만 재검증한다. 5080/5090 원격 재조회와 기존 observer 실행은 종료했다.
- A100 복구는 `search_artifacts/titantpp_contribution_history_recovery_20261010_v1`의 봉인 계약과 한 번 실행한 pipeline을 사용한다. 부모 삭제 Pod에는 접속하지 않는다.
- 이력 보정 execution ID27개는 완료10 재사용, 부분2 재개, 미시작15 fresh다. 첫 batch2개, 이후 batch3개로 최대 동시3 fit을 유지한다.
- 고정 검색은 자원과 동일 기준선 계약 정리 전 대기다. 이 자동화가 새 검색 fit을 활성화하지 않는다.

## 매회 관측

1. 세션 규칙과 README, 계약/승인/permit/launch/control 봉인을 stdlib SHA로 확인한다. mismatch 시 원격0과 사용자조치를 보고한다.
2. 최근 원본 snapshot의 rawSHA·실제 시각과 manager/service heartbeat를 읽는다. 관측15분 이내면 SSH0으로 재사용한다.
3. 새 관측이 필요하고 cleanup/stop 증거가 없을 때만 새 bundle `control/monitor.py`를 회차당 정확히1회 실행한다. 추가 SSH·polling·launch를 하지 않는다.
4. 실제 저장/best epoch와 조건부 ETA를 보고한다. 재개는 E24/E47 이후 epoch만 속도 표본에 넣고 새 첫 저장 전 부모 E23/E46을 진척으로 표시하지 않는다.
5. 조회실패와 학습실패를 구분한다. 소유권/계약 정합 오류, 서비스 회수 실패 또는 stale guard는 미확정·사용자조치 필요다.

## 비용과 종료

부모 청구, Pod 전 저장 비용 상한, 새 생성시각부터 저장 포함 비용을 합산해 기존 총 $50 상한을 유지한다. `manager.py`는 비용 종료, `terminal_service.py`는10분 관측과 live backup, 실제 종료 뒤 immutable 원본 SHA 회수와 Validation 비교, 새 Pod 및 전용 volume 정리를 담당한다. 회수 실패/claim을 보존하며 자동 retry는 없다.

Instacart9와 history27의 실제 종료·원본 SHA·최종 Validation 보고·새 Pod 및 volume 부재가 확인되면 최종관측보고를 남기고 이 자동화만 삭제한다. 고정 검색 대기와 CPU 재추론 감사·Test·새 독립 평가 미완료를 별도로 알린다. 부분/실패/미확정이면 유지한다.
