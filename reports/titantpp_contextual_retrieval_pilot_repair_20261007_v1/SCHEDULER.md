# 시간별 관측 — 운영 수정

자동화 대상은 기존 `titantpp-5-gpu` 하나다. 수정 계약 봉인 및 최초 진입 확인 후 프롬프트를 새 운영 root와 SHA에 연결한다. 매회 로컬 계약·control·source·parent operation·repair operation SHA를 먼저 확인하고, 승인된 local observer를 정확히 한 번 호출한다. 실행된 live host는 읽기 전용 SSH 1회씩, terminal cache는 실제 원관측 시각을 유지하며 SSH 0회다. 별도 SSH·polling·학습·retry·resume·kill·claim 삭제는 하지 않는다. 관측 후 새 finalizer를 한 번 호출한다. 미완료는 local pending/remote0이며 원본 회수는 승인된 별도 보존 작업이다.

보고는 실제 KST 관측시각과 서버별 완료/실패/진행/대기/미확정, 실제 저장/best epoch 및 fit 자체 최근 최대10 완료 epoch 중앙값의 조건부 ETA를 포함한다. 재개 fit 속도는 E149 이후 완료 epoch만 사용한다. 미시작 arm/GPU 속도를 복사하지 않고 전체 큐 ETA는 미확정이다. 모든10 완료·세owner 종료·binary 원본SHA·최종Validation 보고가 검증된 뒤 이 자동화만 삭제한다.
