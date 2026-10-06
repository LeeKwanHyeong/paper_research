# 실행과 관측

**5090 native 검증과 학습을 시작한다 — 다음 작업**
- 원래 Intermittent12 종료 경계 증거와 현재 GPU idle·기존 소유PID 부재·runtime/UUID를 확인한 후 전용 fresh root에 SHA 패키지를 배치한다.
- 자기 호스트 native qualification 및 실제 exit0를 검증하고 호스트별 training permit을 발급해 다섯 fit을 시작한다. 기존 공용 Runtime/인증/source/data는 바꾸지 않는다.

**5080의 기존54 종료·회수를 기다린 뒤 같은 대조를 시작한다 — 외부 작업 대기**
- capacity gate가 미충족이면 로컬 pending만 기록하고 SSH/GPU 실행을 하지 않는다.
- 기존54 originals SHA와 종료/최종 Validation receipt가 모두 확인된 뒤에만 승인된5080 새 대조를 활성화한다. 호스트 자신의8시간 lease를 그때 시작한다.

**시간별 GPU 상태와 완료 결과를 보고한다 — 기존 사용자 요청 유지**
- 각 host monitor는 한번만 수행하고 별도 반복 polling을 하지 않는다. 저장/best epoch와 같은 epoch Validation6지표, fit 자체 측정시간 ETA를 보고한다. pending/failed 원인을 보존한다.
- 전체10 실행완료·양서버실제종료·manifest/archiveSHA 로컬 검증·최종 Validation 비교를 모두 확인한 뒤 검색 pilot 관측만 종료한다. 기존54 자동화는 자기 완료 gate에 따라 독립 처리한다.
