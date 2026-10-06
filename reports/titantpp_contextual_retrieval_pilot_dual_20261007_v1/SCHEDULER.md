# 실행과 관측

**5090에서 검색부 다섯 방식의 학습을 진행한다 — 진행 중**
- 기존 Intermittent12의 종료 경계와 5090 GPU 사용 가능 상태, 기존 Runtime·UUID를 확인했다. 전용 root 배치와 모든 동결 파일의 SHA 대조를 마쳤다.
- 다섯 방식의 native GPU 동작 검증과 사전 비용 기준을 모두 통과했다. 실제 qualification 종료 코드0와 소유·GPU PID 부재를 확인한 뒤 2026-10-07 08:39:05 KST에 다섯 fresh fit dispatcher를 시작했다.
- 평균 검색, 유사도 가중합, 시간·수량 고정 혼합, 가변 혼합, 같은 예산의 일반 검색 순으로 진행한다. 실제 worker·저장 epoch는 hourly 보고에서 확인한다.
- 5090의 시작 허가 마감은 2026-10-07 16:35:48 KST다. 종료 예상 시각이 아니며, 조건7200초·전체28800초·worker1·자동 retry 없음이라는 실행 한도를 유지한다.

**5080의 기존54 종료·원본 회수 후 같은 대조를 시작한다 — 외부 작업 대기**
- 기존 MLP8 구조54조건의 원본 manifest/archive SHA, 최종 Validation 보고, 세 소유root의 실제 종료가 모두 검증되어야 한다. 미충족이면 새 대조는 로컬 pending만 기록하고 원격 호출·GPU 작업을 하지 않는다.
- 승인된 활성화 도우미는 배타적 intent/claim을 보존하며 아직 시도하지 않은 배치·qualification·학습만 이어 간다. 실패나 불완전한 시도는 자동으로 재실행하지 않는다.
- 5080도 다섯 방식을 같은 순서로 새로 학습한다. 자신의 시작 허가에서 8시간 기한을 고정하며 기존 공용 Runtime·인증·데이터는 바꾸지 않는다.

**시간별 상태와 실제 실행 증거를 보존한다 — 진행 중**
- 기존 사용자의 시간별 GPU 보고 요청을 유지한다. 한 회차에서 5080 활성화 도우미를 한번, 두 호스트 관측 도우미를 한번, 원본 회수 finalizer를 한번 실행한다.
- 학습을 시작하지 않은 호스트는 원격 조회0회다. 시작한 live host는 읽기전용 SSH1회씩 독립 관측하며 검증된 종료 cache는 실제 관측시각을 유지해 SSH0으로 재사용한다.
- 실제 저장/best epoch와 해당 fit의 최근 최대10완료 epoch 시간을 이용한 조건부 ETA·300epoch 시나리오를 보고한다. 아직 측정되지 않은 방식의 속도를 복사하지 않으므로 전체 큐 ETA는 미확정이다. 첫 저장 전과 조회 실패를 구분한다.
- 관측 JSON/Markdown 원본 바이트 SHA와 latest 포인터를 기록한다. current.json의 status·training_started는 봉인 당시 상태이며 변경하지 않는다. 현재 진행은 launch/qualification 영수증과 hourly 보고를 기준으로 한다.

**최종 비교와 checkpoint 원본을 회수한다 — 전체10회 종료 후 다음 작업**
- 과학적 구성5개를 seed42로 각 GPU에서 반복하는 총10회 실행이다. 두 seed나 독립 평가로 세지 않는다. 같은 GPU 내 결과와 비용을 비교하며 기존 MLP8 구조54조건 결과를 별도로 보존한다.
- 전체10 실행의 과학적 종료, 양호스트 실제 소유PID 부재·supervisor exit0, 선택/마지막 fullValidation과 작은 원본 SHA를 확인한 후 immutable 원본 manifest/archive SHA 회수를 한번 수행한다.
- 완료 gate가 없으면 finalizer는 로컬 pending만 기록하며 원격 회수나 평가를 하지 않는다. 실패·claim·lock을 보존하고 자동 retry하지 않는다.
- 수량으로 선택한 같은 epoch의 전체·큰수량 RMSE/MAE/TimeNLL과 비용을 보고한다. Test/held-out 성과·예측 행·Validation/Test 혼합 성과를 열지 않는다.
- 원본 SHA와 최종 Validation 보고까지 검증되면 이 검색부 파일럿 자동화만 삭제한다. 기존54 자동화는 자기 완료 gate에 따라 독립 처리한다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.

실행 명령(이미 승인된 범위):

```bash
/usr/local/bin/python3 paper/scripts/deploy_titantpp_contextual_retrieval.py --host 5080 --mode activate
/usr/local/bin/python3 paper/scripts/observe_titantpp_contextual_retrieval_hourly.py
/usr/local/bin/python3 search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/control/finalize_once.py
```

현재 관측: [hourly/latest.json](hourly/latest.json). native 비용 검증: [NATIVE_QUALIFICATION_5090.md](NATIVE_QUALIFICATION_5090.md).
