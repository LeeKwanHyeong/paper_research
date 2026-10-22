# A–D 후속 제어 복구

## 현재 기준선과 보존 대상을 고정한다 — 완료

사용자는 기존 실패 기록과 checkpoint를 보존하면서 5080 결과 회수부터 전체 Validation, 고정 Test와 속도·메모리 측정까지 마무리하도록 새로 승인했다. 학습 재실행과 MAC은 제외한다. 원래 과학 계약·실행 source·runtime·선택 checkpoint·seed·평가 모집단은 변경하지 않는다.

기존 5090 학습/재사용24개와 개별 Validation24개는 실제 종료0 및 최초 회수를 검증했다. 5080은22:32 읽기 전용 조회에서 학습12개·Validation12개와 dispatcher 종료0을 확인했다. 마지막 D42는96epoch 종료·56epoch 선택이다. 정식 5080 회수·양host 전역gate·Test·비용 측정은 아직 미시작이다.

19:47의 실제 SSH255 연결 시간 초과는 원문으로 보존한다. 이후 cached_failure는 새 접속 실패를 의미하지 않는다. 원래 advance가 FAILURE 존재 시 원격0으로 종료하도록 설계돼 있어, 연결이 회복되고 새 승인을 받아도 기존 경로만으로는 후속 단계를 이어갈 수 없다. 네트워크 원인이나45초 설정 대비217초 wall 기록의 원인을 이 변경으로 확정하지 않는다.

PRESERVATION.json은 원래 continuation_v1의 모든 기존 파일과 검증된5090 회수본 전체, 총833개 파일의 bytes/SHA를 기록한다. 과학 source987·입력9·legacy36·배포 archive·기존 계약 검증은 기존 verify를 그대로 재사용한다.

## 원래 실패를 유지하는 새 상태기계를 구현한다 — 진행 중

대상은 새 로컬 recovery_v1 제어뿐이다. 원래 controller의 REPORT 경로를 바꾸거나 원본 FAILURE/LATEST를 삭제·덮어쓰지 않는다. 새 승인·binding·독립QA·round·LATEST·FAILURE·COMPLETE는 새 경로에 쓴다. 원래 gate/collector/comparator 함수와 파일 경로는 그대로 사용한다. 새 Test launch 성공 증적 두 개를 원래 continuation에 배타 생성하는 것만 승인에 명시한다.

공유 advance.lock 안에서 검증과 side effect를 수행한다. source/승인/보존 SHA와 기존 collection, interrupted round·partial retrieval·시도 기록·한쪽 launch만 성공한 상태를 먼저 검사한다. 무결성·신원·부분 실행·통신 실패는 원문을 보존하고 후속 원격을 차단한다. 자동 과학 재시도·claim 삭제·partial 덮어쓰기·학습 resume/kill·원격 재배포는 하지 않는다.

## 종료 증거와 전체 Validation 이후 Test를 연다 — 다음 작업

5080의 고정된 원래 fit_validation 신원이 실제 종료0인지 확인한 뒤 최초900초 상한 pack/전송을 한 번 수행한다. 5090은 검증된 기존 회수본을 재검증한다. 두 host36개 선택본의 train/reuse·개별Validation receipt/exit144개를 묶은 원래 global gate가 통과해야 Test36 및 cost216을 각host에서 한 번 시작한다.

Test/cost는 기존 원격 source의 별도 tmux phase다. cost는 원본 checkpoint의 RAM 사본과 Train전용 고정37batch(5예열/32측정),train/eval각3개 fresh process를 사용한다. 원본 checkpoint는 쓰지 않으며 before/after binary SHA를 확인한다. 이는 모델 재학습이 아니다.

## 로컬 합성 검증 후 실제 실행한다 — 다음 작업

새 코드와 소유자 회귀 테스트, 독립 QA는 원격/GPU/실제추론 없이 수행한다. 실패 marker 보존, 학습 명령 부재, 중복 실행 차단, global gate, marker 내용, 완료 cache 순수 재계산을 검증한다. code/test/QA와 새 승인·보존 원문 SHA를 binding에 고정하고 paper_research의 codex/shared-component-ablation-20261022 브랜치에 별도 커밋한 뒤 실행한다. 기존 사용자 변경은 커밋 대상에서 제외한다.

## 회수·비교와 기록을 마무리한다 — 다음 작업

모든 Test/비용 child 및 dispatcher 종료, 선택 binary/tensor/epoch/source/input/원본/archive/manifest/member SHA를 검증한다. supervisor 자신의 OS 종료를 직접 관측했다고 표현하지 않는다. COLLECTION_COMPLETE와 비교 완료 COMPLETE를 구분한다. 완료 cache는 원격0으로 전체 inventory와 원래 comparator check-only의 순수 재계산을 검증한다.

보고는 세 데이터별 Validation/Test 전체·큰수량 RMSE/MAE·recorded TimeNLL 평균±표본SD, A–B/A–C/B–D/C–D의 모든seed 차이와 악화를 포함한다. 비용은 같은 GPU/seed에서 latency·throughput·allocated/reserved peak·parameters를 비교하고5080/5090 절대시간을 합치지 않는다. 기존 A 다중학습 총시간을 새 단독학습과 직접 비교하지 않는다.

현재 채팅의 A–D heartbeat 하나만 새 제어로 갱신한다. 정상 대기는 조용히 유지하고 주요 전환·완료·차단 실패만 알린다. 새 Notion 개선안3·4·7 하위에 필요한 결과를 기록하고 최종 보고를 전달한 뒤 이 A–D 자동화만 종료한다. 기존2seed/mac9·다른자동화와 MAC 보류는 유지한다.
