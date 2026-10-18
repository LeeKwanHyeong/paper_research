# Cap 외부27 실행 후 확인 순서

## 현재 실험 범위

S2P2·RMTPP·THP × Taxi·RAF·Intermittent × seed42·52·62의 새27개만 포함한다. 기존 CNN과 Cap MLP36은 변경·재학습·재평가하지 않는다. 반복개선 자동화가 아니라 각 서버에서 한 번 실행하고 종료하는 학습→Validation 재현→Test 큐다.

## 상태 확인

프로젝트 root에서 `paper.scripts.manage_titantpp_external_cap27 --mode status --host 5080`과 `--host 5090`을 승인된 상태 확인 요청에 한 번씩 사용한다. 실제 결과와 원본 transport는 새 bundle/report에 기록된다. 이미 시작한 root에서 prepare/deploy를 다시 실행하지 않는다. 과거 GPU 유휴 관측을 현재 상태로 표현하지 않는다.

## 원본 회수

각 서버 final_receipt.status=complete이고 completed_fit_count/validation_replay_count/test_count가 각각 배정 수(5080=12,5090=15), dispatcher actualexit0와 소유 PID·GPU 부재가 확인된 경우에만 같은 관리 도구의 `--mode collect --host <host>`를 한 번 실행한다. 실패/timeout/미완료 retrieval을 자동 재시도하거나 기존 원문을 덮어쓰지 않는다.

## 최종 비교

두 서버 회수 검증 뒤 `paper.scripts.compare_titantpp_external_cap27`을 실행한다. 저장된 Cap MLP36의 계약·전체 원본·평가 대상·선택 binary/state·종료 증거를 재검증해 새27개와 합쳐 별도 비교표를 쓴다. 기존 보고서는 대체하지 않는다.

완료는 새 retrieval/final_receipt.status=complete,new_fit_count=27,candidate_test_count=27,validation_replay_count=27,baseline_reference_count=36,comparison_model_count=63,seed_count=3 및 원본/보고서 SHA까지다. 완료 cache 경로는 기록된 파일 SHA 확인이며 전체 집계 재계산과 구분한다.

Validation/Test 각 데이터셋의7모델 평균±표본SD, 모든seed의 외부−각MLP 및 외부 상호 절대·상대차이를 보고한다. q>cap은 빈 집합/N/A,q==cap을 따로 기록하며 예측은 clipping하지 않는다. 외부3종은 공통 head를 사용하는 백본 비교다. Cap에서의 순위를 원본 수량 과제의 개선으로 해석하지 않는다.
