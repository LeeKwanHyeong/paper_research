# Taxi 분포 감사의 독립 검토

읽기 전용 검토 두 작업을 병렬 수행하고 root의 raw 데이터 감사와 결합했다. 원본이나 모델 상태를 변경하지 않았다.

1. 원본/loader 검토: 세 input receipt의 경로 이외 전체 값 동일, terminal SHA 연결 12개, 전 epoch Train/Validation 노출, 실제 seed 초기화/shuffle 경로, 과학 117개 소스 byte 동일성을 검산했다. 다른 split/bootstrap 경로를 사용하지 않음을 확인했다. 첫 batch digest는 실제 tensor bytes이며 개별 target 인덱스 순서를 제공하지 않는다는 한계를 유지했다.
2. 성능/이력 검토: 회수 원본 135개 파일 SHA, Taxi 세 run의 history·endpoint·진단·timing, fresh Validation/Test 6개 receipt를 검산했다. 전체/5구간 SSE·MAE·bias·NLL 합계와 seed52 추가 SSE 117,194,484.7977562 중 >686 비율 96.46625657522068%를 독립 재계산했다. 457epoch의 기록된 finite·selector·노출·timing 연속성을 확인했다.
3. 새 보고서 검토: rawTrain/targetTrain 구분, 원본 최초 131개 gap0과 실제 target gap 양성 구분, seq 간격 9,799개의 해석, RMSE/bias 부호·추가 SSE 분모·그림·README 숫자를 대조했다. 초기 상태와 registry 결속 및 첫 batch digest 차이 guard를 보강했다. 결과를 뒤집는 문제는 발견하지 않았다.

검토는 데이터/집계 정합성에 한정한다. 개별 예측 오류나 초기화·학습 순서·선택 epoch의 원인 분해는 완료한 것으로 표현하지 않는다. 새 학습·추론·Test 재선택을 실행하지 않았다.
