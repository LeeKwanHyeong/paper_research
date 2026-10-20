# 원본 RAF 공유형의 고정 checkpoint Test 비교

**평가 대상과 선택 기준을 고정한다 — 완료**

- 최신 사용자 요청에 따라 원본 RAF 공유형 seed42·52·62의 Test3을 평가하고, 기존 MLP16·외부6종의 저장 Test21개와 비교한다.
- Validation RMSE로 선택한 epoch9·5·19와 binary·tensor SHA를 고정한다. Test로 epoch나 구조를 다시 선택하지 않는다.
- 원본 데이터·시간/수량 log1p·시간 가중치1·공통 수량/시간 head·메모리를 유지한다. 기존 CNN과 기준선은 변경·재학습·재추론하지 않는다.

**새 평가 경로를 검증한다 — 진행 중**

- 평가 worker와 운영 도구는 서로 다른 새 파일에서 병렬 구현한다. 공통 계약·입력·선택 checkpoint 결속과 실제 실행은 순서대로 진행한다.
- 저장 Test21개와 원본 RAF Test target5226개가 같은 사건·truth인지 검증한다. 세 seed의 모든 차이와 악화도 보고한다.
- 합성 모델의 구조 식별·target 차단·batch 일치·상태 불변·여섯 지표를 검사하고 기존 source·원본을 보존한다.

**기존 학습 종료 후 native Validation과 Test를 실행한다 — 외부 작업 대기**

- 현재5090 원본9개 학습과 기존 Validation9 재현·원본 회수·최종 비교가 완료돼야 한다. 5080 Cap학습은 그대로 둔다.
- 실제5090 유휴와 동일 GPU·Runtime을 다시 확인하고, 새 격리 경로에서 합성 qualification3→Validation3→전체 gate3→Test3 순서로 진행한다.
- Validation의 여섯 지표와 개별 예측을 모두 재현한 뒤에만 Test를 연다. 기존 claim·실패·불일치가 있으면 자동 재시도하지 않고 원문을 보존한다.
- 승인된 RAF3 외 데이터셋 Test·새 학습·기한 연장·공용 Runtime 변경·반복 개선 자동화는 추가하지 않는다.

**원본 회수와 3seed 비교를 마무리한다 — 다음 작업**

- 새 child/pipeline 실제 종료·소유 PID/GPU부재와 archive/manifest/결과 SHA를 확인한다. wrapper 자체 OS exit의 직접 관측 여부는 별도로 표시한다.
- RMSE·MAE 중심의 평균±표본SD(ddof1), seed별 절대·상대차이, 큰수량/시간 지표와 단순 기준선을 보존한다.
- 기존Test를 본 뒤 개발한 탐색적 후속 비교다. 새로운 독립 평가나 보편적 일반화·통계적 유의성으로 확대하지 않는다.
