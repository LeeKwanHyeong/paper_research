# Intermittent3 중간 checkpoint 평가 도구 독립 검토

판정: **PASS**. 열린 중대 실행 오류는 없습니다. 실제 native Validation/Test 실행 전의 로컬 검토입니다.

- seed42·52·62의 snapshot epoch42·40·38에서 best epoch5·5·11을 고정하며, 현재 상태가 아닌 best_state_dict를 읽습니다. 원본 checkpoint와 source983은 변경하지 않습니다.
- 전체 RMSE·MAE·TimeNLL의 과거 Validation 값 세 개만 허용오차1e-5로 재현합니다. 큰수량 지표와 개별 예측은 새 산출물이며 과거 행 단위 재현으로 표현하지 않습니다.
- Validation3의 실제 exit0와 재현 gate 전에는 Test를 실행하지 않습니다. qualification3+Validation3+Test3, 총9개 worker와 pipeline1 범위입니다.
- 5080의 기존 정지 GPU context3개를 보존하며 5090 학습은 그대로 둡니다. 학습 완료나 전체 GPU 유휴를 주장하지 않습니다.
- 외부 Validation18은 저장 집계만 이용한다는 한계를 유지합니다. 기준선 재추론은 없습니다.

population의 audit 필드5개를 worker가 요구하는 identity3개로 전달하지 않던 연결 오류는 담당자가 새 manager에서 수정했습니다. SPEC과 reference 원본은 유지했고 prepare→worker 통합 회귀 검사가 통과했습니다.

독립 합성 검사: **57 tests +2 subtests 통과, 실제 exit0**. 검토한21개 파일 SHA는 검사 전후 동일했습니다. stdout/stderr 원문과 최종 파일 SHA는 INDEPENDENT_QA.json에 기록했습니다.

다음 작업은 부모 작업에서 검토된 파일을 커밋하고 로컬 prepare·계약 검증을 거쳐 승인된 native 평가를 한 번 실행하는 것입니다. 이 검토에서는 원격 호출과 실제 Test를 실행하지 않았습니다.
