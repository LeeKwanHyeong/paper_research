# TitanTPP 시간·수량 분리 설계 검토

상태: 설계 검토 완료. 완전 분리의 성능 효과는 미확인.

현재 근거는 공동 encoder가 있다는 사실을 확인하지만, 그것이 성능 저하의 주원인이라는 결론을 지지하지 않는다. 기존 gradient·shared-block·routing 진단의 부정적 결과, Taxi checkpoint 선택과 학습 궤적, 관측 likelihood·scale·global clipping의 대안 설명을 함께 반영한다.

완전 분리 모델을 다음 본모델로 확정하지 않는다. 동일 B 구조·초기화·입력·학습 budget에서 공동(J), 수량 전용(Q), 시간 전용(T)을 비교하는 진단을 다음 구현 대상으로 권고한다. Q/T의 조합이 완전 분리 시스템이므로 첫 진단에 별도의 네 번째 모델은 필요하지 않다. 본모델 기여 주장에는 전체 capacity/compute와 selector 통제 및 RMTPP·THP에도 같은 기법을 적용한 비교가 필요하다.

기존 clamped 시간 loss를 사용하는 호환성 진단과 정상화된 관측 likelihood를 쓰는 새로운 논문 비교를 구분한다. 이번 검토에서는 코드 변경·학습·추론·원격 실행을 하지 않았다.

전체 근거·대안·반론·구현 계약·남은 작업과 검토 경계 기록은 [상세 검토](../../../reports/titantpp_design_review_20260910/README.md)에 있다.
