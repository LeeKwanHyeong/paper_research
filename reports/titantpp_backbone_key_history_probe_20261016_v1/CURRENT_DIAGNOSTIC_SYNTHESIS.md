# Key-only 과거 tap 진단 결과와 다음 질문

**고정 모델 진단을 검증했다 — 완료**
- 실제 native 종료 관측은 2026-10-16 05:34:03.008048 KST이다. 합성3+실데이터12 worker, pipeline1, supervisor child1 실제exit0와 소유16PID/GPU부재를 확인했다. wrapper 자신의OSexit는 직접 관측하지 않았다. 이 과거 증적은 다음 작업 시점의 GPU 유휴 증명이 아니다.
- 원본167파일/archive161members와 선택·source145·입력·상태불변을 검증했다. 원형 Validation3의6지표·원대상 재현을 통과한 뒤 원형Train3→과거tap차단Train3→Validation3을 수행했다.
- 저장된97,407쌍과 원자료에서 이력·대상을 독립 재구성했다. 994개SHA·180요약·90support·1,560평균/표본SD·36가산등식·12개6지표요약을 재검산했다. 관측이력1개인15,000쌍은 예측/시간손실이 정확히 동일하다. 기존CNN/QueryOnly/Test추론 및 새학습은0이다.

**과거 tap 전체 제거는 채택하지 않는다 — 완료**
- 행가중 Train off−on 평균RMSE−0.023754/MAE+0.008561, Validation−0.039180/+0.013672이다. RMSE는 두split의3seed 모두 개선되지만 TrainMAE는3seed 모두 악화하고 ValidationMAE는42/62악화·52개선이다.
- 공통개체동일가중 Train seed52 ΔRMSE+0.000936 예외를 보존한다. 전체·큰수량 TimeNLL은6개seed×split 모두 악화한다. Validation 큰수량은42/52의RMSE·MAE악화,62만개선이다. 평균tail개선을 일관된기전으로 확대하지 않는다.
- 사전고정15조건 가운데 두split×3seed×두가중방식 모두에서 RMSE·MAE가 동시에 개선된 조건은 없다. 이는 기술적 부호 확인이며 새로운 성능gate나 통계유의성 판정이 아니다.
- off는 기존CNN이나 현재tap만으로 재학습한 모델이 아니다. 현재tap의 선형채널배율은 공통K투영에 흡수 가능하며 신규기여로 주장하지 않는다.

**과거 내용과 수준 배율을 구분한다 — 다음 작업**
- 현재 진단은 과거tap을 없애면서 과거내용과 가용tap계수합을 동시에 바꿨다. 과거이력 자체가 문제라는 결론은 아직 없다.
- 다음 단일 진단은 같은 KeyOnly3의 가용tap계수합을 유지하고 과거K를현재K로대체한다. 원형과 대체조건 차이는 가용한 w1(K1−K0)+w2(K2−K0)이다. 경계에서는 실제 가용mask를 그대로 적용한다.
- reports/titantpp_backbone_key_content_probe_20261016_v1에 명세·독립 설계검토를 남긴다. 새도구 구현과native진단은 아직 미실행이다. 이 진단 자체는 CNN방법론 개선 성공이 아니며 새학습 진입을 자동 결정하지 않는다.
- 입력/log1p·공통손실/head/메모리/seed/학습조건·기존결과는 보존한다. 기존CNN과 외부모델 재학습·재추론·재평가0. 동일한 삭제/DC진단을 근거 없이 반복하지 않는다. 결과 후에는 남은 근거로 중복없는 CNN가설/대조를 고정하거나 현재근거 부족을 명시한다.

증적: native receipt42951c5ffb69639a438a002f0dd9e70eee1f3d8c234ad87b2ad42861c5041ff7; RESULT953a87fa46e0d70b9c624d01a91df33c052c3a3158eecf8f1473c861c11cb2d7; 분석receipt d90f3c67217fbe6eb414e1521fa6e63491f0161a302eb8227cf5b934c58a47eb; 독립분석QA c2edf909e392e0e32835778b6132d82e09520d54c0defe3982e86104cf7a6094.
