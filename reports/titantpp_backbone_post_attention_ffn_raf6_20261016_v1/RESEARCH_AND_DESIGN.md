# Q/K/V 필터 밖에서 CNN backbone을 개선하기 위한 검토

**현재 코드와 진단을 분리해 판단한다 — 완료**

기존 CNN은 첫 encoder의 Q/K/V 관측3tap CNN 뒤에 두 attention 블록과 각 블록의 시점별 FFN을 둔다. 두 번째 FFN은 64→128→64 GELU MLP다. 전체 모델은 이미 causal attention으로 이전 관측을 본다. 따라서 3tap을 전체 모델의 관측 범위로 설명하면 안 된다. 기존 CNN-only는 블록 사이 GRU/HistoryMLP 보정을 호출하지 않는다.

KeyContent의 내용대체 off−on 평균은 Train RMSE −0.023063/MAE +0.010555, Validation −0.040200/+0.029062다. 모든 split·seed·두 가중에서 RMSE 개선과 MAE 악화가 함께 나타났고 시간 손실은 악화했다. 큰수량 오류는 개선됐다. 이는 과거 내용을 일괄 약화하는 방향의 절충을 보여준다. 마지막 FFN이 오류 원인이라는 증거는 아니다.

현재 검토는 저장 Train·Validation과 frozen source145를 사용했다. 이전 독립 집계의 705개 입력 SHA와 진단 receipt의 133개 파일 SHA를 다시 확인했고 고유 파일은705개다. 완료 helper·기존 CNN 추론·새 Test 열람·원격 작업은 실행하지 않았다.

**적용 위치가 다른 세 후보의 우선순위를 정한다 — 완료**

| 순위 | 방법 | 현재 근거와 한계 | 기존 시도와 구별되는 조건 |
|---|---|---|---|
| 1 | 최종 attention 뒤 FFN 내부의 비선형 관측-lag 필터 | 일괄 과거 억제 대신 prefix 표현을 합성하는 통제 가설. FFN 병목은 미확인 | QKV를 그대로 두고 최종 FFN expanded hidden에서만 처리. 기존 HistoryMLP는 블록 사이 위치 |
| 2 | 사건 단위와 짧은 구간 표현을 함께 만드는 계층적 CNN | 단순 긴 lag보다 구간 패턴을 학습하는 질문. RAF의 현재 분석은 장기 범위 부족을 입증하지 못함 | MultiLag 병렬 차분의 재명명이 아닌 실제 구간 인코더/세부 사건 보존. 데이터 대상과 head 변경을 피해야 함 |
| 3 | 실제 관측 시간 간격으로 이력 기여를 정하는 연속시간 CNN | event-lag와 시간 경과를 구분할 수 있지만 현재 간격매칭은 공통표본0으로 효과 추정 불가 | 더 긴 event-lag와 다르나 시간 간격 원인 근거가 생기기 전 학습하지 않음 |

1순위는 작은 위치 변경과 명확한 대조가 가능해서 먼저 선택했다. 성능 우위를 예측했기 때문이 아니다. 2·3순위를 동시에 붙이거나 새 학습 큐에 넣지 않는다.

**문헌을 적용 원리와 우리 증거로 나누어 사용한다 — 완료**

- [LocalViT](https://arxiv.org/abs/2104.05707)는 FFN에 depthwise convolution을 넣는 시각 모델이다. FFN 내부 locality라는 위치 근거만 참고한다. 우리 비선형 observed-lag 필터는 논문 구현 복제가 아니며 이미지 결과가 RAF 개선 증거는 아니다.
- [Conformer](https://arxiv.org/abs/2005.08100)는 attention과 convolution을 결합하는 음성 모델이다. Macaron FFN·GLU·BatchNorm까지 함께 도입하지 않는다.
- [ConvTimeNet](https://arxiv.org/html/2403.01493v1)은 패치와 계층적 합성곱을 검토할 근거다. forecasting 구현의 RevIN·head·입력 패치를 포함한 성능을 backbone 단독효과로 인용하지 않는다.
- [CKConv](https://arxiv.org/abs/2102.02611)와 [공식 구현](https://github.com/dwromero/ckconv)은 불규칙 위치에 정의되는 연속 kernel의 참고다. 우리 저장 행의 시간 간격 효과가 검증됐다는 의미는 아니다.

논문은 메커니즘의 선행 근거다. 새 원리·독립 일반화·인과관계·통계 유의성은 주장하지 않는다.

**두 개의 새 FFN 확장만 비교한다 — 설계 완료 / 구현 다음 작업**

기존 FFN 전체를 교체한다는 초기 표현은 수정한다. 정확한 변경은 마지막 FFN의 첫 선형층 뒤에 expanded channel별 비선형 이력 잔차를 넣는 것이다. 기존 W1/W2가 채널을 혼합하며 새 연산은 채널별로 작동한다.

u=W1·LN(x)+b1, C=(1/3)Σ A_l·c_l·GELU(a_l·u_source(l)+b_l), 이후 기존 GELU(u+C), dropout, W2와 외부 residual을 그대로 사용한다.

- **Temporal:** 현재·직전·그이전의 관측 prefix 표현을 사용한다.
- **Current 대조군:** 세 분기 모두 현재 prefix 표현을 사용한다. 가용성 A와 고정 분모3은 같다.
- 각각 a,b,c의3×128개, 총1152개를 새로 학습한다. a=1,b=[−1,0,1],c=0로 고정하고 추가 RNG를 소비하지 않는다. ±1은 통계량이나 최적화값이 아니다.
- current 대조군도 attention이 요약한 이력과 관측 가용성을 쓴다. 이력이 없는 모델이 아니다. 두 구조의 차이는 이전 위치 hidden 내용의 직접 접근이다.
- 세 선형 tap에 현재 hidden을 반복하는 대조는 계수합으로 축약되므로 거부했다. 이번 대조는 tap별 GELU 전 a,b가 달라 단순 선형 재매개화가 아니다. 동일 파라미터 수는 동일 함수공간·최적화 난이도를 보장하지 않는다.
- 기존 CNN은 저장 결과만 사용한다. 새 대조군은 비선형 FFN 확장으로 별도 함수군이며 기존 CNN 재학습이 아니다.

c=0에서 a,b의 첫 task-gradient는0이다. c가 움직인 뒤 일반적 합성입력에서 gradient 전달을 확인한다. AdamW weight decay로 a,b 값도 첫 step부터 달라질 수 있다.

**효과와 한계를 구별할 기준을 고정한다 — 설계 완료**

Temporal이 CNN보다만 좋고 Current보다 좋지 않으면 명시적 과거 결합의 근거로 보지 않는다. Temporal−Current Validation 평균 RMSE·MAE가 둘 다 낮아야 해당 가설의 Validation 근거이며 Test의 두 지표도 별도로 확인한다. 모든 seed 차이와 큰수량·TimeNLL 악화를 남긴다.

사용자 성능 기준은 별도로 유지한다. 각 후보의 Validation/Test RMSE·MAE 3seed 평균 네 값이 기존CNN과 동일 대상 외부 지표별 최우수보다 모두 엄격히 낮아야 한다. 통과 후보가 여러 개면 Validation RMSE→MAE→arm 문자순으로 확장한다. Current가 선정되면 시점별 FFN 확장의 성과로 표현하고 temporal 효과로 바꾸지 않는다. Test를 통과 자격에 사용했으므로 탐색적 선택이다.

Q/K/V와 head·memory의 구조·규칙을 유지해도 공동 학습으로 공통 가중치가 바뀐다. FFN 출력은 Hard-LMM 앞에 있으므로 prototype 선택도 달라질 수 있다. attention·memory 영향과 독립적인 결과라고 설명하지 않는다. withheld는 새 gather의 관측 경계를 초기화하지만 기존 attention의 과거 정보까지 지우지 않는다.

**구현·검증·실행의 순서를 구분한다 — 다음 작업**

1. paper_research 새 모델/runner 파일에서 마지막 FFN 경로와 원래 observed_context 전달을 구현한다. 기존 source145와 기준선은 보존한다.
2. 초기 tensor/RNG·dropout·mask/경계·미래 차단·비축약·gradient·checkpoint 식별·공통 head/loss를 합성 검사하고 독립 구현 검토를 받는다. 이 문서는 검사를 실행했다는 증적이 아니다.
3. 신규 운영 계약·도구를 별도 경로에서 봉인하고 실제 반환 진입점/SHA만 사용한다. 계약·커밋·봉인·원격 실행은 직렬이다. 독립 모델/운영 구현과 검토는 파일 소유를 분리할 때 병렬 가능하다.
4. 5090 fresh 유휴/GPU/Runtime/입력/3개 동시 admission을 검증한 뒤 Current3→Temporal3, 최대3병렬로 진행한다. 상속 hostdeadline과36h를 연장하지 않는다.
5. 전체6종료·원본회수·고정 checkpoint Validation6 재현 뒤 Test6와 저장 비교군을 비교한다. 실제 native 검증 전 학습 준비 완료로 표시하지 않는다.

현재 승인 부족은 없다. 구현·운영·native 검증이 아직 남아 있으며 새 원격 root·학습·평가 실행은0이다.
