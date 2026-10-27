# Taxi seed42 내부 시간 분리 진단

## 실행 범위 — 확정

기존 Causal Encoder + single_mlp를 Train 앞부분으로 처음부터 한 번 학습한다. 본체와 출력층 모두 이 범위만 사용한다. 구조·손실·학습률 후보 검색은 하지 않는다. 기존 실험·실패·checkpoint를 보존하며 Test 자료와 성능은 읽지 않는다.

| 자료 | 사건 수 | 용도 |
|---|---:|---|
| fit | 23,108 | 본체 및 출력층 학습 |
| 내부 dev | 7,536 | 동일 상태 평가·checkpoint 선택·조기 종료 |
| 내부 audit | 7,487 | 학습·선택 종료 후 고정 epoch 평가 |
| 경계 제외 | 262 | 제외 기록 보존 |
| 기존 Validation | 8,268 | 선택 종료 후 selected·last 평가만 |

## 로컬 분할·평가 검증 — 완료

원본 소스 991개와 안전한 Train/Validation 입력을 고정했다. fit의 정답 및 과거 입력에 보류 대상 정답이 섞이지 않음을 검사했고, 내부 분할 후에도 사건 순서·입력 tensor·target identity가 원본과 같다. 정규화와 시간 초기 통계는 fit만으로 다시 계산하고 독립 산술과 대조했다. fit의 최대 관측 이력은 168건이다.

prepare_01의 context_only 호환성 실패를 보존했다. 검사 전용 사본만 기존 grid 검사기에 전달하도록 고쳤으며 실제 loader나 target은 바꾸지 않았다. prepare_02와 최종 synthetic 24개, 실제 모델 합성 4건의 optimizer 1회 및 평가 검사는 실제 exit 0이다. 이 단계에서 실제 자료 학습은 하지 않았다.

## 서버 배치 — 다음 작업

5080은 읽기 전용 최초 접속에서 timeout(exit 255)이 발생해 실패 기록을 보존했다. 학습 실패가 아니다. 별도 확인한 5090은 원본 소스 SHA와 GPU 신원이 맞고 compute process가 없었다. 승인된 두 서버 범위에서 5090의 새 격리 디렉터리에 실행한다. 시작 직전 신원·유휴 상태를 재검사하며 기존 작업을 중지하지 않는다.

## 학습 및 평가 — 다음 작업

최대 300 epoch, minimum 40·patience 40을 유지하고 내부 dev 수량 RMSE의 최초 엄격 최저점으로 선택한다. fit·dev 전체를 매 epoch 동일 checkpoint·eval/no_grad 조건으로 계산하고 평가 전후 학습 난수 상태를 보존한다. 온라인 Train 손실은 별도 보조 지표다.

내부 audit은 사전 격자 1·5·10·20·40·60·…·300 중 도달한 시점과 selected·last에서만 평가한다. 기존 Validation은 selected·last만 평가한다. 이 평가는 학습과 선택을 종료한 뒤 같은 원격 작업이 수행한다.

## 회수·판독 — 다음 작업

실제 child·supervisor·dispatcher 종료와 원본 불변·회수 파일 SHA를 검증한 후, 동일 checkpoint의 사건별 재계산과 learning curve를 대조한다. 수량 RMSE·MAE·log-MSE와 시간 NLL을 각각 읽는다. fit 지원 범위 안팎을 함께 보고 후기 일반화 악화와 구간 차이가 공존할 가능성을 남긴다. 이 seed42 한 번으로 순수 과적합과 순수 분포 변화의 인과 효과를 확정하거나 새 모델을 채택하지 않는다.
