# 폭8·12 학습 배치와 확인 증거

2026-10-04의 실제 관측은 [상태와 시각](status.json)에 서버별로 보존한다. 폭16을 후속 기준선, 폭4를 대조군으로 유지한다.
Taxi·Intermittent·RAF × 폭8/12 × seed42/52/62 = 18조건; Instacart는 후속으로 미룬다.

| 서버 | 대상 | 초기 실행 확인 | 상태 |
|---|---|---|---|
| 5080 | Taxi 6 + RAF 6 | 15:48:11 KST epoch14, 16:02:59 KST epoch82 | 소유 worker·GPU PID·checkpoint SHA 확인, 학습 중 |
| 5090 | Intermittent 6 | 16:07:16 KST epoch2 | 소유 worker·GPU PID·checkpoint SHA와 첫2epoch 비용 검사 통과, 학습 중 |

두 서버에서 각각 worker1개를 사용한다. 초기 확인과 최신 진행 상태를 구분하며, 완료 집계는 terminal manifest·원본 SHA·selected/last Validation replay가 확인된 뒤 갱신한다.

**채택·시간 진단 — 완료**
- [진단 보고서](../titantpp_width_time_adoption_20261004_v1/README.md)와 채택 기준을 새8/12 결과 전에 고정했다. 시간 악화는validation의 일괄 현상이 아니며, sigma의 일괄 축소 설명도 확인되지 않았다.
- 같은 validation RMSE로 선택된 epoch의 전체·큰 수량 RMSE, MAE, 시간 NLL, 3seed 평균·표본 표준편차·불리한 seed를 함께 보고한다. Train 표본과 전수 validation을 구분한다.

**구현·실행 계약 — 완료**
- 모델·지표75개, 캠페인35개, 시간 진단9개, 총119개 CPU 테스트 통과. 폭4/16 초기 SHA·identity와 원래 분기·/8·출력부·손실·선택은 유지했다.
- 실험소스는paper_research master의`3bec316f2eb0319cf52e062dcf17b3f9518dcb7a`. 114개 소스 SHA를 동결해 기존Runtime을 바꾸지 않는 새폴더에 배포한다. Push하지 않았다.
- 계약SHA `8434041a6a301995a8d22baa77d5ca0da4eb2cd942d091a8aa8cef379406c48d`. source closure `41a4a4ff79e5e17d907b9ee8966723004516ee2e42608d31c7f1a871f0acd3e1`.
- 각 서버에서 worker1개를 사용한다. 전체168시간·조건36시간 상한과 자동 retry/resume 금지를 유지한다. 마감은 ETA가 아니다. Intermittent의 5090 이관 출처를 기록하며 GPU 간 효율 비교에 혼합하지 않는다.

**5080 실행 — 진행 중**
- 배포122파일 SHA·native GPU qualification·기존 폭4의 6개 선택 checkpoint validation replay 통과. 폭8/12의 보정12,288/18,432개와 실제 설정별 전체 파라미터 수·최대 메모리를 측정했다.
- 감독기 PID1142789, worker group1142812, GPU worker1142813. 저장 epoch14와 checkpoint SHA receipt, 같은 fit의 첫2epoch 비용 검사 통과를 확인했다. Mac 연결 없이 서버 lease·timeout으로 실행된다.
- 해당 fit의 최근10epoch 중앙값은13.10초다. 미시작 조건의 자체 속도와 5090 대기시간이 불확실하므로 전체 큐 ETA는 미확정이다.

**5090 전송·환경 검증·학습 연결 — 완료, 학습 진행 중**
- 자동 승인 검토가 회사 Intermittent가 반영된 기존 checkpoint3개와 노출 기록을 5090 전용 폴더에 전송하는 작업을 거부했다. 기존 명시적 전송 승인이 5080에 한정된다는 이유다.
- 이전 거부 명령은 실행되지 않았고 동일 SHA의 기존 사본도 확인되지 않았다. 이후 사용자가 “5090 학습 연결: 기존 체크포인트3개·노출 기록 전송 후 환경 검증과 학습을 시작합니다. 진행하자 승인한다”라고 명시 승인하여 차단을 해소했다. [대상별 승인 기록](transfer_authorization_5090.json)을 추가하고 원래 과학 계약·선택·예산을 바꾸지 않았다.
- 목적지는`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090`이다. 배포126파일 SHA를 대조했고, 서버에 이미 있던 train/validation의 원본 SHA·대상 수·입력 규격을 확인했다.
- [5090 native qualification](qualification_5090.json)은 GPU UUID·엄격한 결정론·메모리·유한 gradient·기존 선택 checkpoint3개의 Validation replay를 통과했다. 각각 약1.98GB의 synthetic peak allocation은 실험 속도 추정이나 실제 학습 전체 메모리 보장이 아니다. 서로 다른 GPU 출처를 유지한다.
- [초기 학습 증거](checkpoint_confirmation_5090.json)에 감독기2297721·worker group2297789·GPU worker2297790, 실제 epoch2 원본 SHA와 비용 검사 통과를 보존했다. 실제 첫2epoch는81.96/81.55초이며, 300epoch 비용 projection은 남은 전체 큐의 ETA가 아니다.

**CNN·GRU 대조 — 구현·CPU 계약 검증 완료**
- [폭16 기준 2×2 설계](../titantpp_cnn_gru_width16_design_20261004_v1/README.md): 기존 MLP16 재사용/CNN+MLP16/GRU54/CNN+GRU54. [구현·검증 기록](../titantpp_cnn_gru_implementation_20261004_v1/README.md)에 새59개·기존101개 검사 통과, 실제 합성 학습과 selected/last Validation replay6건, 실제 전체 파라미터 수를 보존했다. 이번18조건 큐에는 넣지 않았다.
- 완료 원본의 회수·비교 실행기도 준비했다. [18조건 비교 기록](../titantpp_history_capacity_comparison_20261004_v1/README.md)은 원본 SHA·같은 epoch·선택·split·과거 폭4/16 연결을 확인한다. 비교 계약21개와 회수 무결성4개, 기존 지표18개의 추가 검사도 통과했다. 신규 학습 완료 결과는 확인된 범위만 집계한다.

**남은 작업 순서**
1. **18조건 용량 대조를 완료한다 — 진행 중**: 5080·5090의 terminal, selected/last Validation replay와 원본 SHA를 확인하고 기존 폭4/16의 같은 seed 기록까지 연결한다. CPU 재평가는 별도 상태로 남긴다.
2. **서버가 비면 CNN·GRU를 GPU에서 검증한다 — 외부 작업 대기**: CPU에서 확인한 세 후보를 원격 torch2.11 환경에 연결하고 결정론·최대 메모리·처리시간·추론 지연을 실측한다. 실제 검증 전에는 GPU 비용이나 효율을 주장하지 않는다.
3. **후속 CNN·GRU 실행 조건을 고정한다 — 다음 작업**: 완료된 용량 대조와 GPU 검증을 검토한 뒤 새27조건의 별도 실행 계약·배치를 구체화한다. 폭8·12 학습 큐와 분리한다.
