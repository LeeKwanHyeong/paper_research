# 폭8·12 학습 배치와 확인 증거

2026-10-04T15:48:11.379633+09:00 실제 관측 기준. 폭16을 후속 기준선, 폭4를 대조군으로 유지한다.
Taxi·Intermittent·RAF × 폭8/12 × seed42/52/62 = 18조건; Instacart는 후속으로 미룬다.

| 서버 | 대상 | 완료/전체 | 상태 |
|---|---|---:|---|
| 5080 | Taxi 6 + RAF 6 | 0/12 | Taxi 폭8 seed42 저장 epoch14; 진행1·대기11 |
| 5090 | Intermittent 6 | 0/6 | 기존 선택 checkpoint3개·노출 기록의 5090 전송 승인 대기; 학습 미시작 |

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

**5090 배포 — 승인 필요**
- 자동 승인 검토가 회사 Intermittent가 반영된 기존 checkpoint3개와 노출 기록을 5090 전용 폴더에 전송하는 작업을 거부했다. 기존 명시적 전송 승인이 5080에 한정된다는 이유다.
- 거부된 명령은 실행되지 않았다. 동일 checkpoint SHA의 5090 기존 사본도 읽기 전용 검사에서 확인되지 않았다.
- 코드·계약·원본 SHA와 배포 준비는 완료했다. 목적지는`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090`이며 기존 train/validation 자료는 서버에 이미 있다. 전송 승인 후 배포→native 검증→학습을 이어간다. 준비 차단을 학습 실패로 집계하지 않는다.

**CNN·GRU 대조 — 설계 완료, 다음 구현**
- [폭16 기준 2×2 설계](../titantpp_cnn_gru_width16_design_20261004_v1/README.md): 기존 MLP16 재사용/CNN+MLP16/GRU54/CNN+GRU54. 근접한 보정 예산과 마스크·상태 reset·초기화를 명시했다. 이번18조건 큐에는 넣지 않았다.

**남은 작업 순서**
1. **5090 전송과 학습을 연결한다 — 승인 필요**: 기존 checkpoint3개와 노출 기록 전송 승인 후 SHA 배포·교차 GPU validation replay·qualification을 통과하고 6조건을 시작한다.
2. **18조건 용량 대조를 완료한다 — 진행 중**: 5080·5090의 terminal, selected/last validation replay와 SHA를 확인하고 기존 폭4/16의 같은 seed 기록까지 연결한다. 원본 checkpoint 회수·CPU 감사는 별도다.
3. **CNN·GRU를 구현해 계약 검증한다 — 다음 작업**: 국소 CNN과 순환 보정의 효과·상호작용을 분리할 모델을 구현한다. 인과성·padding·출력부·초기화·GPU 메모리를 검증한 뒤 후속 실행 범위를 고정한다.
