# Titan + 이력 MLP 대 외부 인코더: 비교 조건과 validation 결과

2026-09-29. 앞선 B 기준 해석과 구분한다. 현재 비교 대상은 Titan + 이력 MLP다. Taxi·Intermittent 각3seed(42·52·62), 같은 최초 strict raw-RMSE 선택 checkpoint의 MAE·RMSE·시간 점수다. 새 학습이나 held-out 평가는 수행하지 않았다.

**기존 결과 및 실제 조건 대조 — 완료**
- 이력 MLP6조건과 외부 인코더24조건의 원본 summary·selected replay·exposure를 읽었다. 24개 dataset/seed/외부모델 쌍에서 과학적 학습 인자, 관측 시간 head, train 통계, 수량 경계, optimizer 그룹 설정, 겹치는 전체 train epoch 배치 prefix와 validation 순서를 대조했다. 역할·계약 표기는 다르며 JSON에 보존했다.
- 세 데이터셋의 계약 항목은 원본 복제 계약과 core 계약에서 동일하다. 이번 실제 결과 대조는 완료된 Taxi·Intermittent만 포함한다. 소스 공통90파일 중86파일 동일,4개 변경은 core모델 등록·라우팅 및 과거 Instacart 계약/복구 검증 경로 추가다. 공통 trainer/head/loss는 유지됐다.

## Taxi: 평균 ± 표본표준편차

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ | 파라미터 |
|---|---:|---:|---:|---:|
| Titan + 이력 MLP | 25.6237 ± 0.8320 | 79.7111 ± 2.8260 | 1.0387 ± 0.2843 | 96,003 |
| RMTPP | 28.5124 ± 0.6344 | 94.7283 ± 3.4006 | 1.5798 ± 0.9400 | 25,347 |
| THP | 32.4095 ± 0.8144 | 107.5977 ± 4.5040 | 0.6629 ± 0.0174 | 100,355 |
| NHP | 94.2694 ± 1.8752 | 318.8148 ± 5.3011 | 0.6516 ± 0.0021 | 30,211 |
| SAHP | 34.2260 ± 0.4539 | 120.5273 ± 1.9996 | 0.6848 ± 0.0146 | 112,835 |

## Intermittent: 평균 ± 표본표준편차

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ | 파라미터 |
|---|---:|---:|---:|---:|
| Titan + 이력 MLP | 0.7005 ± 0.0541 | 1.6730 ± 0.0819 | 0.4971 ± 0.2095 | 96,003 |
| RMTPP | 0.8949 ± 0.0494 | 2.5499 ± 0.1604 | 0.2748 ± 0.0047 | 25,347 |
| THP | 0.8993 ± 0.0661 | 2.7534 ± 0.1784 | 0.2911 ± 0.0045 | 100,355 |
| NHP | 4.5188 ± 0.1706 | 12.8755 ± 1.0405 | 0.4613 ± 0.0969 | 30,211 |
| SAHP | 1.8619 ± 0.0899 | 7.0298 ± 0.7673 | 0.3448 ± 0.0138 | 112,835 |

수량 MAE·RMSE는 두 데이터 모두 MLP가 기존 네 외부 인코더보다 평균 및 각 seed에서 낮다. 시간 NLL은 Taxi에서 THP/NHP/SAHP보다 높고, Intermittent에서는 네 외부 인코더 모두보다 평균이 높다. 수량 우위를 시간까지 확장하지 않는다.

## 무엇을 통제했고 무엇이 남았는가

| 항목 | 확인 결과 |
|---|---|
| 데이터·split·train 통계·평가 표본 | 동일 |
| 관측 입력 범위·loader·동일 seed의 배치 순서 | 공통 계약 및 실제 겹치는 epoch prefix 대조 |
| 수량·시간 head / loss | 공통 구현·설정 |
| optimizer | AdamW, lr0.001, weight decay0.01, clip1 |
| batch/종료/선택 | 128, max300/min40/patience40, 최초 strict raw-RMSE 최소 |
| seed | 42·52·62; 다른 구조의 tensor 초기화까지 동일하다는 뜻은 아님 |
| 모델 크기/연산량 | 동일하지 않음. MLP96003, RMTPP25347, THP100355, NHP30211, SAHP112835 |
| 튜닝 기회 | 모델별 탐색 없는 고정 설정 비교. 동등 탐색 예산의 최적 설정 비교는 아님 |
| 실제 학습량 | 조기 종료로 epoch·step·시간이 다름 |
| 평가 split | validation만. 선택에 사용한 validation이므로 독립 일반화 확인은 아님 |

**backbone 주장 범위를 확정하는 비교 설계 — 다음 작업**
- 모델 전체의 수량 경쟁력과 Titan 구성요소 자체의 인과적 기여를 구분한다. 후자를 주장하려면 공통 부가 MLP 사용 여부 등 구조 차이를 분리하는 대표 대조가 필요하다. 모든 모델에 무조건 같은 MLP를 붙여야 한다는 뜻은 아니다.
- 동일 hyperparameter 값은 고정 조건 실험에는 적절하지만 각 모델의 최적 성능을 보장하지 않는다. 동등 탐색/계산 예산으로 선정한 설정의 비교와 용량·비용 대조를 별도로 설계할 수 있다. 참고: [Dodge et al. 2019](https://aclanthology.org/D19-1224/). 새 GPU 실행을 시작한 것은 아니다.
- Instacart MLP3seed가 확정되면 불리한 결과를 포함해 세 데이터 결과를 합쳐 판단한다.

기계판독 근거: [mlp_external_comparison.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_core_ablation_execution_20260928_v1/completed_5080_20260929/mlp_external_comparison.json).
