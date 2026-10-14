# GRU 보정 크기 제한 후보의 Train·Validation 실험 계약

이 문서는 기존 CNN+GRU의 GRU 보정 출력을 제한하는 후보 하나를 정의한다. 현재 승인은 계약 작성과 로컬 구현·합성 입력 검증까지다. 실제 학습, 원격 배포·조회, 실제 데이터 행 평가, Test 실행은 시작하지 않았다. 이 계약을 준비했다는 사실은 학습 실행 승인을 뜻하지 않는다.

**기존 실험과 새 후보의 관계를 고정한다 — 완료**
- 기존 구성요소 캠페인의 CNN·FULL 모델은 각 데이터셋·seed별 선택 checkpoint를 그대로 보존한다. 새 후보를 기존 FULL에 합치거나 좋은 결과로 교체하지 않는다.
- 새 후보는 `titantpp_cnn_gru54_residual_cap025` 하나이며, 상한 비율은 `rho=0.25`다. 여러 상한을 탐색하지 않는다.
- 이 설계는 기존 Test 결과와 선택 모델의 오류 진단을 본 뒤 제안한 탐색 후속이다. 기존 Test를 새 독립 평가로 부르지 않으며, 이 실험만으로 일반화 개선이나 순위 반전의 원인을 입증했다고 해석하지 않는다.

**GRU 보정에만 크기 제한을 적용한다 — 로컬 구현·검증 준비**
- CNN+GRU의 block1 기본 표현을 `h`, GRU 보정 출력을 `r`라 하면, 각 배치·이력 위치의 마지막 64차원에 대해 `||r_cap||₂ ≤ 0.25 × ||h.detach()||₂`가 되도록 보정 방향을 유지하며 크기만 제한한다.
- 크기 계산은 float64이며 출력 dtype은 원래 보정 출력과 같다. 상한 안의 출력은 그대로 유지한다. 기본 표현의 크기가 0이면 보정도 0이며, 보정이 0인 초기화에서도 유한한 학습 gradient를 유지한다. 작은 양수 하한을 추가하지 않는다.
- dtype 변환으로 상한을 넘는 경우에만 0 방향으로 최대 8회의 최소 표현 단위 보정을 적용하고 float64로 크기를 다시 확인한다. 이 정책과 변환 gradient 처리는 실제 모델 metadata에 기록한다.
- 추가 학습 파라미터, 학습되는 배율, 정규화 계층, dropout을 넣지 않는다. 기존 FULL의 초기 tensor·생성 난수 상태·CNN·prototype·수량 head·시간 head를 유지한다. 후보 구분용 `cnn_gru_residual_cap_identity` 버퍼만 추가하며, uint8 32바이트로 실제 metadata SHA를 저장한다.
- 계약의 `candidate.metadata`는 모델 모듈의 `metadata(64, arm=..., rho=0.25)`에서 직접 가져온다. 문서에 적힌 수동 설명만으로 모델 신원을 대신하지 않는다.

**동일 데이터·seed·환경 안에서 후보 9개를 비교한다 — 다음 작업·실행 승인 필요**

| 데이터셋 | seed | 새 후보 수 | 계획 GPU·cuDNN | FULL과 후보의 등록·활성 파라미터 수 |
|---|---|---:|---|---:|
| Taxi | 42·52·62 | 3 | 5080·91900 | 각각 115167 |
| RAF | 42·52·62 | 3 | 5080·91900 | 각각 104159 |
| Intermittent | 42·52 | 2 | 5090·92000 | 각각 115167 |
| Intermittent | 62 | 1 | 5080·91900 | 각각 115167 |

- RAF의 원래 최대 이력 길이는 84이고 Taxi·Intermittent는 256이므로 파라미터 수가 다르다. 비교 조건은 각 데이터셋 내부에서 후보와 FULL의 수가 같다는 것이다. CNN은 해당 FULL보다 활성 파라미터가 24732개 적으므로 CNN과의 차이를 GRU 출력 크기 하나의 인과효과로 해석하지 않는다.
- 각 seed가 원래 학습된 GPU·Runtime을 계획에 유지한다. Intermittent seed62와 seed42·52의 GPU·cuDNN 차이는 별도로 명시하며, host 간 학습 시간을 합쳐 비교하지 않는다.
- 현재 실제 Runtime 상태는 조회하지 않았다. 모든 실제 학습 전에 고정 Python·GPU UUID·Runtime·입력 SHA를 해당 host에서 검증해야 한다. 특히 새 5090 학습 실행 승인은 현재 없다.
- 원본 `execution_contract.json`의 loader, 데이터 split·이력 창, Train 통계, 시간 관측 규칙, head 설정을 그대로 상속한다. 입력 파일을 새로 분할하거나 통계를 다시 추정하지 않는다. 준비 단계는 입력 행을 읽거나 입력 Parquet를 복사하지 않으며, 실제 입력 bytes 재검증은 native 검증에서 수행해야 한다.

**기존 CNN·FULL Validation 18개를 변경 없이 참조한다 — 완료·현재 기준선**
- Taxi·RAF 구성요소 평가 캠페인에서 CNN·FULL 12개, Intermittent seed42·52에서 4개, seed62에서 2개를 사용한다. 기존 본실험, 보정27, Instacart9, 외부 비교군 결과를 이 18개에 섞지 않는다.
- 참조마다 원래 캠페인·계약 raw SHA, Validation-only receipt와 원래 결과 파일 목록 manifest의 SHA, 선택 epoch·binary SHA·모델 tensor SHA·선택 history SHA, source131 closure, Train 입력과 Validation 입력 지문, Validation 대상 식별자, Runtime·파라미터 수를 결속한다.
- 선택 history에서 전체 Validation raw 수량 RMSE의 첫 strict finite 최소 epoch를 다시 확인한다. 동률은 최초 epoch다. 이는 기존 선택의 감사이며 새 checkpoint 선택이나 재학습이 아니다.
- 파일 목록 manifest의 Validation 항목만 사용한다. Test 결과 수치와 예측 행은 이번 준비에서 읽지 않는다. 기존 Validation 입력의 파일 구성과 학습 입력 구성이 다를 때는 각 SHA를 따로 보존하고 원래 Validation 대상 식별자가 일치하는지 확인한다.

**학습과 선택 기준을 원래 계약대로 유지한다 — 계약 고정**
- 새 학습이 승인되면 후보 9개를 fresh fit으로 학습한다. 기존 checkpoint에서 재개하거나 warm start하지 않는다.
- 수량 학습은 기존 `log1p` 수량 손실 가중치 1, 시간 손실 가중치 1, tail 가중치 0을 유지한다. 선택·주 평가는 전체 Validation raw 수량 RMSE다. 손실과 주 지표의 척도가 다르다는 점을 유지하고 별도로 보고한다.
- AdamW 학습률 0.001, weight decay 0.01, gradient clip 1, batch128을 유지한다. 최소40·최대300 epoch, patience40이며 최초 strict finite 최소 checkpoint를 선택한다.
- 조건당36시간과 host당168시간의 기존 절대 기한을 유지한다. 제한 실험이라는 이유로 예산을 늘리거나 자동 retry·resume하지 않는다.
- `source131` 원본과 운영 봉인·local40 원본을 보존한다. 신규 모델·runner·prepare 세 모듈의 SHA를 별도로 추가하고, 최종 준비 시 승인된 별도 bundle에 과학 source131과 신규3개를 복제한다. 원래 source·데이터 경로에는 쓰지 않는다.

**비교 지표와 해석 범위를 먼저 고정한다 — 계약 고정**
- 주 지표는 전체 raw 수량 RMSE다. 보조 지표는 전체 MAE, 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL이다. 여섯 지표 모두 같은 Validation 선택 checkpoint에서 계산한다.
- 큰 수량 기준은 원래 Train 상한을 유지한다: Taxi `q>3449`, Intermittent `q>187`, RAF `q>200`. 구간을 후보 결과에 맞춰 바꾸지 않는다.
- 각 데이터셋의 3seed 평균·표본SD(`ddof=1`)와 seed42·52·62 각각의 후보−FULL, 후보−CNN 차이를 보고한다. 낮을수록 좋은 지표에서 음수 차이는 개선을 뜻한다. 불리한 지표, seed별 예외, 실패·미완료 조건도 보존한다.
- 주 지표의 3seed 평균이 FULL보다 낮은지는 보고하되, 개별 seed와 보조 지표를 함께 제시한다. CNN보다 나은지는 별도로 판단한다. 평균 개선만으로 모든 seed·지표·데이터셋의 우월성이나 통계적 유의성을 주장하지 않는다.
- 상한 적용은 GRU 파라미터 수를 바꾸지 않는 통제이지만, 제한으로 학습 경로와 최적화 결과도 달라질 수 있다. 개선이 나와도 과적합, 입력 분포 변화, GRU 용량 효과를 모두 분리했다고 해석하지 않는다.
- 이 계약의 평가 범위는 Train·Validation이다. 후보 Test는 이번 계약에 포함하지 않는다. Test 재선택·재학습, 새 dataset·seed·외부 모델 추가, 상한 sweep은 하지 않는다.

**준비 증적을 보존하고 실제 실행은 별도로 승인받는다 — 다음 작업**
- prepare 도구는 로컬 원본 SHA·경로·계약·기준선 18개를 먼저 검증하고, 불일치면 신규 bundle을 만들기 전에 차단한다. 별도 출력 경로는 한 번만 생성하며 기존 bundle·원본을 덮어쓰지 않는다.
- 출력 계약과 receipt의 상태는 `prepared_training_not_started`, 예정 fit9·시작 fit0이다. `execution_authorized=false`, 실행 승인·학습 permit 없음, 원격0·Test0을 기록한다. 학습 완료 상태로 기록하지 않는다.
- 모델·runner·prepare의 합성 검사와 계약 변조 차단 테스트가 통과한 뒤 최종 source SHA를 고정한다. 이어서 native 검증과 실제 새 fit9개 실행 범위·승인을 확정해야 학습을 시작할 수 있다.
- 기존 claim·failure·학습·공용 Runtime·인증·원고·다른 자동화를 변경하지 않는다.

## 원본 결속

| 항목 | 고정 값 |
|---|---|
| 원래 bundle | `search_artifacts/titantpp_cnn_gru_components_20261012_v5` |
| 원래 계약 canonical SHA | `e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074` |
| 원래 source131 closure | `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53` |
| 원래 operation closure | `12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903` |
| 원래 local40 raw SHA | `16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074` |
| 원래 과학 source revision | `57da14a2a7a5687e1bdef7d3164f3e8c0051da59` |

도구: [prepare](../../paper/scripts/prepare_titantpp_gru_residual_cap.py), [준비 계약 테스트](../../paper/tests/test_prepare_titantpp_gru_residual_cap.py), [모델](../../models/TPPs/CountAwareTitanCNNGRUResidualCap.py), [실행 연결부](../../paper/scripts/run_titantpp_gru_residual_cap.py). 이 문서의 상대 링크는 저장소 파일을 가리킨다.
