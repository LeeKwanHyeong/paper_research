# CNN·GRU 구조 대조 구현과 CPU 계약 검증

**기존 폭16을 기준으로 세 후보를 구현한다 — 완료**

- 대상은 `paper_research`의 첫 Encoder와 두 Encoder 사이의 보정 모듈입니다. 기존 MLP16과 폭4 결과는 재사용합니다.
- `CNN + MLP16`, `GRU54`, `CNN + GRU54`를 별도 모델 이름과 checkpoint 식별자로 구현했습니다. 후보를 다른 구조나 기존 모델로 바꾸어 읽는 경로는 거부합니다.
- CNN은 Encoder 1의 사건 Q/K/V에 채널별 kernel3 잔차 필터를 넣습니다. 현재 사건과 같은 구간에서 관측된 직전 두 사건을 읽습니다. Padding은 건너뛰고, 유효하지만 관측하지 않은 사건에서는 필터의 과거 구간을 초기화합니다. Persistent token과 Encoder 2의 Attention은 필터링하지 않습니다.
- GRU는 기존 8개 MLP 보정 분기를 전체 대체합니다. `64 → 54 → GRU54 → 64` 보정을 원래 64차원 표현에 더합니다. 첫 관측에서는 상태만 갱신하고, 두 번째 관측부터 보정합니다. 표본·prefix·구간 밖으로 순환 상태를 넘기지 않습니다.
- GRU에는 기존 `/8`과 단계별 8분기 활성화를 붙이지 않습니다. 그래서 GRU 대조는 과거 범위·활성화·잔차 크기를 포함한 **보정 모듈 전체 교체** 효과입니다. 순환 연산 하나의 효과라고 해석하지 않습니다.
- CNN의 추가 kernel과 GRU의 출력 projection은 0으로 초기화합니다. 공통 파라미터와 난수 상태를 보존하며, 초기 출력은 기존 MLP16과 같습니다.

실제 모델을 생성해 계산한 파라미터 수는 다음과 같습니다. GRU54는 MLP16보다 보정 파라미터가 156개 많습니다. 54는 순환 상태의 폭이며 MLP의 폭16과 같은 잠재 폭이 아닙니다.

| 구성 | 중간 보정 | CNN 추가 | 합계 보정·CNN | 전체(max length256) | 전체(max length84) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 MLP16 | 24,576 | 0 | 24,576 | 114,435 | 103,427 |
| CNN + MLP16 | 24,576 | 576 | 25,152 | 115,011 | 104,003 |
| GRU54 | 24,732 | 0 | 24,732 | 114,591 | 103,583 |
| CNN + GRU54 | 24,732 | 576 | 25,308 | 115,167 | 104,159 |

수량 출력부·손실, 관측된 정수 시간의 확률질량 NLL, Encoder 2, 두 persistent bank와 마지막 prototype retrieval은 공통 경로를 유지합니다. 실제 연구 학습은 동일한 선택 기준의 계약으로 별도 봉인해야 합니다. 현재 CPU 검증에서 실제 연구 데이터를 학습하지 않았습니다.

**인과성·모델 식별·공통 학습 경로를 검증한다 — 완료**

- `/usr/local/bin/python3`, Python3.12, torch2.14.0의 CPU에서 **160개 검사 통과**: CNN·GRU59, 기존 폭4/8/12/16 모델57, 폭8/12 실행 계약35, 시간 진단9입니다.
- CNN과 GRU의 독립 수식 및 gradient 대조, padding 건너뛰기, 관측 경계 초기화, 미래·target 비노출, 표본·호출 간 상태 분리를 확인했습니다. CNN의 필터 입력이 사건 Q/K/V뿐이며 persistent memory는 원래 경로를 유지하는 것도 확인했습니다.
- seed42·52·62 및 최대 길이84·256에서 공통 초기 tensor·난수 상태·초기 출력을 확인했습니다. GRU의 출력이 0인 최초 backward에서 내부 gradient가 0이고, 출력 projection을 한 번 갱신한 다음 내부 gradient가 생기는 것을 확인했습니다.
- CNN이 들어간 두 후보 모두, 실제 전체 모델의 joint loss에서 Q/K/V 세 kernel의 현재·직전·두 사건 전 행 각각에 유한한 0이 아닌 gradient가 생기는 것을 확인했습니다.
- 세 후보 각각에 합성 자료로 공통 학습기를 2epoch 실행하고, 선택 checkpoint와 마지막 checkpoint의 전체 **Validation** 재평가를 수행했습니다. RMSE·MAE·Time NLL이 해당 epoch의 이력과 일치했습니다. 연구 성능 결과가 아니라 실행 경로 검증입니다.
- 후보별 role, 손실과 선택 기준, 분할 표시, 파라미터 shape 및 architecture identity를 바꾼 checkpoint를 거부했습니다. 불완전한 state를 읽을 때 모델을 부분 변경하지 않습니다. 기존 CNN checkpoint 경로와 폭16의 과거 식별자도 유지됩니다.
- 후보 검사 후 process-local Factory·학습기·role hook을 복원하므로 같은 Python 검사 세션에서 기존 폭 모델 검사와 함께 통과했습니다.

검사 명령:

```sh
/usr/local/bin/python3 -m pytest simple_lab_test/search/tests/test_titantpp_cnn_gru.py simple_lab_test/search/tests/test_titantpp_history_width.py simple_lab_test/search/tests/test_titantpp_history_capacity_campaign.py simple_lab_test/search/tests/test_titantpp_width_time_diagnostic.py -q --junitxml=reports/titantpp_cnn_gru_implementation_20261004_v1/cpu_contract_tests.xml
/usr/local/bin/python3 reports/titantpp_cnn_gru_implementation_20261004_v1/verify_implementation.py
```

증적은 [verification.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_implementation_20261004_v1/verification.json)과 [검사 결과 XML](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_implementation_20261004_v1/cpu_contract_tests.xml)에 있습니다. 각 코드·설계·검사 파일의 SHA와 실제 모델별 파라미터 수를 기록했습니다.

**현재 폭8·12 학습과 GPU 검증의 순서를 연결한다 — 다음 작업**

- 5080·5090에서 실행하는 Taxi·Intermittent·RAF의 폭8·12 학습은 기존 동결 source를 사용합니다. 이번 새 모듈은 그 실행본을 수정하지 않습니다. Instacart는 후속 대상입니다.
- CPU torch2.14.0 검증은 원격 torch2.11.0+cu130에서의 GRU 결정성·동작 검증을 대신하지 않습니다. 서버가 비면 지정 GPU에서 합성 qualification과 peak memory·step 시간·추론 지연을 확인해야 합니다. 이를 아직 완료했다고 표시하지 않습니다.
- 기존 MLP16 9조건을 재사용하고 새 세 구조 × 세 데이터 × 세 seed의 27조건을 구성하는 것은 후속 실행 계약의 범위입니다. 이번 증적은 새 GPU 학습을 시작하거나 결과의 우월성을 입증하지 않습니다.
- 파라미터 수가 가깝더라도 계산량·표현 용량은 같지 않습니다. 성능 개선, 일반화, 논문의 계산 효율에 관한 결론은 실제 완료 결과와 측정 뒤에 판단합니다.

원래 [동결 설계](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_width16_design_20261004_v1/design.json)는 설계 당시 상태로 보존합니다. 이번 구현은 [새 모델](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanCNNGRU.py)과 [명시적 실행 경로](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_titantpp_cnn_gru.py)에 연결합니다. 실제 회사 자료, 연구 checkpoint, Test 성능·예측 및 Validation/Test 혼합 결과는 이 CPU 검사에서 읽지 않았습니다.

**구현 원본과 독립 검토를 보존한다 — 완료**

- [소스 봉인 기록](source_seal.json)은 CPU 검증에서 사용한 현재 공통 코드와 신규 모델·실행기의117개 원본과 SHA를 보존합니다. 실제 CPU 검증본의 closure는`40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd`입니다. 실행 중인 폭8·12 동결본을 수정하지 않았습니다. 원래114개와 다른 현재 라우팅4파일 및 추가 의존성을 명시했고, 처음 부모 소스를 그대로 결합한 보존본은 미검증 구성으로 구분했습니다. 현재117개 보존본을 새 임시 폴더로 복원해 합성 검사·초기 SHA·파라미터 수·8회 optimizer update를 다시 확인했습니다.
- [독립 코드 검토](independent_review.json)에서는 계약·인과성·상태 범위·공통 출력부·checkpoint 연결에 확인된 차단 문제가 없었습니다. GPU 실행 검증은 별도로 남겼습니다.
- 이 보존본은 CPU 구현 증거입니다. 실자료 학습 허가·새27조건의 동결 실행 계약은 포함하지 않습니다.
