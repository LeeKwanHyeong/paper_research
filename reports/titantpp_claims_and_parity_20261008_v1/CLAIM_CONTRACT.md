# TitanTPP 핵심 질문과 검증 계약

이 문서는 논문 본문이 아닌 연구 실행 계약과 근거 기록이다. 기존 결과, 앞으로 검증할 가설, 현재 승인된 실행 범위를 구분한다.

**현재 대표 구조를 고정한다 — 완료 / 기준선**
- 대표 구조는 동결 science revision `4274043717cf24640d9a18fb35231a1f7ba50c8e`의 CNN+GRU54다. source126 closure는 `918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a`다.
- `titantpp_retrieval_mean`은 native 동등성 검사를 통과한 경우 같은 구조 기준선의 실행 이름으로 사용할 수 있다. 두 이름을 독립 모델 성능 근거로 중복 집계하지 않는다.
- 기존 A100 selected checkpoint는 강한 관측 비교 대상으로 보존한다. 새 초기화와 첫 업데이트의 일치는 과거 A100 초기화가 동일했거나 과거 성능 차이의 원인이 밝혀졌다는 뜻이 아니다.
- 모든 데이터에 같은 대표 버전을 사용한다. 데이터마다 사후 최선 버전을 골라 대표 모델처럼 합치지 않는다.

**명시적 이력 보정의 추가 효과를 검증한다 — 중심 가설 / 후속 실행 대기**
- 질문: 인코더가 이미 관측 이력을 요약해도 직전 관측 상태를 명시적으로 연결하는 보정이 추가 수량 이득을 주는가?
- 공통 encoder 구조와 초기 파라미터·관측 마스크·출력부를 맞추고, CNN 보정과 GRU 잔차 경로의 전체 사용, 해당 보정 경로 제거, 같은 활성 파라미터 예산의 현재 상태 전용 보정을 비교한다. 각 arm은 별도로 end-to-end 학습하므로 학습 후 encoder 상태까지 동일하다는 뜻은 아니다.
- 현재 상태에도 prefix 이력이 들어 있으므로 현재 상태 대조를 '이력이 없는 모델'이라고 표현하지 않는다. 제거 대조만으로 용량 효과가 통제됐다고 보지 않는다.
- 같은 데이터·seed의 두 비교 모두에서 전체 raw 수량 Validation RMSE가 엄격하게 낮아야 개선 쌍이다. 같은 두 데이터 이상에서 각 3seed가 모두 개선되면 개발 단계 지지로 판정한다. 세 데이터의 불리한 결과, 개선 크기, 평균과 표본 SD를 함께 보존한다.
- 이 기준은 신규 CNN 대조를 위한 사전 규칙이다. 완료된 MLP8 구조54의 기존 계약·판정을 바꾸거나 그 결과를 CNN 기여의 대체 증거로 쓰지 않는다. 통계적 유의성이나 독립 일반화를 이 부호 기준만으로 주장하지 않는다.

**검색 정보 분리의 추가 효과를 검증한다 — 조건부 후보 / 후속 실행 대기**
- 질문: 시간·수량 정보를 나눠 학습된 정적 prototype을 선택·결합하면 일반 학습 검색보다 추가 수량 이득이 있는가?
- 후보는 fixed 0.5/0.5다. mean, weighted, fixed, fixed와 정확히 같은 활성 추가 파라미터 예산의 일반 학습 검색을 비교한다.
- fixed 추가 파라미터는512개, 기존 generic은521개다. 기존 generic을 fixed의 정확한 동일 예산 대조로 인증하지 않는다. 실제 기능에 사용되는 파라미터와 gradient 경로를 확인하며 사용하지 않는 dummy 파라미터는 예산 대조로 인정하지 않는다.
- prototype64/top4와 과거32 관측 사건 요약을 맞춘다. 실제 과거 사건 bank나 prototype 내용 자체를 개선하는 방법으로 표현하지 않는다.
- score scale·normalization·temperature를 맞추고, 고정 두 점수 합이 concatenated 일반 검색으로 재표현되는지 확인한다. 선택 집합과 가중 결합의 효과도 분리한다.
- 같은 두 데이터 이상에서 각 3seed의 fixed가 mean·weighted·활성512 일반 검색 모두보다 전체 Validation RMSE가 엄격하게 낮고 같은 GPU의 두 길이 비용 기준을 충족하면 개발 단계 지지로 판정한다. 큰 수량 개선은 전체 개선에서 추론하지 않고 별도의 짝 비교로 확인한다.
- adaptive의 기존 파일럿 계약과 실패한 추가 효과 판정은 그대로 보존한다. 신규 fixed 계약을 과거 Test의 성공 판정으로 소급 적용하지 않는다. fixed는 후속 검증 대상이며 최종 채택 모델이 아니다.

**학습과 선택 조건을 맞춘다 — 계약 확정 / 후속 구현 대기**
- B128, 최대 길이256, lookback168, min40/max300/patience40, AdamW lr0.001/wd0.01/betas0.9,0.999/eps1e-8, clip1, time-head LR multiplier1을 유지한다.
- count-only log regression, 수량 손실 계수1, tail 보조 손실0, 동일 lognormal duration head·관측 시간 계약·Train 통계를 유지한다.
- 기록 시간 단위는 Taxi=hour, Intermittent=week, RAF=month다. 각 데이터의 기존 관측 단위를 유지하며 데이터 간 TimeNLL을 하나의 수치로 합산하지 않는다.
- 최초 strict finite 최소 full Validation raw 전체 수량 RMSE로 checkpoint를 선택한다. 동률이면 최초 epoch를 유지하며 시간이나 큰 수량 점수로 다른 epoch를 고르지 않는다.
- 같은 선택 epoch의 전체/큰 수량 RMSE·MAE·TimeNLL 6지표를 보고한다. Train 고정 경계 초과를 큰 수량으로 정의한다: Taxi3449, Intermittent187, RAF200.
- 같은 데이터·seed의 모든 arm은 같은 GPU·Runtime·배치 순서를 사용한다. 공통 초기 tensor의 binding과 RNG를 기록한다. 같은 seed의 이종 GPU 반복은 독립 seed로 세지 않는다.
- 예정 seed는42/52/62다. 새 3dataset/3seed 학습은 이번 승인 범위 밖이며 실행 허가는 별도로 기록한다. GPU 임대·프로세스 wall-clock 제한도 후속 운영 계약에서 정한다. 제한시간 중단을 과학적 완료로 인증하지 않는다.
- 동일 GPU B128/T64,T256에서 mean 대비 추론+20%, 학습step+30%, peak allocation+20%의 기존 비용 기준을 적용한다. TimeNLL 허용치를 결과를 본 뒤 새로 만들지 않는다. 비용 적격과 정확도 효과를 구분한다.

**외부 경쟁력과 새 독립 평가를 분리한다 — 기존 관측 유지 / 데이터 확인 대기**
- 기존 CNN+GRU54 seed42의 공통 출력부 외부 수량 비교는 당시 동결 버전·평가 계약의 관측 근거다. 최종 모델의 모든 seed·지표에 대한 우월성으로 확대하지 않는다.
- 외부 비교에는 RMTPP·THP·NHP·SAHP·AttNHP와 S2P2를 포함한다. 공통 출력부 적용 범위, 데이터·학습 예산·선택 규칙을 기록한다. 새 모델의 결과와 과거 버전의 결과를 혼합하지 않는다.
- 기존 Taxi/Intermittent/RAF Test와 Instacart의 Test 노출 증거가 있어 이 자료를 새 untouched 평가로 인증하지 않는다. 새 모델 선택과 수정에는 개발용 Validation만 사용한다.
- 독립 후보는 이후 기간 Taxi와 Online Retail II다. 둘 다 아직 미노출 인증이 없으며 원본 확보 또는 접근 이력 감사가 필요하다. 모델·학습·selector를 동결한 후에만 예약한 평가 자료의 성과를 확인한다.
- 현재 데이터 조사는 metadata·접근 계보의 확인이며 새 독립 평가 실행은 아니다.

**현재 승인 범위의 기준선 검사를 수행한다 — 완료**
- 5080 기존 ai_env에서 seed42, 두 구현 각각의 자연 초기화와 constructor RNG, 합성 B128/T64,T256 eval/train, 실제 Taxi 첫 Train 배치를 반복한 3회 업데이트를 확인한다.
- 실제 Train optimizer update는 각 구현3회, 합성 업데이트는 각 구현2회가 상한이다. 구현 사이 파라미터 복사를 통해 차이를 숨기지 않는다. Dropout 비교를 위해 분기 시작 RNG를 복원하고 종료 RNG도 비교한다.
- 공통 tensor·gradient·clip 이후 gradient·optimizer state·업데이트 이후 파라미터는 bitwise로 비교한다. 입력, source, RNG, 실제 Runtime과 GPU UUID의 digest를 남긴다.
- native 검사의 합격은 새 환경의 구현 동등성 근거다. 새 full fit, Validation/Test 성과 평가, checkpoint 선택, 기여 대조 확대는 실행하지 않는다.
- 2026-10-08 09:11:52KST, 5080/ai_env의 합성2case와 실제 Train1case에서 총83개 bitwise 비교가 통과했다. 각 구현의 실제 Train 업데이트3회·합성2회, exit0·종료 후 GPU PID 부재와 source/input 보존을 확인했다.
- 전향 계약의 시간 단위 표기는 v2에서 데이터별 단위로 정정했다. native 검사가 rawSHA로 참조한 v1은 그대로 보존하며, 실제 검사는 동결 Taxi 데이터 계약을 사용했으므로 이 문서 표기 정정이 검사 계산이나 결과를 바꾸지 않는다.

근거:
- [현재 기계 판독 계약 v2](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/titantpp_claims_validation_20261008_v2.json)
- [완료된 native 검사 계약](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/execution_contract.json)
- [native 완료 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json)
- [문헌 검토](LITERATURE_REVIEW.md)
- [독립 평가 후보 계보](evidence/independent_evaluation_inventory.json)
- [완료된 기준선 감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/README.md)
