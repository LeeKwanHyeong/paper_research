# TitanTPP 아키텍처 기여 대조 계약 — 2026-10-06

**계약·대표 구조·판정 기준 확정은 완료했다. 새 모델 구현·GPU 검증·54조건 학습은 실행하지 않았다.**

사용자는 이번 단계에서 아키텍처 기여 대조 계약과 대표 모델/성공 기준 고정을
요청했고, 주요 목표를 **“수량 우위가 주목표, 시간 손해는 필수 보고”**로 선택했다.
계약 생성 승인을 새 54조건 원격 학습 승인으로 확대하지 않는다.

## 현재 기준선 — 완료

- 저장소는 `paper_research`, 작성 시 브랜치는 `master`다. 기존 dirty 변경과 연구
  원본은 보존했다. 기존 캠페인·자동화·원격 프로세스를 변경하지 않았다.
- 대표 구조는 **History-MLP 폭8 + 최종 정적 prototype 검색**으로 고정한다.
  후보 폭4·8·12·16의 전용 Validation 표에서 세 데이터 모두 폭8의 3seed 평균
  전체 수량 RMSE가 가장 낮다. 아래 값은 동결 Markdown에 표시된 정밀도다.

| 폭 | Taxi Validation RMSE | Intermittent Validation RMSE | RAF Validation RMSE |
|---|---:|---:|---:|
|4|79.7111|1.67303|33.9507|
|**8**|**77.7352**|**1.66429**|**33.9380**|
|12|79.5101|1.73226|34.0501|
|16|77.8260|1.68670|34.0830|

- 단일 폭 선택 규칙은 데이터마다 최소 평균 RMSE로 나눈 비율의 로그를 데이터
  동등 가중으로 평균하고, 최소 score를 선택한다. 동률이면 더 적은 보정
  파라미터 수를 선택한다. 폭8은 세 데이터에서 각각 유일 최소이므로 데이터
  가중치를 바꿔 다른 폭을 유리하게 만들 필요가 없다.
- MLP12·16·모든 가용 분기 MLP·TitanTPP B와 CNN+GRU54는 과거 참조로 보존한다.
  이번 대조 결과를 보고 대표 폭을 바꾸지 않는다.
- **기존 Test가 이미 노출된 이후의 후속 개발 계약**이다. 과거 Validation을
  근거로 지금 고정한 구조이며, 깨끗한 신규 사전등록/미접근 독립 평가가 아니다.

근거: [전용 Validation 표](../titantpp_cnn_gru54_test_3seed_comparison_20261006_v1/VALIDATION_TABLES.md),
[선택 증적](../../search_artifacts/titantpp_architecture_contribution_contract_20261006_v1/selection_receipt.json).
이번 생성·재검증 프로그램은 전용 Validation 문서만 읽는다. 최초 하위 QA가
혼합 CSV를 순회해 Validation 행만 계산했던 파일 경계 일탈은 선택 증적에
보존했다. 따라서 세션 전체를 “혼합 파일 미열람”으로 인증하지 않는다.

## 구조·용량·직전 이력 입력을 구분한다 — 계약 확정 완료

같은 encoder·입력·시간/수량 출력부·손실·selector에서 아래 여섯 구조만 비교한다.
P는 최종 prototype 검색, H는 두 encoder block 사이의 직접 이력 보정이다.

| Arm | 최종 prototype 위치 | 중간 보정 | 두 모듈의 명목 파라미터 |
|---|---|---|---:|
|P0_H0|제거|제거|0|
|P1_H0|정적 prototype|제거|4,096|
|P0_H1|제거|원래 History-MLP8|12,288|
|**P1_H1**|**정적 prototype**|**원래 History-MLP8**|**16,384**|
|PF_H1|일반 FFN으로 교체|원래 History-MLP8|16,384|
|P1_HC|정적 prototype|현재 상태 FFN으로 교체|16,384|

위 수는 **전체 모델 파라미터 수가 아니라 개입하는 두 모듈의 수**다.
레이어별 16×64 persistent token, 두 층 합계 2,048개는 모든 구조에 유지한다.
P0를 “전체 메모리 제거”라고 부르지 않는다.

- Prototype은 논리적 64×64 bank(native state tensor `[1,64,64]`)에서 cosine 유사도 top4를 고르고 평균을 더한다.
  test-time update를 수행하는 Titans neural memory가 아니다.
- PF는 같은 위치·같은 입력·같은 잔차 연결의 bias 없는 **64→32→64 GELU FFN**으로
  교체한다. 4,096개가 실제 연산에 참여하며 dummy parameter로 수를 맞추지 않는다.
- H1은 분기8개 각각 **128→8→64**, current/직전 관측 state를 입력하며 12,288개다.
  원래 관측 수 경계 `>1/2/4/8/16/32/64/128`와 residual÷8을 유지한다.
- HC는 분기8개 각각 **64→12→64**로 같은 12,288개를 사용한다. 같은 eligibility,
  output-zero 초기화와 residual÷8을 유지하고 명시적인 previous state만 입력에서
  제외한다. current state 자체의 causal encoder 이력은 그대로 존재한다.
- RAF 최대 길이84에서는 경계128 분기가 원래 비활성이다. H1/HC 모두 같은 분기를
  유지하며 잠재적 참여 보정 파라미터는 **7×1,536=10,752개**다. 동적 prototype
  선택·실제 분기 참여와 전체 모델 파라미터 수는 native 검증에서 별도 실측한다.
- 두 개입의 결과는 시간·수량 출력부 모두에 전달된다. 수량 전용 경로로
  표현하거나 TimeNLL이 구조적으로 보존된다고 주장하지 않는다.

History-MLP 자체가 일반 FFN이므로, 같은 입력·mask·폭의 “일반 이력 FFN”은
동일 연산의 재표현이다. 별도 참신성 대조로 만들지 않는다. 이 계약에서
동일 문맥·용량 대조는 P1_H1 대 PF_H1이며, 직접 previous 입력의 추가 효용은
P1_H1 대 P1_HC로 검증한다.

## 학습·평가 조건을 유지한다 — 계약 확정 완료

- Taxi·Intermittent·RAF × seed42/52/62 × 6 arms = **54개 신규 fit 계획**이다.
  기존 완전 모델도 새 대조군과 함께 초기화부터 학습해야 한다. 과거 완료 fit을
  새로운 paired 원인 대조 셀로 재사용하지 않는다. 현재 실행 수는 **0**이다.
- 각 dataset/seed의 여섯 구조를 같은 GPU/runtime에서 실행한다. 잠정 배치는
  5080 Taxi·RAF, 5090 Intermittent다. 서버 소유권·실제 가용성·환경은 아직
  실시간 검증하지 않았으며 배치와 학습 승인도 부여하지 않았다.
- 아홉 dataset/seed 블록의 실행 순서도 계약에 직렬화했다. `schema|dataset|seed|arm`의
  UTF-8 SHA256 오름차순으로 정하며 결과를 본 뒤 순서를 바꾸지 않는다.
- Train-derived 통계·target SHA·시간 단위·normalization·loader·AdamW LR0.001,
  weight decay0.01·batch128·clip1을 부모 계약 그대로 유지한다.
- 손실은 `TimeNLL + 1.0 × log1p(quantity) MSE`, min40/max300/patience40이다.
  최초 strict finite 최소 full Validation raw 수량 RMSE checkpoint를 고른다.
  TimeNLL을 별도 최적 epoch에서 가져오거나 시간 전용 보정 checkpoint를 섞지 않는다.
- 공통 encoder/head 초기 tensor SHA·batch 순서·공통 dropout stream을 구조 사이에서
  맞춘다. 선택 모듈 초기화가 공통 RNG를 밀지 않도록 격리하고 native에서 확인한다.
- PF ordinary Linear 초기화와 원래 prototype `N(0,0.02²)` 초기 잔차 분포는 다르다.
  초기 잔차 norm·실제 gradient 참여를 기록하며 결과를 보고 scale을 맞추지 않는다.
  비교 결과는 사전 고정한 **같은 용량의 모듈 묶음** 차이이며 순수 검색 연산만의
  효과라고 확대하지 않는다. Output-zero의 첫 input gradient=0은 정상일 수 있어
  실제 update 이후 참여도까지 확인한다.
- Test는 잠금 상태다. 전체·큰 수량 RMSE/MAE/TimeNLL을 같은 선택 epoch에서
  보고한다. 큰 수량 경계는 Train 기준 Taxi3449·Intermittent187·RAF200 초과다.
- native 조건36h/전체168h, 서버당 worker1, 자동 retry 금지를 제안했다.
  54조건이 실제 예산·전체 시간에 들어오는지는 검증 전이며 실제 lease/큐 ETA는 없다.
  일부 결과가 좋다는 이유로 불리한 셀을 취소·누락하지 않는다.

## 판정의 종류를 구분한다 — 기준 고정 완료

1. **정합성:** 54셀 전체를 완료·실패·미시작까지 기록한다. 학습 완료 판정은
   scientific_success, selected/last full Validation, 소형 원본 SHA, 실제 소유
   프로세스 부재와 supervisor exit0가 함께 확인돼야 한다. Binary 회수는 별도다.
2. **성분 제거 효과:** dataset/seed별 `q11−q01`, `q10−q00`, `q11−q10`,
   `q01−q00`, `q11−q10−q01+q00`을 보고한다. q는 같은 seed의 선택된 Validation
   RMSE이며 음수가 개선이다. 이 2×2에는 제거된 용량 차이도 포함된다.
3. **용량을 맞춘 구조 근거:** `q11−qPF_H1` 및 `q11−qP1_HC`를 각각 보고한다.
   **같은 두 데이터 이상에서 두 필수 대조가 모두 세 seed 음수**일 때 두 요소가
   함께 뒷받침된다고 기술한다. 각 요소만의 개선 데이터 수도 별도로 보고한다.
   큰 수량/MAE/TimeNLL이 나쁘면 함께 공개한다.
   이는 통계적 유의성이나 두 구조의 순수 시너지를 입증하는 기준이 아니다.
4. **외부 수량 우위:** 하나의 고정 모델이 세 데이터 중 두 개 이상에서 S2P2보다
   평균 전체·큰 수량 RMSE가 모두 낮아야 “대부분 데이터의 수량 평균 우위”를
   논의한다. 같은 seed 차이·개선 seed 수·MAE 및 강한 내부/단순 기준선도 보고한다.
   과거 외부 결과는 개발 문맥이며 새 GPU 대조 셀이나 독립 확인으로 취급하지 않는다.
5. **시간 손해:** 같은 수량-selected epoch의 전체/큰 수량 TimeNLL 평균·표본 SD·
   각 seed 절대 차이를 반드시 기록한다. 임의 2%·5% 허용치를 만들지 않는다.
   시간 손해가 있으면 공동 수량/시간 우월성은 미확립으로 남긴다.
6. **불확실성:** 표본 SD는 seed 변동이며 confidence interval이 아니다. 같은
   target을 세 seed로 평가해도 독립 사건 수는 세 배가 되지 않는다. 미접근
   기간/대상과 외부 모델을 함께 고정한 독립 평가 계약은 별도 후속이다.

구조 근거가 실패하면 불리한 결과를 보존하고, prototype 기여나 메모리 참신성이
확인됐다고 쓰지 않는다. 결과를 본 뒤 폭/selector를 바꿔 같은 계약을 성공시킬 수 없다.

## 검증 증적 — 완료

- [기계 판독 계약](../../search_artifacts/titantpp_architecture_contribution_contract_20261006_v1/design_contract.json)
- [현재 동결 포인터](../../search_artifacts/titantpp_architecture_contribution_contract_20261006_v1/current.json)
- [123개 원본 소스 SHA](../../search_artifacts/titantpp_architecture_contribution_contract_20261006_v1/source_registry.json)
- [오프라인 계약 검증](../../search_artifacts/titantpp_architecture_contribution_contract_20261006_v1/validation_receipt.json)

오프라인 검증은 문서/계약/원본 SHA/정적 파라미터 산술을 확인한다. 실제 모델의
forward·gradient·메모리·GPU deterministic validation을 통과했다는 뜻은 아니다.

## 남은 작업 순서

**여섯 구조를 구현하고 계약 테스트한다 — 다음 작업 / 최우선**
- `paper_research`의 별도 모델 identity와 strict checkpoint 검증을 구현한다.
- 원래 폭8 forward/초기화 일치, mask/reset/인과성, 실제 parameter/gradient 참여,
  공통 tensor 및 RNG 증적을 검증한다. 새 source closure를 동결한다.

**원격 환경과 54조건 실행 예산을 검증한다 — 다음 작업 / 구현 이후**
- 잠정 5080·5090의 소유 작업·GPU/runtime·native memory와 비용/시간 한도를 확인한다.
- 실제 실행 계약·start/training permit과 lease는 이 검증 후 별도로 확정한다.

**54조건 신규 학습을 실행한다 — 다음 작업 / native 검증·실행 승인 이후**
- dataset/seed마다 여섯 구조를 같은 GPU에서 실행하고 모든 조건을 기록한다.
- 과거 학습이나 자동화를 다시 시작하지 않는다. Test는 잠금 상태를 유지한다.

**원인 대조와 외부 비교를 독립 평가로 연결한다 — 다음 작업 / 개발 결과 이후**
- 성분 제거·용량 대조·외부 수량 비교·시간 손해를 각각 판정한다.
- 고정된 후보와 외부 비교군의 미접근 평가를 별도 계약으로 검증한다.

이번 단계에서 커밋·Push·MR·원격 배포는 수행하지 않았다.
