# 현재 논문 주장에 대한 검색 실험 필요성 판단

2026-10-12, paper_research. 사용자 요청은 현재 결과·논문 주장에 따라 검색 실험이 필요한지 판단하고, 필요한 경우 최소 구성과 진행 순서를 정리하는 것이다. 이번 작업은 판단·계획 문서만 작성한다. 열린 `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex`와 PDF는 수정하지 않는다.

**핵심 주장에 필요한 증거를 판단한다 — 완료**

권고는 **현재 논문에는 신규 검색 학습을 추가하지 않는다(0 fit)**이다. 현재 제목·세 기여·결론은 CNN+GRU54의 국소·순환 결합 보정 경로, 외부 인코더와의 성능 위치, Instacart 구조 비교다. 검색 규칙의 우위는 현재 완료 기여에 포함되지 않는다.

- 보정27은 동일한 mean prototype readout을 유지한 상태에서 full·보정 우회·current-only를 비교했다. Taxi·Intermittent의 세 seed 모두 두 대조보다 전체 RMSE가 낮고 RAF52는 실패한다. 이는 **prototype을 포함한 해당 backbone에서의 조건부 보정 효과**다. Prototype 자체의 필요성, prototype을 없애도 같은 보정 이득이 생기는지, CNN/GRU별 인과 효과를 분리한 결과는 아니다.
- Instacart9는 서로 다른 파라미터 예산의 구조 비교다. 작은 평균 수량 이득과 seed52 MLP12 우위, 평균 전체 TimeNLL 악화가 있다. 이 결과가 새 검색 규칙을 요구하거나 검색 효과를 입증하지는 않는다.
- 검색 규칙을 바꾸는 실험은 별도 질문을 추가한다. 현재 전체 TimeNLL11/18 및 큰 수량 RMSE·MAE 각각10/18 손해, 검증에 이미 노출된 데이터, seed 간 GPU·Runtime 차이는 검색 실험을 추가한다고 해결되는 문제가 아니다. 검색으로 손해가 해소된다고 미리 가정하지 않는다.
- 따라서 우선순위는 완료된 보정 근거에 맞춰 원고를 좁히고, 재현성과 미노출 평가 가능성을 정리하는 것이다. 현재 결과만으로 일반화·유의성·참신성·채택 가능성이 확정된다는 결론은 아니다.

이 판단은 로컬 주장–근거 대조에 따른 연구 방향의 권고다. 일반적인 판단 기준으로 [NeurIPS 공식 체크리스트](https://neurips.cc/public/guides/PaperChecklist)의 주장–근거 범위 일치, 한계와 변동 요인·재현성 공개 원칙을 참고했다. 이를 PAKDD의 제출 규정이나 특정 추가 실험 의무로 사용하지 않는다.

**검색을 제외하더라도 실제 모델의 설명은 유지한다 — 다음 작업**

- 대상은 열린 동일 `main.tex`다. 실제 mean top4 prototype readout과 persistent attention bank의 차이, prototype64×64·출력부·학습 계약은 재현성 설명으로 남긴다. 검색 규칙 비교를 제외하는 것과 실제 모델에서 prototype을 제거하는 것은 다르다. 학습된 모델·결과표는 바꾸지 않는다.
- Introduction의 `For retrieval, we compare` 같은 현재형, Related Work의 미완료 검색 연구 설명, fixed/general512 수식·그림, 대기 상태 Results절과 Q2·비용 문단을 후속 연구/부록으로 분리하거나 줄이는 편집안을 준비한다. 완료하지 않은 비교를 본문의 두 번째 실증 기여처럼 제시하지 않는다.
- 과거 MLP8 prototype on/off 관찰은 구조가 다른 역사적 근거로 유지한다. 그 결과를 CNN+GRU54의 prototype 효과로 옮기지 않는다.
- 완료 조건은 제목·기여·Methods·Results·결론이 보정 중심의 같은 질문을 설명하고, 실제 mean readout을 재현할 정보는 남아 있는 것이다. 이번 판단 요청에서는 `.tex`를 편집하지 않았다.

**검색이 필요한 조건과 서로 다른 질문을 구분한다 — 완료 / 조건부 후속 작업**

검색 규칙을 논문의 독립 기여로 남기고 “fixed가 단순 평균·유사도 가중·동일 예산 공동 query보다 낫다”는 주장을 하려는 경우에는 새 비교가 필요하다. 현재 원고 중심 주장을 유지하기 위해 이 확장을 반드시 할 필요는 없다.

| 질문·범위 | 과학 비교 조건 수 | 말할 수 있는 범위 | 현재 권고 |
|---|---:|---|---|
| 현재 보정 중심 주장 유지 | 신규0 | Prototype을 유지한 backbone에서 보정 결합 경로의 개발 효과 | 우선 진행 |
| 검색 규칙 단일 dataset 파일럿: 4arm×3seed | 12 | 해당 dataset의 개발 관찰 | 기존 Q2 완료로 표시 불가, 필수 아님 |
| 검색 규칙 2dataset×4arm×3seed | 24 | 선택한 두 dataset에서 새로 한정한 질문 | 논리상 최소지만 원래3dataset 설계 축소를 공개하는 새 계약 필요 |
| 기존 검색 연구: 3core dataset×4arm×3seed | 36 | 원래 세 dataset을 보존한 Q2와 손해 사례 | 검색을 독립 기여로 유지할 때의 권장 완전 구성 |
| Prototype 자체의 추가 효과: full에서 on/off×3seed×3core | 18 | 해당 full 구조에서 prototype 유무의 효과 | 검색 규칙 Q2와 다른 연구, 현재 필수 아님 |

24조건을 쓰려면 dataset·선택 이유를 검색 성과를 보기 전에 고정하고 기존3dataset 설계를 축소했다는 사실을 공개한다. 이미 보정 성과가 좋다는 이유로 Taxi·Intermittent만 선택하면 사후 선택이다. 새 등록 절차를 마련해도 이미 관찰한 Validation이 독립 확인으로 바뀌지 않는다. 원래 범위를 유지하는 기본 예산은36조건으로 잡는다. Instacart 추가, 학습률·prototype 수·top-k·query 설계 튜닝은 이 최소 구성에 포함하지 않는다.

Prototype on/off18조건은 full의 prototype 추가 효과만 확인한다. Prototype 없는 조건에서도 보정 이득이 관찰되는지와 두 요소의 상호작용을 비교하려면 correction on/off×prototype on/off×3seed×3core=36조건이 필요하다. 이 구성만으로 효과크기의 동등성이 입증되는 것은 아니며, 그런 주장에는 별도의 동등성 기준·분석이 필요하다. Current-only 동일 예산 주장까지 두 prototype 환경에 확장하려면3correction×2prototype×3seed×3core=54조건이다. 같은 예산의 비메모리 대조까지 필요한 메모리 고유 효과 주장도 별도 확장이다. 현재 논문을 위해 이런 추가 주장을 만들지 않는다.

**필요 시 최소 네 검색 규칙과 고정 판정을 정한다 — 승인 필요**

| Arm | 동일 구조의 prototype bank를 읽는 방식 | 추가 활성 query 파라미터 |
|---|---|---:|
| mean | 기존 hidden cosine으로 같은 top4를 선택하고 균등 평균 | 0 |
| weighted | 같은 top4 선택에 softmax 유사도 가중 평균 | 0 |
| fixed | 시간·수량 query를 각각 정규화하고 점수0.5:0.5 결합 | 512 |
| general512 | 같은 시간·수량 요약으로 하나의 공동 query·정규화 | 512 |

- 공통 조건은 CNN+GRU54 full correction, 정적 학습 prototype64×64, top4, 온도1이다. 각 arm은 동일한 공통 파라미터 초기값·입력 계약에서 end-to-end 독립 학습하며, 최종 encoder·prototype 값은 서로 다르다. 관측 과거32개에서 얻는 시간·수량 각각4개 요약은 fixed/general512에만 query 입력으로 쓰인다. Mean/weighted는 hidden 기반 검색이며 이 요약을 query 입력으로 사용하지 않는다. 기존 general521 파일럿을 general512 대조로 쓰지 않는다. Fixed의 분리 정규화 대조는 연결 dot-product 표현과 대수적 동등성이 있으며 새로운 메모리 연산의 발명으로 해석하지 않는다.
- 데이터는 원래 Taxi·Intermittent·RAF, seed42/52/62를 유지한다. B128, min40/max300/patience40, frozen 초기화·data·loss·optimizer·selector를 유지한다. 실행에는 현재 작업트리 기본 코드가 아니라 검증한 frozen source를 연결해야 한다. 현재 evaluator 한 파일은 동결 공통 source와 달라 새 코드 결과를 frozen 재현으로 표시하지 않는다.
- Q2는 같은 두 개 이상 dataset **각각 모든3seed**에서 fixed의 전체 raw 수량 RMSE가 mean·weighted·general512 모두보다 strict 낮은 조건을 요구한다. Dataset별 최종 fixed 비용도 같은 GPU mean 대비 추론≤1.20, 학습 step≤1.30, peak allocation≤1.20를 사전에 고정한 모든 길이에서 통과해야 개발 지지다. 성과나 비용 기준을 통과하지 못해도 모든 arm과 불리한 지표를 보고한다.
- 모든 성능은 최초 strict finite 최소 전체 Validation raw 수량 RMSE epoch에서 전체/큰 수량 MAE·RMSE·TimeNLL을 함께 가져온다. 3seed 평균·표본 SD, seed별 차이와 손해를 보존한다. 시간이나 tail에 좋은 다른 epoch를 따로 고르지 않는다. CPU 감사·Test·새 독립 평가와 구분한다.

**공통 mean 재사용과 실행 계약을 먼저 확정한다 — 승인 필요 / 현재 재사용 확정0**

- 이력 보정 full9는 **A1005+RTX4**다. RTX4는 Intermittent42/52/62 on5090과 RAF62 on5080이다. A100 Taxi3·RAF42/52를 RTX 동일 GPU mean으로 사용할 수 없다. RAF42의 물리 A100 이관도 보존한다.
- RTX4는 재사용 후보다. 동결 검색/RTX 이력 source129씩을 검증했으며 공통126의 SHA와 data·training·claim 계약은 일치한다. 과거 실제 GPU UUID·Python/torch/CUDA/cuDNN·초기화·입력·Train/Validation 순서 원본도 있다. 그러나 **새 검색 native qualification, arm 초기화·공통 batch-prefix 및 단독 비용 bridge 증거가 아직 없어 확정 재사용은0**이다.
- 신규 계약은 `(host,dataset,seed)`·동일 물리 GPU UUID/Runtime·초기화 canonical SHA·동일 입력/target/selector·공통 Train epoch-prefix와 Validation 순서·binary 원본 SHA를 연결한다. 과거 mean의 원관측시각과 원본은 유지하고 새 fit으로 이름을 바꾸지 않는다. 이 검증이 안 되면 해당 mean도 fresh fit으로 잡는다. 같은 seed·코드 일부의 동일성만으로 재사용을 확정하지 않는다.
- 전체36과학 조건의 안전 기본은 **36 fresh fit**이다. RTX4 bridge가 모두 통과하면 **32 신규 fit=검색27+새 RTX mean5**, 과거 mean4를 더해36조건이다. 과거 문서의 “새 검색27만”은 현재 RTX 이관 경로의 비용 계획이 아니다.
- 기존 검색 bundle은 `prepared`, `prior_stage`가 과거 history bundle을 가리키고 workers/host=1이다. 최신 RTX 인계와 parallel3 승인을 자동 연결하는 실행 계약이 아니다. old advance/deploy/qualification/launch를 그대로 재실행하지 않는다. 신규 계약·배포·permit·ownership 및 원본 보존을 먼저 설계한다.

**비용 측정의 길이 불일치를 해결한 뒤만 실행 계약을 확정한다 — 차단됨 / 계약 정합 미완료**

- 현재 문서·원고의 비용 기준은 T64/T256이다. Frozen 검색 operation은 `{min(64,max_seq_len),max_seq_len}`이라 Taxi·Intermittent는64/256, RAF는64/84를 측정한다. RAF84 비용을256 비용으로 표시하지 않는다.
- 기존 gate를 조용히 완화하지 않는다. Frozen 모델의 `_get_pos`는84개 learned position parameter를 선형 보간해256 위치의 tensor를 생성한다. 따라서 **모델 생성·Train cap·초기화는84로 유지하고 operation의 synthetic 측정 길이에256을 추가하는 방식은 코드상 가능**하다. 새 Parameter나 초기화 RNG를 소비하지 않는다. 실제 native 실행의 finite·메모리·측정 성공은 미확정이며, RAF 실제 학습창 밖의 synthetic 비용 스트레스 검사라는 범위를 명시해야 한다.
- 반대로 모델 생성의 max_seq_len을256으로 바꾸면 positional parameter가11,008개 늘고 초기화 RNG도 달라져 같은 과학 모델이 아니다. 이 방식으로 mean 비용을 비교하지 않는다. 새 운영 측정 계약에 T64/T84/T256 구분과 단독 조건을 고정하고, 승인·native 검증 전까지 비용 계약과 launch를 미완료로 둔다. [동결 코드 검토](contract_review.md)에 경로·줄·SHA를 기록했다.
- 비용은 batch128, warmup5/measure20, 해당 GPU에 다른 fit이 없는 단독 조건에서 측정한다. 세 fit 동시 실행시간을 비용비나 구조 효율의 근거로 쓰지 않는다. 시간·전체 큐 ETA·금액은 현재 미확정이며 다른 arm/GPU 속도를 복사해 추정하지 않는다.

**검색을 유지할 때만 의존관계대로 실행한다 — 승인 필요**

1. 주장·dataset 범위, 네 arm, Q2/여섯 지표, 비용 길이와 공통 mean 재사용·source 계약을 문서로 확정한다. 이 단계는 native 실행 없이 준비할 수 있다.
2. 사용자에게 고정된 fit 목록·GPU/Runtime·최대3worker·중단 조건·시간/비용 상한·회수 예산을 제시하고 해당 실행 범위를 승인받는다. 이번 계획 요청은 launch 승인이 아니다. 서버의 현재 유휴·Runtime 상태는 아직 조회하지 않았다.
3. 승인 범위 안에서만 source/입력/초기화·causality·배치 순서·native 단독 비용을 검증하고 실제 GPU UUID·유휴·3worker admission을 확인한다. 실패/claim을 보존하고 자동 retry/resume하지 않는다. 비용 초과 arm도 결과 비교에서 삭제하지 않는다.
4. 각 dataset/seed의 네 arm을 같은 GPU·Runtime에 고정한다. Mean은 먼저 fresh 완료하거나 재사용 bridge를 통과한 원본을 연결한다. 이어 weighted/fixed/general512 세 fit을 최대3개 동시 실행하고, 세 fit 모두 자연종료한 뒤 다음 seed로 간다. 서로 다른 dataset/seed 묶음은 두 host에 병렬 분담할 수 있지만 같은 묶음의 mean과 검색 arm을 다른 GPU로 나누지 않는다.
5. 실제 worker·supervisor 종료와 immutable binary manifest/archive를 확인하고 원본을 회수한다. 노출 순서·GPU·Runtime·선택 epoch를 대조해36개 과학 ID를 한 번씩 집계한다. 모든 여섯 지표·표본 SD·불리한 조건과 Q2/비용 판정을 함께 보고한다.

**현재 권고에 따른 논문 후속 순서를 재정렬한다 — 완료**

- 다음: 실제 mean readout의 재현성 설명은 유지하면서 미완료 검색 기여·현재형·빈 Results의 비중을 줄이는 원고 편집안을 반영한다. 공통 `main.tex` 편집은 한 세션에서 직렬로 진행한다.
- 다음: 보존 checkpoint·frozen source의 CPU 재추론 감사 계약과 실제 미노출 자료의 확보 가능성·평가 계약을 별도로 준비한다. 기존 Test/Validation을 새 독립 평가로 바꾸지 않는다. 실행·held-out 접근은 별도 범위 승인 후다.
- 병렬 가능: 독립 문서에서 문헌·서지·출처·재현성 점검과 미노출 자료 메타데이터 확인을 진행한다. 같은 원고·공통 평가 계약에 의존하는 최종 편집은 직렬이다.
- 이후: 확정된 주장·평가 범위로 Related Work·Appendix·제출 분량을 정리하고 열린 동일 문서를 내장 compiler로 검증한다. 현재33쪽 확장 초안을 제출 규정 충족본으로 표시하지 않는다.
- 검색은 독립 기여로 유지하겠다는 결정이 있을 때 위 조건부 경로로 재검토한다. GPU가 비었다는 이유로 자동 실행하거나 과거 자동화를 다시 만들지 않는다.

근거와 변경 범위는 [evidence.json](evidence.json), 작업 의존관계는 [plan.json](plan.json), 독립 과학 검토는 [claims_review.md](claims_review.md)에 기록한다. 계획과 상태 문서 외의 원고·코드·Runtime·실험 원본은 바꾸지 않았다.
