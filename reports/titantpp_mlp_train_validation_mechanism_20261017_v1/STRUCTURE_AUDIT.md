# 이력 MLP 구조와 입력 변환 공유 후보의 제한적 중복 확인

작성일: 2026-10-17. 상태: 코드 읽기와 구조 비교 완료, 후보 구현·학습 미실행.

**확인 결론**

아래에서 읽은 MLP 관련 소스에서는 **현재·직전 관측 쌍의 비선형 입력 변환 W 하나를 8개 분기가 공유하면서, 분기별 출력 B_j·기존 가용 문턱·고정 /8을 유지하는 구현을 찾지 못했다.** 확인 범위에 한정한 결론이다. 저장소 전체·과거 모든 브랜치·외부 문헌에 해당 방식이 없다는 뜻이나 독창성 주장은 아니다. 구현이 존재한다는 사실과 실제 학습·성능 검증이 완료됐다는 사실도 구분한다.

기존 구현의 `torch.cat([p.weight for p in self.input_projections])`는 **독립된 8개 입력 가중치를 계산용으로 이어 붙이는 연산**이다. 가중치 공유가 아니다.

**현재 기준선과 구조 대조 — 완료**

| 구조 | 현재·직전 관측 관계 처리 | 가용 분기 | 보정 가중치 수 | 확인 코드 |
|---|---|---|---:|---|
| 이력 MLP 폭4 | 분기마다 독립 128→4→64 GELU | 기존 문턱 유지 | 6,144 | 동결 CoreAblation 43~75행 |
| 이력 MLP 폭16 | 분기마다 독립 128→16→64 GELU | 기존 문턱 유지 | 24,576 | 동결 HistoryWidth 46~80행 |
| AllAvailable 폭4 | 독립 8분기가 같은 현재·직전 관측 쌍을 변환 | 직전 관측 존재 시 전부 활성화 | 6,144 | 동결 HistoryControls 30~58행 |
| CurrentOnly 용량 대조 | 현재 표현만 독립 64→6→64 GELU | 기존 문턱 유지 | 6,144 | 동결 HistoryControls 30~58행 |

여기서 현재·직전 표현은 첫 encoder block을 통과한 64차원 상태다. CurrentOnly도 현재 상태에 attention으로 반영된 과거 정보까지 없애는 모델은 아니다. 보정은 첫 block 뒤·둘째 block 앞에 들어가며 기존 메모리와 공통 출력부는 유지한다. 적용 위치는 동결 MultiLagDetail 230~238행에서 확인했다.

기존 가용 문턱은 관측 수가 1·2·4·8·16·32·64·128을 각각 초과할 때다. 각 분기는 해당 거리를 직접 읽지 않고 같은 직전 관측을 읽는다. padding은 건너뛰고 withheld 경계에서는 이 보정 경로의 가용 이력을 초기화한다. 모든 구조가 합을 고정된 8로 나눈다. bias 없는 입력 투영은 기본 초기화, 출력 투영은 0으로 초기화하며 별도 CPU 난수 문맥을 써 공통 모델 초기화 난수 상태를 보존한다.

AllAvailable 폭4는 직전 관측이 존재할 때 가중치를 이어 붙인 단일 128→32→64 GELU MLP와 대수적으로 같다. 폭16도 8개 분기가 모두 열렸을 때 단일 128→128→64 GELU MLP와 같다. 단순한 분기 병합을 새로운 표현 원리로 볼 수 없다.

**이미 확인한 중복과 후보의 차이 — 완료**

| 기존 구현 | 이미 바꾼 내용 | 제안한 W 공유와 구별되는 점 |
|---|---|---|
| ActiveBranchNorm | 고정 /8 대신 활성 분기 수로 나눔 | 입력 가중치는 여전히 8개 독립 W_j |
| AllAvailable | 분기 문턱을 모두 첫 직전 관측으로 맞춤 | 독립 W_j 유지, 활성화 조건 변경 |
| CurrentOnly | 명시적 직전 상태를 제거하고 폭6으로 용량 대조 | W 공유가 아니라 입력 접근 변경 |
| MLPCandidates cross_product | 현재·직전 투영의 곱으로 관계 구성 | 독립 W_j 상속, 결합식 변경 |
| MLPCandidates recent4_mean | 최근 최대4개 관측 평균을 문맥으로 사용 | 독립 W_j 상속, 관측 범위·요약 변경 |
| ArchitectureContribution | 폭8 이력 보정·현재상태 대조·후단 prototype의 유무/대체 | 이력/현재 보정의 입력·출력 투영은 분기별 독립 |

순차 MLP는 별도의 기존 제안으로 부모 작업에서 확인됐으므로 새롭게 발견한 방법으로 표현하지 않는다. 이번 한정 확인에서 해당 제안 문서나 결과는 추가 열람하지 않았다.

**단일 후속 가설과 구별 조건 — 다음 검토**

관측 쌍을 x_i=[h_i;h_prev], 기존 분기 가용성을 A_ij라 두면 현재 구조는 다음과 같다.

`r_i = (1/8) Σ_j A_ij B_j GELU(W_j x_i)`

검토할 공유 후보는 다음과 같다.

`r_i = (1/8) Σ_j A_ij B_j GELU(W x_i)`

이는 관계 특징을 이력 길이별 분기 사이에서 공유하는 구조 제약이다. 가용 문턱, 분기별 B_j, /8을 그대로 두면 AllAvailable·ActiveBranchNorm과 구별할 수 있다. 초기화, 입력·손실·head·메모리·선택 기준도 고정해야 한다. gate·정규화·새 관측 범위·깊이를 동시에 추가하면 이 가설 하나를 비교하기 어렵다.

**용량과 최적화의 차이는 피할 수 없는 해석 제한이다.** 폭16에서 W만 공유하면 보정 가중치는 `8×(128×16+16×64)=24,576`에서 `128×16+8×16×64=10,240`으로 감소한다. 이 단순 비교로 얻는 차이는 공유 제약과 용량 감소, 각 분기에서 공통 W로 모이는 gradient 및 최적화 변화가 합쳐진 결과다. 폭을 늘려 수를 맞추면 폭 변경이라는 추가 차이가 생긴다. dummy 파라미터로 숫자만 맞추면 유효 용량 대조가 되지 않는다. 같은 이력 길이에서는 출력 가중치의 가용 합을 하나의 readout으로 표현할 수 있지만, W 공유는 서로 다른 길이에서도 같은 비선형 특징을 사용하도록 제한한다는 차이가 남는다.

현재 저장된 출력 오류는 이 공유 제약이 실제로 유익할지 직접 보여주지 않는다. 폭16 Train 행과 기존 대조군 seed52·62 행이 부족하며, 이력 길이별 오류는 수량·개체 구성과 얽힐 수 있다. 따라서 가설을 기록하는 것과 새 모델을 채택·학습하는 결정을 분리한다. 이번 구조 확인으로 새 학습을 정당화하지 않는다.

**확인한 소스 원문과 SHA-256 — 완료**

동결 경로의 기준 디렉터리는 `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs`다. 현재 HistoryWidth에는 이후 폭8·12 확장이 있으므로 폭16 원본 근거는 아래 동결 파일을 사용했다.

| 파일·관련 시작 행 | 읽은 원문 SHA-256 |
|---|---|
| [동결 CoreAblation:43](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs/CountAwareTitanCoreAblation.py:43) | `b34b61aadc7a2c91e000493caf2497fc89549a499082ca77ebbc3ed7cab2b120` |
| [동결 HistoryWidth:46](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs/CountAwareTitanHistoryWidth.py:46) | `75f43447b39aecfc1113bf7213d88272b8821beec6655612f8573ff5fc113174` |
| [동결 HistoryControls:30](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs/CountAwareTitanHistoryControls.py:30) | `60ce1ea2675e9f516f3d09663d893a39e4825ede5a2ab5a703779dcb158e640b` |
| [동결 ActiveBranchNorm:33](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs/CountAwareTitanActiveBranchNorm.py:33) | `14ebe590a4190276fd789bb2608520d14e031c79e98afe08511fef4aeb13d7e9` |
| [동결 MultiLagDetail:77](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_width16_replication_20261003_v1/frozen_source/models/TPPs/CountAwareTitanMultiLagDetail.py:77) | `4bcbecdadf921df21a4e7fb379962eb4ecec2cfeb2f8ff1edccbb6b418807eae` |
| [현재 MLPCandidates:35](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanMLPCandidates.py:35) | `eb88a42aa092bb173ccbf0139a1a8469542e5a4cb5f25fd23b098d60f2ca841a` |
| [현재 ArchitectureContribution:108](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanArchitectureContribution.py:108) | `0c586378e11d6f1aacd128786cfaa52e33310f043495056864357eebe43a193b` |

위 SHA는 이번 읽기 시점 파일 바이트의 해시다. 현재 코드의 두 항목은 구현 중복 확인에만 사용했으며, 과거 실행 source와 동일하다는 추가 주장을 하지 않는다. 원격 접근·추론·학습·checkpoint 로딩이나 변경은 수행하지 않았다.

**범위 기록과 남은 작업**

앞선 구조 확인에서 파일명에 CNN이 없는 `CountAwareTitanPostAttentionFFN.py`를 열었다가 CNN 기반임을 확인했다. 해당 파일 내용은 이 보고서의 중복 판정과 후속 가설 근거에서 제외했고 추가 열람하지 않았다. 이번 구조 확인에서 Test·혼합 결과·원고·CNN 결과는 열람하지 않았다.

다음 작업은 순수 Train·Validation 분석과 위 해석 제한을 합쳐 가설의 검증 가능성을 판단하는 것이다. 새로운 모델·학습 계약은 아직 확정하지 않았다. 이 문서 외 다른 파일과 기존 원본은 수정하지 않았다.
