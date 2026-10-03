Here is the result of "fetch" for the Page with URL https://app.notion.com/p/3ebbbe40561381059aa0c5b73d6d47e3 as of 2026-09-30T01:45:23.802Z:
<page url="https://app.notion.com/p/3ebbbe40561381059aa0c5b73d6d47e3">
<ancestor-path>
<parent-page url="https://app.notion.com/p/2e9bbe40561380678b4ff6c7be991d33" title="6. From Initial Experiments to Extensive/Extended Studies"/>
<ancestor-2-data-source url="collection://2e4bbe40-5613-8184-bfa2-000bc4313c64" name="프로젝트"/>
<ancestor-3-database url="https://app.notion.com/p/2e4bbe4056138194996afa28247dd519" title=""/>
<ancestor-4-page url="https://app.notion.com/p/2e4bbe40561380898427d3ac376df1f8" title="졸업 논문"/>
</ancestor-path>
<properties>
{"title":"TitanTPP 논문 설계안 — 사건 수량 표현과 온라인 메모리 단순화 (2026-09-30)"}
</properties>
<iconMetadata>null</iconMetadata>
<content>
작성일: 2026-09-30 · 대상: paper_research · 평가 범위: validation · 목표: PAKDD 투고를 위한 방법·실험·주장 설계
<callout icon="📌">
	**논문 작성을 진행한다.** 중심은 사건의 시간·수량 정보를 이용하는 과제 특화 표현 설계와 온라인 메모리 갱신을 생략한 구조의 정확도·비용 절충이다. 현재 수량 예측 개선은 확인됐으며, 원본 메커니즘 대비 효율·최종 독립 평가는 별도 입증한다.
	5080/5090 승인 학습·validation 재평가는 모두 종료됐다. TitanTPP 스케줄러는 사용자 요청으로 삭제됐다. 이 문서는 새 GPU 실험·스케줄러·held-out 평가의 실행 승인이 아니다.
</callout>
이전 논의: <mention-page url="https://app.notion.com/p/3eabbe405613818dba4dce3c935c8432"/>. 이 설계안의 완료 상태와 Gate 판단이 이전 페이지의 진행 중 서술보다 우선한다.
## 1. 현재 기준선과 논문의 질문 — 완료 / 주장 범위 정리
**연구 질문:** 사건 발생 시점과 수량이 관측되는 데이터에서, 온라인 neural memory를 갱신하지 않는 비교적 단순한 표현 구조로 다음 사건 수량의 정확도와 계산 비용 사이에 유리한 절충을 만들 수 있는가?
- 산업적 동기: 사건 유형뿐 아니라 발생 간격과 발생량이 의사결정에 필요하다. 범주형 사건 유형과 수치형 사건 크기를 구분한다. 수량은 정수 count일 수도 있으므로 모든 수량을 본질적으로 연속값이라고 설명하지 않는다.
- 현재 과제: 관측된 시간·수량 이력에서 다음 사건 수량을 예측하고 시간 분포를 함께 학습·평가한다. 시간 NLL은 예측 대상 자체가 아니라 손실·평가 지표다. 현재 관측 시간 단위와 수량 회귀 정의를 그대로 적는다.
- 주장 범위: 다음 사건 예측의 개선이다. 달력상의 임의 미래 구간 총수요나 장기 시계열 전체를 예측하는 능력까지 검증했다고 확대하지 않는다.
- 완료: core 54조건/108 selected·last 역할, 외부 TPP 추가 36조건, Gate 별도 seed42 6조건의 결과를 확보했다. core와 외부의 고유 학습 조건은 90개이며 중복 B·Full 18개는 한 번만 센다.
- 완료: 1차 분석 계산 검증. 현재 보고서의 verification에는 5090 checkpoint binary와 최종 source 원본 회수 감사가 미완료로 남아 있다. 소형 기록 감사 완료와 구분한다.
- 완료: 2026-09-30 사용자 “오케이 그러면 MLP로 진행하자”에 따라 기존 이력 MLP 구성(`titantpp_history_mlp`)을 논문의 단일 제안 모델 TitanTPP로 확정했다. Taxi·Intermittent·Instacart 모두 같은 구성을 사용한다.
- 완료: 확정 구조의 방법 수식·그림·설계 근거와 기존 MAC 비용 감사를 작성했다. 다음은 별도 승인할 최소 효율 측정과 원고 통합·독립 평가 준비다. 기존 validation 검토 후 모델을 선택했다는 사실을 유지한다.
근거: 로컬 \[L1\], \[L2\]. 전체 파일 경로는 마지막 출처 목록을 따른다.
## 2. Contribution을 무엇으로 고정할 것인가
### C1. 사건 수량 예측을 위한 TitanTPP 표현 구조 — 구현 확인 / 독창성 비교 필요
> 사건 수량 예측을 위한 TitanTPP 표현 구조를 제안한다. 사건 간격과 과거 수량을 인과적으로 통합하고, 학습 가능한 메모리와 저차원 MLP 이력 보완을 통해 다음 사건 수량 예측에 적합한 표현을 학습한다. 이 구조는 추론 중 온라인 신경 메모리 갱신을 요구하지 않는다.
- 입력 설계: 사건 간격과 과거 수량의 log1p 특징을 사용하고, 미래 사건 정보가 들어오지 않도록 인과적 마스킹을 적용한다.
- 표현 설계: persistent memory를 갖는 인과적 encoder와 정적 prototype 검색을 유지하고, 두 encoder 단계 사이에서 현재·직전 표현을 결합하는 저차원 MLP의 잔차 출력으로 이력을 보완한다. 추가 조건부 Gate와 학습 가능한 상수 계수는 사용하지 않는다.
- 초기화 설계: 보완 출력의 영 초기화로 공통 초기 예측을 보존하며, 사건 수량 예측에 필요한 보완 표현을 학습한다.
**서술 원칙:** C1에서는 TitanTPP라는 하나의 제안 방법과 설계 이유를 설명한다. 구성 대안의 비교와 선택 근거는 6.1절의 구성 요소 검증으로 분리한다.
**현재 기준선과 다음 작업:** 채택 구조는 기존 이력 MLP로 고정했다. 해당 구성의 입력·계산식·보완 위치·메모리 상태·복잡도를 방법 절에 기술하고, 원본 Titans 및 일반 causal Transformer/THP와의 차이를 설명한다. 보편적인 구성요소를 결합했다는 사실만으로 독창성을 확정하지 않으며, 설계 이유와 통제된 검증 결과를 연결한다. \[L3\]
### C2. 온라인 메모리 갱신을 생략한 정확도·비용 절충 — 가설 / 효율 일부 근거
원본 Titans는 추론 중에도 관측 입력으로 neural memory를 갱신하는 메커니즘을 포함한다. 채택한 TitanTPP는 그 경로를 사용하지 않지만 persistent memory와 정적 검색은 유지한다. 따라서 표현은 **“모든 Memory 제거”가 아니라 “온라인 neural memory 갱신을 사용하지 않는 재설계”**다. [Titans 원논문](https://arxiv.org/abs/2501.00663)
- 과거 Titans-MAC 이벤트 적응 실험은 이미 있다. 해당 기록을 폐기하거나 처음부터 없는 것으로 취급하지 않는다.
- 과거 조건에서 MAC의 높은 epoch 시간은 확인됐다. 다만 채택한 TitanTPP(기존 이력 MLP) 대비 정확한 가속 배수는 아직 같은 조건에서 확정하지 않았다.
- 속도, GPU 메모리, 파라미터 수는 각각 다른 지표다. 한 지표의 개선을 나머지 개선으로 바꾸어 말하지 않는다.
- B 대 MAC도 segment attention·메모리·파라미터가 함께 달라진다. backbone 전체 비교이며, 온라인 갱신 하나만 제거한 인과적 제거 실험은 아니다.
**완료 조건:** 같은 부하에서 학습 시간과 필요한 메모리·추론 지표를 측정하고 정확도와 함께 제시한다. 속도만 유리하면 “학습 처리 속도 개선”, 모든 자원에서 이득을 확인해야 그 범위의 경량화를 주장한다. \[L4–L6\]
### C3. 통제된 비교에서 수량 예측 경쟁력과 한계를 검증 — validation 근거 확보
기존 TPP 구현의 입력·head·loss·학습·선택 규칙을 맞춘 비교에서 두 데이터의 수량 개선을 확인한다. 구조별 반례, Instacart의 작은 차이, 시간 예측의 상충관계를 숨기지 않는다.
- 공통 head 비교는 표현 효과를 검토하는 근거다. 각 원논문의 native 모델 및 최적 튜닝 결과를 모두 이겼다는 의미는 아니다.
- 대표 구조를 고정한 TitanTPP의 외부 비교 결과와, 내부 설계의 역할을 확인하는 구성 요소 검증을 구분한다.
- 현재 조건부 MLP Gate는 두 데이터에서 plain MLP의 MAE·RMSE를 개선하지 못했다. Gate를 핵심 성능 기여에서 제외하고 제한된 탐색 결과로 제시한다.
- 독립 평가 전에는 validation에서 반복되는 개선과 일반화 성능을 구분한다. \[L1\]
## 3. 논문용 주장 초안과 금지할 과장
**현재 근거로 작성할 문장**
> 본 연구는 사건 발생 시점과 수량을 다루는 예측 문제를 위해, 온라인 신경 메모리 갱신을 사용하지 않는 Titans-inspired 구조인 TitanTPP를 구성한다. 학습 가능한 메모리와 관측 사건 이력의 보완 표현을 결합하고, 공통 예측 head와 통제된 학습 조건에서 구조의 효과를 검증한다. Taxi와 Intermittent에서는 기존 TPP 비교 구현 대비 수량 예측 개선을 확인하며, Instacart의 제한적 이득과 시간 예측의 상충관계를 분석한다.
**효율 검증 후에만 추가할 문장**
> 동일한 데이터·장치·측정 조건에서 Titans-MAC 이벤트 적응 모델 대비 학습 비용을 줄이면서 수량 예측 성능의 경쟁력을 유지한다.
실측 결과에 맞춰 학습 비용의 종류와 개선 범위를 명시한다. 결과가 반대이면 이 문장은 채택하지 않는다.
<table header-row="true">
<tr>
<td>피할 주장</td>
<td>사용할 표현</td>
</tr>
<tr>
<td>TPP는 원래 이산 mark만 다룬다</td>
<td>범주형 사건 유형을 주로 다루는 비교군을 수치형 사건 크기 예측에 맞춰 통제 비교한다.</td>
</tr>
<tr>
<td>연속값을 넣어 최초의 새로운 패러다임을 만들었다</td>
<td>기존 continuous-mark·간헐 수요 연구와 구별되는 표현 설계와 비용 절충을 검증한다.</td>
</tr>
<tr>
<td>Memory를 모두 제거했다</td>
<td>온라인 neural memory 갱신을 생략했다. persistent memory와 정적 검색의 유지 여부는 모델별로 명시한다.</td>
</tr>
<tr>
<td>MLP가 가벼우므로 메모리도 적고 항상 빠르다</td>
<td>파라미터·학습 시간·추론 시간·GPU 메모리를 각각 측정한다.</td>
</tr>
<tr>
<td>세 데이터에서 모두 우월하다</td>
<td>Taxi·Intermittent의 수량 개선과 Instacart의 우위 미확보를 함께 보고한다.</td>
</tr>
</table>
연속값 mark와 이산값을 함께 다루는 [FlexTPP](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html), 발생 간격·수요 크기와 TPP를 연결한 [Deep Renewal Processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)가 이미 있다. 이들은 문제 설정의 관련 연구이자 차별성을 설명해야 할 가까운 비교 대상이다.
## 4. 현재 결과를 본문에 어떻게 배치할 것인가 — 완료 근거
논문의 주 비교 표는 세 데이터 모두 기존 `titantpp_history_mlp` 결과를 TitanTPP로 표기한다. 아래 구성별 수치는 선택 근거와 ablation을 위한 보존 자료다. Full 결과를 TitanTPP의 결과로 대신 사용하거나 구성 간 수치를 합치지 않는다.
모든 표는 같은 validation raw-RMSE 선택 checkpoint의 MAE·RMSE를 사용한다. 3seed 평균의 상대 감소율이며, 지표마다 checkpoint를 다시 선택하지 않았다.
<table header-row="true">
<tr>
<td>데이터</td>
<td>모델</td>
<td>최선 외부 비교군 대비 MAE 감소</td>
<td>RMSE 감소</td>
<td>해석</td>
</tr>
<tr>
<td>Taxi</td>
<td>Full</td>
<td>11.05%</td>
<td>16.84%</td>
<td>기존 외부 네 모델 대비 같은 seed 3/3 수량 개선</td>
</tr>
<tr>
<td>Taxi</td>
<td>History MLP</td>
<td>10.13%</td>
<td>15.85%</td>
<td>고정 MLP 구성 자체의 경쟁력</td>
</tr>
<tr>
<td>Intermittent</td>
<td>Full</td>
<td>16.25%</td>
<td>28.86%</td>
<td>외부 비교 개선과 MLP 대비 열세를 분리</td>
</tr>
<tr>
<td>Intermittent</td>
<td>History MLP</td>
<td>21.72%</td>
<td>34.39%</td>
<td>core 수량 MAE·RMSE 평균 최저</td>
</tr>
<tr>
<td>Instacart</td>
<td>Full / MLP</td>
<td>우위 미확보</td>
<td>우위 미확보</td>
<td>작은 차이를 통계적 동등성으로 해석하지 않음</td>
</tr>
</table>
두 데이터의 최선 외부 수량 비교군은 RMTPP다. Instacart는 지표별 외부 최선이 다르다. 3seed는 표본표준편차와 함께 보고하며 유의성 또는 독립 평가의 대체물로 쓰지 않는다. \[L1\]
**반례와 실패 이력도 논문의 결과다.**
- Taxi에서는 Full 평균 RMSE가 MLP보다 1.17% 낮지만, Intermittent에서는 MLP가 Full보다 7.77% 낮다.
- Full 대 MLP의 기존 복합 보호기준은 모든 데이터에서 0/3 통과다. 실행 성공과 성능 기준 통과를 분리한다.
- MLP 조건부 Gate는 두 데이터 seed42에서 수량 개선이 없었다. Taxi 상수만 MAE 1.42%·RMSE 2.76% 개선했으나 시간 NLL이 0.842735에서 1.304540으로 악화했다.
- Gate 6완료조건은 별도 seed42 탐색이다. Full Taxi C/D 미시작 보류와 실제 실패·복구 이력을 유지한다.
- 데이터 특성은 설명 가설로 쓴다. 예를 들어 짧은 이력이 Instacart의 이득을 제한할 수 있지만 그 인과를 입증한 것은 아니다.
## 5. Epoch당 시간 비교는 어떻게 사용할 것인가
**답: 학습 속도의 주요 지표로 사용할 수 있다. 단, 같은 양의 일을 같은 환경에서 처리한 시간이어야 한다.**
### 5.1 이미 찾은 Titans-MAC 기록 — 재사용 가능한 과거 근거
과거 seed42의 B0 대 B1/Titans-MAC 기록이다. 현재 core의 B·Full·MLP 결과표와 합치지 않는다.
<table header-row="true">
<tr>
<td>데이터</td>
<td>과거 B0 초/완료 epoch</td>
<td>과거 MAC 초/완료 epoch</td>
<td>MAC/B0 시간비</td>
<td>완료 epoch B0/MAC</td>
</tr>
<tr>
<td>Taxi</td>
<td>7.6847</td>
<td>52.7067</td>
<td>6.86배</td>
<td>42 / 46</td>
</tr>
<tr>
<td>Intermittent</td>
<td>107.0674</td>
<td>583.2923</td>
<td>5.45배</td>
<td>240 / 229</td>
</tr>
</table>
산식은 `summary.elapsed_seconds / completed_epochs`다. 과거 source revision의 timer 확인을 완료했다. epoch loop의 train·validation·기록·저장과 종료 후 best-state validation·checkpoint 저장을 포함하는 상각 실행 시간이다. 순수 train-only 시간이나 warm-up 제외 중앙값이 아니다. 별도 짧은 forward 표도 device 전송·head·loss를 포함한 target_outputs 경로로 확인됐다. 상세 감사는 하위 방법·효율 문서와 로컬 `reports/titantpp_method_efficiency_20260930_v1/efficiency_audit.md`에 있다.
과거 짧은 forward 프로파일에는 별도 비용 기록도 있다.
- Taxi: B0 5.284ms, MAC 35.019ms의 steady forward 중앙값. peak allocated는 B0 364.89MiB, MAC 74.30MiB.
- Intermittent: B0 5.276ms, MAC 33.737ms. peak allocated는 B0 364.89MiB, MAC 145.05MiB.
- 각 모델 5 batch 중 cold 1회를 구분하고 나머지 4회로 계산한 작은 프로파일이다. 전체 학습 peak나 현재 모델의 메모리 사용량으로 일반화하지 않는다.
- **이 과거 프로파일에서는 MAC가 느리지만 peak allocated는 더 낮았다.** 메모리 연산을 생략했다고 GPU 메모리까지 절약된다고 결론 내릴 수 없다. \[L5\]
### 5.2 현재 모델의 가속 배수로 바로 사용하지 않는 이유
<table header-row="true">
<tr>
<td>항목</td>
<td>과거 MAC</td>
<td>현재 비교</td>
<td>영향</td>
</tr>
<tr>
<td>시간 head·loss</td>
<td>legacy clamped RMTPP</td>
<td>lognormal duration과 관측 정수 구간 확률질량</td>
<td>연산과 encoder 학습 신호가 다름</td>
</tr>
<tr>
<td>선택·조기 종료</td>
<td>validation joint objective</td>
<td>validation raw 수량 RMSE</td>
<td>성능 선택과 완료 epoch 수, 전체 평균 시간의 구성이 다름</td>
</tr>
<tr>
<td>MAC 내부 gradient</td>
<td>seed42 unbounded, 안정화 seed52/62 clip1</td>
<td>현재 프로토콜 MAC 결과는 없음</td>
<td>세 seed를 하나의 설정으로 합칠 수 없음</td>
</tr>
<tr>
<td>장치·실행 부하</td>
<td>당시 source·Runtime·실행 기록</td>
<td>5080/5090, 병렬·복구 이력 포함</td>
<td>같은 GPU·단독 점유·정밀도인지 대조해야 함</td>
</tr>
</table>
선택 규칙이 다르다는 이유만으로 모든 step 시간 비교가 무효인 것은 아니다. **부하와 timer 범위가 맞는 시간은 재사용할 수 있다.** 다만 현재 자료의 `총시간/완료epoch`를 순수 backbone 속도비로 바꾸거나, 다른 학습 손실로 얻은 정확도를 한 통제 표에 섞어서는 안 된다.
### 5.3 주장별 최소 측정 범위 — 설계안, 미실행
- **“학습이 빠르다”**: 같은 GPU·Runtime·정밀도·batch·실제 입력 길이·train 전체 target·optimizer step 수를 맞춘 초/epoch를 중심으로 제시한다. train-only와 train+validation의 범위를 구분하고, compile/cold 첫 회는 별도 기록한다. 이후 반복 측정의 중앙값·변동 범위를 제시한다.
- **“추론이 빠르다”**: 별도 forward 지연·처리량이 필요하다. 온라인 neural memory를 쓰는 MAC는 추론 중 실제 관측 메모리 갱신을 포함해야 한다. 갱신을 꺼 놓고 원본 MAC 추론이라고 하지 않는다.
- **“GPU 메모리가 적다”**: 같은 조건의 peak allocated와 peak reserved를 구분한다. 파라미터 수도 별도로 보고한다.
- **“적은 비용으로 같은 정확도에 도달한다”**: epoch 시간 외에 도달 성능·수렴까지의 시간·실패 조건을 함께 본다.
- epochs/s만으로 FLOPs 감소나 추론 속도까지 주장하지 않는다. CPU loader·전송·동기화 병목과 checkpoint/ACK 대기·재부팅·동시 실행을 섞지 않는다.
- **속도만 확인하기 위해 9조건의 300epoch 학습을 자동으로 요구하지 않는다.** 기존 기록의 비교 가능성을 먼저 확인하고, 부족한 항목만 짧은 동일 부하 측정으로 보완하는 별도 계약을 만든다. 속도용 짧은 측정 결과를 새 성능 학습 결과로 취급하지 않는다.
- 현재 head 아래에서 MAC의 최종 정확도까지 주장하려면 별도의 공정한 학습 비교가 필요하다. 기존 9조건 실행 준비 사본은 있으나 실행 승인은 없고 새 학습도 시작하지 않았다. 이전 480 GPU시간 제안을 자동 채택하지 않는다. \[L6\]
## 6. 비교군은 연구 질문별로 제한한다 — 기존 결과 재사용 / 추가는 제안
<table header-row="true">
<tr>
<td>역할</td>
<td>모델·비교</td>
<td>상태</td>
<td>검증할 내용</td>
</tr>
<tr>
<td>기존 주 비교</td>
<td>RMTPP · THP · NHP · SAHP</td>
<td>36조건 완료</td>
<td>동일 시간·수량 입력과 공통 head/loss에서 표현의 경쟁력</td>
</tr>
<tr>
<td>내부 구조</td>
<td>B · Full · History MLP · 수준만 · 변화만 · 정적 검색 제거</td>
<td>54조건 완료</td>
<td>이력 보완·분리·검색의 역할과 반례</td>
</tr>
<tr>
<td>원본 메커니즘</td>
<td>Titans-MAC 이벤트 적응 모델</td>
<td>과거 기록 있음 / 현재 규칙 장기 결과 없음</td>
<td>온라인 메모리 포함 구조와 정확도·비용 비교</td>
</tr>
<tr>
<td>최근 TPP backbone</td>
<td>S2P2 우선, AttNHP 보완 후보</td>
<td>적합성 검토 및 별도 승인 필요</td>
<td>기존 네 모델에 없는 최근 state-space·연속시간 attention 비교</td>
</tr>
<tr>
<td>연속 mark 관련 비교</td>
<td>FlexTPP</td>
<td>관련 연구 및 구조 보존 비교 후보</td>
<td>연속값 출력 자체를 독창성으로 주장하지 않고 구체적 차이를 확인</td>
</tr>
<tr>
<td>간헐 수요 관련 비교</td>
<td>Deep Renewal 계열</td>
<td>과제 정합성 검토 필요</td>
<td>발생 간격·수량의 모델링과 다음 사건 평가를 맞출 수 있는지 확인</td>
</tr>
<tr>
<td>저비용 기준선</td>
<td>직전 수량·관측 이력 평균</td>
<td>현재 target에 맞춘 계산 계약 필요</td>
<td>복잡한 표현의 실질적 추가 이득</td>
</tr>
</table>
주 비교는 **같은 관측 정보·head·loss를 사용하는 TPP backbone 비교**로 유지한다. 연속시간 질의나 native decoder가 핵심인 모델을 head만 제거해 원본 전체 방법이라고 부르지 않는다. FlexTPP 등의 구조 보존 비교는 별도 패널이며 수정한 부분과 튜닝 예산을 공개한다. 모든 후보를 실행 목록에 자동 추가하지 않는다.
PatchTST·PatchMixer는 등간격 시계열을 전제로 한 구성과 현재 사건 목록의 시간축이 달라 주 비교에서 보류한다. 변환의 타당성을 별도 입증하지 않고 강제로 넣지 않는다.
근거: [S2P2](https://arxiv.org/abs/2412.19634), [AttNHP](https://arxiv.org/abs/2201.00044), [FlexTPP](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html), [Deep Renewal Processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764). 후보 이름은 실행 완료나 채택 확정을 의미하지 않는다.
### 6.1 구성 요소 검증의 역할 — 기존 실험 완료 / 이력 MLP 채택 확정
논문에서 TitanTPP는 기존 History MLP 구성(`titantpp_history_mlp`)을 가리킨다. Full·수준만·변화만·정적 검색 제거와 B는 설계 비교군으로 배치하며, 원래 구성명과 실험 ID는 ablation과 재현 자료에서 유지한다. 기존 실험 ID `titantpp`는 이력 보완 없는 B이므로 새 논문 명칭과 혼동하지 않는다.
- Full의 수준·변화 분리와 현재/직전 표현을 결합하는 History MLP를 비교했다. MLP는 전체 backbone을 대체하는 독립 MLP가 아니라 내부 이력 보완 모듈이다.
- 보완 위치는 두 encoder 단계 사이이며, 저차원 투영과 출력의 영 초기화로 공통 초기 예측을 보존한다.
- 현재 local 변형의 여덟 분기는 인접 이력을 읽으며 이력 길이에 따른 활성 조건을 가진다. 서로 다른 여덟 시간 지연을 읽는 다중 시간척도 구조라고 쓰지 않는다.
- Full·MLP의 추가 파라미터는 각각 6,144개로 같다. 단순 MLP라는 이름만으로 파라미터가 감소했다고 주장하지 않는다.
- B도 사건 이력을 인코딩한다. B의 의미는 추가 이력 보완 모듈이 없다는 것이며, 이력을 전혀 사용하지 않는 모델이 아니다.
- 주 비교 표에는 Taxi·Intermittent·Instacart의 기존 이력 MLP seed42·52·62 결과를 사용한다. Full은 수준·변화 분리를 적용한 대안 구성으로 둔다. 현재 정적 검색 제거 실험은 Full 기반이므로 MLP에서 그 모듈만 제거한 결과로 해석하지 않는다.
**현재 결정 상태 — 완료:** 사용자 승인으로 기존 plain History MLP 구성을 최종 TitanTPP로 채택했다. Gate·학습 가능한 상수 계수·정적 검색 제거를 추가하지 않는다. Intermittent의 MAE·RMSE 개선과 Taxi의 외부 비교 경쟁력을 선택 근거로 삼고, Taxi에서 Full보다 높은 평균 오차와 Instacart의 우위 미확보·시간 NLL 상충관계도 보존한다. 비용 우월성은 이 선택만으로 입증되지 않는다. 기존 학습 결과를 재사용하며 이번 결정으로 새 학습이나 held-out 평가를 시작하지 않는다. \[L3\]
## 7. 논문 목차와 필요한 표·그림 — 작성 설계
1. **Introduction:** 산업에서 사건 발생량이 중요한 이유, 시간·수량 예측 문제, 온라인 메모리의 비용과 필요한 표현 사이의 질문을 연결한다.
2. **Related Work:** 범주형·수치형 mark TPP, 간헐 수요의 renewal 관점, Titans와 온라인 신경 메모리, 경량 사건 표현을 구분한다.
3. **Problem Formulation:** 관측 이력, 시간 단위·관측 likelihood, 수량 회귀, 인과성·target 마스킹, next-event 평가 범위를 정의한다.
4. **Method:** 채택한 TitanTPP 구조를 시간·수량 입력 → persistent-memory causal encoder → 현재·직전 표현의 저차원 MLP 잔차 보완 → 다음 encoder → 정적 검색 → 공통 시간·수량 head의 흐름으로 제시한다. 각 설계 이유와 온라인 갱신이 없는 부분을 설명하고, 구성 대안의 비교는 Results의 Ablation Study로 옮긴다.
5. **Experimental Setup:** 데이터·split·3seed·head·학습·선택·Runtime·비용 측정 정의를 공개한다.
6. **Results:** 수량 MAE/RMSE 주 표, 시간 NLL 및 구간별 반례, 내부 구조 비교, Titans-MAC 비용·정확도 비교를 증거 수준별로 배치한다.
7. **Limitations and Conclusion:** Instacart의 한계, 시간 지표 악화, Gate 음성 결과, validation 탐색과 독립 평가의 구분을 명시한다.
**주요 표·그림**
- 표 1: 원본 Titans 메커니즘 / MAC 이벤트 적응 / 현재 TitanTPP의 구조 대응.
- 표 2: 단일 고정 구조 TitanTPP와 외부 TPP의 공통 head 비교. 데이터별 MAE·RMSE·시간 NLL, 3seed 평균 ± 표본표준편차.
- 표 3: Full·MLP 등 내부 구성 요소 검증과 Gate 별도 탐색. 제안 모델의 실험 구성명을 명시하고 같은 RMSE-selected checkpoint를 사용한다.
- 표 4: 초/epoch, 추론 지연, 파라미터, peak GPU 메모리. 과거 프로토콜과 현행 측정을 구획한다.
- 그림 1: 구조와 관측 정보의 흐름.
- 그림 2: 데이터별 정확도·시간 또는 정확도·메모리 절충. Instacart와 불리한 조건을 누락하지 않는다.
- 부록: 실패·복구 이력, selected/last, source·계약·데이터 SHA, 타이밍 정의와 구간별 오차.
## 8. 남은 작업 순서 — 현재 세션의 문서 범위와 후속 실행을 구분
**기존 결과와 설계안을 연결한다 — 완료**
- 대상: paper_research의 1차 분석 및 이 Notion 페이지.
- 기존 결과·과거 MAC 비용·현재 코드 설계를 연결했다. C1은 하나의 제안 방법 TitanTPP로 정리하고, Full·MLP의 비교 설명은 구성 요소 검증 절로 옮겼다. 기존 학습을 다시 실행 대상으로 올리지 않는다.
**논문 대표 모델을 고정한다 — 완료**
- 대상: paper_research 논문 설계안과 `representative_model_decision.json`.
- 기존 plain History MLP를 TitanTPP로 채택했고, 세 데이터의 3seed 결과 및 원래 실험 ID와 대응시켰다. 기존 계약·소스·결과는 수정하지 않았다.
**채택 구조의 방법 절과 설계 근거를 작성한다 — 완료**
- 대상: paper_research 방법 절·기호 정의표·구조 그림·코드 대응표.
- 동결 source에 맞춘 9개 수식과 MLP 마스킹·잔차·영 초기화를 작성하고 작은 CPU 텐서로 수식 동치를 확인했다. 설계 목적과 입증된 효과를 구분했다.
- 상세 문서: <mention-page url="https://app.notion.com/p/3ebbbe40561381e2be20fbeb63fd7329">TitanTPP 방법 절·설계 근거·MAC 효율 감사 — MLP 확정 구조 (2026-09-30)</mention-page>. 로컬 `reports/titantpp_method_efficiency_20260930_v1/`에 원본과 SVG·PDF를 보존했다.
- 남은 원고 작업: 기존 주 결과표·ablation 표를 최종 투고 형식에 통합한다.
**기존 MAC 비용을 감사하고 최소 효율 측정을 설계한다 — 감사·범위 제안 완료 / GPU 실행 승인 필요**
- 과거 timer source까지 대조했다. 5.45/6.86배는 과거 MAC/B0의 상각 시간비이며 현재 MLP 가속 배수가 아니다. 현재 MLP 기록, 과거 target_outputs profile 및 메모리 반례를 구분했다.
- 권장 범위는 개인5080의 MLP/MAC × Taxi/Intermittent, 같은 train batch를 반복하는 총30분 상한 profile이다. 새 측정은 아직 실행하지 않았다.
- 실제 전체 초/epoch는 profile 이후 전량 반복의 소요시간을 산정해 별도 승인한다. MAC9조건 장기학습은 속도 측정의 기본값이 아니다.
**주장에 필요한 추가 비교와 독립 평가를 수행한다 — 승인 필요**
- 대상: 확정될 GPU 비교 계약 및 잠긴 held-out 평가.
- 대표 모델은 기존 이력 MLP로 고정했다. 독립 평가 계약과 비교 규칙을 확정한 뒤 관련성이 높은 추가 TPP만 선정하며, 기존 결과를 무조건 재학습하지 않는다.
- 속도 측정과 새 과학적 fit, held-out 평가를 각각 구분하여 승인한다. 이 문서 작성으로 실행하지 않는다.
**최종 무결성과 원고를 확정한다 — 다음 작업**
- 대상: paper_research 최종 감사·논문·Notion 문서.
- 5090 binary/source 감사 잔여분과 결과·그림·인용·한계 표현을 확인한다. 기존 5080/Gate 감사는 재사용한다.
- 학습 스케줄러를 재생성하지 않는다. 커밋·Push·외부 공개·실제 학회 제출은 이번 요청에 포함되지 않는다.
## 9. 근거 파일과 원문
로컬 파일은 이 Mac의 `/Users/igwanhyeong/PycharmProjects/paper_research/`를 기준으로 한다. Notion에서 로컬 파일을 직접 열 수 있다는 의미가 아니며, 재현을 위한 경로 표기다.
- \[L1\] `reports/titantpp_first_analysis_20260930_v1/report.md`, `analysis.json`, `verification.json`: 완료 결과·기여 범위·계산 검증. validation-only.
- \[L2\] `search_artifacts/titantpp_core_ablation_20260928_v1/aggregation/20260929T232132255352Z/ledger.json`: core 54조건의 선택·재평가 소형 기록.
- \[L3\] `models/TPPs/CountAwareTPP.py`, `CountAwareTitanMultiLagDetail.py`, `CountAwareTitanCoreAblation.py`, `models/Titan/backbone.py`, `models/Titan/common/memory.py`: 실제 표현·메모리·보완 경로. 현재 코드와 동결 source revision을 구분한다.
- \[L4\] `reports/titans_mac_reuse_audit_20260928_v1/report.md`, `historical_seed42.csv`, `mac_current_compatibility.csv`: 과거 MAC의 재사용 범위. 해당 보고서의 core 진행 상태는 당시 기록이며 본 문서의 완료 상태가 우선한다.
- \[L5\] `paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/historical_cost.csv`, `runtime_cost.json`; 계산 함수 `paper/scripts/analyze_count_aware_titantpp_mac.py:historical_cost_row`: 평균 epoch 및 별도 forward 프로파일.
- \[L6\] `reports/titans_mac_execution_preparation_20260928_v1/README.md`, `search_artifacts/titans_mac_observed_time_20260928_v1/README.md`: 현행 MAC 9조건은 로컬 준비·동결까지이며 GPU 미실행·별도 승인 필요.
- \[L7\] `reports/titantpp_mlp_gate_execution_20260929_v1/final_report.md`, `comparison.json`: Gate 완료 결과.
- 외부 원문: [Titans](https://arxiv.org/abs/2501.00663), [FlexTPP](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html), [Deep Renewal Processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764), [S2P2](https://arxiv.org/abs/2412.19634), [AttNHP](https://arxiv.org/abs/2201.00044).
**판단:** 논문 작성과 PAKDD 투고 준비를 진행할 근거가 있다. 실제 채택이나 현재 상태의 제출 완성을 보장하는 판단은 아니다. 기여는 “연속값의 최초 도입”이나 “Gate 우월성” 대신 구체적인 사건 표현 설계와 검증된 정확도·비용 절충에 둔다.
<page url="https://app.notion.com/p/3ebbbe40561381e2be20fbeb63fd7329">TitanTPP 방법 절·설계 근거·MAC 효율 감사 — MLP 확정 구조 (2026-09-30)</page>
<empty-block/>
</content>
</page>
