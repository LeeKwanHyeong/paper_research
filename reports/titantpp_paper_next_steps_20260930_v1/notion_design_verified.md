Here is the result of "fetch" for the Page with URL https://app.notion.com/p/3ebbbe40561381059aa0c5b73d6d47e3 as of 2026-09-30T02:05:48.838Z:
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
	**논문 작성을 진행한다.** 중심은 사건의 시간·수량 정보를 이용하는 과제 특화 표현 설계와 온라인 메모리 갱신을 생략한 구조의 정확도·비용 절충이다. 현재 validation 수량 개선과 동일 입력 MAC adapter 대비 짧은 처리 비용 절감을 확인했다. 평가 메모리 증가를 함께 보고하며, 최종 독립 평가는 별도로 남아 있다.
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
- 완료: 확정 구조의 방법 수식·그림·설계 근거와 기존 MAC 비용 감사를 작성했다. 같은 장치·입력의 최소 효율 측정도 완료했다. 다음은 원본 감사·차별성 검토·원고 통합·독립 평가 준비다. 기존 validation 검토 후 모델을 선택했다는 사실을 유지한다.
근거: 로컬 \[L1\], \[L2\]. 전체 파일 경로는 마지막 출처 목록을 따른다.
## 2. Contribution을 무엇으로 고정할 것인가
### C1. 사건 수량 예측을 위한 TitanTPP 표현 구조 — 구현 확인 / 독창성 비교 필요
> 사건 수량 예측을 위한 TitanTPP 표현 구조를 제안한다. 사건 간격과 과거 수량을 인과적으로 통합하고, 학습 가능한 메모리와 저차원 MLP 이력 보완을 통해 다음 사건 수량 예측에 적합한 표현을 학습한다. 이 구조는 추론 중 온라인 신경 메모리 갱신을 요구하지 않는다.
- 입력 설계: 사건 간격과 과거 수량의 log1p 특징을 사용하고, 미래 사건 정보가 들어오지 않도록 인과적 마스킹을 적용한다.
- 표현 설계: persistent memory를 갖는 인과적 encoder와 정적 prototype 검색을 유지하고, 두 encoder 단계 사이에서 현재·직전 표현을 결합하는 저차원 MLP의 잔차 출력으로 이력을 보완한다. 추가 조건부 Gate와 학습 가능한 상수 계수는 사용하지 않는다.
- 초기화 설계: 보완 출력의 영 초기화로 공통 초기 예측을 보존하며, 사건 수량 예측에 필요한 보완 표현을 학습한다.
**서술 원칙:** C1에서는 TitanTPP라는 하나의 제안 방법과 설계 이유를 설명한다. 구성 대안의 비교와 선택 근거는 6.1절의 구성 요소 검증으로 분리한다.
**현재 기준선과 다음 작업:** 채택 구조는 기존 이력 MLP로 고정했다. 해당 구성의 입력·계산식·보완 위치·메모리 상태·복잡도를 방법 절에 기술하고, 원본 Titans 및 일반 causal Transformer/THP와의 차이를 설명한다. 보편적인 구성요소를 결합했다는 사실만으로 독창성을 확정하지 않으며, 설계 이유와 통제된 검증 결과를 연결한다. \[L3\]
### C2. 온라인 메모리 갱신 없는 구조의 처리 비용 절감 — 동일 입력 실측 완료
채택 TitanTPP는 온라인 neural-memory 갱신을 사용하지 않고 persistent memory와 정적 검색을 유지한다. 따라서 “모든 memory 제거”가 아닌 재설계로 설명한다. [Titans 원논문](https://arxiv.org/abs/2501.00663)
- RTX5080 한 장에서 같은 train 입력·공통 head/loss·batch128로 비교한 결과, MAC 이벤트 adapter 대비 학습step은 Taxi **5.74배**, Intermittent **5.58배** 빨랐다. 평가batch는 각각 **8.49배**, **8.21배** 빨랐다.
- 파라미터는 **21.51% 감소**했고 학습 peak allocated는 약 **75.5% 감소**했다. 반면 평가 peak allocated는 **389.23 대97.83MiB**로 TitanTPP가 약 **3.98배 높았다**.
- 예열5+측정32batch를 모델·데이터·학습/평가 모드별3회 실행했다. 이는 seed42의 시간 반복이며 정확도3seed 실험이 아니다. 평가 경로에는 target loss와 MAC 온라인 메모리 갱신이 포함된다.
- 전체epoch 시간·순수 inference·수렴까지 총비용·동일정확도 우위는 이번 측정에서 확인하지 않았다. 두 backbone은 여러 구조가 함께 달라 온라인 갱신 제거 한 요소의 인과효과로 단정하지 않는다.
**주장 범위:** 이 구현과 측정 부하에서 처리 시간·학습 peak 메모리·파라미터 절감을 확인했다. 외부TPP 대비 validation 수량 경쟁력은 C3의 별도 증거로 제시한다. MAC와 같은 정확도를 유지한다는 주장으로 두 결과를 합치지 않는다.
실측·원시기록·timer·메모리 반례: <mention-page url="https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63"/>. 과거 MAC/B0 비용은 이 결과와 구분해 보존한다. \[L8\]
### C3. 통제된 비교에서 수량 예측 경쟁력과 한계를 검증 — validation 근거 확보
기존 TPP 구현의 입력·head·loss·학습·선택 규칙을 맞춘 비교에서 두 데이터의 수량 개선을 확인한다. 구조별 반례, Instacart의 작은 차이, 시간 예측의 상충관계를 숨기지 않는다.
- 공통 head 비교는 표현 효과를 검토하는 근거다. 각 원논문의 native 모델 및 최적 튜닝 결과를 모두 이겼다는 의미는 아니다.
- 대표 구조를 고정한 TitanTPP의 외부 비교 결과와, 내부 설계의 역할을 확인하는 구성 요소 검증을 구분한다.
- 현재 조건부 MLP Gate는 두 데이터에서 plain MLP의 MAE·RMSE를 개선하지 못했다. Gate를 핵심 성능 기여에서 제외하고 제한된 탐색 결과로 제시한다.
- 독립 평가 전에는 validation에서 반복되는 개선과 일반화 성능을 구분한다. \[L1\]
## 3. 논문용 주장 초안과 금지할 과장
**현재 근거로 작성할 문장**
> 본 연구는 사건 발생 시점과 수량을 다루는 예측 문제를 위해, 온라인 신경 메모리 갱신을 사용하지 않는 Titans-inspired 구조인 TitanTPP를 구성한다. 학습 가능한 메모리와 관측 사건 이력의 보완 표현을 결합하고, 공통 예측 head와 통제된 학습 조건에서 구조의 효과를 검증한다. Taxi와 Intermittent에서는 기존 TPP 비교 구현 대비 수량 예측 개선을 확인하며, Instacart의 제한적 이득과 시간 예측의 상충관계를 분석한다.
**동일 입력 효율 실측으로 추가할 문장**
> 동일한 RTX5080·train입력·head/loss·batch128 조건에서 TitanTPP는 로컬 Titans-MAC adapter보다 학습step을5.58–5.74배, loss를 포함한 평가batch를8.21–8.49배 빠르게 처리했다. 파라미터와 학습 peak 메모리는 감소했으나 평가 peak 메모리는 증가했다.
이는 짧은 처리 비용 실측이다. 전체epoch나 수렴비용·동일정확도 효율을 확인했다는 문장으로 확대하지 않는다.
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
<td>clip1 짧은 비용 측정 완료 / 현행 장기 정확도 결과 없음</td>
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
### 5.3 현재 TitanTPP–MAC 동일 입력 실측 — 완료
이번 측정은 과거 B0 비교와 별개로, 확정 TitanTPP(History MLP)와 inner-clip1 MAC 이벤트 adapter를 같은 Native Runtime·동일 train tensor로 비교했다.
<table header-row="true">
<tr>
<td>데이터</td>
<td>학습 ms/batch: TitanTPP / MAC</td>
<td>학습 시간비 MAC/TitanTPP</td>
<td>평가 ms/batch: TitanTPP / MAC</td>
<td>평가 시간비 MAC/TitanTPP</td>
</tr>
<tr>
<td>Taxi</td>
<td>31.447 / 180.582</td>
<td>5.74배</td>
<td>7.632 / 64.827</td>
<td>8.49배</td>
</tr>
<tr>
<td>Intermittent</td>
<td>32.157 / 179.472</td>
<td>5.58배</td>
<td>7.909 / 64.916</td>
<td>8.21배</td>
</tr>
</table>
각32batch 중앙값의3회 반복 평균이며 변동은 상세 보고서에 있다. batch128, max length256, CPU→GPU 전송 제외 CUDA동기화 wall time이다. 학습은 loss/backward/optimizer까지, 평가는 target loss와 MAC 내부갱신을 포함한다.
파라미터96,003 대122,310개, 학습 allocated약1,869.5 대7,631MiB, 평가 allocated389.2 대97.8MiB다. 학습·평가 메모리를 합쳐 “항상 메모리 절약”이라고 쓰지 않는다.
**완료 증거:** 준비 포함248.386초,24개worker,optimizer444step 및eval444batch(예열 포함). source117개·결과118개SHA와 동일Runtime/입력/공통head 검증. 측정32 내 추가 compiled graph0. GPU종료 확인. 기존scientific checkpoint는 보존했다.
**해석 한계:** 실제 초/epoch를 이번값에서 외삽해 실측이라고 쓰지 않는다. 순수추론·batch1 요청지연·동일정확도 비용·FLOPs 감소도 별도다. 과거timer/head/장치가 다른 기록과 통합평균을 만들지 않는다.
**추가 측정의 위치:** 원고에서 실제 초/epoch나 순수 inference를 꼭 주장할 때만 해당 최소 측정을 설계한다. 현재범위의 효율표 작성에 MAC9조건 장기학습은 필요조건이 아니다. 정확도 장기비교는 여전히 미실행·별도승인 대상이다.
상세: <mention-page url="https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63"/>; 로컬 `reports/titantpp_efficiency_5080_20260930_v1/`.
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
<td>과거 기록·현재 동일 입력 비용 측정 완료 / 현행 장기 정확도 결과 없음</td>
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
- 표 4: 현재 동일 입력 학습step·평가batch 시간, 파라미터, 학습/평가별 peak GPU 메모리. 과거 상각 초/epoch는 참고로 분리하고 평가batch를 순수추론이라고 이름 붙이지 않는다.
- 그림 1: 구조와 관측 정보의 흐름.
- 그림 2: 데이터별 정확도·시간 또는 정확도·메모리 절충. Instacart와 불리한 조건을 누락하지 않는다.
- 부록: 실패·복구 이력, selected/last, source·계약·데이터 SHA, 타이밍 정의와 구간별 오차.
## 8. 남은 작업 순서 — 효율 실측 완료 후 재정렬 (2026-09-30)
이번 요청은 결과 문서 반영과 작업 순서 정리다. 아래 후속 실험·held-out 평가를 지금 실행한다는 뜻은 아니다.
**현재 기준선과 대표 모델 — 완료**
- 대상: paper_research 및 이 논문 설계안.
- 대표 모델은 세 데이터 공통 `titantpp_history_mlp`이며, 논문에서는 TitanTPP로 표기한다. Full·B·Gate는 구성 비교/별도 탐색으로 둔다.
- 승인된 core54조건108역할과 외부TPP 추가36조건, Gate6완료조건의 학습·validation 재평가가 종료됐다. core와 외부의 중복 제거 고유 조건은90개다.
- 방법9개 수식·기호·구조 그림·코드 대응표와, MAC 대비 단발 비용 측정24개가 완료됐다. 이번 측정은 정확도 seed 추가나 전체epoch 계측이 아니다.
- 상세 효율 결과: <mention-page url="https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63"/>. 기존 학습·측정을 다시 실행 대상으로 올리지 않는다.
**1. 남은 원본 감사를 마치고 논문용 결과표를 확정한다 — 다음 작업 / 최우선**
- 대상: paper_research 로컬 증적과 종료된5090의 보존 원본.
- 5090 최종 checkpoint binary·source 회수 및 무결성/strict loading/선택·노출·복구 이력 감사의 잔여분을 확인한다. 소형 기록 감사는 완료됐지만, 현재 `verification.json`의 `final_core_binary_audit_complete=false`와 `aggregation/latest.json`의 `final_report_complete=false`를 완료로 바꿀 증거는 아직 확인되지 않았다. 일부 collection artifact의 존재만으로 완료라고 하지 않는다.
- 5080·Gate·재사용 조건의 완료 감사는 재사용한다. 새 학습 없이 기존 결과의 추적 가능성을 마무리한다.
- 산출물: core `final_report.md`와 기계판독 결과, 주 비교 MAE/RMSE/시간NLL의3seed 평균±표본SD, 내부 구조·구간별 반례·실패/복구·실측 비용표. 같은 RMSE-selected checkpoint를 유지한다.
- 완료 조건: 조건 중복/누락 없이 각 표의 값이 원본·checkpoint·계약과 연결되고, 미확인 항목과 성능 기준 미통과를 명시한다.
**2. 차별성과 비교의 공정성을 문헌·코드로 확정한다 — 다음 작업**
- 대상: paper_research Related Work 및 주장–근거표.
- C1은 입력→두 causal encoder 사이의 저차원 이력 보완→고정 학습 bank→수량/시간 head로 설명한다. Titans의 온라인 neural-memory, 일반 causal encoder, 관련 수치형 mark/간헐 수요 연구와 같은 점·다른 점을 분리한다. “최초 continuous 패러다임”이나 모든 memory 제거를 주장하지 않는다.
- 기존 RMTPP/THP/NHP/SAHP의 입력정보·전처리·split·공통 head/loss·선택·학습 예산을 대조한다. head를 바꾼 encoder 비교를 native 전체 모델의 최적 성능 순위로 쓰지 않는다.
- 기존 후보 S2P2/AttNHP 등은 원문·공식 코드·과제 적합성을 확인한 뒤 최소 추가 비교 필요성을 결정한다. FlexTPP/renewal 계열은 가까운 관련 연구로 검토하되, 구조를 훼손하는 head 교체를 강제하지 않는다. 모든 후보를 실행 목록에 넣지 않는다.
- 완료 조건: 기여별로 구현 사실/관측 결과/가설/미검증을 구분한 표와, 추가 비교의 채택·보류 이유 및 공정성 한계가 남는다.
**3. 완료된 방법과 결과를 논문 초안으로 통합한다 — 다음 작업**
- 대상: paper_research의 단일 TitanTPP 논문 원고.
- 이미 작성한 방법 수식과 그림을 재사용하여 Introduction → Related Work → Problem/Method → Setup → Results → Limitations를 연결한다.
- 효율 표는 현행 matched step/평가batch 결과를 본표로, 과거 상각 초/epoch를 별도 참고로 둔다. 학습메모리 감소와 평가메모리 증가를 함께 쓴다. MAC와의 동일정확도 효율은 입증된 것으로 쓰지 않는다.
- 주 결과에는 세 데이터의 고정 MLP 결과를 사용한다. Full/Gate를 데이터별로 골라 대신 쓰지 않는다. validation 탐색 후 대표 구조를 선택했다는 사실과 Instacart/시간NLL 반례를 유지한다.
- 완료 조건: 각 Contribution이 어느 표·그림·source로 지지되는지 원고에서 확인되고, 독립 평가 미완료 부분만 명시적으로 남는다.
**4. 원고 주장에 꼭 필요한 실험 공백만 결정한다 — 다음 작업 / 새 실행은 승인 필요**
- 대상: 원고 검토 후 작성할 최소 보완 계약.
- 추가 TPP 비교가 필요한지, 이미 확보된 짧은 비용 표로 효율 주장을 충분히 뒷받침하는지 먼저 판단한다. 실제 초/epoch 또는 순수 inference 지연을 본문에서 주장할 때만 해당 측정을 제안한다.
- Full 기반 정적검색 제거로 MLP 모듈의 필수성을 증명하지 않는다. 이 주장을 하지 않는다면 추가 MLP ablation을 자동 필수로 만들지 않는다.
- 필요한 항목마다 모델·데이터·seed·입력/head/loss·Runtime·시간 상한·분석 규칙을 구체화하고 GPU 실행 승인을 구분한다. 기존 MAC9조건 장기학습·새 Gate 탐색·추가seed·튜닝은 기본 계획에서 제외한다.
- 완료 조건: “추가 불필요” 판단 또는 검토 가능한 최소 실행 계약. 실행 승인이 있을 때만 그 범위를 수행한다.
**5. 대표 구조와 평가 규칙을 동결한 뒤 독립 평가한다 — 준비는 다음 작업 / held-out 열람·실행은 승인 필요**
- 대상: 확정 모델·비교군·checkpoint 목록과 잠긴 held-out split.
- 앞 단계의 모델/비교군/튜닝 결정을 마친 뒤 한 번의 최종 평가 프로토콜을 고정한다. validation에서 선택한 동일 checkpoint로 MAE·RMSE·시간NLL·구간별 결과를 함께 산출하도록 한다.
- 데이터 split과 과거 열람 이력을 확인해 독립성의 범위를 공개한다. 결과를 보고 대표모델·checkpoint·평가 규칙을 다시 고르지 않는다.
- 완료 조건: 승인된 범위의 모든 조건을 빠짐없이 보고한 독립 평가표와 불확실성/한계 설명. 현재 validation만으로 대체 완료 처리하지 않는다.
**6. 독립 결과를 반영하고 투고본을 마감한다 — 이후 작업 / 실제 제출은 별도 승인**
- 대상: paper_research 원고·그림·재현 부록 및 PAKDD 제출본.
- 표/본문/초록의 숫자와 주장 일치, 인용의 원문 근거, 익명성·분량·필수 제출사항을 공식 지침으로 최종 확인한다. 채택 가능성을 보장하거나 모든 데이터 우승을 요구하지 않는다.
- 데이터/코드 공개 가능 범위와 재현 명세를 정리한다. 실제 학회 제출·외부 공개는 이번 계획 요청에 포함하지 않는다.
- 완료 조건: 검토 가능한 제출본과 제출 체크리스트. 사용자의 제출 승인 뒤에만 외부 제출한다.
**병렬 가능 여부와 바로 다음 작업**
- 1번 원본 감사와2번 문헌·비교 검토는 입력/산출물이 달라 독립적으로 진행 가능하다. 이 계획 작성에서 별도 세션이나 agent를 시작하지 않는다.
- 3번의 방법·서론 초안은2번과 병행 가능하지만 결과표 확정은1번을 따른다. 모델/비교 계약 확정 → 필요 보완 → held-out 개방 → 최종 제출은 직렬이다.
- 바로 다음 작업은 **1번: 5090 잔여 원본 감사와 core 최종 결과표 확정**이다. 그동안 기존 방법·효율 자료를 원고로 연결한다. 새로운 학습을 먼저 시작하지 않는다.
- 스케줄러는 재생성하지 않는다. 커밋·Push·공용 Runtime 변경·실제 제출은 이번 문서 작업에 포함하지 않는다.
## 9. 근거 파일과 원문
로컬 파일은 이 Mac의 `/Users/igwanhyeong/PycharmProjects/paper_research/`를 기준으로 한다. Notion에서 로컬 파일을 직접 열 수 있다는 의미가 아니며, 재현을 위한 경로 표기다.
- \[L1\] `reports/titantpp_first_analysis_20260930_v1/report.md`, `analysis.json`, `verification.json`: 완료 결과·기여 범위·계산 검증. validation-only.
- \[L2\] `search_artifacts/titantpp_core_ablation_20260928_v1/aggregation/20260929T232132255352Z/ledger.json`: core 54조건의 선택·재평가 소형 기록.
- \[L3\] `models/TPPs/CountAwareTPP.py`, `CountAwareTitanMultiLagDetail.py`, `CountAwareTitanCoreAblation.py`, `models/Titan/backbone.py`, `models/Titan/common/memory.py`: 실제 표현·메모리·보완 경로. 현재 코드와 동결 source revision을 구분한다.
- \[L4\] `reports/titans_mac_reuse_audit_20260928_v1/report.md`, `historical_seed42.csv`, `mac_current_compatibility.csv`: 과거 MAC의 재사용 범위. 해당 보고서의 core 진행 상태는 당시 기록이며 본 문서의 완료 상태가 우선한다.
- \[L5\] `paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/historical_cost.csv`, `runtime_cost.json`; 계산 함수 `paper/scripts/analyze_count_aware_titantpp_mac.py:historical_cost_row`: 평균 epoch 및 별도 forward 프로파일.
- \[L6\] `reports/titans_mac_execution_preparation_20260928_v1/README.md`, `search_artifacts/titans_mac_observed_time_20260928_v1/README.md`: 현행 MAC 9조건은 로컬 준비·동결까지이며 GPU 미실행·별도 승인 필요.
- \[L7\] `reports/titantpp_mlp_gate_execution_20260929_v1/final_report.md`, `comparison.json`: Gate 완료 결과.
- \[L8\] `reports/titantpp_efficiency_5080_20260930_v1/report.md`, `analysis.json`, `verification.json`, `repetitions.csv`, `per_batch.csv`: RTX5080 동일 입력 비용 측정과118파일 무결성 검증. train-only, 정확도 평가 없음.
- 외부 원문: [Titans](https://arxiv.org/abs/2501.00663), [FlexTPP](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html), [Deep Renewal Processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764), [S2P2](https://arxiv.org/abs/2412.19634), [AttNHP](https://arxiv.org/abs/2201.00044).
**판단:** 논문 작성과 PAKDD 투고 준비를 진행할 근거가 있다. 실제 채택이나 현재 상태의 제출 완성을 보장하는 판단은 아니다. 기여는 “연속값의 최초 도입”이나 “Gate 우월성” 대신 구체적인 사건 표현 설계와 검증된 정확도·비용 절충에 둔다.
<page url="https://app.notion.com/p/3ebbbe40561381e2be20fbeb63fd7329">TitanTPP 방법 절·설계 근거·MAC 효율 감사 — MLP 확정 구조 (2026-09-30)</page>
<empty-block/>
</content>
</page>
