Here is the result of "fetch" for the Page with URL https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63 as of 2026-09-30T01:44:30.642Z:
<page url="https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63">
<ancestor-path>
<parent-page url="https://app.notion.com/p/3ebbbe40561381e2be20fbeb63fd7329" title="TitanTPP 방법 절·설계 근거·MAC 효율 감사 — MLP 확정 구조 (2026-09-30)"/>
<ancestor-2-page url="https://app.notion.com/p/3ebbbe40561381059aa0c5b73d6d47e3" title="TitanTPP 논문 설계안 — 사건 수량 표현과 온라인 메모리 단순화 (2026-09-30)"/>
<ancestor-3-page url="https://app.notion.com/p/2e9bbe40561380678b4ff6c7be991d33" title="6. From Initial Experiments to Extensive/Extended Studies"/>
<ancestor-4-data-source url="collection://2e4bbe40-5613-8184-bfa2-000bc4313c64" name="프로젝트"/>
<ancestor-5-database url="https://app.notion.com/p/2e4bbe4056138194996afa28247dd519" title=""/>
<ancestor-6-page url="https://app.notion.com/p/2e4bbe40561380898427d3ac376df1f8" title="졸업 논문"/>
</ancestor-path>
<properties>
{"title":"TitanTPP–MAC 동일 입력 효율 실측 — 5080, 2026-09-30"}
</properties>
<iconMetadata>null</iconMetadata>
<content>
상태: **complete**. 개인 RTX5080에서 단독 직렬 실행했다. 이 보고서는 **train 표본의 계산 비용**만 평가하며 validation/held-out 정확도를 계산하지 않았다.
**결론:** 동일 입력·공통 head/loss의 이번 측정에서 채택 TitanTPP는 MAC adapter보다 학습 step이 **5.58–5.74배**, 평가 batch가 **8.21–8.49배** 빨랐다. 학습 peak allocated는 약 **75.5% 감소**했다. 반면 평가 peak allocated는 **389.23 대97.83 MiB, 약3.98배**로 TitanTPP가 더 컸다. 따라서 “계산 시간과 학습 메모리를 줄인다”는 범위를 명시하고, 모든 실행 단계에서 메모리가 적다고 주장하지 않는다.
## 1. 직접 측정 결과
아래는 각32 batch의 중앙값을 구한 후, 독립 process 반복3회의 평균 ± 표본표준편차를 표시한 것이다. 이는 동일seed42 시간 반복이고 학습3seed가 아니다. 속도비는 각 반복 MAC/MLP 중앙값의 비율3개의 평균 ± 표본표준편차다. CPU→GPU 전송 제외, GPU 동기화를 포함한 host wall time이다.
<table header-row="true">
<tr>
<td>데이터</td>
<td>경로</td>
<td>TitanTPP ms/batch</td>
<td>MAC ms/batch</td>
<td>MAC/TitanTPP</td>
</tr>
<tr>
<td>Taxi</td>
<td>학습 step</td>
<td>31.447 ± 0.106</td>
<td>180.582 ± 6.056</td>
<td>5.74 ± 0.18</td>
</tr>
<tr>
<td>Taxi</td>
<td>평가 batch</td>
<td>7.632 ± 0.009</td>
<td>64.827 ± 0.622</td>
<td>8.49 ± 0.08</td>
</tr>
<tr>
<td>Intermittent</td>
<td>학습 step</td>
<td>32.157 ± 0.023</td>
<td>179.472 ± 0.555</td>
<td>5.58 ± 0.01</td>
</tr>
<tr>
<td>Intermittent</td>
<td>평가 batch</td>
<td>7.909 ± 0.030</td>
<td>64.916 ± 0.645</td>
<td>8.21 ± 0.07</td>
</tr>
</table>
단위는 batch128이다. 학습은 forward+head+loss+zero_grad+backward+outer clip1+AdamW step이며, 평가는 eval/no-grad forward+head+target loss다. **평가 batch latency이며 순수 inference latency가 아니다.** MAC의 관측 메모리 갱신은 양쪽 경로에서 유지된다. 한 번의 마지막 사건 target을 예측하므로 처리량은 targets/s로 표시한다.
## 2. 메모리와 parameter
<table header-row="true">
<tr>
<td>데이터</td>
<td>경로</td>
<td>MLP allocated MiB</td>
<td>MAC allocated MiB</td>
<td>MLP reserved MiB</td>
<td>MAC reserved MiB</td>
</tr>
<tr>
<td>Taxi</td>
<td>train</td>
<td>1869.5</td>
<td>7631.3</td>
<td>2042.0</td>
<td>7670.0</td>
</tr>
<tr>
<td>Taxi</td>
<td>eval</td>
<td>389.2</td>
<td>97.8</td>
<td>416.0</td>
<td>151.3</td>
</tr>
<tr>
<td>Intermittent</td>
<td>train</td>
<td>1869.5</td>
<td>7631.0</td>
<td>2042.0</td>
<td>7646.0</td>
</tr>
<tr>
<td>Intermittent</td>
<td>eval</td>
<td>389.2</td>
<td>97.8</td>
<td>416.0</td>
<td>130.0</td>
</tr>
</table>
Peak는 warm-up5 뒤 reset하여32회 동안 얻은 값의 반복3회 평균이다. 프로세스별 allocated/reserved baseline·분산·증분은 `repetitions.csv` (repetitions.csv)에서 계산할 수 있다. reserved는 allocator cache를 포함하며 peak allocated와 다른 양이다. 측정 전 allocator를 임의로 비워 특정 모델에 유리하게 만들지 않았다. parameter 수와 peak 메모리는 서로 다른 비용이다.
두 데이터 모두 TitanTPP **96,003개**, MAC **122,310개** parameter로 TitanTPP가 **21.51% 적다**. 학습 allocated 약1,869.53 vs7,631 MiB와 평가389.23 vs97.83 MiB를 별개로 해석한다. MAC의 순차 상태 경로와 MLP의 attention 활성값 등 전체 구조가 달라, 수치의 원인을 온라인 update 한 요소에만 귀속하지 않는다.
## 3. 입력·실행·시간 경계
- 데이터: 원본 parquet 및 split manifest SHA를 계약과 대조한 뒤 **train predicate를 적용한 행만 materialize**했다. 기존 loader의 dataset/collate를 사용했다. 시간·수량 통계는 기존 train 통계를 재사용했다.
- 각 데이터 전체 train target에서 dedicated torch seed42 permutation의 첫4,736개를 선택했다. 예열640+측정4,096 target이다. 이는 원래 fit loader prefix를 재현한 것은 아니다. `inputs.csv` (inputs.csv)에 실제 길이·padding을 기록했다.
- 모든 모델/반복/모드가 동일37개 CPU tensor cache와 target ID 순서를 사용했다. 각 batch tensor SHA와 초기 state SHA를 검사했다. 같은 head/loss/optimizer/batch/precision을 사용하고 encoder 구조는 비교 대상에 따라 달라진다.
- 모델별3회×train/eval을 새 process에서 실행했다. 반복2는 MAC→MLP, 반복1/3은 MLP→MAC 순서다. 동일모델 초기tensor는 매회seed42로 같고 공통 head 초기tensor도 정확히 같다. encoder 간 같은 tensor나 같은 파라미터 수를 강제하지 않는다.
- 전송 시간은 pageable CPU tensors→GPU 및 synchronize다. compute 직전/직후 synchronize로 GPU 비동기 실행을 기다렸다. 두 경계 때문에 실제 비동기 pipeline throughput을 재현하지는 않는다. loader·serialization·로그·검증은 steady timer에서 제외된다. transfer 포함 수치와 targets/s는 `repetitions.csv` (repetitions.csv)에 있다.
- MAC는 원본을 사건 입력에 적용한 로컬 adapter이며 inner clip1, fixed-shape compiled scan을 사용한다. online update를 생략한 대체물이 아니다. MLP에는 online neural-memory update가 없지만 persistent/static learned banks는 남는다.
- 각 worker 첫 batch·warm-up5·process 전체 시간을 공개했다. 새 격리 cache를 실행 내에서 공유했다. 첫 batch는 compilation뿐 아니라 초기 CUDA/allocator 효과도 섞이므로 순수 compile time이라고 부르지 않는다. 측정32 내 추가 compiled graph 수는 **0**이다. 원시32 batch를 삭제하거나 빠른 구간만 고르지 않았다.
- nvidia-smi로 worker 전/후 및10초 간격 장기worker 중 GPU compute PID를 확인했다. graphics/display 활동까지0이었다는 주장은 하지 않는다. GPU 클럭·온도·전력 순간값은 원본 관측에 있지만 전력 적분·전기료는 측정하지 않았다.
## 4. 완료·비용·증거
- 총 실행 시간(원격 준비 포함): **248.386초 / 4.14분**. 고정 상한1,800초. 완료 worker **24/24**, optimizer **444/444** step, eval **444/444** batch(예열 포함).
- runtime/precision/공통 head/입력 및 source117개 검증, 출력 manifest **118개** 파일 SHA 회수 검증. 종료 후 GPU compute process가 없었다. `verification.json` (verification.json).
- 새 cloud 임대료0달러. 개인5080 전기료 미측정. 기존 scientific checkpoint·예측·선택규칙·Runtime을 수정하지 않았으며 scratch 업데이트는 저장하지 않았다.
- contract canonical `d7d635cb939406ebe70777daf1f5287e3f68902b73cf2f6781a9ebf11d55fe26`; source117 closure `16f94a408a5317ff3ba7100ac462e128c07fab513b6be1333239032af61077ce`. 준비된 MAC source만 재사용했으며 과거 미승인 장기9fit 계약은 실행하지 않았다.
- 로컬 작성 검사에서 표준라이브러리와 이름이 충돌한 entrypoint를 [benchmark.py](http://benchmark.py)로 변경하고 검사tensor의 view를 clone으로 분리했다. 과학적 모델 소스 변경은 없고 원격 측정 전의 로컬 수정이다.
- 원본: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_efficiency_5080_20260930_v1` (실험 폴더), `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_efficiency_5080_20260930_v1/retrieved/terminal_manifest.json` (회수 manifest), `analysis.json` (기계판독 집계), `per_batch.csv` (batch별 시간).
## 5. 논문에 사용할 수 있는 문장
> On a single RTX 5080, using identical train-input batches, shared prediction heads and losses, and batch size 128, TitanTPP reduced median training-step time by factors of 5.74 on Taxi and 5.58 on Intermittent relative to our Titans-MAC event adapter. Evaluation-batch processing, including the target loss and the adapter's online memory updates, was 8.49 and 8.21 times faster, respectively. TitanTPP used 21.5% fewer parameters and approximately 75.5% less peak allocated GPU memory during these training steps. This did not extend to evaluation memory: TitanTPP required 389.2 MiB compared with 97.8 MiB for the adapter. These measurements characterize short, matched workloads rather than time to convergence or equal-accuracy efficiency.
첫 MAC training batch는49.920초, 첫 MAC evaluation batch는10.744초였다. 격리 compiler cache를 재사용한 반복에서는 각각 약3.3초와1.9초였다. 이 차이를 그대로 보존하지만 순수 compilation 시간을 분리 계측한 것은 아니다. 본문 속도비는 예열5개 이후의32개에 근거하며, cold-start 비용을 포함한 반복당 전체 시간은 별도 CSV에 공개한다.
## 6. 논문에서 사용할 범위
이 표로 같은 장치·입력·공통 head/loss 아래의 **짧은 학습 step 및 평가 경로 비용**을 비교할 수 있다. 특정 구현과batch128/max length256의 결과이며, 전체 epoch 시간·수렴까지 총학습비용·동일정확도 효율·모든 길이의 복잡도를 입증하지 않는다. 연산 경로 삭제의 독립적인 인과 효과를 분리한 ablation도 아니다.
실제 초/epoch가 필요하면 두 모델이 전체train target을 처리하는 별도 계측 범위를 산정한다. 이번32 batch를 전체 step수에 곱한 값을 실측 epoch 시간으로 쓰지 않는다. 추가 GPU 실행과 스케줄러는 자동으로 시작하지 않는다.
</content>
</page>
