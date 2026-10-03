# 완료된 5080 결과의 원본 감사

감사 완료: **2026-10-01 08:03 KST**. 대상은 추가 TPP 12조건(Taxi·Intermittent × S2P2·AttNHP × seed42·52·62)과 Instacart 활성 분기 정규화 seed42 1조건이다.

**13조건 모두 통과했다.** 원본 466파일의 SHA와 캠페인별 source 106개·104개를 확인했다. 최종 selected/last checkpoint 26개와 Instacart의 기존 epoch2 복구 checkpoint 1개를 CPU에서 검증했다. 두 source 묶음에는 공통 파일이 있으므로 210개 고유 파일이라는 의미는 아니다.

**기존 전체 비교표의 수치는 바뀌지 않았다.** 13조건의 선택 epoch·validation 표본 수·MAE·RMSE·시간 NLL이 기존 CSV와 정확히 일치한다. 이번에 해소한 것은 원본 checkpoint/source 감사의 미완료 상태다. 감사 통과와 성능 우월성·독립 평가 완료는 구분한다.

**원본 회수와 무결성을 확인한다 — 완료**

- 추가 TPP: 325파일·106개 source·12조건·24개 최종 checkpoint. [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5080_20261001_v1/retrieval_receipt.json) · [CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5080_20261001_v1/terminal_audit.json).
- Instacart 정규화: 141파일·104개 source·1조건·2개 최종 checkpoint와 보존된 복구 checkpoint 1개. [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_instacart_norm_5080_20260930_v1/retrieved/terminal_20261001_v1/retrieval_receipt.json) · [CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_instacart_norm_5080_20260930_v1/retrieved/terminal_20261001_v1/terminal_audit.json).
- 계약·승인·실제 환경의 사전 검증·관리자 완료 상태·terminal manifest·전체 파일 SHA를 연결했다. 추가 TPP 계약은 `a39790e6…`, Instacart 5080 운영 계약은 `f93a456e…`다. 정규화 checkpoint의 과학 계약 `ae7a1bff…`는 최초 4090 계약을 유지하므로 운영 계약과 구분했다. 전체 SHA는 audit_summary.json과 각 terminal_audit.json에 기록했다.
- 회수 당시 종료된 실험 경로의 소유 프로세스가 없음을 확인했다. 같은 5080의 RAF는 별도 실행으로 유지했고, 5090에는 새 조회나 실행을 하지 않았다. 원격에서는 저우선순위 읽기·전송만 수행했다.

**checkpoint와 학습·재평가 기록을 대조한다 — 완료**

- 동결 source의 별도 CPU 사본으로 모델을 구성했다. selected/last와 last 안의 best tensor를 strict loading하고, 복원한 tensor의 SHA까지 확인했다. 현재 작업 트리의 변경된 과학 코드는 사용하지 않았다.
- optimizer 상태의 모양·유한값·학습률·누적 step을 검사하고 CPU 복원을 확인했다. Python·NumPy·Torch RNG와 shuffle 상태는 CPU에서 복원했다. CUDA RNG는 저장된 바이트만 확인했다.
- 최초 최소 validation raw 수량 RMSE 선택과 최초 patience40 종료, batch128/min40/max300, 전체 train/validation 노출과 배치 순서를 검증했다. 초기화 동일성은 당시 실제 환경에서 통과한 검증 기록과 checkpoint의 초기 SHA를 연결해 확인했다.
- 저장된 selected/last validation 재평가 26역할을 tensor SHA 및 학습 이력과 대조했다. 수량·이력 길이별 구간의 표본 수·SSE·MAE·시간 NLL 합계, body/tail과 빈 구간까지 확인했다. 새 예측이나 재평가는 실행하지 않았다.
- epoch 영수증은 마지막 epoch 직후의 노출을, terminal manifest는 종료 후 추가 validation을 포함한 최종 파일을 가리킨다. 마지막 validation 기록을 제외한 노출 prefix SHA도 확인했으며 원본은 변경하지 않았다.
- S2P2 첫 층의 `delta_net.weight`는 원본 연산에서 입력이 None인 경로여서 optimizer 상태가 생기지 않는다. 동결 vendor/adapter 코드로 확인한 이 항목만 허용했다. 또한 summary의 `history_rows`는 epoch 수가 아닌 이력 길이 구간표다. summary가 생략한 빈 구간은 endpoint의 명시적인 0개 구간과 대응했다.

**Instacart의 4090→5080 복구를 확인한다 — 완료**

- 4090에서 저장한 epoch1~2의 history·exposure·timing·train.log와 epoch2 checkpoint SHA가 최종 원본에 보존됐다. epoch2의 optimizer 31,114steps, RNG·shuffle·모델·내부 best 상태를 CPU로 검증했다. 5080은 epoch3부터 이어졌다.
- **종료 70epoch·선택 30epoch, 전체 optimizer 1,088,990steps**다. 시간·수량 head, 과학 계약·source·초기 SHA·복구 식별 정보가 유지됐다. GPU를 바꾼 뒤의 결과가 중단 없이 4090에서 실행한 결과와 비트 단위로 같다는 보장은 하지 않는다.
- summary elapsed **16,276.045800초는 5080 복구 이후 시간**이다. 전체 70epoch의 기록된 시간 합은 **17,458.497193초**이며, 첫 2epoch는 4090 기록이다. 원래 RunPod의 예산 검사 중단과 복구 대기·장비 차이를 삭제하거나 추정 보정하지 않았다. 원래 마감 안에서 종료했다.

**선택 checkpoint의 결과를 원본까지 연결한다 — 완료**

모든 값은 validation이며 낮을수록 좋다. 선택/종료는 epoch 번호다. 추가 TPP는 공통 head 비교이고, Instacart 정규화는 seed42 탐색이다.

| 데이터 | 모델 | seed | 선택 / 종료 | MAE | RMSE | 시간 NLL | CPU 감사 |
|---|---|---:|---:|---:|---:|---:|---|
| Taxi | S2P2 | 42 | 70 / 110 | 32.645699164 | 104.412231572 | 0.692848087 | 통과 |
| Taxi | AttNHP | 42 | 59 / 99 | 36.974967408 | 136.554972544 | 0.668676909 | 통과 |
| Taxi | S2P2 | 52 | 50 / 90 | 33.662705402 | 106.411925794 | 0.658092424 | 통과 |
| Taxi | AttNHP | 52 | 156 / 196 | 36.843305551 | 126.186778094 | 0.707504075 | 통과 |
| Taxi | S2P2 | 62 | 22 / 62 | 32.761044040 | 108.237978078 | 0.646354425 | 통과 |
| Taxi | AttNHP | 62 | 38 / 78 | 34.124992614 | 118.645348143 | 0.686730554 | 통과 |
| Intermittent | S2P2 | 42 | 60 / 100 | 0.940676398 | 2.487482405 | 0.253593401 | 통과 |
| Intermittent | AttNHP | 42 | 36 / 76 | 1.136849742 | 3.383345721 | 0.264412043 | 통과 |
| Intermittent | S2P2 | 52 | 93 / 133 | 0.925992913 | 2.622541654 | 0.215749142 | 통과 |
| Intermittent | AttNHP | 52 | 62 / 102 | 1.037497053 | 3.069178716 | 0.259251461 | 통과 |
| Intermittent | S2P2 | 62 | 12 / 52 | 0.940988998 | 2.617502052 | 0.235457204 | 통과 |
| Intermittent | AttNHP | 62 | 145 / 185 | 0.949442101 | 2.787287756 | 0.360201573 | 통과 |
| Instacart | 활성 분기 정규화 | 42 | 30 / 70 | 3.990505654 | 5.871660721 | 2.802718516 | 통과 |

기존 [전체 비교 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/report.md)의 07:07 KST 수치와 114개 고유 조건 집계는 그대로 보존했다. 당시 verification의 pending은 분석 생성 시점의 상태이며, [후속 감사 상태](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/audit_completion.json)가 최신 증거를 연결한다. 추가 TPP는 **5080의 12조건에 대한 최종 감사가 완료**됐으며 5090 최종 감사는 남아 있다.

**감사 범위와 한계를 기록한다 — 완료**

- 로컬 CPU 환경은 Python3.12.10·Torch2.14.0이다. 당시 CUDA 환경을 재현한 실행은 아니다. 실제 환경·공통 초기화·인과성·마스킹 검증은 SHA로 확인한 기존 기록을 재사용했다. 로컬에서 만든 초기 tensor가 native CUDA 초기화와 같다고 주장하지 않는다.
- 원본 데이터를 다시 로드하지 않았다. 계약의 데이터 SHA, 실제 입력 통계·표본 식별 정보, 노출·배치 순서 기록을 대조했다. held-out/test 성능은 열람하지 않았다.
- 추가 TPP의 합성 60updates와 5080 정규화의 복구 검증 3updates는 준비 검증 증거다. 실제 fit step에 포함하지 않는다.
- Core 54조건의 최종 감사와 Titans-MAC 효율 측정은 완료 증거를 재사용했다. 새로운 학습·평가·Pod·스케줄러·과학 코드 변경·commit/push는 없다.

**1. 방법과 관련 연구를 원고로 연결한다 — 다음 작업 / 지금 착수 가능**

- 대상은 paper_research의 단일 논문 원고다. 완료한 수식·구조 그림·코드 대응·효율 자료와 이번 감사 결과를 재사용해 기여, 공통 head 비교 범위, RMSE 선택 이유를 연결한다. 기존 GPU 큐와 독립적으로 병행 가능하다.
- 완료 조건은 방법·관련 연구·실험 설정 초안에서 각 주장의 근거와 미검증 범위가 드러나는 것이다. 최종 결과 수치는 3번에서 확정한다.

**2. 승인된 5080 RAF와 5090 Instacart 추가 TPP를 마친다 — 진행 중 / 서버 작업 대기**

- 기존 서버별 큐를 독립적으로 유지한다. 이번 감사에서 중단·재시작·추가 실험을 하지 않았다. 서버 진척은 마지막 07:35 관측 이후 다시 조회하지 않았다.
- 종료 후 완료·실패·미시작 조건을 빠짐없이 회수하고 같은 원본 감사를 수행한다. 기존 세 데이터의 정규화 seed52·62는 보류하고, 자동 스케줄러는 삭제 상태를 유지한다.

**3. 네 데이터의 최종 비교와 주장–근거표를 확정한다 — 다음 작업 / 2번 이후**

- 고정 대표 TitanTPP MLP와 외부 6개 모델을 같은 평가 기준으로 비교한다. 정규화·Gate는 별도 탐색, Core는 구성 비교로 두고 시간 NLL 악화·구간별 반례·실측 비용을 포함한다.
- 완료 조건은 기존 세 데이터와 RAF의 전체 결과가 원본까지 추적되고, 같은 seed 비교와 3seed 집계의 주장 범위가 명확해지는 것이다.

**4. 최종 평가 규칙을 고정하고 독립 평가한다 — 이후 작업 / 실행·열람 승인 필요**

- 모델·checkpoint·지표·비교 범위를 먼저 고정한 실행 계약을 작성한다. held-out 결과를 현재 validation 분석에 섞지 않는다. 실제 독립 평가의 대상과 예산을 구체화한 뒤 승인 범위에 따라 실행한다.

**5. PAKDD 제출본을 완성한다 — 이후 작업 / 실제 제출 승인 필요**

- 독립 평가를 반영해 원고·표·그림·재현 부록의 주장과 근거를 대조한다. 완료 조건은 검토 가능한 제출본이며, 외부 제출·공개는 별도 승인 대상으로 둔다.

[기계판독 감사 요약](audit_summary.json) · [검증 결과](verification.json) · [현재 기준선과 전체 작업 순서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_baseline_reset_20261001_v1/README.md)
