# 이력 보완 TitanTPP: 최종 검증 결과와 PAKDD 2027 준비 계획

**연구 결과 통합 문서 | 2026-09-27 기준**

이력 보완 TitanTPP를 논문의 주모델로 삼기 위한 현재 근거, 기여 후보, 투고 판단, 추가 비교와 독립 test 평가 계획을 정리한다. 기존 B는 이력 보완 경로를 제거한 내부 대조군으로 유지한다.

이 문서는 현재 대화의 연구 방향과 final_report.md 전체 내용을 포함한다. 새 실험을 실행하거나 held-out 결과를 열람한 문서가 아니다. 학습 성공, 성능 기준 통과, 논문 채택 가능성은 각각 구분한다.

| 구분 | 현재 상태 |
| --- | --- |
| 기존 승인 실험 | 세 데이터셋 × 여섯 모델 × 세 seed, 54조건 완료 |
| 평가 증거 | selected/last checkpoint의 validation 재평가 108회 완료 |
| 주모델의 강한 근거 | Taxi에서 세 seed 모두 수량 RMSE 1위 |
| 주요 한계 | 데이터셋별 반례, 시간 예측 손실, 추가 용량 효과 미분리 |
| 아직 없는 결과 | 독립 test 성능, 새 구조 비교, 모델별 튜닝 비교 |
| 투고 검토 | PAKDD 2027 연구 트랙; 채택을 보장하는 평가는 아님 |

**문서 순서**

1. 현재 연구 판단과 논문 기여 후보
2. PAKDD 투고 판단과 보강할 비교
3. 독립 test 평가와 최종 분석 설계
4. 출처와 증거의 해석 범위
5. final_report.md 전체 수록
6. 마지막 정리: 다음 작업의 실행 순서와 완료 조건

<!-- pagebreak -->

## 1. 현재 연구 판단과 논문 기여 후보

### 1.1 이력 보완 TitanTPP를 주모델로 정하는 의미

이력 보완 TitanTPP를 후속 연구의 기준 모델로 선택할 수 있다. 다만 이는 연구 방향의 선택이며, 모든 데이터셋과 지표에서 B보다 우월하다는 실험 결론은 아니다. B는 논문의 주모델 대신 내부 절제 실험의 기준으로 남긴다.

현재 구현은 관측된 인접 이력의 수준과 변화량을 별도 경로로 처리하여 encoder 블록 사이에 잔차로 더한다. local 모드의 실제 참조 간격은 1이다. 여러 이력 길이에 따른 경로 사용 조건이 있다고 해서 여러 시간 간격을 직접 읽는 모델이라고 설명해서는 안 된다. 정적 메모리 검색을 사용하며, test 중 메모리를 학습하는 구조로 표현하지 않는다.

### 1.2 확인된 결과와 그 의미

| 데이터 | B 대비 평균 수량 RMSE 변화 | 해석 |
| --- | --- | --- |
| Taxi | -12.96% | 세 seed 모두 수량 개선. 시간 NLL 악화가 동반됨 |
| Intermittent | +1.88% | 평균 RMSE 악화. 수량 크기와 seed에 따라 결과가 달라짐 |
| Instacart | +0.108% | 세 seed 모두 B보다 RMSE가 소폭 높음 |

공통 head/loss를 사용하는 외부 인코더 네 개 중 가장 좋은 모델과 비교하면 local의 평균 수량 RMSE는 Taxi에서 약 16.8%, Intermittent에서 약 28.9% 낮다. Instacart에서는 THP보다 약 0.47% 높다. 이 수치는 각 원논문의 최적 구현에 대한 승리나 SOTA 달성을 의미하지 않는다. B와 비교해야 이력 보완 경로의 추가 효과를 직접 논의할 수 있다.

세 데이터셋 × 세 seed의 기존 종합 성능 기준은 9개 모두 미통과했다. 연구 범위를 수량 예측 중심으로 설명할 수 있지만, 과거 기준을 사후 수정해 통과한 것으로 보고하지 않는다.

### 1.3 논문에 제시할 기여 후보와 필요한 증거

| 기여 후보 | 현재 근거 | 더 필요한 증거 |
| --- | --- | --- |
| 관측 이력의 수준·변화 경로를 결합한 수량 예측 구조 | 구현과 3-seed validation 결과 | 기존 방법과의 차별성, 같은 용량의 대조군, 경로별 절제 |
| 시간·수량을 공통 조건에서 평가하는 문제 설정과 관측 처리 | 같은 head/loss와 기록 시간의 likelihood 사용 | 기존 공동 예측 연구와의 관계. 최초의 공동 예측이라는 주장은 하지 않음 |
| 효과와 실패 조건을 함께 보여주는 실증 분석 | 전체·수량·이력 구간 및 seed별 반례가 있음 | 독립 test, 강한 비교 모델, 비용과 불확실성 분석 |

권장 연구 메시지는 “관측 이력의 수준과 변화 정보를 분리해 보완하는 설계가 이벤트 수량 예측에 어떤 조건에서 도움이 되는지 평가한다”이다. 모든 데이터에서의 우월성, 정보 손실 복원의 증명, 메모리와 국소 경로의 필수적 시너지는 아직 확정된 기여가 아니다.

<!-- pagebreak -->

## 2. PAKDD 투고 판단과 보강할 비교

### 2.1 투고 목표의 위치

PAKDD를 현실적 투고 후보로 검토하고, CIKM·ECML PKDD 수준의 설득력을 목표로 연구를 보강하는 방향을 권한다. AAAI는 독창성과 일반성을 더 확보했을 때의 도전 후보다. 현재 증거만으로 KDD·ICLR·ICML·NeurIPS 본회의 경쟁력을 확정하기는 어렵다. 이는 공식 등급이나 채택 확률이 아닌 현재 연구에 대한 판단이다.

PAKDD 2027 연구 트랙은 독창적인 데이터 마이닝·학습 연구와 새로운 통찰을 요청하며 시계열 학습을 범위에 포함한다. 공식 제출 마감은 2026-11-20 23:59 AoE이다. 2026-09-27 기준 약 8주가 남았으며, 실제 제출 전 공고 변경을 다시 확인한다. 아래 실험 목록은 학회가 지정한 필수 목록이 아니라 이 논문을 위한 권장 보강안이다.

### 2.2 구조의 효과를 분리하는 네 가지 비교

전체 local 모델과 B의 기존 결과를 현재 기준선으로 사용한다. 다음 네 가지는 동일한 현재 조건의 최종 비교 근거를 새로 마련할 대상이다.

| ID | 추가 모델 | 검증할 질문 | 통제할 조건 |
| --- | --- | --- | --- |
| A1 | 같은 용량의 단순 이력 MLP | 분해 구조의 효과인가, 단순 용량 증가인가? | 같은 두 관측 상태, 추가 파라미터 수, 삽입 위치, 이력 사용 조건과 초기 잔차 |
| A2 | 수준 경로만 사용 | 변화량 경로가 필요한가? | head/loss와 데이터 유지, 변화량 경로만 제거 후 새 학습 |
| A3 | 변화량 경로만 사용 | 수준 경로와의 결합이 필요한가? | head/loss와 데이터 유지, 수준 경로만 제거 후 새 학습 |
| A4 | 메모리 검색 경로 제거 | 전체 구조에서 메모리 검색이 기여하는가? | 이력 보완 경로 유지, 검색 경로를 제거한 효과와 비용 기록 |

A2·A3·A4에서 제거에 따라 파라미터 수가 달라지면 이를 명시한다. 사용하지 않는 파라미터를 넣어 수치만 맞추지는 않는다. 용량 효과를 가장 직접적으로 다루는 비교는 A1이다. A4 하나만으로 메모리와 이력 보완 사이의 상호작용을 증명할 수는 없다. 시너지를 핵심 주장으로 삼을 경우 메모리 유무 × 이력 보완 유무의 2×2 설계를 추가로 검토한다.

네 변형을 세 데이터셋과 seed42·52·62에 적용하면 최종 학습은 36조건이다. 이는 제안 규모이며 아직 실행 계약이나 새 GPU 사용 승인이 아니다. 기존 결과를 재사용할 때 데이터·head/loss·선택 규칙·source identity의 호환성을 먼저 확인한다.

### 2.3 외부 비교의 경쟁력 보강

- 마지막 관측 수량과 과거 평균 등의 단순 수량 기준선을 추가한다. 평균과 창 길이 등은 train/validation에서만 정한다.
- 수량 전용 신경망 한 종류를 선정해 시간 예측을 함께 학습하는 설계의 실용적 필요성을 확인한다. 수량 전용 모델의 시간 NLL은 미해당으로 표기한다.
- 최근 관련 방법 한 종류를 입력·목표·데이터 조건이 맞는지 확인한 뒤 선정한다. 구체적인 논문과 구현은 아직 미확정이다.
- 주요 모델에 공정한 validation 튜닝 예산을 사전에 정한다. 현재 고정 설정 결과와 모델별 튜닝 결과는 별도 표로 구분한다.
- 공통 head를 붙인 encoder 비교와 원 모델의 전체 구조 비교를 구분한다. 관측 공간이나 시간 likelihood가 다르면 NLL을 같은 점수처럼 직접 순위화하지 않는다.

새 학습 모델 두 종류를 모두 세 데이터셋·세 seed에 적용하면 18조건이다. 구조 비교 36조건과 합쳐 추가 최종 학습은 54조건이라는 규모 예시가 된다. 튜닝, 합성 검증, 기존 모델의 설정 변경에 따른 재학습은 포함하지 않았다. 실제 GPU 시간은 확정 모델의 실측 epoch 시간과 조기 종료 조건을 바탕으로 계산한다.

<!-- pagebreak -->

## 3. 독립 test 평가와 최종 분석 설계

### 3.1 test를 열기 전에 확정할 사항

모델 구조, 비교군, 설정 선택 절차, seed, 주 지표, 보조 지표, 구간 경계와 평가 checkpoint를 먼저 동결한다. 주 지표는 현재 맥락에서는 원단위 수량 RMSE이며 MAE·시간 NLL·구간별 손실·비용을 함께 보고하는 구성이 적절하다. 수량과 시간을 모두 개선한다는 주장을 유지하려면 그에 맞는 추가 근거가 필요하다.

과거 연구에서 test가 모델 선택에 사용됐는지는 성능 파일을 열지 않는 메타데이터 감사부터 확인한다. 독립성이 훼손된 test라면 미사용 최종 평가 범위를 별도 계약으로 정해야 하며, 이미 사용한 데이터를 새 held-out이라고 부르지 않는다.

### 3.2 별도 평가기의 구현과 검증

현재 local_detail 실행기는 validation 전용이므로 기존 금지 조건을 우회하지 않는 별도의 test 평가 경로가 필요하다. 원본 checkpoint와 학습 계약을 수정하지 않고 평가 입력과 출력의 출처를 별도 기록한다.

- 시간순 target 분할, 관측 가능한 과거 이력, target 시점의 수량·시간 차단을 합성 데이터로 확인한다.
- 정규화·시간 관측 규칙·수량 구간 경계는 기존 train 기준을 유지한다.
- 순차 평가에서 이전 test 사건을 이후 사건의 관측 이력으로 사용할지 사전에 정한다. 허용하더라도 미래 사건이나 현재 target은 입력할 수 없다.
- padding, 짧은 이력, top-code 시간, 빈 구간, 표본 중복과 누락을 검증한다.
- 동일 checkpoint의 validation 재평가가 기존 기록과 일치하는지 확인한 후 test 입력만 별도 승인 범위에서 연다.

### 3.3 최종 test 실행

validation raw-RMSE로 고른 checkpoint 하나를 각 조건의 최종 평가 대상으로 고정한다. 같은 checkpoint에서 수량·시간·구간 지표를 계산한다. test 결과를 보고 best/last를 바꾸거나 새로운 모델과 seed를 골라내지 않는다.

현재 54조건의 checkpoint가 계약과 호환되고 재사용 가능하면 기존 범위의 test는 54회 평가다. 기존 selected/last validation 108회를 test에서도 그대로 반복할 필요는 없다. 새 비교 모델과 절제 모델의 선택 checkpoint 평가는 그 조건 수만큼 추가된다. 학습 설정을 바꾼 모델은 변경된 계약의 결과로 구분한다.

평가가 끝나면 모든 사전 지정 모델과 seed를 포함한다. 기술적 오류가 발견되면 원인·영향 범위·수정·재평가 근거를 별도 기록하고, test에 맞춘 구조 변경과 구별한다.

### 3.4 논문에 넣을 분석 묶음

| 분석 | 보고할 내용 |
| --- | --- |
| 주 결과표 | 데이터셋별 수량 RMSE·MAE, 시간 NLL, 세 seed 평균·표본 SD |
| 구조 비교표 | B, 전체 local, A1~A4의 동일 조건 결과와 파라미터 수 |
| 반례 분석 | 수량 크기·이력 길이별 성능, 구간 표본 수, 시간 예측 손실 |
| 비용 분석 | 같은 장비·batch·길이에서 파라미터·메모리·추론 처리량과 지연 |
| 불확실성 | seed별 paired 차이; 필요 시 시퀀스·시간 블록 기반 신뢰구간 |
| 재현 증거 | source·데이터·checkpoint identity, 실제 학습량, 실패·복구 이력 |

54조건과 108회 재평가를 독립된 과제 수로 해석하지 않는다. seed 표본 SD는 신뢰구간이 아니다. 신뢰구간에는 이벤트의 종속성을 반영해야 하며, 블록 bootstrap용 그룹별 증적 수집은 아직 남은 작업이다.

<!-- pagebreak -->

## 4. 출처와 증거의 해석 범위

### 4.1 로컬 근거

- [최종 validation 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/final_report.md): 전체 결과와 한계의 기준 문서.
- [전체 실행 감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/terminal_audit.json): 조건별 감사와 source receipt.
- [조건별 상세 결과](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/condition_results.json): 원본 연결, 실제 학습량과 checkpoint 정보.
- [세 seed 집계](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/three_seed_summary.json): 전체·구간별 평균과 표본 SD.
- [seed별 B 대비 차이](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/paired_seed_changes.json): 유리하거나 불리한 seed를 모두 포함한 비교.
- [실험 세션 규칙](/Users/igwanhyeong/PycharmProjects/paper_research/TEST_SESSION_PROTOCOL.md): validation-only 및 held-out 평가 범위.
- [현재 이력 보완 구현](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanMultiLagDetail.py): local/multilag 분기와 수준·변화 경로 정의. 과거 결과 재현에는 해당 실행의 동결 source를 사용한다.

로컬 증거 파일의 전체 경로와 링크는 함께 제공하는 Markdown에 포함된다. PDF는 보고서 내용 전체와 클릭 가능한 공식 웹 출처를 제공한다.

현재 작업 브랜치는 codex/hard-lmm-causal-qkv이며 기존 사용자 변경이 존재한다. 본 문서 작성은 학습 코드·기존 결과·Runtime을 변경하거나 커밋·Push하지 않는다. v0.7의 과거 head 결과를 이번 결과표와 혼합하지 않는다.

### 4.2 공식 학회 근거

- [PAKDD 2027 Research Track](https://pakdd2027.org/pages/calls/research): 범위, 독창성 요구, 제출 마감. 2026-09-27 확인.
- [CIKM 2026 Full Research Papers](https://cikm2026.diag.uniroma1.it/full-research-papers/): 연구 주제 적합성 참고. 2026년 공고를 향후 제출 일정으로 해석하지 않는다.
- [ECML PKDD 2025 채택 논문](https://ecmlpkdd.org/preprints/2025/): Interpretable Hybrid-Rule Temporal Point Processes 등 최근 TPP 연구의 주제 적합성 참고.
- [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines): 독창성·의의·근거에 대한 참고. 해당 기준을 PAKDD의 공식 체크리스트로 대체하지 않는다.

### 4.3 원문 수록 방식

다음 5절에는 final_report.md의 모든 본문과 표를 수록한다. 통합 문서에 맞춰 제목 단계와 링크 경로만 조정했으며 수치·판정·반례는 변경하지 않았다. 원문 안의 “남은 작업 순서”는 원문 작성 당시 기록이다. 현재 대화까지 반영한 최종 작업 순서는 이 문서 맨 마지막 6절에 있다.

원본 SHA256: 25e2888ce1e3e4d18f3024e24f2a3238d3b8ca60856fbc3b3cf844ec32507ce1

<!-- pagebreak -->

## 5. final_report.md 전체 수록

<!-- ORIGINAL_REPORT_BEGIN -->
## 이력 보완 TitanTPP — 3-seed validation 최종 결과

최종 학습 종료: **2026-09-27 21:57:01 KST**. 범위: 세 데이터셋 × 여섯 모델 × seed42·52·62, **54조건·108개 selected/last validation 재평가**.

**판단 — 제한을 명시하면 사용 가능**
- Taxi 수량 예측에서는 이력 보완 TitanTPP를 이어갈 근거가 강화됐다. 세 seed 모두 여섯 모델 중 RMSE 1위이며, B 대비 평균 RMSE 12.96%, MAE 11.80% 감소했다.
- 전체 데이터셋의 B 대체 모델로 확정할 근거는 부족하다. Intermittent는 평균 RMSE가 1.88% 증가했고, Instacart는 세 seed 모두 B보다 RMSE가 소폭 높았다(평균 +0.108%).
- 모든 지표를 함께 보는 사전 기준은 B 대비 9개 데이터셋·seed 조합 모두 미통과다. Taxi의 시간 NLL 악화와 다른 데이터셋의 수량·구간·안정성 반례를 유지한다. 실행 성공과 성능 기준 통과는 별도다.
- 연구 방향은 **Taxi의 수량 개선을 중심 근거로 삼고, 시간 예측 및 데이터셋별 한계를 명시한 후보**로 정리하는 것이 타당하다. “여러모로 항상 더 좋다” 또는 “범용적으로 B보다 우월하다”는 주장은 이 결과가 지지하지 않는다.

### 검증 범위와 현재 기준선 — 완료

| 출처 | 조건 | 재평가 | 기록된 optimizer step |
| --- | --- | --- | --- |
| seed42_reused | 18 | 36 | 7,678,963 |
| 5080_new | 24 | 48 | 3,448,376 |
| 5090_original_completed | 8 | 16 | 7,342,904 |
| 5090_recovery | 4 | 8 | 4,635,986 |
| 합계 | 54 | 108 | 23,106,229 |

복구 seed62 paired 파일은 원본 B/local 2조건·4재평가와 복구 4조건·8재평가를 결합한 6모델·12재평가다. 원본 B/local을 복구 신규 조건으로 중복 계수하지 않았다. 정지한 RMTPP 첫 시도는 별도 실패 이력으로 보존되며, 저장되지 않은 부분 update 수는 위 23,106,229 step 합계에 포함할 수 없다.

다음 항목을 54조건 모두의 소형 JSON 증적으로 검증했다: 첫 허용 조기 종료 시점, 최초 strict raw-RMSE 최소 checkpoint, 실제 epoch·step·train/validation 표본수, 초기 상태 SHA, 동일 seed의 공통 학습 batch prefix, validation 순서, selected/last 재평가와 history 일치, 구간 count/SSE/MAE/time NLL 합산, first40/last30 평균·표본 표준편차, paired 비교와 복구 partition 집계. 신규 복구는 네 개의 독립 worker PID와 60회 CUDA qualification, 고정 Runtime·공통 deadline, 정상 프로세스/tmux 종료를 확인했다.

5090 원본 완료 결과·입력·체크포인트 61개를 원격 바이트 SHA로 재검증했다. 기존 16개 checkpoint 파일 SHA와 복구 8개 checkpoint SHA를 보존했다. 로컬 감사는 checkpoint를 역직렬화하거나 데이터·held-out 예측을 읽지 않았으며, 이미 수행된 validation 재평가 증적을 검증했다. B/local 공통 초기 tensor 검증은 동결 qualification/실행 증적에 근거하고, 이번 감사에서는 초기 상태 SHA의 일치를 확인했다.

### 세 seed의 동일 선택 checkpoint 비교 — 완료

모든 수량·시간·구간 지표는 각 조건의 **동일한 validation raw-RMSE 선택 checkpoint**에서 나온다. 아래 ±는 seed42·52·62 간 표본 표준편차(ddof=1)이며 신뢰구간이 아니다. 세 seed를 동일 가중치로 평균했다. RMSE·MAE·시간 NLL은 모두 낮을수록 좋다. 데이터셋 간 수량 단위와 척도가 다르므로 하나의 RMSE로 합산하지 않는다. RMTPP·THP·NHP·SAHP는 공통 head/loss 및 고정 학습 규칙을 적용한 비교 구현이며, 각 원논문의 최적 튜닝 결과에 대한 순위가 아니다.

#### Taxi

| 모델 | 수량 RMSE | 수량 MAE | 시간 NLL |
| --- | --- | --- | --- |
| 기존 B | 90.5093 ± 0.5207 | 28.7545 ± 0.4526 | 0.7247 ± 0.0668 |
| 이력 보완 TitanTPP | 78.7774 ± 1.0305 | 25.3623 ± 0.3037 | 1.1822 ± 0.1372 |
| RMTPP | 94.7283 ± 3.4006 | 28.5124 ± 0.6344 | 1.5798 ± 0.9400 |
| THP | 107.5977 ± 4.5040 | 32.4095 ± 0.8144 | 0.6629 ± 0.0174 |
| NHP | 318.8148 ± 5.3011 | 94.2694 ± 1.8752 | 0.6516 ± 0.0021 |
| SAHP | 120.5273 ± 1.9996 | 34.2260 ± 0.4539 | 0.6848 ± 0.0146 |

#### Intermittent

| 모델 | 수량 RMSE | 수량 MAE | 시간 NLL |
| --- | --- | --- | --- |
| 기존 B | 1.7806 ± 0.0439 | 0.7588 ± 0.0658 | 0.3846 ± 0.1034 |
| 이력 보완 TitanTPP | 1.8140 ± 0.0854 | 0.7495 ± 0.0298 | 0.4714 ± 0.1898 |
| RMTPP | 2.5499 ± 0.1604 | 0.8949 ± 0.0494 | 0.2748 ± 0.0047 |
| THP | 2.7534 ± 0.1784 | 0.8993 ± 0.0661 | 0.2911 ± 0.0045 |
| NHP | 12.8755 ± 1.0405 | 4.5188 ± 0.1706 | 0.4613 ± 0.0969 |
| SAHP | 7.0298 ± 0.7673 | 1.8619 ± 0.0899 | 0.3448 ± 0.0138 |

#### Instacart

| 모델 | 수량 RMSE | 수량 MAE | 시간 NLL |
| --- | --- | --- | --- |
| 기존 B | 5.8796 ± 0.0166 | 3.9932 ± 0.0018 | 2.8065 ± 0.0045 |
| 이력 보완 TitanTPP | 5.8859 ± 0.0162 | 3.9926 ± 0.0013 | 2.8068 ± 0.0034 |
| RMTPP | 5.8775 ± 0.0213 | 3.9852 ± 0.0052 | 2.8051 ± 0.0048 |
| THP | 5.8582 ± 0.0253 | 4.0013 ± 0.0025 | 2.8056 ± 0.0016 |
| NHP | 6.7613 ± 0.0538 | 4.4923 ± 0.0304 | 2.8132 ± 0.0080 |
| SAHP | 5.8776 ± 0.0067 | 3.9979 ± 0.0099 | 2.8018 ± 0.0013 |

### B 대비 seed별 차이와 반례 — 완료

수량 변화율은 (local/B − 1) × 100이며 음수가 개선이다. 시간 NLL은 local − B다. 사전 기준을 사후 수정하지 않았다.

| 데이터 | seed | RMSE 변화 | MAE 변화 | 시간 NLL 차이 | 수량 RMSE 순위 |
| --- | --- | --- | --- | --- | --- |
| Taxi | 42 | -14.609% | -14.456% | +0.443485 | 1/6 |
| Taxi | 52 | -11.491% | -10.662% | +0.601540 | 1/6 |
| Taxi | 62 | -12.770% | -10.203% | +0.327648 | 1/6 |
| Intermittent | 42 | +2.102% | +11.333% | -0.191058 | 2/6 |
| Intermittent | 52 | +4.198% | -9.304% | +0.381007 | 2/6 |
| Intermittent | 62 | -0.761% | -3.987% | +0.070435 | 1/6 |
| Instacart | 42 | +0.139% | -0.001% | +0.001073 | 4/6 |
| Instacart | 52 | +0.066% | -0.020% | -0.000966 | 4/6 |
| Instacart | 62 | +0.119% | -0.025% | +0.000860 | 5/6 |

Taxi의 세 seed 모두 수량 기준은 통과했으나 시간 NLL 허용 증가 +0.01을 초과했다. 평균 시간 NLL은 B 0.724678 → local 1.182236이다. Intermittent는 seed42·52의 RMSE가 나빠졌고 seed62에서만 개선됐다. Instacart는 시간 NLL 차이는 작고 허용 범위 이내지만, 전체·tail RMSE 개선 조건은 모든 seed에서 미통과했다.

#### 사전 기준의 미통과 항목 전체

- Taxi seed42: `recorded_time_nll_guard`.
- Taxi seed52: `recorded_time_nll_guard`.
- Taxi seed62: `recorded_time_nll_guard`.
- Intermittent seed42: `raw_rmse_improves`, `raw_mae_guard`, `body_mae_guard`, `tail_mae_guard`, `middle_bin_1_qty_mae_guard`, `middle_bin_2_qty_mae_guard`, `middle_bin_3_qty_mae_guard`, `middle_bin_3_qty_rmse_guard`, `last30_rmse_mean_guard`, `last30_rmse_sd_guard`.
- Intermittent seed52: `raw_rmse_improves`, `tail_mae_guard`, `tail_rmse_improves`, `middle_bin_3_qty_mae_guard`, `middle_bin_3_qty_rmse_guard`, `recorded_time_nll_guard`.
- Intermittent seed62: `tail_mae_guard`, `tail_rmse_improves`, `middle_bin_3_qty_mae_guard`, `recorded_time_nll_guard`, `last30_rmse_sd_guard`.
- Instacart seed42: `raw_rmse_improves`, `tail_mae_guard`, `tail_rmse_improves`, `middle_bin_3_qty_mae_guard`, `middle_bin_3_qty_rmse_guard`.
- Instacart seed52: `raw_rmse_improves`, `tail_rmse_improves`.
- Instacart seed62: `raw_rmse_improves`, `tail_rmse_improves`.

기준: 전체 RMSE 및 tail RMSE strict 개선, 전체 MAE ≤ B×1.01, body/tail MAE 및 각 중간 수량 구간 MAE/RMSE ≤ B×1.02, 시간 NLL ≤ B+0.01, last30 RMSE 평균·표본 SD ≤ B×1.05. 모든 비교 모델 대비 전체 판정은 [terminal_audit.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/terminal_audit.json)의 jobs에 보존했다.

### 수량·이력 구간의 개선과 반례 — 완료

아래는 B와 local의 선택 checkpoint에 대한 세 seed 평균이다. 경계는 오른쪽 값을 포함한다. 각 표본수는 seed마다 같은 validation 대상 수이며, 세 seed를 더한 수가 아니다. 모든 모델·구간의 평균과 표본 SD는 [three_seed_summary.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/three_seed_summary.json)에 있다. 빈 구간은 측정 불가로 유지하며 0점으로 처리하지 않았다.

- Taxi: 다섯 수량 구간의 평균 RMSE가 모두 개선됐다. 그러나 이력 65–128 구간(n=1,475)의 RMSE는 2.5126 → 2.7916으로 악화됐다. 시간 NLL도 전체에서 악화됐으므로 수량 개선을 모든 과제의 개선으로 일반화할 수 없다.
- Intermittent: 수량 46 초과–187 이하 구간(n=3,405)의 RMSE는 4.3077 → 4.7665, tail 187 초과(n=1,141)는 8.6230 → 9.4091로 악화됐다. 이력 64 이하(n=25,088)도 1.4217 → 1.5840으로 악화됐다.
- Instacart: 수량 20 이하 두 구간은 평균 RMSE가 개선됐지만, 20 초과 세 구간은 악화됐다. tail 35 초과(n=6,036) RMSE는 24.4246 → 24.7895다. 이력 64 초과 구간은 관측이 없고, 추가 세부 이력 구간에서도 개선과 악화가 섞여 있다.

#### 수량 구간 전체

| 데이터 | 구간 | 표본수 | B RMSE ± SD | local RMSE ± SD | MAE B → local | 시간 NLL B → local |
| --- | --- | --- | --- | --- | --- | --- |
| Taxi | ≤ 7 | 4,364 | 12.7981 ± 1.3666 | 10.5102 ± 2.7023 | 2.6377 → 2.4792 | 1.356882 → 2.214957 |
| Taxi | (7, 686] | 3,138 | 59.0971 ± 6.0618 | 48.7841 ± 0.9026 | 26.0823 → 22.9509 | 0.022371 → 0.034625 |
| Taxi | (686, 1562] | 386 | 188.5006 ± 5.8580 | 154.4968 ± 6.2935 | 130.6082 → 110.6076 | 0.000000 → 0.000000 |
| Taxi | (1562, 3449] | 301 | 302.2241 ± 9.1653 | 272.3415 ± 5.6555 | 221.3809 → 197.9421 | 0.000000 → 0.000000 |
| Taxi | > 3449 | 79 | 431.6904 ± 19.1679 | 386.2413 ± 6.5165 | 346.0092 → 311.1526 | 0.000000 → 0.000000 |
| Intermittent | ≤ 2 | 45,036 | 0.2246 ± 0.0358 | 0.2033 ± 0.0144 | 0.1156 → 0.1070 | 0.645914 → 0.614685 |
| Intermittent | (2, 31] | 29,579 | 1.5106 ± 0.0234 | 1.4134 ± 0.0802 | 0.9628 → 0.9152 | 0.088785 → 0.241135 |
| Intermittent | (31, 46] | 7,124 | 2.7554 ± 0.5346 | 2.4861 ± 0.2520 | 1.9934 → 1.7600 | 0.182883 → 0.340016 |
| Intermittent | (46, 187] | 3,405 | 4.3077 ± 0.2186 | 4.7665 ± 0.3571 | 2.9910 → 3.5083 | 0.049534 → 0.920971 |
| Intermittent | > 187 | 1,141 | 8.6230 ± 0.6017 | 9.4091 ± 0.6078 | 6.4862 → 7.2715 | 0.000009 → 0.265107 |
| Instacart | ≤ 8 | 247,651 | 3.8310 ± 0.0811 | 3.7941 ± 0.0810 | 2.7298 → 2.7209 | 2.796248 → 2.797069 |
| Instacart | (8, 20] | 202,534 | 4.7029 ± 0.0509 | 4.6474 ± 0.0560 | 3.7065 → 3.6658 | 2.809358 → 2.808678 |
| Instacart | (20, 25] | 27,322 | 9.0693 ± 0.1305 | 9.1131 ± 0.1224 | 7.9511 → 8.0319 | 2.840424 → 2.841370 |
| Instacart | (25, 35] | 20,190 | 13.0677 ± 0.2643 | 13.2262 ± 0.1191 | 11.6992 → 11.9450 | 2.843134 → 2.845796 |
| Instacart | > 35 | 6,036 | 24.4246 ± 0.4907 | 24.7895 ± 0.1571 | 21.7567 → 22.2456 | 2.852952 → 2.855815 |

#### 관측 이력 길이 구간 전체

| 데이터 | 구간 | 표본수 | B RMSE ± SD | local RMSE ± SD | MAE B → local | 시간 NLL B → local |
| --- | --- | --- | --- | --- | --- | --- |
| Taxi | ≤ 64 | 1,241 | 0.8474 ± 0.0277 | 0.8466 ± 0.0083 | 0.5084 → 0.5200 | 2.368297 → 3.036464 |
| Taxi | (64, 128] | 1,475 | 2.5126 ± 0.0839 | 2.7916 ± 0.1625 | 1.2213 → 1.2768 | 1.079773 → 2.046370 |
| Taxi | > 128 | 5,552 | 110.4424 ± 0.6359 | 96.1224 ± 1.2567 | 42.3829 → 37.3139 | 0.262953 → 0.538198 |
| Intermittent | ≤ 64 | 25,088 | 1.4217 ± 0.0359 | 1.5840 ± 0.0733 | 0.8518 → 0.9090 | 0.157532 → 0.299525 |
| Intermittent | (64, 128] | 28,456 | 2.7153 ± 0.1228 | 2.7085 ± 0.1248 | 1.3317 → 1.2719 | 0.282653 → 0.511615 |
| Intermittent | > 128 | 32,741 | 0.6069 ± 0.1905 | 0.5748 ± 0.2589 | 0.1895 → 0.1733 | 0.647254 → 0.568191 |
| Instacart | ≤ 1 | 261 | 10.6599 ± 0.1034 | 10.6405 ± 0.1028 | 7.2796 → 7.2128 | 2.827294 → 2.824448 |
| Instacart | (1, 3] | 130,279 | 6.0118 ± 0.0259 | 6.0134 ± 0.0240 | 4.0489 → 4.0524 | 2.853580 → 2.854072 |
| Instacart | (3, 7] | 205,041 | 5.7845 ± 0.0136 | 5.7837 ± 0.0208 | 3.9493 → 3.9488 | 3.064047 → 3.065293 |
| Instacart | (7, 15] | 140,970 | 5.8911 ± 0.0479 | 5.9073 ± 0.0018 | 4.0374 → 4.0337 | 2.566378 → 2.566192 |
| Instacart | (15, 31] | 26,298 | 5.7839 ± 0.0496 | 5.8210 ± 0.0439 | 3.7863 → 3.7855 | 1.907938 → 1.902538 |
| Instacart | > 31 | 884 | 6.9249 ± 0.0821 | 6.8681 ± 0.0716 | 4.0758 → 3.9748 | 1.133006 → 1.146269 |

### 안정성과 계산 비용 — 완료

last30은 각 조건의 실제 종료 전 30개 epoch라 서로 다른 학습 위치를 비교한다. 같은 학습 구간인 first40도 함께 제시한다. 아래 값은 각 seed의 epoch별 RMSE 평균·SD를 계산한 뒤 그 통계를 세 seed 평균한 것이며, 위 선택 checkpoint의 seed 간 SD와 다르다.

| 데이터 | 모델 | first40 평균 | first40 내 SD | last30 평균 | last30 내 SD |
| --- | --- | --- | --- | --- | --- |
| Taxi | 기존 B | 120.6436 | 24.8898 | 106.9275 | 18.4128 |
| Taxi | 이력 보완 TitanTPP | 109.7630 | 24.4153 | 92.0632 | 13.6165 |
| Intermittent | 기존 B | 2.5642 | 0.9420 | 2.2992 | 0.4605 |
| Intermittent | 이력 보완 TitanTPP | 2.9188 | 0.9783 | 2.9836 | 0.8674 |
| Instacart | 기존 B | 6.0744 | 0.1049 | 6.0826 | 0.0931 |
| Instacart | 이력 보완 TitanTPP | 6.0619 | 0.0996 | 6.0634 | 0.0865 |

Taxi local의 last30 평균·변동은 B보다 낮지만 실제 학습 epoch가 훨씬 길다. Intermittent local은 seed42의 불안정성을 포함해 평균 last30 변동이 크게 나빴다. Instacart local은 first40/last30 통계가 개선됐어도 선택 checkpoint 전체 RMSE 우위로 이어지지 않았다.

| 데이터 | 파라미터 B → local | 평균 epoch 초 B → local | 평균 peak allocated MiB B → local | 실제 학습시간 local/B |
| --- | --- | --- | --- | --- |
| Taxi | 89,859 → 96,003 | 12.31 → 13.35 | 1848.3 → 1878.8 | 2.79× |
| Intermittent | 89,859 → 96,003 | 117.09 → 127.77 | 1848.6 → 1878.8 | 1.26× |
| Instacart | 77,571 → 83,715 | 206.43 → 250.41 | 404.4 → 411.5 | 1.20× |

local 파라미터는 +6,144개다. epoch당 시간은 Taxi +8.47%, Intermittent +9.12%, Instacart +21.30%였다. 실제 학습시간 비율은 조기 종료 시점 차이를 포함한다. 데이터셋 내부의 같은 서버 비교이며, 전용 성능 벤치마크나 동등 학습량·동등 용량의 인과 실험은 아니다. 학습시간은 대기·정지한 첫 시도·별도 endpoint 재평가 시간을 제외한다.

### 실제 종료·선택·학습량 전체 목록 — 완료

| 데이터 | seed | 모델 | 실제 종료 epoch | 선택 epoch | step | train/epoch | validation/replay |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Taxi | 42 | 기존 B | 86 | 46 | 25,800 | 38,393 | 8,268 |
| Taxi | 42 | 이력 보완 TitanTPP | 215 | 175 | 64,500 | 38,393 | 8,268 |
| Taxi | 42 | RMTPP | 163 | 123 | 48,900 | 38,393 | 8,268 |
| Taxi | 42 | THP | 110 | 70 | 33,000 | 38,393 | 8,268 |
| Taxi | 42 | NHP | 48 | 8 | 14,400 | 38,393 | 8,268 |
| Taxi | 42 | SAHP | 62 | 22 | 18,600 | 38,393 | 8,268 |
| Intermittent | 42 | 기존 B | 111 | 71 | 341,547 | 393,824 | 86,285 |
| Intermittent | 42 | 이력 보완 TitanTPP | 45 | 5 | 138,465 | 393,824 | 86,285 |
| Intermittent | 42 | RMTPP | 66 | 26 | 203,082 | 393,824 | 86,285 |
| Intermittent | 42 | THP | 108 | 68 | 332,316 | 393,824 | 86,285 |
| Intermittent | 42 | NHP | 42 | 2 | 129,234 | 393,824 | 86,285 |
| Intermittent | 42 | SAHP | 75 | 35 | 230,775 | 393,824 | 86,285 |
| Instacart | 42 | 기존 B | 70 | 30 | 1,088,990 | 1,991,192 | 503,733 |
| Instacart | 42 | 이력 보완 TitanTPP | 68 | 28 | 1,057,876 | 1,991,192 | 503,733 |
| Instacart | 42 | RMTPP | 46 | 6 | 715,622 | 1,991,192 | 503,733 |
| Instacart | 42 | THP | 70 | 30 | 1,088,990 | 1,991,192 | 503,733 |
| Instacart | 42 | NHP | 68 | 28 | 1,057,876 | 1,991,192 | 503,733 |
| Instacart | 42 | SAHP | 70 | 30 | 1,088,990 | 1,991,192 | 503,733 |
| Taxi | 52 | 기존 B | 59 | 19 | 17,700 | 38,393 | 8,268 |
| Taxi | 52 | 이력 보완 TitanTPP | 195 | 155 | 58,500 | 38,393 | 8,268 |
| Taxi | 52 | RMTPP | 195 | 155 | 58,500 | 38,393 | 8,268 |
| Taxi | 52 | THP | 158 | 118 | 47,400 | 38,393 | 8,268 |
| Taxi | 52 | NHP | 51 | 11 | 15,300 | 38,393 | 8,268 |
| Taxi | 52 | SAHP | 104 | 64 | 31,200 | 38,393 | 8,268 |
| Taxi | 62 | 기존 B | 67 | 27 | 20,100 | 38,393 | 8,268 |
| Taxi | 62 | 이력 보완 TitanTPP | 135 | 95 | 40,500 | 38,393 | 8,268 |
| Taxi | 62 | RMTPP | 128 | 88 | 38,400 | 38,393 | 8,268 |
| Taxi | 62 | THP | 121 | 81 | 36,300 | 38,393 | 8,268 |
| Taxi | 62 | NHP | 47 | 7 | 14,100 | 38,393 | 8,268 |
| Taxi | 62 | SAHP | 101 | 61 | 30,300 | 38,393 | 8,268 |
| Intermittent | 52 | 기존 B | 51 | 11 | 156,927 | 393,824 | 86,285 |
| Intermittent | 52 | 이력 보완 TitanTPP | 159 | 119 | 489,243 | 393,824 | 86,285 |
| Intermittent | 52 | RMTPP | 87 | 47 | 267,699 | 393,824 | 86,285 |
| Intermittent | 52 | THP | 74 | 34 | 227,698 | 393,824 | 86,285 |
| Intermittent | 52 | NHP | 67 | 27 | 206,159 | 393,824 | 86,285 |
| Intermittent | 52 | SAHP | 116 | 76 | 356,932 | 393,824 | 86,285 |
| Intermittent | 62 | 기존 B | 83 | 43 | 255,391 | 393,824 | 86,285 |
| Intermittent | 62 | 이력 보완 TitanTPP | 79 | 39 | 243,083 | 393,824 | 86,285 |
| Intermittent | 62 | RMTPP | 64 | 24 | 196,928 | 393,824 | 86,285 |
| Intermittent | 62 | THP | 64 | 24 | 196,928 | 393,824 | 86,285 |
| Intermittent | 62 | NHP | 48 | 8 | 147,696 | 393,824 | 86,285 |
| Intermittent | 62 | SAHP | 96 | 56 | 295,392 | 393,824 | 86,285 |
| Instacart | 52 | 기존 B | 55 | 15 | 855,635 | 1,991,192 | 503,733 |
| Instacart | 52 | 이력 보완 TitanTPP | 55 | 15 | 855,635 | 1,991,192 | 503,733 |
| Instacart | 52 | RMTPP | 55 | 15 | 855,635 | 1,991,192 | 503,733 |
| Instacart | 52 | THP | 51 | 11 | 793,407 | 1,991,192 | 503,733 |
| Instacart | 52 | NHP | 55 | 15 | 855,635 | 1,991,192 | 503,733 |
| Instacart | 52 | SAHP | 51 | 11 | 793,407 | 1,991,192 | 503,733 |
| Instacart | 62 | 기존 B | 75 | 35 | 1,166,775 | 1,991,192 | 503,733 |
| Instacart | 62 | 이력 보완 TitanTPP | 75 | 35 | 1,166,775 | 1,991,192 | 503,733 |
| Instacart | 62 | RMTPP | 75 | 35 | 1,166,775 | 1,991,192 | 503,733 |
| Instacart | 62 | THP | 75 | 35 | 1,166,775 | 1,991,192 | 503,733 |
| Instacart | 62 | NHP | 73 | 33 | 1,135,661 | 1,991,192 | 503,733 |
| Instacart | 62 | SAHP | 75 | 35 | 1,166,775 | 1,991,192 | 503,733 |

54조건 모두 max300/min40/patience40 규칙의 첫 허용 시점에서 조기 종료됐다. 조건별 원본 경로·source/contract SHA·초기 및 선택 상태 SHA·selected/last 지표는 [condition_results.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/condition_results.json)에 있다.

### 한계와 원본 보존 — 완료

- 이번 결론은 validation에 한정된다. held-out 평가나 새로운 seed/모델별 튜닝을 하지 않았으며, seed 세 개만으로 통계적 유의성·범용성을 주장하지 않는다.
- 기존 RMTPP seed62 첫 시도의 native 정지 원인은 미확정이다. 저장된 진척이 없어 부분 update 수는 알 수 없다. 각 모델의 새 프로세스 실행은 격리 복구이며 확정된 라이브러리 버그 수정이 아니다.
- 이전 Taxi 완료 후 Intermittent 시작 전 중단된 v2 wrapper와 Intermittent v4 복구도 출처로 보존했다. 사용한 완료 partition만을 계수했으며 이번 RMTPP 정지와 혼동하지 않는다.
- 동결 Runtime의 THP attention backward 비결정성 경고를 보존했다. 동일 checkpoint의 selected/last 재평가와 기록 지표 일치는 통과했지만, 학습 전체의 비트 단위 결정성을 새로 입증한 것은 아니다.
- B/local 간 추가 파라미터 용량 효과를 분리하지 않았다. 현재 결과는 이력 보완 구조의 독립 인과 효과를 증명하지 않는다.

### 남은 작업 순서

**승인된 학습·증적 감사·최종 비교 — 완료**
- 5080 신규24조건과 5090 원본8조건·복구4조건, 재사용 seed42 18조건을 모두 통합했다. 이번 범위에 남은 학습이나 자동 재실행은 없다.

**시간별 모니터링 종료 — 완료**
- 최종 보고서를 작성하고 heartbeat `5080-5090`을 삭제해 시간별 모니터링을 종료했다. 기존 `vnc-hard-lmm-5090-screening`은 PAUSED로 유지했으며 이 작업은 archive하지 않았다.

**후속 연구 범위 확정 — 다음 작업, 새 실행은 승인 필요**
- Taxi 수량 개선을 중심으로 논문 주장 범위를 정리할 수 있다. 범용 B 대체를 목표로 한다면 시간·tail 손실과 동일 용량/학습량 비교를 다루는 별도 설계가 필요하다. 추가 학습·held-out·Runtime/head/loss 변경은 이번 승인에 포함되지 않는다.

### 증거와 재계산

- [전체 실행 감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/terminal_audit.json), [54조건의 원본 연결 및 상세 지표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/condition_results.json), [세 seed 평균·표본 SD와 전 구간](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/three_seed_summary.json), [seed별 B 대비 차이](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/paired_seed_changes.json).
- [순수 Python 감사 코드](/Users/igwanhyeong/PycharmProjects/paper_research/reports/local_detail_3seed_final_20260927_v1/audit_titantpp_3seed_final.py): `--verify-only`로 저장 결과와 재계산 값을 대조한다. 모델·checkpoint·실제 데이터를 로드하지 않는다.
- [복구 완료 원본 사본](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/local_detail_instacart_remaining_5090_20260926_v1/monitor/20260927T130927Z/terminal_5090_recovery/collection_manifest.json), [5080 완료 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/local_detail_replication_5080_20260924_v1/monitor/20260925T204245Z/terminal_5080/terminal_audit.json).
- 모든 source root와 canonical/source SHA는 전체 실행 감사의 source_receipts에 보존했다. 원본 파일을 수정·삭제하지 않았고 커밋·Push·외부 공개를 수행하지 않았다.

<!-- ORIGINAL_REPORT_END -->

<!-- pagebreak -->

## 6. 마지막 정리: 다음 작업의 실행 순서와 완료 조건

### 현재 기준선 보존 - 완료

- 대상: paper_research의 기존 5080·5090 실험과 final_report.md.
- 54조건·108회 validation 재평가, 최종 감사와 결과 보고는 완료됐다. 이 범위의 남은 학습은 없다.
- heartbeat 5080-5090은 종료됐고 기존 screening 자동화는 PAUSED 상태로 보존됐다. 이번 문서 작성에서 원격 상태를 다시 점검하거나 자동화를 변경하지 않았다.
- 기존 결과와 실패·복구 이력을 후속 실험의 출처로 유지한다.

### 1단계. 주장과 비교표를 먼저 확정 - 다음 작업

- 대상: paper_research의 후속 연구 설계 문서.
- 이력 보완 TitanTPP를 주모델, B를 내부 대조군으로 명시한다. 수량 중심 주장과 시간 예측의 한계를 함께 적는다.
- 관련 연구를 대조하고 A1~A4의 입력·용량·경로 제거 규칙, 외부 모델 두 종류, 단순 기준선, 튜닝 예산을 정한다.
- 완료 조건: 모델·데이터·seed·head/loss·선택 규칙·예산·보고 지표가 들어 있는 검토 가능한 실험표. 36+18조건은 이 단계에서 확정하거나 근거를 붙여 조정한다.

### 2단계. 계약과 로컬 검증을 준비 - 다음 작업

- 대상: paper_research 로컬 코드와 별도 후속 실험 경로.
- 기존 동결 source와의 차이를 정리하고, 새 비교 모델 및 별도 test 평가기를 구현한다. 현재 문서 작성 범위에서 이 구현을 수행한 것은 아니다.
- 합성 데이터로 인과적 입력 차단, 초기 잔차, 파라미터 수, 경로 제거, checkpoint 저장·선택·재평가, split 누수와 집계를 검증한다.
- 문헌·외부 모델 검토와 평가기 준비는 독립 산출물이므로 병렬 진행 가능하다. 공통 head/loss·데이터·선택 계약의 변경은 직렬로 합의한다.
- 완료 조건: 로컬 검증 결과, 고정 source 묶음, 실행·평가 계약과 비용 산정 근거. 기존 held-out lock을 우회하지 않는다.

### 3단계. 새 GPU 실행 범위를 승인받고 validation 비교 수행 - 승인 필요

- 대상: 우선 후보 5080. 5090 추가 사용은 대상 서버와 작업을 명시한 별도 범위 확정이 필요하다.
- 실험표, 서버별 작업, Runtime, 저장 경로, 종료 상한과 실패 시 처리 규칙을 검토한 후 승인된 범위만 실행한다.
- 기존 실험의 2026-10-02 21:28:48 KST 종료 상한과 승인 범위가 새 실험으로 자동 연장·승계되는 것은 아니다.
- 같은 데이터의 비교는 장비 조건을 맞추고 서버당 소유 worker 수를 제한한다. 자동 retry/resume이나 다른 GPU 작업 종료는 새 승인에 포함됐다고 가정하지 않는다.
- 완료 조건: 사전 지정한 비교 결과가 실패·불리한 seed까지 모두 기록되고, 추가 용량 효과와 경로별 효과를 판단할 수 있다.

### 4단계. validation 결과로 주장과 최종 평가 대상을 동결 - 다음 작업

- 대상: paper_research의 논문 결과표와 최종 평가 manifest.
- A1~A4와 외부 비교 결과로 주장 범위를 결정한다. 분해 구조의 우위가 없으면 그 결과를 숨기지 않고 기여를 축소하거나 validation 단계에서 재설계한다.
- 모델·설정·checkpoint·split·구간·주 지표·불확실성 분석을 잠근다. 기존 체크포인트의 재사용 가능성과 test의 과거 사용 이력을 확인한다.
- 완료 조건: test 결과를 보지 않아도 평가 대상과 보고 방식이 확정돼 있고, 평가기의 검증이 통과한 상태.

### 5단계. 승인된 held-out test를 최종 실행 - 승인 필요

- 대상: 4단계에서 동결한 모델과 데이터, 새 평가 계약에 지정할 서버.
- 선택 checkpoint로 추론 평가하며 기존 54조건을 관성적으로 다시 학습하지 않는다. 추가 조건도 동일한 원칙으로 평가한다.
- test 결과로 구조·하이퍼파라미터·epoch·seed를 재선택하지 않는다. 기술적 오류에 대한 수정은 별도 사유와 영향을 기록한다.
- 완료 조건: 모든 사전 지정 조건의 전체·구간별 test 결과와 표본 수, 비용, 불확실성 및 평가 출처가 확보된다.

### 6단계. 논문과 재현 자료를 완성하고 투고 여부를 결정 - 다음 작업

- 대상: paper_research 논문 원고, 도표·부록·재현 문서, PAKDD 2027 연구 트랙 제출 자료.
- 방법의 차별성, 주 결과, 절제 비교, 시간·수량 trade-off, 실패 조건과 제한을 하나의 논리로 연결한다.
- 공식 페이지 제한·양식·익명화·제출 마감을 다시 확인한다. 핵심 근거가 준비되지 않으면 마감만을 이유로 주장 수준을 높이지 않는다.
- 완료 조건: 모든 수치와 주장이 원본 증적으로 추적되고, 독립 test와 반례가 포함된 검토 가능한 제출본. 실제 외부 투고·게시와 paper_research 저장소의 커밋·Push는 별도 요청 범위에서 처리한다.

**가장 먼저 할 일:** 추가 학습을 시작하기 전에 A1~A4, 외부 비교 모델, 공정한 튜닝 예산을 포함한 최종 실험표를 확정한다. 그다음 로컬 검증과 실행 계약을 준비하고, validation 비교가 끝난 뒤 test를 연다.
