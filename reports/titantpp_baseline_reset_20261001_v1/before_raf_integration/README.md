# TitanTPP 현재 기준선과 작업 순서 — 2026-10-01

**원고 통합 반영(2026-10-01T08:26:52.972206+09:00)**: [방법·관련 연구·실험 설정·효율 통합 초안](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_history_mlp_manuscript_20261001_v1.md)을 작성하고 원본 수치·수식·인용을 대조했다. 학습·원격 상태를 새로 조회하지 않았으며 아래 서버 관측시각은 유지한다.

**후속 완료 반영(2026-10-01T08:03:05.327180+09:00)**: 5080 추가 TPP 12조건과 Instacart 정규화 1조건의 원본 회수·CPU 감사를 마쳤다. [감사보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_5080_audit_20261001_v1/report.md). 서버 진척은 아래 07:35 관측 이력으로 유지한다. 감사 완료 시각을 새로운 진척 관측 시각으로 표시하지 않는다.

최초 요청: 불필요한 스케줄러를 삭제하고, 현재 증거를 기준으로 남은 작업을 재정렬한다. 서버 상태는 **2026-10-01 07:35 KST 한 번의 읽기 전용 조회** 기준이다. 학습·재평가·튜닝을 새로 실행하지 않았다. 원본 감사의 후속 완료는 위에 반영했다. 방법·관련 연구·실험 설정·효율의 영문 통합 초안은 후속 작업에서 완료했다. 진행 중 두 캠페인의 최종 감사와 전체 결과표·독립 평가는 남아 있다.

**자동 관측을 정리한다 — 완료**

- 앱 도구로 `titantpp-tpp`(TitanTPP·추가 TPP 시간별 비교)와 `5090-j-q-t`(종료된 J/Q/T 학습 점검)를 삭제했다. 전자는 오래된 범위와 승인 확인 대기 상태를 반복했고, 후자는 2026-09-11 종료 증거가 있다. 삭제 전 설정은 `automation_before/`에 보존했다.
- 남은 자동화 8개는 모두 PAUSED다. 다른 채팅 소속 7개와 기존에 PAUSED 유지가 명시된 `vnc-hard-lmm-5090-screening`을 유지했다. **활성 스케줄러는 0개다.**
- 학습은 각 서버의 자체 실행기·내부 lease·절대시간 guard가 담당한다. 관측 스케줄러 삭제로 진행 중인 학습을 종료하지 않았다. 이후 상태 조회는 사용자 요청 시 수행한다. 새 정기 관측은 이번 요청에서 생성하지 않았다.
- 과거 자동 관측의 승인 확인 대기는 스케줄러 삭제로 더 이상 재개 대상이 아니다. 이번 단발 조회는 현재 기준선 확인 요청에 따른 것이며 시간별 조회 재승인으로 간주하지 않는다.

**대표 모델·평가 방식·논문 주장을 고정한다 — 현재 기준선**

- 대표 모델은 세 기존 데이터 공통 **`titantpp_history_mlp`**, 논문 표기는 **TitanTPP**다. 활성 분기 정규화는 탐색 변형, Full/B/수준·변화/정적 검색 제거는 구성 비교, Gate는 별도 seed42 탐색이다. 데이터마다 유리한 변형을 대표 모델로 갈아 끼우지 않는다.
- checkpoint는 **최초 strict 최소 validation raw 수량 RMSE**로 선택한다. 그 동일 checkpoint의 MAE·RMSE·시간 NLL을 함께 보고한다. RMSE 선택의 목적은 큰 절대 수량 오차에 더 큰 비중을 두는 것이며, 모든 데이터가 대량 수요라는 가정이나 RMSE가 모든 목적에 최적이라는 주장은 하지 않는다. 학습 손실·시간 NLL과 checkpoint 선택 기준은 구분한다.
- 입력·split·공통 head/loss·optimizer·학습 상한·선택 규칙은 각 동결 계약을 유지한다. 외부 6개 비교군은 RMTPP·THP·NHP·SAHP·S2P2·AttNHP이며 공통 head에 연결한 비교다. 원논문 전체 모델의 최적 튜닝 순위로 일반화하지 않는다.
- validation만 사용한다. seed42·52·62가 모두 끝난 묶음만 평균±표본표준편차로 집계한다. 표본표준편차와 같은 seed 승패를 통계적 유의성·동등성 증명으로 표현하지 않는다. held-out 성능은 아직 잠겨 있다.

| 주장 | 현재 확보한 근거 | 주장 범위와 남은 검증 |
|---|---|---|
| 사건 간격·수량 이력을 이용하는 가벼운 TitanTPP 구조 | 인과적 encoder 사이의 저차원 MLP 잔차, 수식·코드 대응·초기화 검증 완료 | 온라인 신경 메모리 갱신은 생략하지만 정적 검색/학습 bank는 유지한다. 8분기는 같은 직전 사건을 읽으며 서로 다른 8개 지연이 아니다. continuous mark의 최초 도입을 주장하지 않는다. |
| 다음 사건 수량 예측의 경쟁력 | Taxi·Intermittent에서 외부 6개 모델과 같은 seed 36쌍 모두 MAE·RMSE가 낮음 | validation 공통 head 조건의 결과다. Instacart 우위·시간 NLL의 일관된 우위는 미확보. RAF와 추가 Instacart 비교는 진행 중이다. |
| Titans-MAC 대비 계산 비용 절감 | 같은 5080·입력의 학습 step 5.58–5.74배, 평가 batch 8.21–8.49배 빠름; 학습 peak allocated 약75.5% 감소 | 평가 peak allocated는 약3.98배 증가한다. 짧은 batch 측정을 전체 epoch·순수 inference·동일 정확도 효율로 확대하지 않는다. |

**완료와 진행 상태를 분리한다 — 현재 기준선**

| 대상 | 07:35 KST 기준 상태 | 남은 일 |
|---|---|---|
| Core54조건·108 selected/last 역할 | 학습·재평가·최종 checkpoint/source 감사 완료 | 재사용한다. core5090 감사를 다시 잔여 작업으로 올리지 않는다. |
| 기존 외부4모델36조건·Gate6조건 | 기존 완료 증거 재사용 | 주 비교·구성 비교·단일seed 탐색을 구분한다. |
| 추가 TPP / 5080 | Taxi·Intermittent 12/12조건과24역할 완료; 기존 terminal 증거 재사용 | 후속 13조건 감사에서 회수·CPU 감사 완료; 재사용 |
| 추가 TPP / 5090 | Instacart 3/6조건 terminal 완료. AttNHP seed52는51epoch/선택11 저장, 소유worker·GPU 존재, terminal 미완료. seed62 두 조건 대기 | 남은 기존 큐 완료 후 원본 회수·최종 감사 |
| 정규화 / 기존3데이터 seed42 | Taxi·Intermittent는4090 완료. Instacart는 승인된5080 복구로70epoch/선택30 및selected/last 재평가 완료 | Instacart 로컬 binary/source 최종 감사 완료; 재사용. 기존3데이터의seed52·62 여섯 조건은 사용자 보류 유지 |
| RAF / 5080 | 8모델×3seed=24조건 중3조건 terminal 완료: plain MLP42(49epoch/선택9), 정규화42(49/9), RMTPP42(70/30). THP42 실행 중 | 나머지21조건에는 현재 THP를 포함한다. 전체종료·원본 회수·감사·3seed 비교 필요 |
| RunPod | 이전 캠페인 원본 회수·CPU 감사·비용 보고·소유Pod 삭제 완료 | 새 임대 보류. 개인5080의 후속 Instacart 성공으로 이전 RunPod 중단 이력을 덮지 않는다. |
| 방법·효율 자료 | 수식·그림·코드 대응표·직접 비용 측정 및 영문 통합 초안 완료 | 미완료 캠페인 종결 후 최종 결과·독립 평가·초록/결론 반영 |

진행 중인 두 서버에서 이번 조회의 새 실패는 없었다. RAF 원래 수동 중단과 fit 전 운영 guard 정정은 보존된 과거 이력이다. 추가 TPP의 전체 마감은 **10월5일12:16 KST**, RAF는 원래 시작 기준 **10월2일09:05:42 KST**다. 이 시각은 ETA가 아니라 실행 상한이다. 이번에 예산·마감을 바꾸지 않았다.

기존 전체 비교 보고서는 **07:07 KST 기준114개 고유 완료조건**을 담고 있으며 RAF3조건은 포함하지 않는다. 본 문서는 운영 기준선을07:35로 갱신한 것이고 기존 보고서의 관측시각이나 수치를 소급 변경하지 않았다.

**새 완료 결과의 원본 감사를 확정한다 — 완료 / 현재 기준선**

- 대상: 5080 추가 TPP 12조건과 Instacart 정규화 1조건. 원본 466파일 SHA, 캠페인별 source 106개·104개, 최종 checkpoint 26개와 복구 checkpoint 1개의 CPU 감사를 통과했다.
- 전체 노출·공통 배치 prefix·optimizer/RNG·최초 RMSE 선택·기존 selected/last 26역할을 검증했다. 기존 비교표의 13조건 수치는 변경 없다. [최종감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_5080_audit_20261001_v1/report.md).
- Core와 효율 측정은 완료 증거를 재사용한다. 5080 RAF와 5090 추가 TPP의 남은 실행·최종 감사는 별도다.

**방법·관련 연구·실험 설정을 한 원고로 연결한다 — 완료 / 현재 기준선**

- 대상: paper_research의 [새 영문 원고](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_history_mlp_manuscript_20261001_v1.md). 기존 v0.7 원고를 보존하고 TitanTPP 이력 MLP 한 구조를 대표로 작성했다.
- 식9개·기호표·기존 그림2개·표6개·1차 문헌9개를 연결했다. 회수된 동결 방법 source8파일 SHA, 수식, seed별 집계, 실측 반복별 시간/메모리를 검증했다. [주장–근거표와 작성 검토](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/claim_evidence_and_writing_notes.md).
- 온라인 memory 갱신 생략과 정적 bank 유지, 공통 head 적응, RMSE 선택, 시간 NLL·평가 메모리 반례를 본문에 명시했다. continuous mark 최초성이나 보편적 우월성을 주장하지 않는다.
- 완료 범위는 통합 초안이다. 최종 네 데이터 결과표·초록·결론·held-out 검증·제출본 확정은 아래 작업에 남긴다.

**1. 승인된 RAF와 Instacart 추가 비교군 큐를 마치고 감사한다 — 진행 중 / 서버 작업 대기**

- 대상: 기존 5080 RAF24조건과5090 Instacart S2P2·AttNHP6조건. 서로 독립적인 서버 실행을 유지하며, 문장 검토는 병행할 수 있다.
- 등록된 조건과 종료 규칙을 유지한다. 완료 조건 재학습, 유리한 모델·seed만 선택하는 제외, 자동 retry·resume은 수행하지 않는다. 종료 후 실패·미완료까지 보존해 원본 회수·CPU 감사를 한다.
- 기존3데이터의 정규화seed52·62 여섯 조건은 사용자 보류다. RAF의3seed는 별도 승인된24조건에 포함된다. 현재 원고 작업은 새 학습이나 정기 관측의 승인이 아니다.
- 완료 조건: 캠페인별 terminal 상태·모든 조건 목록·원본 SHA·checkpoint/source 감사 결과 확보. 이번 원고 통합에서는 서버 상태를 새로 조회하지 않았다.

**2. 네 데이터의 결과표와 최종 주장–근거표를 확정한다 — 다음 작업 / 1번 감사 이후**

- 대상: 기존3데이터+RAF와 현재 영문 원고. 주 결과는 TitanTPP MLP 대 외부6모델, core는 구성 비교, 정규화·Gate는 별도 탐색으로 구성한다.
- 동일 checkpoint의 MAE·RMSE·시간 NLL,3seed 변동, selected/last, 수량 구간·이력 길이 반례, 실측 비용과 복구 이력을 연결한다. 결과가 불리하다는 이유로 RAF나 모델을 제외하지 않는다.
- 이번 통합 원고의 표는 완료된 기존 증거에 한정된다. 최종 전체 표로 확장할 때 cutoff와 감사 수준을 갱신하고, 통계적 유의성이나 확정 원인으로 과장하지 않는다.
- 완료 조건: 전체 조건의 비교와 주장 범위가 확정됨. 추가 모델·seed·ablation을 자동으로 실행하는 단계가 아니다.

**3. 최종 평가 규칙을 동결하고 독립 평가한다 — 이후 작업 / 실행·열람은 승인 필요**

- 대상: 고정된 대표·비교군 checkpoint와 잠긴 held-out split. 모델/지표/집계/불확실성 방법을 먼저 고정하고 split·과거 열람 이력에서 독립성의 범위를 확인한다.
- 결과를 본 뒤 대표나 checkpoint를 바꾸지 않는다. 현재 validation 개발 결과와 원본 무결성 감사는 독립 평가를 대체하지 않는다.
- 완료 조건: 구체적 계약 승인 후 고정한 전체 범위의 평가·반례·불확실성 보고.

**4. PAKDD 제출본을 편집한다 — 이후 작업 / 실제 제출은 승인 필요**

- 대상: paper_research 원고·그림·표·재현 부록. 독립 평가를 반영해 초록·최종 결론을 작성하고 제출 당시 공식 분량·익명성·필수항목을 확인한다.
- 로컬 사용자 경로를 익명화하고 상세 구현·추적표를 부록으로 정리한다. Intermittent 상류 데이터의 출처·배포권과 각 데이터의 구성 한계도 명시한다.
- 완료 조건: 검토 가능한 제출본. 이번 작업은 외부 게시·제출, paper_research의 커밋·Push, 공용 Runtime 변경을 포함하지 않는다.

**다음 단계**는 기존 두 캠페인의 종료 후 원본 회수·감사이며, 그 전에는 현재 영문 원고를 검토할 수 있다. 최종 결과표 → 독립 평가 → 초록·결론·제출본의 순서를 유지한다. 새 스케줄러·학습·sub-agent는 만들지 않았다.

**확인 근거**

- [운영 상태·계약·출처 SHA](baseline.json) · [자동화 삭제 증거](automation_deletion_receipt.json) · [확인 결과](verification.json)
- [기존 전체 비교](../titantpp_completed_external_comparison_20261001_v1/report.md)
- [core 최종 감사](../titantpp_core_ablation_execution_20260928_v1/verification.json)
- [방법 수식](../titantpp_method_efficiency_20260930_v1/method_draft.md) · [직접 효율 측정](../titantpp_efficiency_5080_20260930_v1/report.md)
