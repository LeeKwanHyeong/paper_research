# 2026-10-01 현재 계획으로 이관

현재 기준선과 남은 작업은 [2026-10-01 기준선·작업 순서](../titantpp_baseline_reset_20261001_v1/README.md)를 따른다. Core 원본 감사와 MAC 효율 측정은 완료됐고, 추가 TPP와 RAF는 이미 실행 중이다. 아래 최초 계획의 미완료·후보 검토 서술은 당시 이력으로 보존한다. 오래된 시간별 관측과 J/Q/T 스케줄러는 사용자 요청으로 삭제됐다.

# 2026-09-30 작업 완료 반영

**원본 감사와 최종 결과표 — 완료**
- 5090 잔여 checkpoint·source 감사와 core54조건108역할의 최종 표를 확정했다. 외부 중복 제거, MAE/RMSE/시간 NLL·3seed 변동·반례·비용도 포함했다.
- 기준 보고서: `reports/titantpp_core_ablation_execution_20260928_v1/final_report.md`. 아래 최초 계획의 1번과 미완료 상태는 당시 이력이다.

**차별성과 비교 공정성 검토 — 다음 작업 / 최우선**
- 문헌·코드에서 방법 차이와 기존 TPP 비교 조건을 확인하고, 추가 비교의 필요성을 최소 범위로 판단한다.

**방법·최종 결과·효율 근거의 원고 통합 — 다음 작업**
- 확정된 MLP 방법과 결과를 연결한다. 방법 서술은 문헌 검토와 병행 가능하며, 공통 주장과 표는 같은 세션에서 정리한다.

**필요 보완과 독립 평가 — 승인 필요**
- 모델·비교군·평가 규칙을 동결한 뒤 승인된 범위에서 수행한다. 스케줄러·새 학습을 자동 시작하지 않는다.

---

# TitanTPP 논문 작업 순서 — 2026-09-30

사용자 요청: 효율 실측 내용을 Notion에 반영하고 이후 작업 순서를 재정렬한다. 이번 작업은 문서 갱신과 계획이며 추가 실험을 실행하지 않았다.

[Notion 논문 설계안](https://app.notion.com/p/3ebbbe40561381059aa0c5b73d6d47e3) · [효율 실측 보고서](https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63)

효율 기준선: 학습step Taxi5.74배/Intermittent5.58배, 평가batch8.49배/8.21배 가속. 파라미터21.51%·학습peak allocated약75.5% 감소. 평가peak allocated는389.23대97.83MiB로3.98배 증가. batch128의 동일train입력 짧은 측정이며 전체epoch·순수inference·동일정확도 효율을 입증한 것은 아니다.

## 8. 남은 작업 순서 — 효율 실측 완료 후 재정렬 (2026-09-30)
이번 요청은 결과 문서 반영과 작업 순서 정리다. 아래 후속 실험·held-out 평가를 지금 실행한다는 뜻은 아니다.

**현재 기준선과 대표 모델 — 완료**
- 대상: paper_research 및 이 논문 설계안.
- 대표 모델은 세 데이터 공통 `titantpp_history_mlp`이며, 논문에서는 TitanTPP로 표기한다. Full·B·Gate는 구성 비교/별도 탐색으로 둔다.
- 승인된 core54조건108역할과 외부TPP 추가36조건, Gate6완료조건의 학습·validation 재평가가 종료됐다. core와 외부의 중복 제거 고유 조건은90개다.
- 방법9개 수식·기호·구조 그림·코드 대응표와, MAC 대비 단발 비용 측정24개가 완료됐다. 이번 측정은 정확도 seed 추가나 전체epoch 계측이 아니다.
- 상세 효율 결과: [Notion 근거 페이지](https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63). 기존 학습·측정을 다시 실행 대상으로 올리지 않는다.

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


## 계획에 사용한 로컬 상태

- `reports/titantpp_first_analysis_20260930_v1/verification.json`: 결과 계산 검사 통과, final_core_binary_audit_complete=false.
- `search_artifacts/titantpp_core_ablation_20260928_v1/aggregation/latest.json`: 신규36/36·재사용18/18 소형 감사 완료, final_report_complete=false.
- `reports/titantpp_efficiency_5080_20260930_v1/verification.json`:24worker·118결과파일·동일Runtime/입력 검증 완료.
- `reports/titantpp_core_ablation_execution_20260928_v1/final_report.md`: 이번 확인 시 파일목록에 없음. 감사/최종보고 완료를 가정하지 않는다.
