# TitanTPP CNN+GRU54 투고 형식 작업 초안

2026-10-08 작성. [익명 LNCS 원고](main.tex)는 현재 대표 구조를 중심으로 초록, 서론, 관련 연구, 방법, 실험 설계, 결과 자리, 논의, 결론, 참고문헌을 포함한다. 기존 [MLP8 중심 LNCS 원고](../titantpp_pakdd_2027_draft/main.tex)와 [CNN+GRU54 Methods 연구 초안](../drafts/titantpp_contribution_methods_20261008_v1.md)은 보존했다.

이 원고는 **제출용 구성의 작업 초안**이다. Instacart 실험은 작성 시점 진행 중으로, 대표 구조 보정27조건과 검색27조건은 결과 대기다. 본문의 결과표에 있는 pending은 성과 수치가 아니며, 실제 원본 검증 후 저자가 채워야 한다. 이전 Test 성과와 새 Validation 결과를 섞지 않았다.

## 주장과 근거의 경계

| 원고 내용 | 상태 | 근거·다음 확인 |
|---|---|---|
| 문제와 Methods: 국소 Q/K/V 3위치 보정, GRU54 잔차, static prototype64/top4 | 동결 source·계약으로 확인 | [기존 주장·코드 근거 지도](../../reports/titantpp_contribution_writing_20261008_v1/WRITING_NOTES.md) |
| MLP8의 이력 보정 동기 | 완료된 개발 Validation | [구조54 분석표](../../reports/titantpp_structure_contribution_results_20261007_v1/VALIDATION_TABLES.md); CNN+GRU54 효과로 전용하지 않음 |
| CNN+GRU54와 mean 구현의 제한된 동등성 | 검증 완료 | [83개 bitwise check receipt](../../reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json); 과거 A100 학습 동일성은 아님 |
| Instacart 폭8/12/대표 구조9개 | 진행 중 | 최종 원본·같은 선택 epoch Validation 결과 대기. 보정·검색 추가 효과를 직접 판정하는 조건이 아님 |
| 대표 구조의 보정 추가 효과27개 | 대기 | full, disabled, current-only의 같은 dataset/seed 대조 |
| 고정 검색 추가 효과27개 및 mean9 재사용 | 대기 | mean, weighted, fixed, generic512; 성능과 양길이 비용 판정 |
| S2P2 외부 비교 | 구현의 adaptation 확인, 과거 전체 비교 정합 미완료 | [문헌·adapter 대조](../../reports/titantpp_contribution_writing_20261008_v1/LITERATURE_COMPARISON.md) |
| 새 독립 평가·SOTA 주장 | 미확인 | 미노출 자료와 접근 이력 확인, 대표 모델 및 선택 규칙 동결 후 별도 평가 |

원고의 중심 질문은 이력 조건부 인코더에 **명시적 국소·순환 보정 경로가 추가 수량 이득을 주는가**이다. CNN+GRU54는 단일 직전 state 연결 이상의 연산을 포함하므로, lag-1만의 인과 효과라고 쓰지 않았다. 보조 질문은 정적 prototype bank의 별도 정규화·고정 점수 결합이다. Instacart에서 상대적 이득이 나더라도 1위가 아니면 그 격차를 공개하고, 세 core 데이터의 대조 결과에 맞춰 주장 범위를 정한다.

## 검증과 사용 범위

- Codex 내장 LaTeX 컴파일러: 성공.
- 로컬 TeX Live 2026 LNCS v2.26: [PDF](main.pdf) **8쪽**, A4; undefined reference/citation 또는 overfull box 없음.
- 원고 구조와 수식은 [동결 source와 계약의 기존 독립 검토](../../reports/titantpp_contribution_writing_20261008_v1/REVIEW.md)에 기반했다. 새 학습·원격 조회·Test 성과 열람은 수행하지 않았다.
- 현재 [PAKDD 2027 Author's Kit](https://pakdd2027.org/pages/calls/authors-kit)는 Research Track 12쪽 상한, 익명 심사, 생성형 AI 사용 신고 및 문법 교정 용도 제한을 명시한다. **사실 확인과 원고 사용 적격성은 별개다.** 이 AI 작성 초안을 저자가 수정한다는 사실만으로 정책 충족을 인증하지 않는다. 최종 제출문 작성 방식은 저자가 규정에 맞춰 결정해야 한다.

## 남은 작업

**실험 결과를 같은 선택 epoch에 연결한다 — 진행 중**
- paper_research의 기존 Instacart→보정→검색 캠페인이 각 단계의 원본과 Validation 결과를 확인한 뒤 pending 표를 채운다. 이 문서 작업은 캠페인과 서버를 변경하지 않았다.

**주장 범위와 외부 비교를 고정한다 — 다음 작업**
- 보정 full 대 두 대조군을 먼저 판정하고, fixed 대 세 검색 대조군의 성능·비용을 별도로 판정한다. 불리한 데이터·TimeNLL·큰수량 결과를 함께 보고한다.
- S2P2 등 외부 비교는 공통 출력부 adaptation과 native 모델을 구분하고, 과거 실행의 source·학습·Runtime·선택 규칙을 재확인한다.

**독립 평가와 최종 편집을 마친다 — 외부 작업 대기**
- 새 평가 자료의 미노출 여부를 확인하고 대표 모델과 선택 규칙을 동결한 뒤 별도 평가한다. 그 이후 Results·Abstract·Conclusion을 실제 증거에 맞게 다시 쓴다.
- 저자는 문헌 범위, 익명화, 학회 AI 사용 규정, 최종 LNCS 12쪽 제한을 점검한다. 현재 8쪽은 최종 표·그림을 넣을 여유를 보장하지 않는다.
