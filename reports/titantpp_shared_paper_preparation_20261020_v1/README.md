# 공유형 MLP 논문 연구자료 준비

**최종 선택을 확정했다 — 완료**
- 5090 원본9 학습·Validation9 재검증·원본 회수를 확인했다. Intermittent 최종선택 seed42/52/62의epoch5/5/11과 canonical 가중치SHA가 앞서 Test에 사용한 고정본과 모두 같다.

**확정3seed 수치와 방법 근거를 모았다 — 완료**
- [결과표](RESULT_TABLES.md): 원본3데이터셋·공유형·외부6종, MLP16은 별도 내부대조 역할. Validation/Test 평균±표본SD와 seed차이를 JSON/CSV에도 보존했다.
- [방법 사실](METHOD_FACTS.md), [재현 자료](REPRODUCIBILITY.md), [미해결 질문](OPEN_QUESTIONS.md), [저자 작성 빈 구조](AUTHOR_OUTLINE.md)를 제공한다. 투고용 본문·초록을 자동 작성한 문서가 아니다.
- Taxi와Intermittent는 전체수량 RMSE/MAE의3seed 평균이 외부 지표별 최저보다 낮지만, RAF 전체우세 및 시간NLL의 일관된개선은 확인되지 않았다. 모든불리한결과와seed예외를 보존한다.

**추가2seed를 완료하고5seed로 재검토한다 — 진행 중**
- 5080에서 추가42개를 최대3병렬로 진행한다. 신규42종료→Validation42 재현→고정Test42→별도5seed 집계 순서다. 기존3seed 결과와 이번 연구자료는 덮어쓰지 않는다.

**사람이 본문을 작성하고 미측정 근거를 보완한다 — 다음 작업**
- PAKDD2027의 생성형AI 문법교정 제한에 맞춰 사람이 본문을 작성한다. 정확한 효율·방법론 신규성·용량대조 해석·5seed 안정성은 아직 해결할 사항이다. 측정되지 않은 속도 우세나 보편적 일반화를 주장하지 않는다.
