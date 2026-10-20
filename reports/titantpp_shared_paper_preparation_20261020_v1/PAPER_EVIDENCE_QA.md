# Review Result: PASS

확정3seed 연구용 수치표의 원본 연결과 계산을 독립 검산했다. 열린 CRITICAL·HIGH·MEDIUM·LOW 문제는 없다.

## CRITICAL

- 없음.

## HIGH

- 없음.

## MEDIUM

- 없음.

## LOW

- 없음.

## 완료 조건

- [x] 8모델 ×3데이터셋 ×3seed ×Val/Test =144개 고유 결과를 확인했다. MLP16은 내부 대조로 분리되어 있다.
- [x] 108개 결과는 원본 JSON과 정확히 일치한다. 나머지36개 외부 Validation 결과는 선택된 summary와 최초 유한 RMSE 최솟값의 history 기록에 일치한다.
- [x] 72개 Val/Test 쌍의 선택 epoch가 같다. Inter seed42/52/62는 원본 receipt의 state SHA와 epoch5/5/11이 각각 일치한다.
- [x] 288개 평균·표본SD(ddof1),648개 전체 차이,216개 주요 지표 CSV 차이,12개 외부 최저 평균 비교를 독립 계산했다.
- [x] RAF·Taxi 외부 Validation 큰수량 지표36칸은 N/A다. 부분 seed 평균이나0 대체는 없다.
- [x] CSV와 JSON이 일치하고 Markdown288칸의 표기가 정확하다. 미소 비영 평균10칸은 과학적 표기를 유지한다.
- [x] 표 source binding244개를 전후 검증했다. E1/E2 완료 receipt의 결과 파일 SHA, 원본524개(205+92+227), SOURCE_VERIFICATION의251개 연결도 직접 대조했다.
- [x] 봉인 source987과 계약이 일치한다. 코드 핵심 Ruff 검사와 독립 검산 프로세스의 실제 종료 코드는0이다.

## 검토 범위

`build_evidence_tables.py`, `RESULT_TABLES.json/.md/.csv`, `SEED_DIFFERENCES.csv`, `SOURCE_VERIFICATION.json`을 검토했다. 각 파일 SHA와 원본 완료 증적 SHA는 동명의 JSON 보고서에 보존했다.

검토자는 원본 코드·기존 결과·봉인 파일을 수정하지 않았다. 생성 파일은 `PAPER_EVIDENCE_QA.json`과 `PAPER_EVIDENCE_QA.md`다.

## 검증의 한계

저장된 결과의 재집계 검산이며, 예측 parquet에서 오차를 새로 산출하거나 모델을 실행하지 않았다. 체크포인트는 파일 SHA만 읽었고 역직렬화하지 않았다. 최종 Inter 상태 일치는 원본 Validation/Test receipt에서도 직접 확인했다. SSH·모델 구축·새 추론은 모두0회다.

일부 부동소수점 뺄셈에는1e-12 절대 허용오차를 적용했다. 미소 TimeNLL과 표기 검사는 별도로 유지했다. 추가72/82는 포함하지 않았으며, 이 보고서는5seed 완료나 통계적 유의성을 주장하지 않는다.

## 다음 작업

- 검증된3seed 수치표를 연구자료로 사용한다. 추가42 완료 후에는 별도 비교기로5seed 근거를 검증한다.
