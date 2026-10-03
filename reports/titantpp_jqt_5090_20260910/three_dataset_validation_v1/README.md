# TitanTPP B: 세 데이터셋 J/Q/T 통합 검증

- 분석·독립 검토 완료: [통합 해석](validation_report.md), [전체 수치](validation_results.json), [검증 기록](validation_receipt.json).
- 로컬 대화형 보고서: [미리보기](http://127.0.0.1:4173/). 데이터셋별 최종 표·학습 곡선·clipping과 출처 검사를 제공한다.
- 앱 수치·화면 검증: [수치 기록](app_data_verification.json), [화면 기록](app_verification.json).
- 기존 최종 JSON과 frozen source만 사용했다. seed 42 validation이며 새 학습·모델 평가·held-out 평가는 실행하지 않았다.

미리보기 서버가 종료된 경우 로컬에서 다시 열 수 있다.

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/three_dataset_validation_v1/report_app/dist
```

수치 재계산은 `python3 build_report.py`로 수행한다. 이 명령은 원본 snapshot으로 `validation_results.json`을 추출하며, 분석 문장이나 앱을 자동으로 다시 작성하지 않는다.
