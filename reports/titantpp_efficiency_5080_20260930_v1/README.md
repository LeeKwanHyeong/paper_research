# TitanTPP 효율 근거자료 — 완료

[Notion 실측 보고서](https://app.notion.com/p/3ebbbe405613819b8977c607bd93ea63)

[보고서](report.md) · [기계판독 분석](analysis.json) · [무결성 검증](verification.json) · [반복별 CSV](repetitions.csv) · [원시 batch 시간](per_batch.csv) · [입력 표본](inputs.csv)

[논문용 PDF 그림](paired_efficiency.pdf) · [SVG](paired_efficiency.svg) · [PNG](paired_efficiency.png)

현재 기준선: 5080 단독24개 측정 완료, 준비 포함248.386초. 동일seed42 시간반복3회. train/eval 각각 예열5+측정32batch, optimizer444step/eval444batch. 모델별 성능 평가나 전체epoch 측정은 하지 않았다.

실행 원본은 `search_artifacts/titantpp_efficiency_5080_20260930_v1/`에 있으며 117개 source SHA와118개 결과 파일 SHA 검증을 통과했다. 기존 scientific 결과와 별도다.

재분석: `/usr/local/bin/python3 reports/titantpp_efficiency_5080_20260930_v1/analyze.py`
그림: `/usr/local/bin/python3 reports/titantpp_efficiency_5080_20260930_v1/draw.py`

**남은 작업 — 다음 작업**
- 이 결과의 범위를 논문 효율 절에 반영한다. 속도/학습peak 감소와 평가peak 증가를 함께 보고한다.
- 실제 초/epoch가 필요할 경우에만 별도 전량 계측 범위를 산정한다. 추가 GPU 실행은 이번 범위에 포함하지 않는다. 스케줄러를 새로 두지 않는다.
