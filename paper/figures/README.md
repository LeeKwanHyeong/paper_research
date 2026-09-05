# Figures

논문용 publication figure와 figure별 source data를 둔다. 현재 v0.7 본문은
mark-free architecture F1과 데이터셋별 validation error F2를 사용한다.

- [Figure register](F1_F3_figure_register.md): 질문, caption, 파일과 생성기
- `F1_v0_7_mark_free_architecture.{svg,pdf,png}`: 공통 TitanTPP-T0 구조
- `F2_v0_7_dataset_validation_errors.{svg,pdf,png}`: RMTPP 대비 validation MAE·RMSE
- `source_data/F2_v0_7_dataset_validation_errors.csv`: F2의 frozen numeric source

```bash
python paper/scripts/generate_v0_7_mark_free_figures.py
python paper/scripts/build_v0_7_paper_artifacts.py
```

과거 F1/F2/F3 파일은 v0.6 provenance다. 현재 그림과 파일명이 겹치지 않으므로
재현 기록을 위해 보존한다.
