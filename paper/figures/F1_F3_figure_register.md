# TitanTPP figure register

> Status: v0.7 validation freeze. 현재 본문 그림은 F1과 F2다. 두 그림 모두
> held-out prediction을 사용하지 않는다.

## F1 v0.7. Mark-free TitanTPP architecture

**Purpose.** 공통 continuous-count event interface와 Backbone 비교에서 고정한
TitanTPP-T0 구조를 설명한다.

**Caption.** Mark-free Count-aware TitanTPP. Each observed event token contains the
log-transformed inter-event time and quantity. Two causal memory-attention layers
combine learned persistent tokens with static Hard-LMM top-4 prototype retrieval to form
a shared history state. Common clamped RMTPP time-score and direct log-count heads predict the next
event. RMTPP and THP replace only the history encoder under the matched comparison.

**Files.** `F1_v0_7_mark_free_architecture.{svg,pdf,png}`

**Generator.** `paper/scripts/generate_v0_7_mark_free_figures.py`

## F2 v0.7. Dataset-level validation errors

**Analytical question.** 같은 T0 구조의 상대 quantity error가 세 데이터셋에서
어떻게 달라지는가?

**Chart contract.** 각 데이터셋의 3-seed mean MAE와 RMSE를 해당 RMTPP mean으로
나눈 두 panel grouped bar chart다. 비율 1은 RMTPP 기준선이며 1보다 낮으면 비교
encoder의 평균 오차가 작다. Sample standard deviation은 T3에서, paired-seed
일관성은 T4에서 별도로 보고한다.

**Caption.** Three-seed validation quantity MAE and RMSE relative to RMTPP within each
dataset. TitanTPP produces a large reduction relative to recurrent RMTPP on Intermittent-5000,
a small mean-RMSE reduction with higher MAE on Taxi, and no reduction on Instacart.
Absolute values and sample standard
deviations appear in T3; paired-seed outcomes appear in T4.

**Files.** `F2_v0_7_dataset_validation_errors.{svg,pdf,png}` and
`source_data/F2_v0_7_dataset_validation_errors.csv`

**Generator.** `paper/scripts/build_v0_7_paper_artifacts.py`

## Rebuild

```bash
python paper/scripts/generate_v0_7_mark_free_figures.py
python paper/scripts/build_v0_7_paper_artifacts.py
```

## Archived v0.6 figures

`F1_problem_formulation.*`, `F2_titantpp_architecture*` and
`F3_quantity_sequence_distributions.*` describe or support the former
magnitude-mark/residual revision. They remain as provenance and are not current v0.7
manuscript figures.
