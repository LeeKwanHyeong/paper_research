# TitanTPP 논문 Structure 및 표·그림 계획 (August 14 초안)

Notion page: https://app.notion.com/p/3b4bbe40561381d1be1ecf83819ea5f5

## Decision

Paragraph-level outline보다 manuscript structure를 먼저 고정한다. 이 페이지는 각 장의 역할, 핵심 주장, 필요한 실험, 표·그림의 증거 계약을 정하는 Structure v0.1이다. 문단별 outline과 실제 영문 초안은 이 구조를 기준으로 작성한다.

## Draft target

- 초안 기준일: 2026-08-14
- 평가 원칙: 현재 단계는 validation-only이며 held-out test는 설정 고정 후 한 번만 평가한다.
- 주장 원칙: “TitanTPP가 모든 데이터셋과 지표에서 항상 우월하다”는 주장은 사용하지 않는다.

## Working research question

간헐 수요를 다음 사건의 발생 시점과 수요량이 함께 결정되는 marked temporal point process로 표현할 때, 긴 사건 이력과 연속적이며 heavy-tailed한 수량을 기존 RMTPP보다 안정적으로 모델링할 수 있는가?

## Problem-to-solution chain

1. 간헐 수요는 일정 간격의 일반 시계열이라기보다, 수요가 발생한 사건 시점과 발생량을 함께 예측해야 하는 문제다.
2. RMTPP는 과거 사건을 recurrent hidden state에 순차적으로 압축하므로 긴 이력의 정보 보존과 병렬 처리에 제약이 있다.
3. 수요량은 범주형 event type과 달리 연속적이고 꼬리가 길다. 수량 전체를 categorical mark로만 다루면 크기 정보를 잃고, raw quantity를 직접 회귀하면 규모별 오차와 극단값 때문에 학습이 불안정해질 수 있다.
4. 본 연구는 수량을 로그 스케일의 mark와 구간 내 residual로 분해하고, Titan 계열 인코더와 quantity-aware objective를 결합한다.
5. V3b는 mark-conditioned quantity experts와 분리된 gradient path를 이용해 mark 분류와 수량 회귀의 학습 충돌을 줄인다.
6. 효과는 matched RMTPP와 THP, 데이터셋별 ablation, sequence-length 및 quantity-scale breakdown으로 검증한다.

## Core claims

- C1. 연속적 수요량을 단일 categorical mark나 raw regression만으로 다루는 데 생기는 한계를 mark-residual decomposition으로 완화한다.
- C2. TitanTPP는 장기 사건 이력을 표현하는 encoder와 수량 오차를 직접 반영하는 objective를 결합한다.
- C3. V3b는 Taxi에서 mark와 quantity objective 간 간섭을 줄여 marker 및 quantity 성능을 함께 개선한다.
- C4. matched validation evidence에서 개선의 크기와 위치는 데이터셋별로 다르며, 현재 가장 강한 효과는 Taxi의 marker NLL과 quantity MAE에서 관찰된다.
- 금지할 과장: 모든 metric, 모든 seed, 모든 dataset에서의 보편적 우월성.

## Manuscript structure

The original Notion page proposed a longer manuscript structure:

1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. TitanTPP Methodology
6. Experimental Setup
7. Results
8. Discussion
9. Conclusion
10. Appendix / Supplement

After professor feedback, this was compressed into the 4-page version mirrored in [04_four_page_structure_revision.md](04_four_page_structure_revision.md).

## Tables

- T1 Dataset statistics and task construction
- T2 Matched model and training configuration
- T3 Main matched comparison across three datasets
- T4 Quantity formulation and V2/V3b ablation
- T5 History-length and quantity-scale breakdown
- T6 Efficiency and convergence
- A1 Full hyperparameter and experiment contract
- A2 Per-seed results

Current local files:

- `paper/tables/T1_dataset_statistics.md`
- `paper/tables/T2_model_training_contract.md`
- `paper/results/e300_matched_20260808/tables/preliminary_summary.md`

## Figures

- F1 Intermittent demand as a marked event sequence
- F2 RMTPP-to-TitanTPP architecture and V2/V3b heads
- F3 Why quantity transformation is needed
- F4 Main matched performance comparison
- F5 Long-history and quantity-scale breakdown
- F6 Mark-quantity trade-off and V3b effect
- F7 Convergence and NLL decomposition

Current local files:

- `paper/figures/F1_F3_figure_register.md`
- `paper/results/e300_matched_20260808/figures/validation_nll_comparison.png`
- `paper/results/e300_matched_20260808/figures/quantity_mae_comparison.png`
- `paper/results/e300_matched_20260808/figures/paper_applicability_matrix.png`

## August 14 minimum output

1. T1 and T2 first.
2. F1-F3 while training continues.
3. T3 after e300 18-run completion.
4. Fresh strict Titan execution for final fair comparison.
5. T4-T6 and F4-F7 after results are stable.
6. Do not leave empty tables in the August 14 draft.
7. Open held-out test only once after configuration freeze.
