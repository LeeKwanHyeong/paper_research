# Historical external panel restoration: numeric evidence

Status: **completed**. Scope: saved **Validation-only** aggregate records; no training, inference, SSH, prediction-row access, or held-out/mixed performance access.

## Verified panel

- Recomputed 28 dataset/model groups from 84 distinct selected runs (seeds 42, 52, 62).
- All 168 displayed mean/SD numbers match the earlier manuscript to four decimals. SD uses `ddof=1`.
- The same selected endpoint supplies MAE, RMSE, and TimeNLL for each run. This reuses prior endpoint/binary audits; it is not a new binary or prediction audit.
- The representative historical arm is `titantpp_history_mlp`: **MLP4**, bottleneck width 4, eight branches, 6,144 correction parameters. It is distinct from MLP8 and CNN+GRU54.
- Taxi and Intermittent use `completed_external_comparison/comparison.json`; RAF uses its final `comparison.json`; Instacart uses the final `analysis.json` selected endpoints. The earlier combined comparison was incomplete for Instacart seed 62 and is not used for that dataset.

## Source integrity

The main three aggregate sources match the registered integration manifest exactly. All per-seed referenced artifact hashes are recorded in JSON; registered expected hashes and the reused audit receipts were checked separately. No original artifact is overwritten.

## Results that must be preserved

| Dataset | Historical MLP4 vs comparator | Quantity interpretation |
|---|---|---|
| Taxi | RMTPP | MAE reduction 10.131391%; RMSE reduction 15.852939%; RMSE seed wins 3/3 |
| Intermittent | RMTPP | MAE reduction 21.719216%; RMSE reduction 34.388979%; RMSE seed wins 3/3 |
| RAF | S2P2 | MAE reduction 0.241414%; RMSE reduction 0.129504%; RMSE seed wins 2/3 |
| Instacart | S2P2 | MAE reduction 0.178065%; RMSE reduction -0.520970%; RMSE seed wins 0/3 |

- Negative reductions mean the historical MLP4 result is worse.
- External timing scores are lower in several datasets; no quantity-only blanket superiority claim is supported.
- Instacart S2P2 and AttNHP have lower RMSE in all three paired seeds; RAF MLP4 vs S2P2 RMSE wins are 2/3.

## Simple quantity rules

All eight deterministic rows match both the validation-only native aggregate JSON and the preparation comparison. They are single deterministic results, not three seeds. On Intermittent, last quantity MAE (0.5647) is better than MLP4 (0.7005); MLP4 RMSE improves by 2.035874%. No duration NLL is supplied for these rules.

## Scope of reuse

The restored panel establishes the archived MLP4 variant’s position among matched numerical-input/common-head encoders. It cannot stand in for the new CNN+GRU54 result, establish native-model dominance, or certify independent evaluation. The parent contract audit covers alignment with current campaigns.

## Files

- `panel_evidence.json`: all 84 per-seed records, 28 groups, hashes, paired seed comparisons, and eight deterministic rules.
- `panel_table.tex`: the complete external panel and simple-rule table, labels `tab:external-legacy` and `tab:external-simple`.
