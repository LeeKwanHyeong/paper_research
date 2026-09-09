SELECT
  model || ' · F' || substr(scope, 6, 1) AS variant_fold,
  model AS variant,
  CAST(substr(scope, 6, 1) AS INTEGER) AS fold,
  CASE history WHEN '2-3' THEN 'H2–3' WHEN '8-15' THEN 'H8–15' END AS history_slice,
  CASE history WHEN '2-3' THEN 0 ELSE 1 END AS history_order,
  CAST(count AS INTEGER) AS target_count,
  CAST(mae AS REAL) AS mae,
  CAST(mse AS REAL) AS mse,
  CAST(rmse AS REAL) AS rmse,
  CAST(bias AS REAL) AS bias,
  CAST(centered_mse AS REAL) AS centered_mse,
  CAST(delta_mse_vs_B AS REAL) AS delta_mse_vs_B,
  CAST(delta_mse_vs_residual_off AS REAL) AS delta_mse_vs_residual_off
FROM history_metrics
WHERE model IN ('residual_off', 'q_only', 'k_only', 'v_only', 'full_qkv')
  AND scope IN ('fold_0', 'fold_1')
  AND history IN ('2-3', '8-15')
ORDER BY
  CASE model
    WHEN 'residual_off' THEN 0
    WHEN 'q_only' THEN 1
    WHEN 'k_only' THEN 2
    WHEN 'v_only' THEN 3
    ELSE 4
  END,
  fold,
  history_order;
