# Count-aware Tail-aware Loss Train-only Audit

- 판정: **CONTINUE**
- 분석 범위: train split `398,824` events only
- 수량 p90 / p95 / p99: `31` / `46` / `187`
- p95 초과 표본 비율: `4.9423%`
- p95 초과 log-MSE loss 비율: `25.8091%`
- p95 초과 absolute log-location gradient 비율: `12.2645%`

## Tail 구간

| 구간 | 표본 수 | 비율 | Log-MSE 비율 | Log gradient 비율 |
| --- | ---: | ---: | ---: | ---: |
| <= p90 (31) | 362,291 | 90.8398% | 66.1306% | 81.2904% |
| p90-p95 (31, 46] | 16,822 | 4.2179% | 8.0604% | 6.4451% |
| p95-p99 (46, 187] | 15,739 | 3.9464% | 17.3936% | 9.0643% |
| > p99 (187) | 3,972 | 0.9959% | 8.4155% | 3.2002% |

## 중복성 판정

- Q3b/Q3c는 marked TPP의 `direct_raw_qty + causal shrinkage RevIN`에 log2 Huber를 더한 실험이다. 이번 계약은 mark와 RevIN이 없는 `log1p-MSE` decoder에 capped raw Huber를 더하므로 동일 실험이 아니다.
- K=1은 log-normal NLL이 기존 log-MSE를 대체하고 shared encoder를 크게 변경했다. 이번 계약은 log-MSE를 그대로 유지하고 tail term만 보조하므로 K=1 반복 실험이 아니다.

## 고정 계약

- Tail: `q > 46` (train p95)
- Raw normalization scale: `46`
- Target/prediction cap: `187` (train p99)
- Huber delta: `1.0`
- Body sample의 보조 손실은 0이며, reduction은 전체 target event 평균이다.
- `lambda_tail`은 validation을 보지 않고 별도 train-only gradient calibration으로 고정한다.

## 결론

기존 log-MSE의 p95 초과 gradient가 사전 중단선 50%를 넘지 않았다. 따라서 tail-aware auxiliary 구현과 train-only coefficient calibration을 진행한다.
