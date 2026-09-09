# VNC-Hard-LMM Instacart seed42 screening

상태: **후보 탈락**. VNC-Hard-LMM은 RTX 5090의 Instacart validation screening에서 실행·감사 계약은 통과했지만, 사전에 고정한 성능 gate 중 raw RMSE를 충족하지 못했다.

학습은 최대 300 epoch 계약으로 시작해 epoch 112에서 조기 종료됐다. validation raw RMSE의 가장 이른 엄격한 최솟값인 epoch 72를 선택했다. Held-out test는 사용하지 않았다.

## B 대비 결과

| 지표 | VNC | B | VNC - B | 판정 |
|---|---:|---:|---:|---|
| Raw RMSE | 5.8743577925 | 5.8722169305 | +0.0021408620 | 실패: B보다 엄격히 작지 않음 |
| Overall MAE | 3.9933433509 | 3.9937810783 | -0.0004377274 | 통과 |
| Body MAE | 3.4405897817 | 3.4386015032 | +0.0019882786 | 통과 |
| `>p99` MAE | 21.8170765719 | 21.9177947653 | -0.1007181934 | 통과 |
| Legacy Time NLL | 3.2153963533 | 3.2155228955 | -0.0001265422 | 통과 |

MAE, tail MAE, Time NLL은 B와 동등하거나 소폭 개선됐고 body MAE도 허용 범위 안이었다. 그러나 raw RMSE가 B보다 `0.0021408620`, 약 `0.0365%` 높았다. 이 차이가 작더라도 결과 전에 고정한 strict gate를 변경하지 않는다.

## 감사 결과

[Strict CUDA audit](jobs/seed42_screening_insta_market_basket/audit_strict.json)는 다음을 확인했다.

- 5090 CUDA에서 전체 validation `503,733`개 target을 독립 재생했다.
- 보고 지표와 재계산 지표가 일치했다.
- source·data identity, checkpoint, optimizer, RNG, full train 처리 건수와 earliest raw-RMSE selector가 계약과 일치했다.
- 모델 state는 validation 재생 전후 동일했고 gradient가 생성되지 않았다.
- `evaluation_scope=validation_only`, `held_out_test_evaluated=false`였다.

Audit code는 결과 전에 commit `22727ad591d383b5a7d07189b0e6d0e94b98f65d`로 동결했다. Audit JSON의 SHA-256은 `d4c22a616caf0114bc264df2fe8bce1c7967357f95b74b5e1a0d6787ed883989`이다.

## 후속 판정

Stage 1 gate가 실패했으므로 VNC용 normalized-duration fit, Taxi·Intermittent seed42, seed52·62와 held-out 평가는 실행하지 않는다. Machine-readable 결론은 [decision.json](decision.json)에 기록했다. 대용량 checkpoint는 5090 artifact에 보존하고 Git 증적에서는 제외했다.
