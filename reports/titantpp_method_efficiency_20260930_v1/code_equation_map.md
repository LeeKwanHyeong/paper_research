# 수식과 동결 구현의 대응 — 확인 완료

현재 파일이 원본 core 계약의 해당 8개 source SHA와 일치함을 먼저 검사했다. 단순히 최신 코드의 동작을 과거 결과에 소급하지 않았다. 전체 97개 manifest closure를 대조했으며, 이번에는 아래 방법 관련 8개 파일의 실제 bytes를 검증했다. 5090 전체 원격 binary/source 회수 감사와는 별개다.

| 수식·설명 | 실제 구현 | 확인한 경계 |
|---|---|---|
| (1) 입력·관측 마스크 | `models/TPPs/CountAwareTitanMultiLagDetail.py`, `_encode_base` | dt/q 모두 observed 위치만 사용, log1p, projection+position |
| 마지막 관측에서 다음 사건 예측 | `paper/scripts/count_aware_tpp_backbone/core.py`, `target_outputs` | `target_positions=lengths-1`, `history_positions=lengths-2`, target write 금지 |
| (2) 두 causal block | `models/Titan/backbone.py`, `TitanBackbone.forward`; `models/Titan/common/memory.py`, `MemoryAttention.forward` | pre-LN·causal/observed mask·persistent K/V·dropout·두 residual |
| (3) predecessor·가용성 | `models/TPPs/CountAwareTitanMultiLagDetail.py`, `lag_source_indices(mode='local')` | 8개 모두 직전 관측, padding skip, withheld valid row에서 eligibility reset |
| (4) MLP 잔차 | `models/TPPs/CountAwareTitanCoreAblation.py`, `HistoryCorrection.__init__/forward` | 8×(128→4→64), bias 없음, exact GELU, 고정 /8, output zero, RNG 보존 |
| (4) 삽입 위치 | `models/TPPs/CountAwareTitanMultiLagDetail.py`, `_encode_base` | 첫 block 출력에 잔차 추가한 뒤 두 번째 block |
| (5) 정적 검색 | `models/Titan/common/memory.py`, `HardLocalMemoryMatcher.forward` | cosine top4, raw prototype 산술평균, tied K/V, residual |
| (5) 동일 head state | `models/TPPs/CountAwareTitanMultiLagDetail.py`, `encode_task_states` | time/quantity에 동일 검색 후 state 반환 |
| (6) 수량 head | `models/TPPs/CountAwareTPP.py`, `predict_quantity`, `quantity_outputs`, head 초기화 | softplus log prediction, expm1, log-MSE, 수량 분포 모델 아님 |
| (7) 시간 head | `models/TPPs/CountAwareTPP.py`, duration parameter 계산 및 time likelihood 경로 | heteroscedastic lognormal, sigma floor .001 |
| (8) 관측 시간 질량 | `models/TPPs/positive_integer_time.py`, `positive_integer_log_mass` | 첫 bin (0,1.5), 중간 폭1, Instacart top-code30 |
| (9) loss/selector | `paper/scripts/count_aware_tpp_backbone/core.py`, `target_outputs`; `training.py`, `train_one` | 시간 NLL+log 수량 MSE; 선택은 별도 raw validation RMSE |

원본 파일 경로의 기준은 저장소 루트 `/Users/igwanhyeong/PycharmProjects/paper_research`다. 정확한 byte 해시는 [verification.json](verification.json)에 있다.

## CPU 수식 검증

[audit.py](audit.py)는 동결 SHA가 일치하는 `HistoryCorrection`을 CPU에서만 호출해 논문 수식과 구현을 비교한다. 작은 합성 텐서에 대해 출력 행렬을 영 초기값과 비영 값으로 각각 설정한다. optimizer·학습·checkpoint replay는 실행하지 않는다.

- 6,144개 파라미터, 영 초기 잔차, 호출자 RNG 보존 확인.
- 명시적으로 현재/직전 표현을 concat하는 수식과 project-before-gather 구현의 일치 확인.
- padding을 건너뛰는 predecessor, withheld 뒤 가용성 reset, 여덟 branch 동일 predecessor 및 고정 8 나눗셈 확인.
- 숨겨진 위치의 NaN이 결과에 들어오지 않는지와 미래 hidden 변경이 과거 correction에 영향을 주지 않는지 확인.
- 이 검사는 correction 수식의 동치 검증이다. 전체 모델 GPU 재검증이나 인과성 새 실험으로 확대 해석하지 않는다.
