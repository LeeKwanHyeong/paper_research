# 전체 관측 회수·검증 완료

총 1,920,436건을 확인했다. 기존 best Validation 172,570건, 실패한 v4 dispatcher 안에서 정상 종료·독립 검산된 seed42 best Train 393,824건, 승인된 별도 기준의 v6 5080 관측 480,109건과 v8 5090 관측 873,933건이다. v4 전체 실행 및 v6 5090 qualification의 실패 판정과 PARTIAL_COLLECTION은 보존한다. v7 관측은 포함하지 않는다.

| seed | checkpoint | epoch | split | rows | source |
| --- | --- | ---: | --- | ---: | --- |
| 42 | best | 1 | train | 393,824 | v4_completed_child_of_failed_dispatcher |
| 42 | best | 1 | validation | 86,285 | historical_validation_reuse |
| 42 | last | 41 | train | 393,824 | execution_v6 |
| 42 | last | 41 | validation | 86,285 | execution_v6 |
| 52 | best | 5 | train | 393,824 | execution_v8 |
| 52 | best | 5 | validation | 86,285 | historical_validation_reuse |
| 52 | last | 45 | train | 393,824 | execution_v8 |
| 52 | last | 45 | validation | 86,285 | execution_v8 |

재사용 Validation의 μ·σ는 float32 export, NLL은 원래 float64다. 신규 기록의 μ·σ·NLL은 native float64다. 재사용 자료에 없는 native log-MSE·학습 손실·head preactivation·hidden은 채우지 않았다. 전체 회수 완료는 Prototype의 추가 설명력이나 시간 손실 악화 원인이 확인됐다는 뜻이 아니다. 정량 해석은 다음 분석 단계에 남긴다.

관측 Parquet 434개 134,948,477 bytes와 best Train hidden NPZ 194개 187,080,460 bytes를 합하면 **322,028,937 bytes (307.11 MiB)**다. hidden은 seed42·52 best Train 각393,824건, 합계 787,648건이다. archive·소스 사본·로그·manifest 중복은 이 용량에 포함하지 않는다.
