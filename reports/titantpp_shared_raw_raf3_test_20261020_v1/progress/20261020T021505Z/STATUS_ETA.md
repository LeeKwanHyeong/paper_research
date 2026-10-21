# 5090 원본 학습과 RAF Test 시작 예상

관측: 2026-10-20T11:15:04.999691+09:00

완료4·진행3·대기2·실패0·미확정0. RAF3 및 Taxi42 성공 종료, Taxi52/62·Intermittent42 실제worker/GPU신원 확인. 원본 Validation9와 RAF Test는 아직0개다.

| 데이터 | seed | 상태 | 저장/best | best 미갱신 조건 종료epoch |
|---|---:|---|---:|---:|
| raf_spare_parts | 42 | complete | 49/9 | — |
| raf_spare_parts | 52 | complete | 45/5 | — |
| raf_spare_parts | 62 | complete | 59/19 | — |
| yellow_trip_hourly | 42 | complete | 152/112 | — |
| yellow_trip_hourly | 52 | running | 160/148 | 188 |
| yellow_trip_hourly | 62 | running | 154/127 | 167 |
| intermittent_frozen_5000 | 42 | running | 1/1 | 41 |
| intermittent_frozen_5000 | 52 | queued | None/None | — |
| intermittent_frozen_5000 | 62 | queued | None/None | — |

## 조건부 학습 ETA

실행 중3개 fit의epoch_timing.json/timing.json이 존재하지 않고 train.log에도epoch별elapsed가없다. 최근10완료epoch의중앙값기반ETA는계산불가다. 아래Taxi값은동일fit의10:24:54→11:15:04 실제관측사이121완료epoch/3010.11초의구간평균(약24.88초/epoch) 참고값이다. 부하변화와epoch경계오차가있어최근epoch중앙값과구분한다.

| seed | best 미갱신·구간속도 유지 시 학습종료 | 300epoch 시나리오 |
|---|---|---|
| Taxi52 | 2026-10-20T11:26:41.554195+09:00 | 2026-10-20T12:13:07.772209+09:00 |
| Taxi62 | 2026-10-20T11:20:28.399996+09:00 | 2026-10-20T12:15:37.033888+09:00 |

Intermittent42는 저장1·best1로, best가 갱신되지 않으면 41epoch에서 종료해 40epoch가 남는다. 최근 완료 epoch 소요 시간이 없고 seed52/62는 미시작이다. 다른 fit 속도를 빌리지 않으므로 Intermittent 전체 종료 및 RAF Test 시작 시각은 미확정이다.

학습9종료→Validation9→회수/SHA·로컬비교·finisher종료→새5090유휴/합성qualification3→RAFValidation3gate→Test3 순서다. 각후속단계의실측시간도아직없어임의분/시간을붙이지않는다. best가갱신되면종료예상도뒤로이동한다. 기존기한연장은없다.

학습/코드변경/추론0. 상태SSH1회와누락timing확인읽기SSH1회만수행했으며, 로컬저장형식오류는보존된transport원문에서복구했고추가원격재시도는없다.
