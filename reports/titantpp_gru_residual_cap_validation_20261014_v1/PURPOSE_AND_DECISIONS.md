# GRU 보정량 제한 실험의 목적과 판단 기록

## 목적 — 진행 중

이 실험은 기존 구성요소 대조에서 CNN+GRU의 Validation 우세가 Test에서 안정적으로
이어지지 않았다는 관찰 뒤에 시작한 탐색 후속이다. 여기서 확인하려는 가설은 **GRU
보정 출력의 크기가 과도해지는 현상이 이 불안정성에 기여할 수 있다**는 것이다.

따라서 `titantpp_cnn_gru54`의 GRU 보정만 `rho=0.25`로 제한한
`titantpp_cnn_gru54_residual_cap025`를 새로 학습한다. 기준 CNN과 기존 FULL의
checkpoint·결과는 보존하며, 새 후보가 그 결과를 교체하지 않는다.

## 고정한 결정 — 완료

- 상한은 `rho=0.25` 하나이며, 상한값 탐색은 하지 않는다.
- 손실, 수량·시간 head, 파라미터 수, seed, 입력 split, 선택 규칙, 최소·최대 epoch와
  학습 예산은 원래 계약을 유지한다.
- 후보 9개는 fresh fit이다. 기존 CNN·FULL checkpoint를 재개하거나 warm start하지
  않는다.
- 비교는 데이터셋별 3 seed의 Validation 결과에서 후보·FULL·CNN을 나란히 둔다.
  여섯 지표는 선택된 같은 checkpoint에서 계산하고, 후보−FULL 및 후보−CNN의
  seed별 차이와 3 seed 평균·표본 SD를 남긴다.
- 실행 전에 각 host의 GPU UUID, Python·Torch·cuDNN Runtime, 고정 입력 SHA를
  확인한다. 검증이 실패하면 학습을 시작하지 않는다.

## 해석 경계 — 완료

후보와 FULL의 차이는 GRU 보정 크기 제한 하나지만, 제한은 최적화 경로도 바꾼다.
따라서 개선이 관찰되어도 과적합, 입력 분포 변화, GRU 용량 효과를 모두 분리했다고
결론내릴 수 없다. CNN은 FULL보다 활성 파라미터가 적으므로, 후보와 CNN의 차이도
보정량 제한 하나의 인과효과로 해석하지 않는다.

이번 단계는 Validation 전용이다. 결과가 좋아도 Test 일반화 개선이나 기존
Validation→Test 순위 반전의 원인을 입증하지 않는다. Test를 보고 checkpoint를
재선택하거나 이 계약에 Test를 추가하지 않는다.

## 사용자 승인과 다음 순서 — 진행 중

사용자는 다음 범위를 명시적으로 승인했다.

> 고정 계약과 기존 CNN·FULL 결과는 유지하고, GPU·Runtime·입력 SHA 검증을
> 통과하면 GRU 보정량 제한 후보 9개를 학습하자. 끝나면 3seed Validation 결과를
> 기존 CNN·FULL과 비교해서 정리해줘.

1. 5080에서 Taxi 3개, RAF 3개, Intermittent seed62 1개와 5090에서 Intermittent
   seed42·52 2개의 실행 자격을 검증한다.
2. 검증을 통과한 경우에만 후보 9개를 시작하고, 기존 학습·claim·실패 기록은
   변경하지 않는다.
3. 각 후보의 terminal manifest, 선택 checkpoint, 저장 history, Validation replay를
   SHA로 결속한 뒤에만 3 seed 비교를 생성한다.

아직 후보 학습과 비교 결과는 완료되지 않았다.
