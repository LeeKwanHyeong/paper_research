# 루프4 Q/K-only CNN 결과와 다음 진단

**고정 checkpoint의 평가를 마쳤다 — 완료**
- 학습 종료·원본 회수 관측: 2026-10-16 04:09:05.824525 KST.
- native Validation3 통과 후 Test3 완료: 2026-10-16 04:10:35.583264 KST.
- seed42/52/62 선택 epoch는 5/5/22, 종료 epoch는 45/45/62다. 기존 CNN은 저장 결과만 재사용했다.

| split | 모델 | 전체 raw RMSE, 평균±표본SD | 전체 MAE, 평균±표본SD |
|---|---|---:|---:|
| Validation | 기존 CNN | 34.431408±0.073201 | 9.218271±0.028002 |
| Validation | Q/K-only | 34.475385±0.083828 | 9.231898±0.025188 |
| Test | 기존 CNN | 39.600663±0.023654 | 9.972138±0.036428 |
| Test | Q/K-only | 39.535394±0.064265 | 10.011760±0.067976 |

**전체 성능 기준을 충족하지 못했다 — 완료**
- 기존 CNN과 같은 대상 외부 지표별 최우수를 모두 넘어야 하는 네 조건 중 Test RMSE만 통과했다.
- 외부 기준은 Validation RMSE 33.994704(S2P2 matched-head), MAE 9.136205(RMTPP), Test RMSE 39.772935(S2P2 matched-head), MAE 9.906347(RMTPP)다.
- Validation RMSE는 모든 seed에서 악화했다. Test RMSE는 seed52에서 악화했고, Test MAE는 seed42만 개선했다. 큰수량·시간 지표의 평균 이득과 seed별 악화 모두 평가 보고서에 보존했다.
- Taxi/Intermittent로 확장하지 않는다. checkpoint를 바꿔 결과를 대체하지 않는다.

**현재 모델의 Q/K 이력 보정 영향을 분리해 확인한다 — 진행 중**
- 수량 구간별 Validation SSE의 손익 방향이 seed마다 달라, 특정 이력 패턴이나 다음 필터 구조를 바로 원인·해결책으로 확정할 근거가 없다.
- 별도 `titantpp_backbone_qk_branch_probe_20261016_v1`에서 같은 선택 모델의 Q/K 지역 잔차만 함께 켬/끔으로 비교하는 진단을 구현·검증 중이다. 기존 개별 예측 행이 없으므로 on Validation 재현과 on Train 추출 후 off를 실행한다.
- Test는 추가로 읽지 않고 Train/Validation의 같은 대상, 고정 15개 이력 조건·Train 경계·공통 개체 가중으로 분석한다. 새 학습과 기존 CNN 추론은 없다.
- 같은 모델의 분기 민감도이므로 Q와 K 각각의 효과, 분기 없이 재학습한 효과, 원인이나 일반화 개선으로 해석하지 않는다. 다음 학습 구조는 진단 근거를 검토한 뒤 결정한다.

**증적과 한계**
- 학습 receipt SHA: `e16e983b571a271a8bc86abe82136023e863e0d8cfdc7caa15fe9d8a1bbf059e`.
- 평가 receipt SHA: `f3c6823b69a5ad8f0e3faea727828f291563d88dba0082a4724a2fd263e83f1a`.
- 평가 287파일 SHA, worker6/pipeline2 실제 exit0, Validation→Test 순서·선택 불변 및 242개 독립 산술 대조를 확인했다.
- 기존 CNN 5080/cuDNN91900 대 후보 5090/cuDNN92000, local CNN 계수576→384의 용량 감소와 공동 최적화 변화가 있다. Test를 참고한 탐색적 반복이며 독립 평가·통계유의성을 주장하지 않는다.
- 평가 완료 cache는 현재 GPU 전역 유휴를 증명하지 않는다. wrapper 자신의 OS exit도 직접 관측하지 않았다.
