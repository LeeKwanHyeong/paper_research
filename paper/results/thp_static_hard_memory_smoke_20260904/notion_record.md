# Notion Result Readback

Page: https://www.notion.so/3d1bbe40561381f996b4c9677cfb4446

Parent verified by fetch: 졸업 논문 / 프로젝트 / 5. Model Design Enhancement

Title: 2026-09-04 Count-aware THP Static Hard Memory CUDA and Full e1

## 2026-09-04
### 상태
완료. 5080 CUDA 계약 테스트와 Taxi·RAF 전체 데이터 e1 검증을 통과했다. 실행 시각은 2026-09-04 19:49:53\~19:50:14 KST이며, 로컬 artifact 재검증도 완료했다. e300 screening은 시작하지 않았고 별도 승인이 필요하다.
### 목적
기존 THP encoder에 원래 Hard-LMM의 정적 prototype retrieval 하나를 추가한 대조 구조가 CUDA와 실제 데이터에서 정상 동작하는지 확인한다. 이번 단계는 정확도 개선 판정이 아니라 실행 가능성 검증이다.
### Variant 계약
Count-aware THP + Static Hard Memory: 마지막 THP 층 이후 cosine top-4로 고른 raw prototype 평균을 residual로 더한다. Quantity와 time head가 같은 결합 표현을 사용한다. Persistent token과 online memory update는 없다. 기존 THP보다 parameter가 4,096개 추가된다.
### 고정 조건
Taxi·RAF, seed 42, 각각 전체 train·validation 1 epoch, batch 128, lr 0.001, hidden 64. Context는 Taxi 168/256, RAF 84/84. Direct log1p quantity MSE, legacy_clamped_rmtpp, validation joint objective checkpoint 선택을 유지한다. Held-out test는 사용하지 않는다. Artifact: `search_artifacts/count_aware_thp_static_memory_cuda_e1_20260904`.
### 실행 명령어
5080의 격리 snapshot에서 `paper/scripts/run_thp_static_hard_memory_smoke.py`를 실행한다. CUDA 계약 테스트 통과 후 Taxi, RAF 순으로 별도 프로세스에서 e1을 수행하고 저장된 checkpoint와 결과를 검증한다.
### 결과
CUDA 계약 테스트 35개가 모두 통과했다. Taxi는 train 38,393개·validation 8,268개, RAF는 train 25,779개·validation 6,690개 target을 빠짐없이 처리했다. 수치·checkpoint·optimizer가 finite이고, 저장·복원 및 quantity/history 구간별 집계도 일치했다. Held-out test는 열지 않았다.
<table header-row="true">
<tr>
<td>Dataset</td>
<td>e1 Quantity MAE</td>
<td>e1 Quantity RMSE</td>
<td>e1 Time NLL</td>
<td>e1 Joint objective</td>
</tr>
<tr>
<td>Taxi</td>
<td>59.732959</td>
<td>229.053097</td>
<td>1.367272</td>
<td>1.593199</td>
</tr>
<tr>
<td>RAF</td>
<td>9.430184</td>
<td>38.125877</td>
<td>3.298434</td>
<td>3.882918</td>
</tr>
</table>
이 표는 1 epoch 동작 확인용이며, 기존 e300 baseline 대비 성능 우위나 후보 채택을 판단하는 결과가 아니다. 다음 단계는 별도 승인 후 Taxi·RAF seed42 fresh e300 screening이며 e1 가중치를 이어 쓰지 않는다.
CUDA attention backward의 비결정성 경고가 있어 전체 학습의 비트 단위 재현성은 주장하지 않는다. OOM·Xid·실제 학습 오류는 없었고 종료 후 CUDA 프로세스는 없다. 모델·loss·time head·판정 기준과 시스템 서비스는 변경하지 않았다.

