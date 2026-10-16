# 5090 Intermittent 완료8개 선택 모델의 부분 Test 비교

## 승인 범위와 현재 기준선

사용자 원문: “우선 5090에서 이미 학습한 모델들에 대해 Test 결과까지 미리 뽑아놓고 그걸로 먼저 분석하자.”

대상은 Intermittent seed42·52의 NONE·CNN·GRU·FULL, 고유8개다. 새 학습은 0개다. 원36개 계약 및 전체 Validation gate를 변경하지 않으며 seed62를 2seed 집계에 섞지 않는다. 기존 chronological Test의 접근 이력을 유지하고 새 독립 평가로 표현하지 않는다.

## 선택과 지표를 고정한다 — 완료

전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch 선택 binary를 사용한다. binary·실제 tensor·과학 source131·Train 통계·전처리를 고정했다. 종료 binary8개도 감사용 원본으로 보존한다. 주 지표는 전체 raw 수량 RMSE, 보조 지표는 전체 MAE, q>187 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL이다. 동일 checkpoint에서 여섯 지표를 계산한다. 직전 수량 반복과 관측 이력 평균은 같은 Test 대상으로 비교하며 TimeNLL은 N/A다.

## 저장 모델과 실제 입력을 감사한다 — 완료

120개 원본 파일과 archive 로컬 SHA를 검증하고 네이티브5090의 선택 tensor8개를 재검증했다. Runtime은 Python3.12.13 / torch2.11.0+cu130 / CUDA13.0 / cuDNN92000, 고정 GPU UUID다. Validation86,285건·Test88,019건의 대상·truth·recorded gap 지문을 고정했다. 이 단계의 모델 예측은 0개다.

첫 입력 감사는 source root의 이미 동결된 빈 sample_data 표식 누락으로 checkpoint·자료 접근 전 실패했다. 첫 claim·stderr·exit1을 보존했다. 원 source131을 변경하지 않고 parent input manifest의 빈 파일 SHA를 대조해 보충했다. 별도 attempt2 입력 감사는 실제 exit0을 확인했다. 이는 수치 학습 실패나 실제 Test 재실행이 아니다.

## Validation8을 재현한 뒤 Test8을 실행한다 — 완료

native5090에서 8개를 순차 평가한다. 모든 Validation 재현의 여섯 지표·대상 지문·실제 worker exit0이 확인돼야 별도 gate8을 생성하고 Test로 넘어간다. 저장 파라미터와 평가 전후 tensor 불변을 확인한다. 평가 claim·failure·원본은 삭제하지 않으며 자동 평가 retry는 0회다. 평가 도구에는 부모 중단·launch 영수증 저장 실패 시 자기 소유 자식만 종료하고 실제 exit를 남기는 검증을 추가했다.

## Test의 2seed 결과와 한계를 정리한다 — 완료

각 arm의 평균·표본SD와 seed별 FULL 대비 차이를 함께 제시한다. 전체·큰 수량과 시간 지표의 불리한 결과도 보존한다. 두 seed 비교이므로 3seed 최종 판정과 통계 유의성·외부 우월성을 주장하지 않는다. Intermittent62는 5080 학습 완료 후 별도 최종 계약으로 연결한다. 현재 main.tex는 변경하지 않는다.

## 자료 출처

Material Passport: local user-approved experiment originals and frozen dataset; source revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`; source131 closure `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`; partial contract `432b300eab9025fc4a0b7c5670c0e8ba3f64c51eda50811d678f5363b9307615`. 원본과 추론은 같은5090 Runtime으로 확인한다. source와 input bytes는 변경하지 않는다.

최종 분석: [VALIDATION_TEST_ANALYSIS.md](VALIDATION_TEST_ANALYSIS.md). Test 완료 KST 2026-10-14T08:47:45.885628+09:00; 로컬 원본·결과 SHA와 독립 QA PASS. 전체36 학습 완료와 3seed 최종 집계는 이8개 완료와 구분한다.
