# Intermittent 현재 best 고정본 Validation·Test 완료

**현재 best 상태의 별도 보존 — 완료**

- 2026-10-20 14:07:56 KST에 원본5090 학습의 resume binary3개를 읽기 전용으로 복사했다. snapshot epoch42/40/38에 포함된 best epoch5/5/11을 고정했다. 이후 최신 상태나 Test 성능으로 선택을 바꾸지 않았다.
- 원본 resume binary의 best_state_dict만 읽어 평가했고, 학습 원본·optimizer·checkpoint·head·공통 조건과 원래5090 종료 후 후속 절차는 변경하지 않았다.

**고정본의 native Validation·Test와 회수 — 완료**

- source commit ec4ff74a. 로컬 합성57tests+2subtests 및 독립 구현 검토를 통과했다. 실제 고정 snapshot3개를 배포 source 경로에서 CPU tensor/metadata로 검증한 뒤 native 평가를 실행했다.
- 5080/cuDNN91900에서 qualification3 → Validation3 → 전체3개 gate → Test3 순서로 최대3개 병렬 실행했다. 원래 학습은5090/cuDNN92000이다.
- Validation3의 기록된 전체 RMSE·MAE·TimeNLL 세 지표를 고정 rtol=atol=1e-5로 재현했다. 원형 학습 중 저장되지 않은 큰수량·개별 예측의 과거 재현을 주장하지 않으며 이번 평가에서 새로 산출했다.
- 실제 Test3 종료 관측14:14:26 KST, 회수 관측14:14:28 KST다. qualification3+Validation3+Test3의 native9개와 pipeline1 실제 exit0 및 평가 소유PID/GPU부재, 기존 일시중지 CUDA문맥 보존을 확인했다. supervisor 자신의 OS exit는 직접 관측하지 않았다.
- 새 평가 archive/manifest와 원본92파일을 검증했다. 전체 원본 raw9 학습 종료·회수를 뜻하지 않는다.

**외부 비교군과의 수량 성능 비교 — 완료**

| split | 공유형 RMSE 평균±표본SD | 공유형 MAE 평균±표본SD | 외부 최우수 대비 RMSE | 외부 최우수 대비 MAE |
|---|---:|---:|---|---|
| Validation | 1.848139±0.057006 | 0.786280±0.026005 | rmtpp 대비 -0.701789 (-27.5219%) | rmtpp 대비 -0.108605 (-12.1362%) |
| Test | 2.966841±0.248890 | 1.197329±0.036735 | s2p2_matched_head 대비 -0.248514 (-7.7290%) | thp 대비 -0.114952 (-8.7597%) |

- 외부6종의 평균 중 지표별 최우수와 비교하면 이 중간 고정본은 Validation/Test의 RMSE·MAE 네 조건 모두 개선했다.
- Test seed42 RMSE는 S2P2보다 +0.008103 악화했고 seed52·62는 개선했다. Test MAE는 세 seed 모두 외부 평균 최우수 모델 THP의 같은 seed보다 낮다. 모든 seed·모델 쌍의 값과 절대·상대 차이는 JSON에 보존했다.
- 보조 대조인 MLP16 대비 Test 평균 RMSE는 +0.031111, MAE는 -0.025427다. Validation 두 지표는 MLP16보다 나쁘다. 외부 모델 대비 성능과 구조 대조의 추가 이득을 구분한다.
- 시간 성능은 수량과 다르다. 공유형 Test 전체 TimeNLL22.553775±15.114807은 THP8.326340±1.229744보다 나쁘며, 공유형 seed42/52/62는38.320870/8.188991/21.151466이다. 이 불리한 결과도 보존했다.
- 비교군 Validation21·Test21은 저장 결과만 재사용했다. 이 중 외부 Validation18은 기존 selected endpoint 집계이므로 개별 예측의 행별 재계산까지 검증됐다고 주장하지 않는다. 외부6종은 공통 head 적용 구현이다.
- 학습 종료 전 캡처한 best 상태의 탐색적 평가다. 최종 학습 선택 모델, 새로운 독립 평가, 통계적 유의성·인과관계·모든 seed 우세를 뜻하지 않는다.

**남은 작업 — 다음 작업**

- 기존5090 학습과 최종 Validation·회수 절차를 유지한다. 최종 best 선택이 이번 고정본과 같은지 종료 뒤 별도로 확인한다.
- 추가2seed 확장과 원고 작성은 이 평가 결과 확인과 구분한다. 이번 실행에는 새 학습·반복 자동화·기존 CNN 변경이 없다.
