# 오류 진단을 먼저 끝내는 최소 추가 실험 계획

2026-10-12, paper_research. **계획만 작성했다.** 저장 결과·입력 메타데이터 분석은 완료이고 GPU 추론·새 학습은 실행하지 않았다. 기존 Test36에 대한 승인을 새 모델·계측·자료 평가로 확장하지 않는다. 프로젝트 [세션 규칙0.1](../../TEST_SESSION_PROTOCOL.md)은 “계획 요청에는 계획만 작성한다”와 “로컬 문서 검토나 모델 설정 점검은 GPU 실험 실행을 포함하지 않는다”고 정한다.

## 확인된 오류 위치와 보존할 기준선을 정한다 — 완료

- 대상: 이력 보정27의 원래 full/보정 제거/current-only 선택 checkpoint, 최초 strict 최소 Validation raw 수량 RMSE selector, 동결 source129·입력·초기화·loss·optimizer.
- Taxi52는 큰 수량 과소예측, Intermittent42는 주로2<q≤31의 과대예측, RAF는 모든 seed의 큰 수량 과소예측이 확인됐다. 좋은 seed와 불리한 지표도 포함한다. 전처리·Runtime·선택 오류는 현재 기록에서 발견되지 않았다.
- 기존 Test는 탐색적 원인 진단의 기준선이며 새 독립 평가가 아니다. 현재 논문 main.tex는 Validation 기준선으로 남아 있고 Test 결과는 아직 통합하지 않았다.
- 완료 근거: [통합 진단](README.md), 수량18쌍·원본 학습27개·frozen loader target digest 검증. 원본·checkpoint는 수정하지 않는다.

## 1. 동결 모델의 오류와 보정 출력을 함께 계측한다 — 다음 작업 / 평가 실행 승인 필요

- 목적: 같은 실제 입력에서 어떤 series·이력·시간 조건의 손해가 CNN·GRU 출력과 함께 발생하는지 좁힌다. 이 단계로 원인을 모두 확정하거나 성능을 개선했다고 주장하지 않는다.
- 대상: Taxi·Intermittent·RAF×seed42/52/62×full/제거/current-only = **27checkpoint**, 각 Validation/Test를 확인하는 **54 split pass**. 신규 fit0개. Instacart는 이번 seed 반전 진단 범위 밖이며 완료된9개를 다시 실행 대상으로 넣지 않는다.
- 실행 대상 후보: **5080의 기존 native Runtime, GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`**. 별도의 새 격리 root·계측 계약을 고정한다. 기존 Test root·과학 source·Runtime·checkpoint·원본은 보존한다. 5090·Runpod의 새 작업은 요청하지 않는다. 현재 유휴·건강은 실행 전 확인할 사항이며 이 계획에서 원격 조회하지 않았다.
- 계약: 원래 selected ID·checkpoint/data/source SHA와 입력 population/truth/history digest, 계측 코드의 별도 봉인, prefix-only 입력, 출력 항목, 허용 수치 오차·finite 검사·중단·회수 범위를 먼저 고정한다. model.eval·gradient0·optimizer0, 저장 가중치 SHA 불변을 확인한다. native Validation 전체 재현을 먼저 통과한 뒤 같은 계측의 기존 Test를 진단용으로 평가한다. Test로 epoch를 다시 선택하지 않는다.
- 계측: 수량 head 출력과 signed error/SSE/SAE, recorded duration과 시간 head 위치·scale·NLL, 입력에서 알 수 있는 이력 길이·cap/truncation·직전/평균 수량, series·시점별 충분통계를 연결한다. full9에는 CNN Q/K/V residual·GRU correction의 norm, 공통 hidden 대비 비율과 최종 head 변화 관계를 남긴다. norm은 크기만 보여주므로 부호·효과를 동일시하지 않는다.
- 구간: Train만으로 고정한 입력 이력·gap 구간과 기존5개 실제 수량 구간을 사용한다. 실제 수량으로 나눈 구간은 사후 오류 설명용이다. 추론 시점에는 알 수 없는 true quantity로 gate를 만들지 않는다.
- 집중 확인: Taxi 큰 수량4series·83건의 입력 상황; Intermittent2<q≤31와 truncated3,655건·긴 gap의 교차; RAF의 Validation 공유/비공유 series·입력 이력≤14/>14와 큰 수량39건의 교차. 전체 지표와 나머지 대상도 남겨 작은 subset만 보고하지 않는다. small-N 교차구간에서는 count와 합을 우선하고 유의성 증명으로 확대하지 않는다.
- 직전 수량 반복·이력 평균의 기존 전체 지표는 기준선으로 유지한다. 필요하면 같은 계측 구간에서 결정적 기준선 충분통계를 계산하되 모델 fit은 없고, 실행 수·split 수를 별도로 기록한다.
- 완료 조건: 모든 target/truth/history digest와 원래 전체 지표가 사전 허용 오차로 일치하고 계측 과정의 저장 가중치 변화가0이다. 각 손해가 특정 series·input stratum·시간 head 또는 보정 출력에 집중되는지, 좋은 seed와 어떻게 다른지 요약한다. 불일치·nonfinite이면 멈추고 원인을 기록하며 자동 retry하지 않는다.
- 수치 재현·입력 계약 검토와 문헌 검토는 독립 파일로 병렬 가능하다. 계측 계약·공통 runner 구현·실행은 하나의 순서로 진행한다. 승인 전에는 runner 구현·GPU 실행을 진행하지 않는다.

## 2. 남는 구조 가설에만 저장 모델의 경로 우회를 적용한다 — 조건부 다음 작업 / 평가 실행 승인 필요

- 선행 조건: 단계1의 재현·계측이 통과하고 어느 경로의 출력 의존성인지 여전히 구분할 필요가 있을 때만 진행한다. 단계1로 충분한 한계 설명이 가능하면 생략한다.
- full9에 CNN만 우회/GRU만 우회 = **18추론 설정**, Validation/Test 모두 읽으면 **36추가 split pass**, 신규 fit0. full 정상 출력은 단계1의 결과를 사용한다. 저장 가중치를 새로 학습하거나 replacement 값을 맞추지 않는다.
- 해석: 결합 상태로 학습한 모델을 바꾼 출력 의존성·분포 밖 개입 진단이다. independently trained CNN-only/GRU-only 성과와 다르며, 우회 시 개선됐다고 구성요소가 항상 불필요하다고 단정하지 않는다.
- 완료 조건: 원래 모델 정상 경로 재현과 개입 식별이 통과하고 양대조·좋은 seed·전체 및 큰 수량/시간 손해를 함께 보고한다. 원래 최종 Test 표를 교체하지 않는다.

## 3. 구성요소별 기여를 논문 주장으로 유지할 경우 RAF에서 독립 학습한다 — 조건부 다음 작업 / 새 학습 승인 필요

- 목적: 진짜 CNN-only·GRU-only·둘 다·둘 다 제거를 학습해, 결합 구조의 이득과 상충을 비교한다. RAF는 이미 Test에서 고른 실패 사례이므로 사후 선택한 탐색적 구조 연구라고 공개한다. RAF만으로 Taxi·Intermittent까지의 기여를 증명하지 않는다.
- 최소 구성: CNN on/off×GRU on/off의 **4arm×3seed(42/52/62)=12fresh fit**. 같은5080 GPU/Runtime에서 전부 새로 학습한다. 과거 A100/RTX full·제거6조건을 재사용해6신규만으로 동일환경 비교를 완성했다고 쓰지 않는다. 기존 결과는 과거 기준선으로만 둔다.
- 정확한 대조: CNN-only에는 inherited MLP16을 넣지 않는다. 기존 `titantpp_cnn_mlp16`은 CNN+MLP16이다. 네 arm 모두 동일 모듈 등록·구성 RNG 순서를 유지하고 inactive 경로를 동결·우회해 공통 파라미터 초기값·shuffle가 같게 한다. arm 식별 buffer가 달라질 수 있으므로 전체 state hash 동일을 요구하지 말고 공통 tensor 초기화를 검증한다.
- 예상 RAF 활성 수는 제거78,851 / CNN-only79,427 / GRU-only103,583 / full104,159다. 이는 frozen RAF의 sequence cap84와 CNN576·GRU24,732에 근거한 계획값이며 실제 새 runner native prelaunch 검증으로 확정해야 한다. Taxi의115,167 full을 RAF의 파라미터 수로 복사하지 않는다. 구조별 용량이 달라 equal-parameter 연구라고 부르지 않는다.
- 유지 계약: 기존 RAF Train/Validation·position cap84·data/head/loss/optimizer/LR·selector, B128·min40/max300/patience40. 최소40은 종료 하한이며 선택 epoch 하한이 아니다. 새 fit은 fresh이고 자동 retry/resume0이다. 미사용 경로에 gradient나 optimizer 활성 값이 없음을 검사한다.
- 운영 계약: 같은 seed의4arm을 같은 물리 GPU·Runtime에서 수행한다. 최대3동시 조건을 유지하면3+1로 나뉘므로 concurrency 차이를 기록하고 wall time을 구조 효율 근거로 쓰지 않는다. fit36시간/host168시간 등 lease·전체 예산·회수·실패 중단 기준을 새 계약에 명시하고 native 자격 검사 후 실행한다. 현재 실행 시간·비용 ETA는 측정하지 않았으며 기존 다른 workload 값을 복사하지 않는다.
- 선택·평가: 새 구조의 Validation 결과로만 checkpoint를 고른다. 전체/큰 수량 RMSE·MAE·TimeNLL과 각 seed·3seed 평균/표본SD를 함께 보고한다. 이미 본 Test는 구조 탐색 결과로만 읽으며 설계의 독립 일반화 증거로 부르지 않는다. Test가 좋은 seed·epoch만 남기지 않는다.
- 완료 조건:12조건 모두 공통 초기화·입력·운영 계약과 실제 종료·원본 SHA 검증을 통과한다. 두 단일 경로와 결합의 이득·손해를 모두 남기고, 구조별 활성 용량 차이와 재학습에 따른 backbone 적응을 공개한다. 이것만으로 용량·정보·최적화 중 하나를 인과 원인으로 완전히 분리하지 않는다.
- 확대는 선택 사항: 두 다른 자료에서도 구성요소 기여가 논문 핵심이면 Taxi·Intermittent의4arm×3seed×2자료 = **24fresh 추가**를 별도 판단한다. 초기 권고12개에 자동으로 더하지 않는다. 반대로 단계1이 특정 loss/head 결함을 가리키면 이 factorial을 자동 실행하지 말고, 그 단일 가설에 맞는 더 작은 대조 계약을 다시 작성한다. LR/loss/seed/search sweep은 실행하지 않는다.

## 4. 최종 구조를 동결한 뒤 새 자료에서 확인한다 — 다음 작업 / 자료·평가 계약 승인 필요

- 선행 조건: 선택한 구조 작업 완료 또는 구조 변경 생략을 결정하고, 모델·전처리·selector·비교군·여섯 지표를 고정한다. 기존 Test로 설계를 바꾸면 해당 Test는 후속 개발 자료다.
- 모델을 유지한다면 기존 full/제거/current-only×3seed의9개 선택 모델과 직전/평균 기준선을 새 미노출 기간에 추론만 하는 후보를 검토한다. 새로운 구조를 채택했다면 checkpoint 목록을 새로 동결해야 하며 기존9개로 새 모델의 확인을 대신하지 않는다.
- 아직 사용하지 않은 자료·기간인지 기록과 처리 경로를 확인하고, 원래 전처리의 호환성과 충분한 평가 대상이 있는지 점검한다. 이번 요청에서는 새 파일 다운로드·성과 열람·평가를 하지 않는다. Taxi 한 기간 확인은 모든 자료 일반화의 증거가 아니다.
- 완료 조건: 자료를 성과 선택에 쓰기 전에 고정한 계약에서 최종 결과를 한 번 보고하고, 불리한 수량·시간 결과와 단순 기준선 비교도 보존한다. 새 결과를 보고 다시 튜닝하면 그것도 개발 결과로 재분류한다. CPU binary 재현 감사는 별도이며 미노출 확인을 대신하지 않는다.

## 5. 확정된 범위로 원고를 정리한다 — 다음 작업 / 공통 main.tex는 직렬 편집

- 대상: `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex`. 이번 요청은 계획이므로 편집하지 않았다. 현재 Test 예외를 반영하는 제한된 서술 초안은 진단 계약 검토와 병행할 수 있지만 최종 원인·기여 문장은 증거 확정 뒤 쓴다.
- 실제 mean prototype 설명은 유지하고 미완료 검색의 현재형·빈 Results 비중을 줄인다. Taxi·Intermittent의 평균 개선과 seed 반전, RAF 악화, 시간·큰 수량 손해와 단순 기준선 우위를 함께 설명한다. Validation의3seed 일관성을 Test로 확장하지 않는다.
- 추가 구조 학습을 생략하면 주장은 ‘prototype mean이 유지된 backbone에서 결합 경로의 조건부 관찰’로 좁힌다. CNN·GRU 각각의 필요성이나 광범위한 우월성을 적지 않는다. 추가 학습을 해도 사후 선택·용량 차이·적용 범위 제한을 남긴다.
- 완료 조건: Abstract·Methods·Results·Conclusion과 부록·split·checkpoint·source·Runtime 출처가 같은 범위를 갖고 내장 LaTeX 컴파일·배치 검증이 통과한다. 문헌·재현성 검토는 독립 파일로 병렬 가능하고 제출 규정 확인·실제 제출 승인은 마지막 별도 단계다.

이번 바로 다음 대상은 **단계1의27checkpoint 진단 계약**이다. 신규 검색0개, 즉시 새 학습0개이며 RAF12fresh는 구조별 기여를 주장할 때만 선택한다. 학습·추론을 자동 실행하지 않고 승인된 범위를 구체적으로 확정한 뒤 진행한다.
