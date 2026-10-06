# CNN+GRU54 수량 경로 동결·시간 출력부130개 재적합

**완료 원본과 두 보정값 결과를 연결한다 — 완료 / 현재 기준선**
- Taxi·RAF seed42/52/62의 원래 수량 선택 checkpoint6개를 그대로 씁니다. 이전 bias2 단계는6완료·원본SHA회수·실제종료가 확인됐으며 Taxi 개선0, RAF 평균TimeNLL0.725% 감소입니다. 새130개 학습은 이전 bias2 선택 상태가 아닌 원래 quantity-selected E0에서 시작합니다.

**시간 출력부130개를 학습하도록 준비한다 — 완료**
- 실제 native 파라미터는 v_t.weight64개, b_t1개, w_raw1개, time_scale_weight.weight64개입니다. CNN·GRU·HardLMM·수량 출력부·모든 나머지 weight/buffer/eval을 고정합니다. 원래 시간 단위와 recorded-positive-integer 시간 likelihood를 유지합니다.
- Train만 fitting에 쓰고 full Validation TimeNLL 최초 strict finite최소를 선택합니다. E0를 포함해 동률/불개선이면 원본을 유지합니다. Adam LR0.001·batch128·clip1·최대40epoch·patience10을 bias2 단계와 같게 유지합니다.
- 실행 대상은5080 별도root·worker1, 조건1시간/전체6.5시간, CPU16GiB/GPU80%/root8GiB, 자동retry 없음입니다. Test/5090/공용Runtime/인증/과학소스는 변경하지 않습니다.

**실제 GPU 검증·학습·원본회수로 비교한다 — 진행 중**
- native 대표 Taxi52 검증 후 각6조건의 원본/E0/population/startup gate를 확인해야 fitting이 가능합니다. 모든 Train/Validation 수량 예측 byteSHA와 동결 weight/buffer SHA가 정확히 유지돼야 합니다.
- 학습완료 후 selected/last130개 head tensor·이력·원래 selected checkpoint 원본manifest/archiveSHA를 회수합니다. 완료 전에 성능을 확정하지 않습니다.
- 결과는 Validation 개발 선택이며 독립 calibration이 아닙니다. S2P2의 동일 재적합·Intermittent 후속·별도Test/독립평가/CPUbinary감사는 별도 미완료입니다.

- 로컬 CPU 테스트73개와 subtest33개 PASS. 원본6개·source123 SHA와 native4tensor/130개 일치 읽기 검토PASS. 5080 실제 native qualification PASS(대표Taxi52, 원본whole-state 복구SHA 일치·수량 반복forward 동일, real optimizer update0). 2026-10-06 10:17:32KST에 학습을 실행했습니다.

**실제 실행 증거 — 진행 중**
- 10:17:44KST 관측: 완료0/실패0/진행1/대기5/미확정0. Taxi seed52 특징추출 중이며 첫새head 저장전입니다. controller17491·worker17510/GPU PID17510의 정확 명령·PPID·GPUUUID를 확인했습니다.
- 5090의 기존 학습·관측은 유지합니다. 이번작업에서5090 추가SSH나중단은 하지 않았습니다.
- 기존 매시간 자동화에 full130관측·원본회수·3stage비교를 추가하고 앱readback 확인했습니다.

**첫 실제 head checkpoint 확인 — 완료 / 전체 학습은 진행 중**
- 10:20:07KST: 완료4/실패0/진행1/대기1/미확정0. Taxi3seed 모두head E10/bestE0로 원본이 선택됐습니다. RAF52는head E12/bestE2, TimeNLL3.479681→3.460704이며 나머지RAF62 E3/best3 진행·RAF42 대기입니다.
- 완료4조건 모두 full Train/Validation 수량예측 byteSHA·동결 weight/bufferSHA·원본CP SHA·selected/last/terminal 증거가 일치합니다. RAF3seed 결과·원본전체회수·서버종료는 아직 미완료입니다.
- 새비교 CPU22개 PASS. 전체95개+33subtests 검증(73+22). 현재 finalize는pending을기록했고 원격회수/평가를실행하지않았습니다.
