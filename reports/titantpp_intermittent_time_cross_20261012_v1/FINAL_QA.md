# Intermittent 시간 위치·분포 폭 분리 실제 종료 및 수치 독립 QA

## Review Result: PASS

- 검증시각: 2026-10-12T13:15:44.318084+09:00
- 상태: 완료. 실제6개 joint OS worker exit0·pipeline OS exit0와18개 model-split pass를 확인했다. 신규 학습0·Test read/evaluation0·calibration fit0.
- 변경 파일: 이 FINAL_QA.md만 작성. source/code/선택 checkpoint/Runtime/main.tex/.agents 수정0, 별도 SSH·추론0.

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

## 확인한 완료 조건

- [x] canonical `bc785c5ea669fe0f23d06604710dcc7ed9267b18996423298857279438d293c4`, operation `62b6dd1d6d76685b013c453445c267caf9eebfc70b16a3a388a9f9e5255f8be6` 및 input219개/source129/checkpoint9/reference18개 봉인 일치.
- [x] H1 v2 실제18개 평가·pipeline 종료·357개 원본·분석·FINAL_QA PASS가 H2보다 먼저 완료됐고 증적7개가 계약에 봉인됨. v1 synthetic failure 기록은 그대로 보존.
- [x] native qualification3 PASS 및 원래 forward bitwise 검증, 실제 native launch/exit/receipt의 argv·PID·PPID를 독립 대조.
- [x] joint6의 launch/exit/receipt argv(Python 포함/제외 정규화)·PID·PPID 정확 연결, 각 joint3개 model receipt의 shared PID/PPID 확인. native 이후 시작·actual exit0·pipeline complete/last phase complete·owner/GPU PID 부재의 원본 확인.
- [x] Train393824개·Validation86285개에 대해9모델의 동일 target/full-history SHA 및 원래 수량·시간6지표/state 재현 확인. good/bad seed와 두 대조를 모두 포함.
- [x] 실제309개 archive member SHA와 manifest·materialized/verified 두 로컬 복사본 전체 byte SHA 재대조. archive `0386dd5f928e13e6e2a1771fff904e10c6b0f330df5ffab050b990e7e0bad208`, manifest `17b068a81ee4c5edb2c29d4c8b14cf3d216679b21b246fb98347f4955647d1d3`.
- [x] analyzer.summarize()를 읽기 전용 독립 호출해 실제6 joint/18 pass/48 paired score cell/24 new cross cell 검증. 전체12행+gap bin60행=72개 strata row의 충분통계·합/제곱합·원래 ff/cc 점수 일치 확인.
- [x]72개 행의 mean=sum/n, scale=fc−ff, location=cf−ff, interaction=cc−fc−cf+ff, total=scale+location+interaction을 별도 계산·허용오차 내 일치 확인.4개 split×대조의 모든3seed 평균·표본SD 정확 일치.
- [x] 최종 analysis completion rawSHA `a611c3f0c54db2f7b889557261551d020b162d040beaa65a64855a0a0c2e8dcb`, conditions/report SHA 대조 통과.
- [x] 원래 CDF integer mass·time scale3·topcodeNone·native σ 변환 유지. 실제event μ·σ는 RAM에서 pairing하고 예측 raw arrays 영구저장0. 신규 checkpoint 선택·floor/τ 탐색·학습0.

## 확인한 주요 수치와 해석 범위

Validation에서 full 위치를 유지하고 두 대조의 σ만 사용한 점수는 모든3seed×2대조에서 낮아진다. 보정 제거 σ 교체 Δ는42/52/62에서−0.374876/−0.054623/−0.069222, current-only σ 교체 Δ는−0.364995/−0.043548/−0.059926이다. μ만 교체하면 같은6조건에서 NLL이 모두 높아진다. Train에서는 σ 교체도6조건 모두 손해다.

3seed 평균 Validation full NLL0.409740, 보정 제거 σ 교차0.243500·current-only σ 교차0.253584이다. 반면 원래 두 대조는0.342995·0.347906이다. 원래 full과 두 대조의 seed별 우열은42에서 full 손해,52/62에서 full 이득으로 유지된다. μ·σ 상호작용은 모든 조건·gap bin에 보존했으며 전체 평균만으로 하위 구간을 단정하지 않는다.

이 결과는 위치를 고정했을 때 다른 모델의 분포 폭을 사용하는 점수 민감도에 관한 사후 관측이다. 새 단일 모델이나 학습된 calibration의 성능이 아니며, σ만이 원인이라는 인과 증명·CNN/GRU 필요성·통계 유의성·독립 일반화·기존 Test 반전 해결로 확대하지 않는다. 교체 순서와 상호작용 때문에 μ/σ 원인을 유일한 비율로 분할하지 않는다.

## 남은 작업 순서

**Train·Validation 진단 결과를 논문 보완 판단에 연결한다 — 다음 작업**
- H1 수량 목표 관측과 H2 분포 폭 민감도를 각각 구분해 원인 후보를 좁힌다. 새 학습/보정 모델/재평가가 필요하면 별도 최소 계약과 승인 후 진행한다. 이 검증으로 새 학습이나 Test 실행을 자동 시작하지 않는다.
