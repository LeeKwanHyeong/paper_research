# Instacart 완료 확인과 A100 이력 보정 복구 — 2026-10-10

사용자 승인: “문제 확인해서 수정하고 남은 학습 진행하자”. 이전 A100 비용을 포함한 총 $50 상한과 최대 3개 동시 fit을 유지한다. Test/held-out 예측·성과와 Validation/Test 혼합 파일은 열지 않았다.

## 현재 기준선과 복구 범위

- Instacart: 9/9 학습 완료. 새 완료 회수 bundle `search_artifacts/titantpp_instacart_completion_recovery_20261010_v1`에 614개 원본, selected/last checkpoint 9쌍 및 manifest/archive rawSHA 검증을 보존했다. 5080 NVML과 kernel driver 버전 불일치를 학습 실패와 구분했고, 공용 driver/Runtime은 수정하지 않았다. 전역 GPU 유휴는 별도 미확정이나 실제 실험 owner 종료와 supervisor exit0, 원본은 확인됐다. 상세는 [Instacart 진단](INSTACART_DIAGNOSIS.md).
- 이력 보정: 원래 27개 execution ID를 유지한다. 완료10개 재사용, RAF seed42 full E23/best9와 current_only E46/best9의 2개 재개, 미시작15개 fresh. 첫 batch는 미완료2개, 이후 동일 dataset/seed의 full/disabled/current_only 3개씩 동시 실행한다.
- 고정 검색: 후속 자원·공통 mean 기준선 계약 정리 전 대기. A100 full 결과를 RTX 동일 GPU mean 결과로 표시하지 않는다.

## 복구 변경과 검증

정상 worker 종료와 GPU PID 목록 갱신 사이의 경합을 기존 검사에서 재현했다. 실제 사고 시각의 PID/argv snapshot이 없어 당시 원인을 단정하지 않는다. 새 운영 검사는 stat/cmdline/start ticks와 최신 GPU 목록을 교차 확인하며, 계속 살아 있는 불명확·외부 PID는 거부한다.

새 root와 운영 계약 번호 때문에 checkpoint identity가 달라지는 문제도 수정했다. 재개2개만 입력의 정확한 경로 연결과 부모 과학 interface identity 연결을 별도 receipt로 기록한다. 원본 .pt, receipt, source129/input7, data/loss/optimizer/selector는 보존했다. exposure/timing 기록을 원래 prefix에 이어 붙이고 optimizer/RNG/loader/best/history 연속성을 확인한다.

로컬 검증: 회귀15개, 봉인 gate8개, 볼륨 정리4개 시나리오 PASS. 독립 검토는 [PRELAUNCH_REVIEW.md](PRELAUNCH_REVIEW.md). 실제 native qualification·3 worker admission·첫 신규 checkpoint는 아직 실행 전이다.

## 새 전용 자원과 비용

- Pod `z278euk8d74nqt`, A100-SXM4-80GB 1개, GPU `GPU-17e020b0-7bbe-d835-af19-881e511f9a2a`, Secure US-MD-1.
- root `/workspace/titantpp_contribution_history_recovery_20261010_v1_a100`. network volume `pkjlrapcbm` 10GB를 `/workspace`에 연결한다. 기존 삭제된 Pod를 재시작하지 않았다.
- Python3.12.3 / Torch2.11.0+cu130 / CUDA13.0 / cuDNN91900 / NumPy2.4.4 / Polars1.39.3: 실제 설치와 학습 전 GPU idle 검증. native 학습 적합성 검사와 별개다.
- 부모 실제 청구 $30.124985998. 새 GPU $1.79/h + container20GB $0.002777778/h + network10GB $0.000972222/h = $1.79375/h. network 생성 시각이 API에 없어 Pod 전 1시간분 $0.000972222를 보수적으로 별도 반영했다.
- 새 청구 시작 2026-10-10 11:10:41 KST. 준비·학습 lease 마감 2026-10-10 19:28:12 KST, 절대 compute 비용 종료 2026-10-10 22:15:27 KST. 회수·정리 예비 $5 유지. 마감은 ETA가 아니며 17개가 모두 비용 안에서 완료될지는 아직 미확정이다.
- 새 원본과 부모 원본 SHA를 검증한 뒤 새 Pod와 전용 volume만 삭제한다. 중간/실패 원본과 claim은 보존하며 자동 학습 retry는 없다. 계정 자동충전 설정의 live 검증은 불가했고 자동충전·충전·계정 변경은 수행하지 않았다.

## 봉인

- canonical `da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899`
- source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`
- operation closure `8d7a5bec2af2f7d746fdb38ff618a8aceda7ffff9391d9252f4e5b46cf70f8c0`
- 부모 canonical `4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a`
- 부모 original archive `83d0d3f4fabd89554a0cc5acc86dc447a9919b22d5247e27a1aaea65ecc01235` / manifest `d25ca3393c1c9a295ebf6b37d0695f8b700adf8af68b777a4e09b57c245f2091`

새 물리 GPU로 재개하므로 RAF42 disabled와 두 재개 arm의 대조에 GPU 이관 표식을 붙인다. 무중단 수치 동일성 또는 같은 GPU 대조를 주장하지 않는다. 시간은 기존/신규 물리 GPU 구간별로 보존한다.

## 남은 순서

1. 새 비용 manager 봉인 및 native qualification·3 worker admission — 다음 작업. 실제 exit0와 GPU/owner idle 후에만 fit을 시작한다.
2. RAF42 E24/E47의 첫 새 checkpoint 저장과 연속성 확인 — 위 검사 통과 후 진행.
3. RAF52/62와 Intermittent42/52/62의 3개 동시 batch — 비용 lease 안에서 진행.
4. 원본 회수·27개 동일 선택 epoch Validation 비교·Q1 판정 — 종료 증거 확인 후. 큰 수량 RMSE/MAE와 TimeNLL 손해도 보고한다.
5. 고정 검색 자원·mean 비교 계약 — 대기. CPU binary 재추론 감사, Test, 새 독립 평가 — 별도 미완료.
