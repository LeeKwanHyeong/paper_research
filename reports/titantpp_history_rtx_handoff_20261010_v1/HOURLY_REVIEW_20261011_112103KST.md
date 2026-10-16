# 이력 보정 RTX 관측 — 2026-10-11 11:21:03 KST

원래 이력 보정27개 중 **완료26·진행1·실패0·대기0·미확정0**이다. Instacart9도 완료다. Intermittent seed62 current_only가 실제 exit0으로 종료했고, 같은 seed의 full만 5090에서 실행 중이다. 사용자 조치는 필요하지 않다.

## 주제와 원관측시각

| 대상 | 상태 | 실제 원관측시각 / 이번 확인 범위 |
|---|---|---|
| Instacart9 | 완료9; 로컬614원본 SHA 일치 | 회수 원관측 5080 2026-10-10 11:09:08.629287 KST / 5090 11:09:11.799019 KST; 이번 SSH0 |
| A100 이력 보정15 | 완료15 재사용 | 과학 실제 종료 2026-10-10 12:17:18 KST; checkpoint 캐시12:25:00 KST; 이번 원격0 |
| RAF62 / 5080 세 arm | 완료3; 로컬233원본 SHA 일치 | 2026-10-10 13:53:36 KST terminal cache; 이번 SSH0 |
| Intermittent42/52/62 / 5090 | 완료8·진행1 | 2026-10-11 11:21:03 KST; 읽기 전용 SSH1회 |
| 고정 검색 | 대기 | 자원/공통mean 계약 전 자동시작0 |
| CPU binary 재추론 감사·Test·새 독립 평가 | 별도 미완료 | 실행0 |

원래 ID27개의 중복 없는 집계를 확인했다(A10015+RTX12). 5090 완료8개는 실제 worker exit0 및 원격 binary terminal manifest 검증에 근거한다. 5090 전체9개 종료 전 해당 host의 로컬 binary 원본 회수는 대기다. history의 로컬 회수 완료 fit은 A10015+5080 세 개다.

## RTX dataset·arm·seed의 저장 상태

| host | dataset | seed | arm | 상태 | 저장 / best epoch |
|---|---|---:|---|---|---:|
| 5080 | raf_spare_parts | 62 | full | 완료 | 60 / 20 |
| 5080 | raf_spare_parts | 62 | 보정 제거 | 완료 | 66 / 26 |
| 5080 | raf_spare_parts | 62 | current_only | 완료 | 62 / 22 |
| 5090 | intermittent_frozen_5000 | 42 | full | 완료 | 55 / 15 |
| 5090 | intermittent_frozen_5000 | 42 | 보정 제거 | 완료 | 70 / 30 |
| 5090 | intermittent_frozen_5000 | 42 | current_only | 완료 | 46 / 6 |
| 5090 | intermittent_frozen_5000 | 52 | full | 완료 | 51 / 11 |
| 5090 | intermittent_frozen_5000 | 52 | 보정 제거 | 완료 | 53 / 13 |
| 5090 | intermittent_frozen_5000 | 52 | current_only | 완료 | 96 / 56 |
| 5090 | intermittent_frozen_5000 | 62 | full | 진행 중 | 36 / 13 |
| 5090 | intermittent_frozen_5000 | 62 | 보정 제거 | 완료 | 45 / 5 |
| 5090 | intermittent_frozen_5000 | 62 | current_only | 완료 | 78 / 38 |

current_only 실제 worker 종료 관측: **2026-10-11 10:47:45 KST**, returncode0·actual_child_exit_observed=true. 저장78/best38, terminal rawSHA `524a6c25023c4292ca5d7a945345f01849897140cc02e53c9c7221a24a83fb2c`. condition timeout/worker group은357882였다. 보정 제거는 기존08:58:06 KST exit0, 저장45/best5를 재검증했다.

이전10:18:08 KST 대비 full 저장29→36/best13 유지, current_only 저장69→78/best38 유지 및 실제 종료를 확인했다. 조회 실패와 학습 실패를 구분하며 이번에는 둘 다 없다.

## 실제 실행과 조건부 ETA

GPU UUID `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`에서 실제 compute PID는 **357843 하나**다. worker PPID357842, timeout357842, dispatch supervisor13235를 확인했다. 이번 시각에는 세 개 동시 실행으로 보고하지 않는다. 기존 qualification/admission의 세 worker 실제 GPU 동시 증거·각 실제 exit0·idle 후 학습 시작은 원관측을 유지한다. admission은 과학 학습이 아니다.

| fit | 자기 완료 epoch 표본 | 중앙값 seconds/epoch | best·속도 유지 시 잔여 | 300 epoch 시나리오 |
|---|---|---:|---|---|
| full / seed62 | E27–E36, 10개 | 588.998297037 | E53까지 2시간 47분 | 43시간 12분 |

ETA는 각 fit 자체 최근 최대10 완료 epoch의 finite 양수 elapsed_seconds 중앙값을 사용했다. best 미갱신 목표는 min(300,max(40,best+40))이며 full은(53−36)×588.9982970370038초다. max300은(300−36)×같은 중앙값으로 별도 계산했다. 다른 arm/GPU 속도 및 합성시간을 복사하지 않았다. 현재 concurrency가2개에서1개로 바뀌어 이후 속도도 달라질 수 있으며 동시 실행 시간을 구조 효율로 해석하지 않는다. 전체 큐 ETA는 미확정, 회수/평가파일/CPU 감사 시간은 제외다.

B128/min40/max300/patience40·condition36시간/host168시간 lease·자동 retry/resume0을 유지한다. deadline은 절대 종료 상한이며 ETA가 아니다. full의 max300 현재 속도 시나리오는 남은 condition lease보다 길어 해당 시나리오 완료를 보장하지 않는다.

## 계약·원본 검증

과학source129/input7/operation2, frozen project_files, current/contract/approval, host control·local_tools·monitor 봉인 파일, start/deploy/launch receipt, ownership_release/source를 stdlib SHA로 대조했고 root·GPU·계약·포인터 불일치0이다.

- canonical `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`
- source closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`
- operation closure `8a7ff350488fe8b03289ea6e6683a41ace4c95c2bdf0d24f2c06f3d78f5c83da`
- science revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`

native qualification/admission 실제 child exit0 및 idle→training permit 연결을 검증했다. qualification/admission receipt는 permit의 rawSHA, start/qualification_exit/concurrency_exit는 봉인 pipeline 규칙의 canonical JSON SHA로 연결되며 native rawSHA는 snapshot에 별도 보존된다. canonical과 rawSHA를 혼동하지 않는다.

실제 관측시각에서 lease 유효·정확 worker/timeout/supervisor argv·claim 계약·GPU PID·stable checkpoint receipt epoch/history/timing이 일치한다. 첫 checkpoint는 이미 저장되어 있으며 새 epoch36을 실제 저장 진척으로 기록했다. pipeline heartbeat는 dispatch이고 원관측시각과90초 미만이다. host 전체 dispatcher exit0·pipeline_complete·최종phase·owner/GPU PID 전체 부재는 아직 대기다. pipeline OS exit 자체는 직접 관측하지 않았다는 사실을 유지한다.

full lease 발급 2026-10-11 11:21:01 KST / 만료 2026-10-11 11:22:31 KST; condition deadline 2026-10-12 17:13:38 KST. lease 판단은 native 관측시각 기준이며 보고서 작성시각으로 대체하지 않는다.

checkpoint receipt rawSHA `468b32bd1d82955fb4acf15fe28c1585d1efc4b0e36f2faf816e8f45ad30bb29`.

```json
{
  "job": {
    "id": "intermittent_frozen_5000__62__titantpp_cnn_gru54",
    "host": "5090",
    "dataset": "intermittent_frozen_5000",
    "seed": 62,
    "arm": "titantpp_cnn_gru54"
  },
  "worker_pid": 357843,
  "worker_ppid": 357842,
  "timeout_pid": 357842,
  "supervisor_pid": 13235,
  "exact_worker_argv": [
    "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12",
    "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py",
    "--contract",
    "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json",
    "--host",
    "5090",
    "--mode",
    "fit",
    "--job",
    "intermittent_frozen_5000__62__titantpp_cnn_gru54",
    "--deadline",
    "1791792818.4858577"
  ],
  "lease": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "expires_unix": 1791685351.5441718,
    "issued_unix": 1791685261.5441718,
    "job": "intermittent_frozen_5000__62__titantpp_cnn_gru54",
    "supervisor_pid": 13235,
    "worker_group_pid": 357842
  },
  "claim": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "deadline_unix": 1791792818.4858577,
    "job": {
      "arm": "titantpp_cnn_gru54",
      "dataset": "intermittent_frozen_5000",
      "host": "5090",
      "id": "intermittent_frozen_5000__62__titantpp_cnn_gru54",
      "seed": 62
    },
    "maximum_concurrency": 3
  },
  "checkpoint_receipt": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "epoch": 36,
    "files": {
      "epoch_timing.json": "882e1e1ae68c3d5487bf36995368fb1b1e232f79fb1b0a08c781a7fbd953c114",
      "exposure.json": "3ae394e8bc2365497eb575e4f520abb51e7c62fb76b33fb2e09e1c7664fb8107",
      "history.json": "77a8d7c63c1a71b2ac90e938c32f5cf609d7b39baef890391d4501a99676d735",
      "last_epoch_state.pt": "d523b481e32681b48e822ec022c5a17887e614cea61148f3c53721c9f69cccc7",
      "train.log": "ff6f643e5fc40e42892747eb956707746bb7d6b09d38446c2ff74e7f3e2778c4"
    },
    "mac_ack_required": false,
    "time": 1791685057.6864004
  }
}
```

독립 로컬 원본 재감사: 고유1,701개 파일·1,737회 SHA 비교 PASS, 불일치[]. 원본1,682개는 Instacart614+A100 partial442+parent393+RTX5080 233이다. archive/manifest/receipt/analysis 참조 및 selected/last checkpoint12쌍·24개, archive receipt 의미적 연결2개, 이전 cleanup rawSHA3개를 재검증했다. 성과 파일 내용을 열거나 checkpoint를 역직렬화하지 않았다.

Instacart completion receipt rawSHA `459ae9f15f6f4247584a8d47bd8e3901ab6e7ba91d6a98df3f64b09dde95cea6`, A100 release source `e7d6fbaf59b1caa7763d2662ac1ed29d65f003f6846f8bc3aaff96fa6d3a8ea1` 및 보존 행정 claim `607cd6721d3c6bbfca7817d63e5eb0b329e1caf9b25e66add1718e8728ed41b8`의 연결을 유지한다. Instacart의 실제 owner/supervisor 종료와 과거 전역 NVML idle 미확정을 학습 실패와 구분한다.

A100 handoff의 FileExistsError·실제 운영 dispatcher/pipeline exit1·failure/claim 및 partial 원본을 보존했다. 완료15만 재사용하며 부모 전체27 완료 또는 새로운 수치 실패로 집계하지 않는다. 삭제 부재 증거는 과거 로컬 receipt에 근거한다: 부모 qq2uk2dpfi2rum 2026-10-09 12:23:55.886020 KST, 복구 z278euk8d74nqt 2026-10-10 12:25:29.800147 KST, volume pkjlrapcbm 12:25:29.799423 KST. 새 SSH/API 부재 조회0이며 폐쇄 service heartbeat를 live 실패로 판정하지 않는다.

cleanup rawSHA는 recovery `79669962c84b3f3cd66d883250a4f158742a851deb25c44af108161dd72d8238`, volume `d69a476e68597494567a995e1e8f080a589e89b39b0982300daa95f96641766b`, parent `00b49d1c4f5d9c11ac5c99b5fba2de2004a9bf53eb9c721bea5d4246bcfbac4c`로 이전 계산값과 동일하다. 사용자 지시의 독립 SHA anchor가 없는 이 세 값은 과거 삭제 기록의 보존 확인으로 표시한다.

Runpod 누적 추정32.36242173899348달러, 최종 청구 미확인, 기존50달러 상한 유지다. 이미 삭제된 Pod·volume를 다시 조회/접속/시작하지 않았으며 새 Pod·충전·가격/Runtime/source 변경0이다.

이번 실행은 승인 advance 정확히1회(exit0), live5090 읽기 전용 SSH1회, 5080cache/Instacart/A100 SSH0, Pod API0, deploy/qualification/launch/kill/retry/resume0이다. 내부 finalizer는1회 pending·remote0·user_action_required=false다. 회수 claim/failure를 보존하며 자동 회수 retry0이다.

## 남은 작업 순서

**마지막 fit 종료와 원본을 확정한다 — 진행 중**
- 대상: 5090 Intermittent seed62 full. 실제 worker 종료 및 dispatcher exit0·pipeline completion·최종phase·owner/GPU PID 부재를 확인한다.
- 완료 조건: 전체9개 terminal/binary manifest 검증 후 immutable 원본을 한 번 회수하고 local archive/manifest/rawSHA와 원래27 ID를 확정한다.

**보정 기여를 같은 선택 epoch의 여섯 지표로 판정한다 — 다음 작업**
- 대상: A10015+RTX12. 최초 strict finite 최소 fullValidation 전체 raw수량RMSE epoch의 전체/큰수량 RMSE·MAE·TimeNLL, 3seed 평균·표본SD와 불리한 조건을 보존한다.
- 완료 조건: full이 두 대조보다 같은2개 이상 dataset의 각3seed 전체RMSE를 strict 개선할 때만 Q1 개발지지를 기록한다.

**확정 결과에 맞춰 논문 초안과 후속 계약을 정리한다 — 다음 작업**
- 대상: paper_research CNN+GRU54 중심 Results, Instacart 및 Methods/주장 정합. 원본·기여 판정 후 작성한다.
- 이후 순서: 검색 포함 여부/필요시 새 공통 계약 → 미노출 평가 가능성·CPU 감사 계약 → 관련연구/References/Appendix → 컴파일·투고 점검.
- 고정 검색은 자원/공통mean 계약 전 대기, 자동시작0이다. A100 full을 RTX 동일GPU mean과 혼합하지 않는다.

RAF42 full/current_only의 물리 A100 이관과 cross-seed GPU/Runtime 차이를 명시하며 시간은 물리 GPU 구간별로 분리한다. current_only hidden에는 prefix 이력이 포함된다. 활성 파라미터 수가 같아도 계산 경로/깊이/활성화 비용은 같다고 볼 수 없다. 최종 판정은 Validation 개발 증거이며 통계유의성·참신성·외부 우월성·새 독립 평가 증명으로 확대하지 않는다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.

최종 Validation 보고·원본 회수·전체 실제 종료 조건이 아직 완료되지 않아 자동화 titantpp-instacart-63을 유지한다. 다른 자동화/서버/Runtime/인증/기존 원본은 변경하지 않았다.

## 봉인된 관측 원본

- advance: [advance_1791685264053969000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791685264053969000.json); rawSHA `c8ef9f5c5f58774892628a955d58f38bd14511e6b0845d1e6dd742bc50f9e3ad`.
- 5090: [snapshot_5090_1791685263977344000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/snapshot_5090_1791685263977344000.json); rawSHA `568d7d757ec64482339e22d05fb7c1e3bd5df37ca99d4143f25c212fada8b020`.
- 5080 기존 terminal: [snapshot_5080_1791608016546790000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/snapshot_5080_1791608016546790000.json); rawSHA `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`.
- 이전10:18:08 KST advance: [advance_1791681488811400000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791681488811400000.json); rawSHA `38e7ab70b32e32763fb9832a3adf117fcca49ab458e80d55a4d7b444050c4558`.
- [PAPER_WORK_ORDER.md](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md).
