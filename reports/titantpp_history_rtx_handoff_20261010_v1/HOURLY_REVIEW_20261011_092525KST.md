# 이력 보정 RTX 분담 관측 — 2026-10-11 09:25:25 KST

**현재 확인된 상태:** 원래 이력 보정27 ID 중 완료25·진행2·대기0·실패0·미확정0이다. Instacart9는 완료이며, 고정 검색은 대기다. 이번에는 5090 seed62 보정 제거의 실제 exit0 종료를 새로 확인했다. 사용자 조치는 필요하지 않다.

## 주제별 상태와 실제 관측시각

| 대상 | 상태 | 실제 원관측시각 / 근거 |
|---|---|---|
| Instacart9 | 완료9; 로컬614원본 SHA 일치 | 원본 회수 당시 5080 2026-10-10 11:09:08.629287 KST / 5090 11:09:11.799019 KST; 이번 확인은 로컬 SHA 재검증 |
| A100 이력 보정15 | 완료15 재사용 | 과학 실제 종료 2026-10-10 12:17:18 KST; checkpoint 캐시 12:25:00 KST |
| 5080 RAF seed62 세 arm | 완료3; 로컬233원본 SHA 일치 | 2026-10-10 13:53:36 KST terminal cache; 새로운 SSH0 |
| 5090 Intermittent seed42/52/62 | 완료7 / 진행2 | 2026-10-11 09:25:25 KST; 읽기 전용 SSH1회 |
| 고정 검색 / CPU 재추론 감사 / Test / 새 독립 평가 | 대기 / 별도 미완료 | 실행0 |

완료25는 A10015 + RAF62/5080 세 개 + Intermittent/5090 일곱 개이다. 원래 execution ID27개에 중복이 없음을 검증했다. 5090의 완료7개는 terminal/binary manifest가 검증된 원격 완료 증거이며, 해당 host 전체9개 종료 전 로컬 binary 원본 회수는 대기다. 이미 로컬 회수된 history fit은 A10015 + 5080 세 개다.

## 5090 마지막 seed62 묶음

| arm | 상태 | 실제 저장 epoch / best | worker PID / PPID | best·현재 속도 유지 시 잔여 | 300 epoch 시나리오 |
|---|---|---:|---|---|---|
| full | 진행 중, checkpoint 저장됨 | 23 / 13 | 357843 / 357842 | E53까지 5시간 18분 | 48시간 59분 |
| 보정 제거 | 완료, 실제 worker exit0 | 45 / 5 | 종료: timeout/worker group357874 | 완료 | 해당 없음 |
| current_only | 진행 중, checkpoint 저장됨 | 52 / 38 | 357883 / 357882 | E78까지 1시간 26분 | 13시간 39분 |

보정 제거의 실제 worker 종료 관측은 **2026-10-11 08:58:06 KST**이며, terminal manifest rawSHA는 `a8d105adfe4c7cbe0f500908d7728a4760016412b89c5cb627368ced25b619a8`이다. 학습 자체는 E45/best5에서 종료했다. 이전 08:41:45 KST 관측 대비 full 저장19→23/best13 유지, 보정 제거 저장41→45/best5 유지 및 종료, current_only 저장41→52/best38 유지다.

관측 당시 실제 GPU compute PID는 **357843 / 357883 두 개**다. 두 fit은 같은 5090에서 동시에 실행 중이며 순차 실행 두 개로 집계하지 않는다. GPU UUID는 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`이다. 앞선 세 worker admission의 실제 동시 GPU 증거·각 exit0·idle 후 학습 시작은 기존 원관측으로 확인한다. admission은 과학 fit으로 세지 않는다.

ETA는 각 fit 자체 최근 최대10개의 완료 epoch finite elapsed_seconds 중앙값으로만 계산했다. 전체 큐 ETA는 미확정이며, 원본 회수·최종 평가 파일 생성·CPU 감사 시간은 포함하지 않는다. 현재 실행 concurrency가 세 개에서 두 개로 변했으므로 앞으로 속도는 다시 달라질 수 있다. 동시 실행 시간은 구조 효율의 증거로 해석하지 않는다.

| arm | 자기 완료 epoch 표본 | 중앙값 seconds/epoch | 계산 |
|---|---|---:|---|
| full | E14–E23 / 10개 | 636.511482751 | (min(300,max(40,13+40))−23) × 중앙값 / max300 별도 |
| current_only | E43–E52 / 10개 | 198.179755329 | (min(300,max(40,38+40))−52) × 중앙값 / max300 별도 |

조건36시간 및 host168시간의 봉인 lease는 유지한다. 각 condition deadline은 종료 상한이며 ETA가 아니다. full의 현재 속도 유지 max300 시나리오는 남은 condition lease보다 길어 해당 시나리오 완료를 보장하지 않는다. 예산·deadline·retry/resume 계약을 늘리거나 변경하지 않았다.

## 프로세스·계약·저장 증거

정확 argv, PID/PPID, UUID, claim, lease, checkpoint receipt와 stable checkpoint/history/timing의 연결을 로컬 raw snapshot으로 다시 검증했다. lease 유효성은 실제 native 관측시각에서 판정했으며, 이 보고서를 쓰는 시각으로 lease와 관측시각을 대체하지 않았다.

pipeline heartbeat는 dispatch 단계이며 원관측시각과의 간격이90초 미만이다. host 전체가 아직 완료되지 않아 dispatcher exit0·pipeline_complete·host owner/GPU PID 전체 부재는 대기다. pipeline OS exit 코드는 직접 관측하지 않았으며 이후 완료 판정에서도 별도 사실로 유지한다.

과학source129·input7·operation2, current/contract/approval/ownership_release, host control, local tools, monitor project_files·봉인 파일, start permits, deploy/launch receipt의 stdlib SHA 및 root/UUID/canonical 계약을 대조했다. 봉인 불일치0이다.

canonical `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`; source closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`; operation closure `8a7ff350488fe8b03289ea6e6683a41ace4c95c2bdf0d24f2c06f3d78f5c83da`; frozen science revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`.

qualification 및 concurrency receipt rawSHA는 native permit와 일치한다. start/qualification_exit/concurrency_exit의 permit 필드는 봉인 pipeline의 규칙에 따라 canonical JSON SHA로 연결되며, 해당 native 파일의 rawSHA는 snapshot에 별도로 보존되어 있다. canonical SHA와 rawSHA를 같은 값으로 비교하지 않는다. 실제 qualification/admission child exit0 및 실제 idle 증거도 검증했다.

### full — 진행 중

- job `intermittent_frozen_5000__62__titantpp_cnn_gru54`; worker 357843/PPID 357842; timeout 357842; dispatch supervisor 13235.
- lease 발급 2026-10-11 09:25:25 KST / 만료 2026-10-11 09:26:55 KST; condition deadline 2026-10-12 17:13:38 KST.
- checkpoint receipt rawSHA `a0005aacfebe3231ef024d5cea3a202b30fd8966bd18de68bcbd5c87790a545d`; 저장 E23/best13.

```json
{
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
  "lease": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "expires_unix": 1791678415.1680894,
    "issued_unix": 1791678325.1680894,
    "job": "intermittent_frozen_5000__62__titantpp_cnn_gru54",
    "supervisor_pid": 13235,
    "worker_group_pid": 357842
  },
  "checkpoint_receipt": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "epoch": 23,
    "files": {
      "epoch_timing.json": "2246f62b69b3e1af07e9c3f4da7d9099be24887891f5b4e7b5b356856297abe4",
      "exposure.json": "4fbca928e4995f6f074858d4e134f4a74754709e7758359075b2d45320bfc04e",
      "history.json": "3ca21bd7eec67d57fe2b857eaa2a50f112eca5f7201263be99c0055098e0c4f0",
      "last_epoch_state.pt": "6af0a43c88237d6c5551575a12e1098e3e4f6760d69ff957fc2ae0a47dae76b9",
      "train.log": "a6d9d83518a99f4213b9d574c9171f670ac2f9e68b81d2c04e33c2d372732d3f"
    },
    "mac_ack_required": false,
    "time": 1791677733.3572364
  }
}
```

### current_only — 진행 중

- job `intermittent_frozen_5000__62__titantpp_cnn_gru54_current_only`; worker 357883/PPID 357882; timeout 357882; dispatch supervisor 13235.
- lease 발급 2026-10-11 09:25:25 KST / 만료 2026-10-11 09:26:55 KST; condition deadline 2026-10-12 17:13:38 KST.
- checkpoint receipt rawSHA `56df62a1e185dc3d2c181333f5a5bc78c637b91a2bc18f4d80dba619d410794e`; 저장 E52/best38.

```json
{
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
    "intermittent_frozen_5000__62__titantpp_cnn_gru54_current_only",
    "--deadline",
    "1791792818.5613778"
  ],
  "claim": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "deadline_unix": 1791792818.5613778,
    "job": {
      "arm": "titantpp_cnn_gru54_current_only",
      "dataset": "intermittent_frozen_5000",
      "host": "5090",
      "id": "intermittent_frozen_5000__62__titantpp_cnn_gru54_current_only",
      "seed": 62
    },
    "maximum_concurrency": 3
  },
  "lease": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "expires_unix": 1791678415.1680894,
    "issued_unix": 1791678325.1680894,
    "job": "intermittent_frozen_5000__62__titantpp_cnn_gru54_current_only",
    "supervisor_pid": 13235,
    "worker_group_pid": 357882
  },
  "checkpoint_receipt": {
    "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
    "epoch": 52,
    "files": {
      "epoch_timing.json": "ed695ab9b90628da5f2d55c2a48fc121724f8b27261a480357d10fffdab8b2ca",
      "exposure.json": "2002535009d2f9e38c6d79f58e910b0459dc1285009b1ba28130584253802e44",
      "history.json": "1447a80cf2cbde60afdb188336561469f8b60525d22b1ff5a64e72f96886f295",
      "last_epoch_state.pt": "f91bf3a8a0ebbe13187368428a3aa38a95490d50f0674a95ce83531afba3e73c",
      "train.log": "d166b4ff749580850f3f09c419224bef957545f0d4b2792cff4b6fe920522917"
    },
    "mac_ack_required": false,
    "time": 1791678134.7940893
  }
}
```

기타 실제 pipeline/dispatch/timeout의 정확 argv와 PID/PPID는 아래 봉인 raw snapshot의 `processes`에 보존했다.

## 완료 원본과 Runpod 종료 증거

독립 읽기 전용 감사에서 고유 로컬 파일1,698개(재확인 포함 SHA 비교1,734회)가 모두 일치했다. 원본은 Instacart614 + A100 partial442 + A100 parent393 + RTX5080 233 =1,682개이며 archive/manifest/receipt/analysis 참조도 함께 검증했다. Instacart 선택/last checkpoint9쌍·18개 및 RTX5080 3쌍·6개를 별도 확인했다. 성과/예측행 내용을 열거나 checkpoint를 역직렬화하지 않았다.

Instacart의 실제 소유 프로세스 종료 및 supervisor exit0와 전역 NVML idle 미확정을 구분했다. 이 회차의 확인은 과거 terminal 증거 및 로컬 SHA 재검증이며 서버의 현재 전역 idle을 새로 관측하지 않았다.

A100 부모의 handoff exclusive claim FileExistsError와 실제 운영 dispatcher/pipeline exit1·failure·claim은 보존되어 있다. 완료15의 과학 fit은 재사용하며 부모 전체27 완료 또는 새로운 수치 실패로 변경하지 않았다. recovery는 partial 원본 보존이다.

삭제 부재 증거는 과거 local receipts에 근거한다: 부모 Pod qq2uk2dpfi2rum 2026-10-09 12:23:55.886020 KST, 복구 Pod z278euk8d74nqt 2026-10-10 12:25:29.800147 KST, 전용 volume pkjlrapcbm 12:25:29.799423 KST. 현재 원격 재조회0. cleanup rawSHA 계산값은 recovery `79669962c84b3f3cd66d883250a4f158742a851deb25c44af108161dd72d8238`, volume `d69a476e68597494567a995e1e8f080a589e89b39b0982300daa95f96641766b`, parent `00b49d1c4f5d9c11ac5c99b5fba2de2004a9bf53eb9c721bea5d4246bcfbac4c`이다. 이 세 값에는 이번 지시의 별도 독립 anchor가 없으므로 현재 bytes 계산값으로 표시한다.

Runpod 누적 추정 비용은 기존32.36242173899348달러이며 최종 청구는 미확인이다. 기존50달러 상한과 삭제된 자원을 유지하며 새 Pod·충전·가격/Runtime 변경은 수행하지 않았다.

이번 실행: 승인 advance 도우미 정확히1회(exit0), 5090 읽기 전용 SSH1회, 5080 terminal cache SSH0, Instacart SSH0, A100 SSH0, Pod API0, deploy/qualification/launch/retry/resume0. 내부 finalizer는1회 pending/remote0으로 종료했으며 사용자 조치 필요=false다. 회수 claim/failure의 삭제와 자동 회수 retry도0이다.

## 남은 작업 순서

**원래27개 학습의 종료와 원본을 확정한다 — 진행 중**
- 대상: 5090 마지막 Intermittent seed62 full/current_only. 실제 worker 종료와 dispatcher exit0, pipeline 완료·최종 phase·owner/GPU PID 부재를 확인한다.
- 완료 조건: 5090 전체9개의 terminal/binary manifest를 검증한 뒤 승인된 immutable 원본 회수를 한 번 수행하고 모든 local archive/manifest/rawSHA를 확정한다.

**같은 선택 epoch의 여섯 지표로 보정 기여를 판정한다 — 다음 작업**
- 대상: paper_research A10015+RTX12의 원래27 ID. 전체 Validation 최초 strict finite 최소 raw수량RMSE epoch의 전체/큰수량 RMSE·MAE·TimeNLL을 함께 보존하고 3seed 평균·표본SD를 계산한다.
- 완료 조건: full이 두 대조보다 같은2개 이상 dataset의 각3seed 전체RMSE에서 strict 개선일 때만 Q1 개발지지를 기록한다. 불리한 조건·큰수량/시간 손해를 제외하지 않는다.

**확정 증거에 맞춰 논문 초안과 후속 계약을 정리한다 — 다음 작업**
- 대상: paper_research의 CNN+GRU54 중심 Results, Instacart 및 Methods/주장. 원본·기여 판정 후 문구를 일치시킨다.
- 이후 순서: 검색 포함 여부/필요시 새 공통 자원·mean 계약 → 미노출 평가 가능성·CPU 감사 계약 → 관련연구/References/Appendix → 컴파일·투고 점검.
- 고정 검색은 승인된 후속 자원/공통mean 계약 전 대기이며 빈 GPU를 이유로 자동 실행하지 않는다. A100 full을 RTX 동일GPU mean과 혼합하지 않는다.

RAF42 full/current_only의 물리 A100 이관과 cross-seed GPU/Runtime 차이를 최종 해석에 표시하며 epoch timing은 물리 GPU 구간별로 분리한다. 현재상태 대조의 hidden에는 prefix 이력이 포함되고, 파라미터 수가 맞아도 계산 경로/깊이/활성화 비용이 같다는 뜻은 아니다. 이 결과는 Validation 개발 증거이며 통계적 유의성·참신성·외부 우월성·새 독립 평가 증명으로 확대하지 않는다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.

최종 Validation 보고 및 모든 원본/실제 종료 조건이 아직 충족되지 않았으므로 자동화 titantpp-instacart-63을 유지한다. 다른 자동화·Pod·서버 작업·Runtime·인증·기존 원본은 변경하지 않았다.

## 봉인된 관측 원본

- advance: [advance_1791678326130298000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791678326130298000.json); rawSHA `5175106e8fc2972cdf948fe985ff9c4bed21db52560ffa4c646a7730b9e42ae0`.
- 5090 snapshot: [snapshot_5090_1791678326000723000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/snapshot_5090_1791678326000723000.json); rawSHA `96bfaa0d0f2692b4f25b269841817d8049964497f645c585d40e9df02af5d842`.
- 5080 terminal snapshot: [snapshot_5080_1791608016546790000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/snapshot_5080_1791608016546790000.json); rawSHA `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`.
- 이전08:41:45 KST advance: [advance_1791675706124856000.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791675706124856000.json); rawSHA `b1731c7dab50b8ae52ac7948457983739e7072b2dbca45f4650c6b99bf215c5b`.
- 논문 작업 순서 기준선: [PAPER_WORK_ORDER.md](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md).
