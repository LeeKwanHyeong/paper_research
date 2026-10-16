# Instacart9·이력 보정27 최종 관측 및 논문 작업 순서

관측 자동화: `titantpp-instacart-63`. 평가 범위는 **Validation-only**다. Test/held-out 예측·성과와 Validation/Test 혼합 파일을 열지 않았다.

**학습 종료와 원본 보존을 확인합니다 — 완료**

- 5090 실제 관측: **2026-10-11 14:21:09 KST**. 완료 원본 회수·Validation 집계 기록: **14:21:13 KST**. 이 문서의 작성 시각은 실제 원격 관측 시각을 대신하지 않는다.
- 이력 보정은 A100 완료15 + RTX5080 완료3 + RTX5090 완료9 = 원래 실행 ID **27개 모두 완료**다. 수치 실패0, 진행0, 학습 대기0, 미확정0이다. Instacart도 기존9개가 완료됐다.
- 마지막 Intermittent seed62/full은 저장 E53, 선택 E13이다. worker 실제 exit0는 **2026-10-11 13:39:31 KST**, supervised dispatcher 실제 exit0는 **13:39:32 KST**, native pipeline_complete 기록은 **13:39:33 KST**다. 이전 저장 E50/선택 E13에서 실제 E53 저장과 종료로 진행했다.
- 5090 관측에서 학습 소유 프로세스와 GPU PID는 비어 있었다. historical worker PID357843/group357842는 종료 증거이며 현재 실행 PID가 아니다. PID·PPID·정확 argv, GPU UUID, lease/claim/receipt 연결은 봉인 snapshot과 worker/supervisor 원문으로 검증했다. **pipeline OS exit 코드를 직접 관측한 것은 아니다.**
- 5080 완료 관측은 **2026-10-10 13:53:36 KST**를 재사용했다. A100 완료15의 원관측과 2026-10-10의 회수·정리 시각도 그대로 유지했다. 종료한 service heartbeat를 live 장애로 해석하지 않았다.
- Instacart 과거5080 전역 GPU idle 미확정(NVML/kernel 버전 불일치)은 이미 완료한 학습 실패와 별개다. 해당 원문과 실제 owner 종료·supervisor exit0 증거를 그대로 보존했다.
- 첫 batch의 세 arm 동시 실행은 기존 admission/native GPU PID 증거로 확인됐다. 이번 관측은 모든 worker 종료 후이며 동시 학습0이다. 종료된 fit의 ETA는 해당 없음이며, 회수·CPU 재추론 감사를 학습 ETA에 합치지 않는다.

| 주제/자원 | 완료 | 실패 | 진행 | 대기 | 미확정 | 관측 근거 |
|---|---:|---:|---:|---:|---:|---|
| Instacart 기존5080/5090 | 9 | 0 | 0 | 0 | 0 | 이전 실제 종료·614개 원본 로컬 재검증 |
| A100 이력 보정 | 15 | 0 | 0 | 0 | 0 | 보존 완료 원본·원관측 재사용 |
| RTX5080 RAF seed62 | 3 | 0 | 0 | 0 | 0 | 2026-10-10 13:53:36 KST terminal cache |
| RTX5090 Intermittent seed42/52/62 | 9 | 0 | 0 | 0 | 0 | 2026-10-11 14:21:09 KST 실제 종료·idle |
| 후속 고정 검색 | 0 | — | 0 | 계약·포함 여부 결정 대기 | — | 이번 학습27과 별도 |

**각 조건의 저장·선택 epoch를 기록합니다 — 완료**

아래 셀은 `최종 저장 epoch / 선택(best) epoch`다. arm 순서는 full, 보정 제거, current_only다. 선택 기준은 전체 Validation의 최초 strict finite 최소 raw 수량 RMSE이며, 동일 선택 epoch의 여섯 지표를 사용한다.

| Dataset | Seed | 완료 원본 host | full 저장/선택 | 보정 제거 저장/선택 | current_only 저장/선택 |
|---|---:|---|---:|---:|---:|
| intermittent_frozen_5000 | 42 | 5090 | 55/15 | 70/30 | 46/6 |
| intermittent_frozen_5000 | 52 | 5090 | 51/11 | 53/13 | 96/56 |
| intermittent_frozen_5000 | 62 | 5090 | 53/13 | 45/5 | 78/38 |
| raf_spare_parts | 42 | a100 | 49/9 | 49/9 | 49/9 |
| raf_spare_parts | 52 | a100 | 45/5 | 45/5 | 83/43 |
| raf_spare_parts | 62 | 5080 | 60/20 | 66/26 | 62/22 |
| yellow_trip_hourly | 42 | a100 | 166/126 | 181/141 | 122/82 |
| yellow_trip_hourly | 52 | a100 | 114/74 | 108/68 | 59/19 |
| yellow_trip_hourly | 62 | a100 | 124/84 | 132/92 | 83/43 |

- RAF seed42/full·current_only는 물리 A100 이관을 거친 재개 조건이며, 보정 제거와 물리 GPU 구간이 완전히 같지 않다. cross-seed에는 A100/RTX와 Runtime 차이가 있다. 원래27 ID를 중복 없이 한 번씩 집계했고 부모의 운영 중단을 새로운 수치 실패로 더하지 않았다.
- 같은 dataset/seed의 RTX 세 arm은 같은 GPU/Runtime에서 최대3개씩 실행했다. B128/min40/max300/patience40·초기화·data/loss/optimizer/selector는 유지했다. full/current_only의 활성 보정 파라미터25308은 같지만 계산 경로·깊이·활성화 차이, current hidden의 prefix 이력이라는 한계는 남는다.

**봉인과 binary 원본을 확정합니다 — 완료**

- canonical `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`, source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, operation closure `8a7ff350488fe8b03289ea6e6683a41ace4c95c2bdf0d24f2c06f3d78f5c83da`가 모두 일치했다. science revision은 `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`다.
- 최초 science integrity, local tools, host별 control, monitor integrity, current/contract/approval 및 존재하는 permits/deployment/native qualification/admission·ownership release 연결을 stdlib SHA로 확인했다. source129/input7/op2와 봉인 project_files/control 전체 대조가 통과했다.
- RTX5090 UUID는 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, RTX5080 UUID는 `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`다. root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_<host>`이며 원문 root·GPU·계약 일치를 확인했다. host별 기존 Runtime·cuDNN91900/92000을 유지했다.
- 기존 Instacart614 + A100 recovery442 + 부모393 + RTX5080 233 = 보존 원본1,682개를 먼저 재검증했다. 이 감사는 고유1,701개 파일·1,737회 SHA 비교를 통과했다. 부모393개는 보존 원본이며 새로운 실행393개로 세지 않는다.
- 새 RTX5090 원본 **341/341개**와 선택/last checkpoint **9쌍·18개**를 검증했다. manifest 밖 추가·누락 각각0, SHA 불일치0이다. 새 감사는 고유356개 파일·386회 SHA 비교를 통과했고 완료 receipt의 local10/external3 참조도 일치했다. 전체 보존 원본 수는 **2,023개**다.
- Instacart 선택/last9쌍과 RTX5080 선택/last3쌍은 기존 로컬 원본 재검증을 통과했다. checkpoint는 bytes SHA로만 감사했으며 CPU binary 재추론은 수행하지 않았다.
- 실제 학습 종료·worker/supervised dispatcher exit0·native pipeline_complete·owner/GPU PID 부재·native qualification/admission·binary manifest의 승인 조건을 통과한 뒤에만 5090 원본을 회수했다. native pipeline_complete의 당시 `binary_originals_retrieved:false`는 원문 그대로 보존하고 후속 로컬 회수 receipt를 별도로 사용했다.

| 증거 | SHA-256 |
|---|---|
| 최종 advance | `096304e3cb5a003945573996dcd99b49ce6cf27237f928da10acaf479950a53f` |
| 5090 실제 snapshot | `b24b327ff1d8cd16aab3524e59d3b49dbb093daf708dab96e1ca970fe635e406` |
| 5090 원본 회수 receipt | `3b9daba36fc441001938650a076579f815dd4b7d946abb06ae6a041f149679f8` |
| 5090 archive | `a7e84c1ca3f91934fcbcdd972dcb9fe4aba5662bd07b8628cc5ddf7d7365ff64` |
| 5090 manifest | `7ed72444970003811ef9cd8a69fe4a56bf34000c6e7fd239286dd58286b971d0` |
| 최종 completion receipt | `656eef38d3192d57f89e86aaa2ed65e69ae5c256288668c1819db7e838c48362` |
| 선택 조건27 | `41ddf10b5379b2ba647e1c8500b7e02e2214bfa127e797d0ee293c1050668742` |
| 최종 Validation 보고 | `8d1bbb13dde7d1ef529176daef15251ad647fbd86bf9ea308a253d66ea5ff87a` |

증거 링크: [advance](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791696073570334000.json), [5090 snapshot](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/snapshot_5090_1791696069884430000.json), [5090 회수 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/originals/5090/receipt.json), [최종 completion receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/analysis/completion_receipt.json), [선택 조건27](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/analysis/selected_conditions.json), [여섯 지표·평균/표본SD·불리한 조건 전체 보고](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/analysis/VALIDATION_HISTORY_REPORT.md). 역할을 분리한 독립 원본 감사와 Validation 재계산 결과는 아래에 기록했다.

**보정의 개발 기여와 불리한 조건을 함께 판정합니다 — 완료**

**독립 Validation 계산 감사 PASS — 불일치 없음.** 원래27 ID(A10015+RTX12), 27개 최초 strict finite 최소 raw 수량 RMSE epoch, 동일 epoch 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표, 3seed 평균·표본SD를 재계산했고 선택 조건 및 최종 보고서와 일치했다.

전체 Validation raw 수량 RMSE의 seed42/52/62 평균 ± 표본SD는 다음과 같다.

| Dataset | full | 보정 제거 | current_only |
|---|---:|---:|---:|
| Taxi | 80.408886 ± 2.982982 | 87.611674 ± 1.416615 | 88.000769 ± 1.375826 |
| RAF | 34.171127 ± 0.212086 | 34.296418 ± 0.084585 | 34.310912 ± 0.101014 |
| Intermittent | 1.695738 ± 0.042287 | 1.740864 ± 0.027858 | 1.756219 ± 0.057444 |

사전 정의한 **Q1 개발 지지 기준은 True**다. Taxi(`yellow_trip_hourly`)와 Intermittent(`intermittent_frozen_5000`) 각각 모든3seed에서 full의 전체 RMSE가 두 대조군보다 엄격히 낮다. RAF52는 full34.401169가 보정 제거34.341489·current_only34.334924보다 높아 RAF는 같은 기준을 충족하지 않는다. RAF 평균이 낮다는 사실로 seed별 strict 판정을 대신하지 않는다.

전체18개 대조(full 대 두 대조×세 dataset×세 seed) 중 전체 TimeNLL 손해11개, 큰수량 RMSE·MAE 손해 각각10개를 보존했다. Intermittent52/62의 큰수량 손해, Taxi52/62의 큰수량 손해와 Taxi의 전체 TimeNLL 손해, RAF52의 전체/큰수량 손해를 제외하지 않는다. 전체 six-metric 값과 모든 불리한 조건은 최종 Validation 보고서에 남아 있다.

원격·학습·재추론·저장소 변경 없이 local Validation 원본 history와 selected/last 지표로 감사했다. 통계 유의성 및 독립 확인은 false다.

독립 계산 감사 원문: [Validation 감사 JSON](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_rtx_handoff_20261010_v1/FINAL_VALIDATION_AUDIT_20261011.json), rawSHA `849a1986577cf87aa896650bc547ab1f906c6655e34e1d8e9fef9df63839e397`.

- 이는 고정된 Validation 개발 판정이다. 통계 유의성·참신성·외부 우월성·새 독립 평가를 입증하지 않는다. A100 full9를 RTX 동일GPU mean에 섞지 않는다. 시간은 host·물리GPU 구간으로 분리하며 세 fit 동시 wall time을 구조 효율로 해석하지 않는다.

**A100 자원 정리와 관측 종료 조건을 확인합니다 — 완료**

- 부모 Pod `qq2uk2dpfi2rum`는 이전 삭제 receipt, recovery Pod `z278euk8d74nqt`와 전용 network volume `pkjlrapcbm`은 **2026-10-10 12:25:29 KST**의 삭제 receipt로 부재가 확인되어 있다. 해당 receipt와 archive/manifest·부모393 로컬 SHA를 재검증했으며 삭제된 Pod의 API·SSH 재조회는0이다.
- 승인 인계 release 원문과 exclusive admin claim·FileExistsError·실제 부모 dispatcher/pipeline exit1·failure는 그대로 보존했다. 부모 운영 중단을 과학 결과 실패로 조작하거나 부모27 전체 완료로 표현하지 않는다.
- Runpod 누적 비용은 **추정32.36242173899348달러**, 최종 청구는 미확인이다. 기존 총50달러 상한을 유지하며 새 Pod·자동충전·예산연장·가격/Runtime 변경은0이다.
- 이번 회차의 승인 helper 실행은 정확히1회, 내부 finalizer 호출1회다. live5090 읽기 전용 관측 SSH1회 후 승인된 종료 원본 collector/archive transfer2회가 실행됐다. 5080·Instacart·삭제 A100 관측 SSH는0이다. 별도 observer/polling/launch/qualification/학습/kill/retry/claim 삭제/driver·Runtime·인증 변경은0이다.
- Instacart9/history27 실제 종료, 모든 archive/manifest 로컬 SHA, 최종 Validation 보고와 A100 Pod/volume 삭제 증거 조건이 모두 충족됐다. 이 최종 관측 보고를 보존한 뒤 **이 자동화 `titantpp-instacart-63`만 삭제**한다. 삭제 여부는 도구 반환으로 별도 확인하여 사용자에게 알린다.

**CNN+GRU54 중심으로 논문의 결과와 주장을 맞춥니다 — 다음 작업**

- 대상: paper_research 논문 초안. 완료된 CNN+GRU54 기준선과 Instacart9·이력 보정27을 구분해 Results에 연결하고, Methods의 보정 정의·선택 기준·GPU 이관 및 Runtime 한계를 결과와 맞춘다.
- 완료 조건: Validation-only 근거, 동일 epoch 여섯 지표, 3seed 평균/표본SD와 불리한 조건이 표·본문 주장에 일치한다. 완료된 학습을 다시 실행 목록에 올리지 않는다.

**고정 검색을 논문에 포함할지 결정합니다 — 다음 작업 / 실행 승인 필요**

- 초안의 핵심 주장과 기존 기준선에 필요한 비교인지 먼저 판단한다. 검색이 필요하면 새 공통 GPU/Runtime·초기화·데이터·선택·자원 계약을 확정한 뒤 승인된 범위로 진행한다.
- 고정 검색27/36개는 대기이며 이번 종료를 이유로 자동 실행하지 않는다. A100 full을 RTX 동일GPU mean과 혼합하지 않는다. 완료 조건은 포함 여부와 필요시 공통 계약·실행 승인의 명시다.

**미노출 평가와 CPU 재추론 감사의 범위를 확정합니다 — 승인 필요**

- 실제 미노출 평가 가능성을 먼저 확인하고 평가 오염·범위·source revision·binary 재추론 감사 계약을 정리한다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.
- 완료 조건: 승인된 독립 평가와 감사 범위가 고정되고, 개발 Validation 판정과 확인 평가가 명확히 구분된다.

**관련연구와 재현 자료를 결과에 연결합니다 — 다음 작업**

- 대상: paper_research Related Work/References/Appendix. 최종 주장의 근거와 참고문헌을 대조하고 조건·선택·원본 provenance·운영 한계를 재현 자료에 반영한다.
- Results·Methods 정합 작업 이후 병렬로 문헌/References와 Appendix의 독립 파일을 검증할 수 있다. 공통 표·주장 계약 수정은 현재 세션에서 직렬로 진행한다.

**원고를 컴파일하고 투고 조건을 점검합니다 — 다음 작업**

- 앞 단계가 완료된 원고를 기준으로 표·그림·References·Appendix 연결 및 컴파일을 확인한다. 투고·공개 등 외부 실행은 구체적 대상과 승인을 확인한다.
- 이번 관측 작업은 논문 작성 순서만 재정렬했다. 저장소 커밋·Push·MR·공용 서버 배포는 수행하지 않았다. 기존 봉인 PAPER_WORK_ORDER.md와 실험 source/계약·원본을 변경하지 않았다.

현재 기준선 작업 순서: [봉인 PAPER_WORK_ORDER.md](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md). 이번 완료 후 다음 즉시 작업은 CNN+GRU54 중심 Results·Methods/주장 정합이다.
