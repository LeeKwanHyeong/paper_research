# TitanTPP Backbone 후보 6조건 — A100 SXM 실행 보고

관측: **2026-10-03 08:03:16 KST**. 학습 시작 및 첫 체크포인트 저장을 확인했다. 성능 비교 결과는 아직 확정되지 않았다.

**구현과 실행 검증 — 완료**

- 별도 실험 사본에 두 Backbone 후보를 구현했다. 공유 출력부·손실·AdamW·입력·학습 대상·선택 기준을 유지한다. 기존 History MLP는 재학습하지 않는다.
- 최근 4개 이력 가중 결합: 현재 상태와 과거 상태의 기존 폭4 투영을 query/key/value로 재사용하고, 과거 최대4개 관측 상태를 softmax 가중 합산한다. 미래·현재 타깃·withheld 경계를 넘지 않는다.
- 블록 뒤 보정: 첫 번째 블록에서 구한 기존 인접 상태 보정 신호를 두 번째 블록 뒤, 정적 메모리 앞에 더한다.
- 두 후보 모두 보정6,144개·전체96,003개 파라미터, 고정 /8, 기존 분기 가용성, 출력zero 초기화를 유지한다.
- 로컬 CPU 테스트9개, native GPU 수식·gradient·padding·withheld 경계·인과성·초기 상태·체크포인트 식별 검사를 통과했다. 세 데이터의 기존 MLP 체크포인트 SHA와 validation 재현도 통과했다.

**승인된 6조건 학습 — 진행 중**

- 소유 Pod `c88rn6dvhdafj0`: A100 SXM4 80GB **1장**, Secure Cloud,32vCPU·251GB RAM.
- seed42, 최소40/최대300epoch, patience40. 최초의 엄격한 최소 finite validation raw quantity RMSE epoch를 선택하며 MAE·RMSE·time NLL은 같은 epoch를 사용한다.
- 현재3조건 실행·3조건 대기. 빈 슬롯은 배포된 dispatcher가 채운다. 첫2epoch의 시간 전망은 기록하지만 그 추정만으로 중단하지 않는다. 실제48시간·비용 한도는 유지하며 자동 retry/resume은 없다.

| 데이터 | 후보 | 상태 | 저장 epoch |
|---|---|---|---:|
| Intermittent | 최근 4개 이력 가중 결합 | 학습 중 | 미저장 |
| Intermittent | 두 번째 블록 뒤 보정 | 학습 중 | 미저장 |
| Taxi | 최근 4개 이력 가중 결합 | 학습 중 | 1 |
| Taxi | 두 번째 블록 뒤 보정 | 슬롯 대기 | 미저장 |
| RAF | 최근 4개 이력 가중 결합 | 슬롯 대기 | 미저장 |
| RAF | 두 번째 블록 뒤 보정 | 슬롯 대기 | 미저장 |

실제 GPU 학습 PID1436·1438·1440, dispatcher1431을 확인했다. Taxi 가중 결합 후보의1epoch 체크포인트는1,658,165bytes이며 저장 영수증의 SHA가 남아 있다. Intermittent 두 후보는 관측 당시 첫epoch 진행 중이었다.

**동시 실행 수 검증 — 완료**

| 동시 실행 수 | 동일6조건의 합산 작업 완료 시간(초) |
|---:|---:|
| 1 | 9.9448 |
| 2 | 7.2207 |
| 3 | 6.3608 |
| 6 | 6.1211 |

6개 구성이 가장 빨랐지만3개 구성도5% 이내였다. 사전에 정한 최소 동시 실행 수 규칙에 따라3개를 선택했다. 이 값은 synthetic 측정이며 프로세스 시작·워밍업을 제외한다. 본학습 ETA나 논문의 효율 측정으로 쓰지 않는다.

**비용과 원본 보존 — 진행 중**

- 사용자 승인 총상한$100, 자동충전 없음. 작업$80·회수와 정리 여유$20. 준비·승인 대기·본학습·저장료를 포함한다.
- GPU$1.59/h, 저장료를 포함한 보수 요율$1.5969444444/h. 계정 잔액 변화는 다른 Pod·충전이 섞일 수 있으므로 개별 청구로 해석하지 않는다.
- 기존 시간 상한은 그대로이며 로컬 종료 한도는 **2026-10-05 07:41:29 KST**다. 이는 ETA가 아니다. 미완료 시 부분 기록을 보존한다.
- 제공자 stopAfter 요청은 **2026-10-05 08:41:29 KST**로 제출됐으나 실제 적용은 확인되지 않았다. 승인 대기 중 같은 Pod를 일시정지 후 재가동한 이력을 보존했다. 이 요청만으로 과금 종료를 보장하지 않는다.
- 비용·회수 관리자PID53515 생존 및 **2026-10-03 08:04:07 KST** 기록을 확인했다. 원본 archive의 SHA를 대조한 뒤 이 소유 Pod만 stop/delete한다. 원본 회수 실패 시 디스크를 보존한다.
- 학습은 원격에서 지속하지만 Mac 절전·통신 장애 시 제공자 API를 통한 회수·과금 종료가 지연될 수 있다.

**완료 결과의 원본 회수와 검증 — 다음 작업**

- 여섯 조건 각각 terminal manifest, selected/last validation replay, 파일 SHA를 확인한다. 완료·실패·미시작을 모두 남긴다.
- 배포된 관리자가 원본을 회수하고 소유 Pod 삭제·비용 보고를 마친 뒤 checkpoint binary CPU 감사와 조건별 검증 목록을 정리한다.

**기존 MLP와 개선 폭 비교 — 다음 작업 / 원본 검증 이후**

- 같은 seed42·validation 기준선과 MAE·RMSE·time NLL을 비교한다. 단일seed·이종GPU 탐색 결과로 취급한다. held-out 성능과 예측을 열지 않으며3seed나 독립 최종평가 결과로 표현하지 않는다.

**증거**

- Bundle: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_routing_placement_a100_20261003_v1`
- Contract canonical SHA: `1db5a7e0cf97b8a667439b86cca8455a6e98158775c46ef1d19efdd26a5cbea0`
- Frozen110-source closure: `2b8156fab28cf44e9c2fd4cd1f2888110f89f98874af5820a60037d44cc51870`
- Native qualification original-file SHA: `6bf64ef2af6b9f51e6b5cfa8fc08ac20b2867c70e5e3625deb86190b6e0193aa`
- Observation: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_routing_placement_a100_20261003_v1/observations/20261002T230316Z/snapshot.json`
- `launch_receipt.json`, `verification/first_checkpoint_confirmation.json`, `transfer_authorization.json`, `verification/cpu_tests.xml`, `verification/source_delta.json`을 함께 보존한다.
- 이 사이드 실험의 사본과 기록만 변경했다. 기존 논문·공유 모델·이전 캠페인·모니터는 변경하지 않았다.
