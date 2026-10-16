# TitanTPP 원본 보존과 RTX 재개 시간별 관측

**A100 원본 회수와 실제 잔여 조건 확정 — 완료**

- 사용자2026-10-05 “진행하자” 승인으로 원본428파일·archive SHA를 검증했다. TaxiCNN+GRU54는166/best126으로 완료된 것을 확인했다. 원래9조건 중7개 scientific_success·소형파일SHA·selected/last full Validation이 검증됐다.
- 소유 Pod okbl76hxfr7a9d는 원본 회수 후 정지·삭제했고 목록 부재를 확인했다. 더 이상 A100 SSH·manager·provider 관측을 하지 않는다. 보수 비용 약$25.82는 청구서와 구분한다.
- [이관 계약과 증거](../../search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/README.md), [원본 목록](../../search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery_inventory.json), [cleanup](../../search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/cleanup_receipt.json)을 참조한다.

**미완료 두 조건의 병렬 재개 — 진행 중**

- 5080 Intermittent GRU54 seed42: 원본24/best10, 다음25epoch. 운영 계약6581c156867c7ad1b317d1af6fb3dfffc29bcf55def953f6264ff6b2de4f6a02.
- 5090 Intermittent CNN+GRU54 seed42: 원본12/best12, 다음13epoch. 별도 전용 Python·NumPy2.4.4 운영 계약6b85d3ea3b5b096653e2d5ba2ffbf35ceeefd5c58f3fe1699c0be29ae9eaf5fb. 공용 ai_env는 바꾸지 않았다.
- 현재 root는 /Users/igwanhyeong/PycharmProjects/paper_research, 신규 bundle은 search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1이다. current.json의 host_contracts가 서버별 운영 계약을 지정한다. 부모46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae와117source40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd는 동일하다.
- 각 서버의 native 초기화·결정성·인과성·메모리와 source selected/last fullValidation 검증을 통과한 receipt에 training permit을 묶었다. 동결117소스·worker4·학습·손실·선택·원본 history·optimizer·난수상태는 유지한다.
- 전용root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5080 및 같은 이름의_5090이다. single worker, once-only claim, 자동retry없음, 조건36h·전체168h의 native timeout이 적용된다. 마감은 ETA가 아니다.

**서버별 한 번의 읽기 전용 관측 — 진행 중**

- 매시간 아래 명령을 독립적으로 각각한번 호출한다. 필요한 읽기 전용 네트워크 권한을 사용한다. 다른SSH/관측기/qualification/학습/replay/재시작/claim삭제를 추가하지 않는다.

```text
/usr/local/bin/python3 paper/scripts/observe_titantpp_cnn_gru_resume_hourly.py --host 5080
/usr/local/bin/python3 paper/scripts/observe_titantpp_cnn_gru_resume_hourly.py --host 5090
```

- TEST_SESSION_PROTOCOL 0·1·9·12와 신규bundle README/current·서버별계약·approval·start/training permit·qualification receipt·hourly latest를 읽는다. 계약/current/GPU/root가 달라지면 과거대상으로 접속하지 않고 변경 이유를 확인한다.
- 각실제KST관측시각,조건·seed·완료/실패/진행/대기/미확정,저장/best epoch와본GPU병렬수를 짧은한국어표로 알린다. 원본7개와재개2개를canonicalID마다한번만집계한다. 원본완료관측/완료시각을 새시각으로바꾸지 않는다.
- 실제supervisor/worker argv와PID/GPUUUID·PID,source continuation/checkpoint receipt·history·exposure·epoch timing을대조한다. 저장JSON만으로진행/완료를선언하지 않는다. PID부재·checkpoint불일치는미확정으로보고하며자동재시작하지 않는다. SSH실패는조회실패이고학습실패가아니다.
- 성능은 최초최소fullValidation raw수량RMSE epoch의RMSE·MAE·TimeNLL을같은epoch에서best-so-far로읽는다. Test·혼합파일·원시예측은읽지않는다. 큰수량Train경계Intermittent187초과; Taxi3449/RAF200은원본참조값이다. 단일seed·이종GPU 탐색이며3seed/논문효율결론으로쓰지않는다.
- ETA는source_epoch이후해당RTX fit의최근최대10개완료epoch중앙값만사용한다. best미갱신·현재속도유지가정으로min(300,max(40,best+40))와300epoch시나리오를별도표시한다. 첫새epoch가없으면자체속도/ETA미확정이다. A100속도를복사하지않는다. endpoint/회수/CPU감사는ETA와별도다.

**폭8·12와 기존 Test 완료 증거 — 현재 기준선**

- 과거폭8/12의18조건과Intermittent6의Validation gate·Test평가는완료다. [최종 비교 보고](../titantpp_three_dataset_final_comparison_20261005_v1/README.md)를 재사용한다. 이전학습관측기·finalize Test·추가평가를 실행하지 않는다. 이번CNN/GRU에서Test는승인범위가아니다.

**재개 완료 확인과 최종 원본 회수 — 다음 작업**

- 두조건 모두terminal scientific_success·소형원본SHA·selected/last fullValidation·서버종료증거가확인돼야완료로집계한다. 실패/미시작/불리한결과를빠뜨리지않는다. 종료server는terminal캐시를사용해반복SSH를끝내며원래관측시각을보존한다.
- 새로운RTX checkpoint 원본manifest/SHA 회수는학습완료와별도로확인한다. 재개2조건과서버terminal,새원본회수,기존A100삭제·비용·원본SHA,기존IntermittentTest완료가모두확인된뒤최종보고후이통합자동화만삭제한다. 미확정이면유지한다.
- binary내용검사와CPU재추론감사는별개다. CPU감사가남으면전체연구감사완료로표현하지않는다. 다른자동화·PAUSEDmonitor·Pod·서버작업·chatarchive상태를바꾸지않는다.
