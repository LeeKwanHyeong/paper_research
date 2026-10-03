# PRO4500 이관3조건 종료 관측 — 2026-10-03

실제 종료 관측 2026-10-03T05:58:04+09:00을 재사용한다. 현재 회차의 새 원격 조회가 아니다.

완료3·최종실패0·진행0·대기0·미시작0. 기존36조건 중 Intermittent seed62의3조건이며 추가 과학조건으로 집계하지 않는다.

| 모델 | 저장/best epoch | 완료 KST | validation MAE | RMSE | 시간 NLL |
|---|---:|---|---:|---:|---:|
| titantpp_current_only_param_matched | 82/42 | 2026-10-02T19:35:05+09:00 | 0.686230 | 1.706125 | 0.557191 |
| titantpp_all_available_history_mlp | 107/67 | 2026-10-03T01:09:38+09:00 | 0.683365 | 1.674399 | 0.380655 |
| deep_renewal_event_native_nb | 142/102 | 2026-10-03T05:57:40+09:00 | 0.605378 | 2.686848 | 1.083007 |

모든 지표는 최초 최소 raw 수량 RMSE의 같은 selected epoch에서 읽었다. Terminal manifest·소형 JSON SHA·selected/last validation replay를 확인했고, 회수된270파일과 압축본의 SHA를 다시 검증했다. Deep Renewal은 고유 NB 시간·수량 손실을 사용하며 원고 비교에서만 제외한다.

소유 Pod eu8yh29yympuf4는 2026-10-03T05:58:34+09:00 삭제·목록부재 receipt가 있다. 첫 준비시도 시작부터 삭제까지 저장료 포함 요율을 연속 적용한 보수 추정은 $10.9083, 승인상한 $25 이내다. 시도 사이 공백도 포함하는 추정이며 개별청구서가 아니다.

Current-only1조건의 기존 CPU감사를 재사용한다. All-available 및 Deep Renewal2조건의 checkpoint CPU감사는 별도 미완료다. 로컬 provider manager PID79797의 부재는 종료·삭제 receipt와 일치하며 재시작하지 않았다.

원본: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/retrieved/original`
회수: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/retrieved/retrieval_receipt.json`
삭제: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_intermittent_seed62_runpodpro4500_20261002_v1_retry1/control/cleanup_receipt.json`
