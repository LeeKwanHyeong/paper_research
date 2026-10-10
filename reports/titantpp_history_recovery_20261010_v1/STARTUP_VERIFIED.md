# 남은 학습 복구 실행 확인

실제 관측: **2026-10-10 11:25:17 KST**. snapshot rawSHA `75ebf63110ae415f8aebe35f244a4e3a30a15a53db03e368d2a702532fdb3d88`.

| 작업 주제 | 상태 | 확인 결과 |
|---|---|---|
| Instacart9 | 완료 | 9/9 학습·614개 원본 및 선택/최종 checkpoint9쌍 SHA 검증 |
| CNN+GRU54 이력 보정27 | 진행 중 | 완료11(부모10 재사용+재개1)·진행1·대기15·실패/미확정0 |
| 고정 검색 추가 효과27 | 대기 | 이력 보정 완료와 후속 자원/공통 mean 기준선 계약 확인 후 진행 |

| A100 RAF seed42 | 상태 | 실제 저장 / best epoch | GPU worker PID |
|---|---|---|---|
| full | 진행 중 | E24 / best9 | 1109 |
| current_only | 완료 | E49 / best9 (E47부터 재개) | 종료 원본 terminal 검증 |
| disabled | 부모 완료 재사용 | 기존 완료 원본 유지 | 새 GPU에서 재실행0 |

새 Runtime은 부모 버전과 일치한다. native qualification actual exit0, 동시에 GPU PID3개(579/581/582) 확인, 각 admission child exit0 및 idle을 확인하고 fit 단계에 진입했다. admission은 과학 fit 완료 수량에 포함하지 않는다. 첫 scientific batch는 미완료 RAF42 2개였고 current_only가 먼저 종료했으므로 이 관측의 실제 학습 GPU PID는1개다. 이후 미시작 RAF52/62·Intermittent42/52/62는 batch당3개 동시 실행하도록 봉인됐다.

full의 새로운 E24 저장은 부모 E23을 진척으로 재표시한 것이 아니다. optimizer/RNG/loader/history/best 연속성 검사를 유지한 frozen trainer가 새 checkpoint receipt를 발행했다. current_only는 E46 원본에서 이어 E49 종료와 selected/last terminal manifest를 검증했다. 물리 GPU 이관을 같은 GPU/무중단 수치 동등성으로 주장하지 않는다.

full ETA는 **신규 완료 epoch1개**의 실측 44.36초만 사용했다. best9가 더 갱신되지 않고 같은 속도가 유지되면 E49까지 약 18.5분, E300 시나리오는 약 3.40시간이다. 단1개 표본의 조건부 값이며 전체 queue ETA는 미확정이다. 회수·집계·CPU 감사 시간은 포함하지 않는다.

누적비용 guard의 최근 추정 $30.6365, 총 상한 $50. 준비·학습 lease 마감 **2026-10-10 19:28:12 KST**, 절대 compute 비용 종료 2026-10-10 22:15:27 KST. 준비·학습에 회수 예비$5를 배정하지 않았고 새로운 GPU 비용만으로 $50을 초기화하지 않았다. 비용 상한 내 남은 모든 fit 완료 가능성은 아직 미확정이다.

운영 수정은 `paper_research`의 `codex/contextual-retrieval-pilot-20261007` 로컬 commit **4a52b17**에 보존했다. frozen scientific source revision은 **57da14a2a7a5687e1bdef7d3164f3e8c0051da59**로 유지했다. Push는 수행하지 않았다.

## 남은 순서

1. RAF42 full과 나머지15 fit — 진행 중. 봉인 dispatcher가 실제 종료 후 다음3개 batch를 시작한다.
2. 선택/최종 binary 원본 회수·27개 Validation 비교와 Q1 판정 — 학습 종료 후. 큰 수량 RMSE/MAE와 TimeNLL 손해를 함께 보고한다.
3. 새 Pod/전용 network volume 정리 — 전체 원본 SHA 확인 후. 비용 manager와 terminal service를 활성화했고 기존 시간별 자동화 하나만 새 bundle로 갱신했다.
4. 고정 검색 자원·동일 기준선 계약 — 대기. Test·CPU binary 재추론 감사·새 독립 평가는 별도 미완료다.
