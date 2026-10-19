# 5090 복구 실행 상태

**원인과 기존 원본 보존 — 완료**

- 2026-10-19 17:24:09 KST의 5090 서버 재부팅으로 원 dispatcher와 진행3개 worker가 함께 사라졌다. 재부팅을 유발한 구체적 원인은 시스템 로그 권한 제한으로 미확정이다.
- 원본1126파일과 완료 RAF6 결과·선택 checkpoint, 중단3개 전체 epoch checkpoint를 v1 origin_archive에 보존했다. 이번 새 복구 경로를 설치한 뒤에도 원 원격1126파일이 모두 같은 SHA임을 확인했다.
- 복구 v1 사전 검사 실패는 빈 sample_data 디렉터리 누락에 따른 import 실패였으며 실제 학습0이다. v1 계약/코드/실패/원격 경로는 그대로 두고, v2 별도 배포 구성만 수정했다.

**5090 새 경로에서 복구 학습 — 진행 중**

실제 관측: **2026-10-19 20:27:28 KST**. 이 시각을 후속 현재 관측으로 재사용하지 않는다.

| 작업 | 복구 기준 | 직접 관측 상태 |
|---|---|---|
| Intermittent52 관측 전달 | 원 epoch15 전체 상태 | worker365313·GPU 실행 중 |
| Intermittent52 용량 대조 | 원 epoch13 전체 상태 | worker365314·GPU 실행 중 |
| Intermittent62 관측 전달 | 원 epoch13 전체 상태 | worker365315·GPU 실행 중 |
| Intermittent62 용량 대조 | 미시작, 원 seed의 fresh fit | 대기 |

- native 합성 child6개 실제 exit0와 qualification process exit0 이후 시작했다. 최대3병렬이다.
- worker의 PID·argv·start_ticks·lease·실제 GPU 연결, 원 checkpoint 복사본 SHA와 입력 승인 receipt를 확인했다. 당시 학습 stderr는 모두0byte이며 새 history 저장은 아직 없었다. 새 epoch 완료나 optimizer update를 이 관측만으로 확정하지 않는다.
- source963 closure와 입력6, 모델·손실·head·공통 조건·예산·선택·기한은 그대로다. runtime/과학worker SHA도 동일하다. 기존 CNN/MLP16·완료 RAF 학습을 다시 실행하지 않는다.
- 학습 진단용 읽기 전용 관측의 첫 호출은 로그 경로 오기로 exit1이었다. 실제 logs/JOB.train.stderr.log로 수정한 관측이 통과했다. 원문을 보존했으며 학습을 재시작하지 않았다.

**학습 종료 후 Validation과 회수 — 다음 작업**

- 이번4개 종료 후 선택 checkpoint를 고정해 Intermittent4 + 저장 RAF6의 native Validation10을 실행한다. Test는 범위에 없다.
- 실제 종료와 원본·checkpoint·같은 대상/여섯지표 재현·소유 프로세스 부재를 확인한 뒤 별도 retrieval archive/manifest로 회수하고 모든 SHA를 로컬에서 재검증한다.
- 로컬 유한 완료 대기 프로세스 **91637**를 분리 실행했다. 학습→Validation10→회수로 종료하며 반복 자동화를 만들지 않았다. 부모 PID1로 유지되는 것도 직접 확인했다.
- 원격 자식·dispatcher 종료와 wrapper 자신의 OS 종료는 구분한다. 재부팅·미확정 종료·실패 시 원문을 보존하고 자동 재시작하지 않는다.
- 이 문서는 최종 완료 보고가 아니다. 학습/Validation/회수 완료 판정은 retrieval/final_receipt.json과 원본/SHA 검증 이후에만 한다.

**증적**

- NATIVE_START.json / START_RECEIPT.json / START_INDEPENDENT_QA.json
- POST_START.json / COMPLETION_PROCESS.json
- PREPARED.json / LAUNCHED.json / INDEPENDENT_PACKAGING_QA.json
- source commit c18aea2f (paper_research, codex/contextual-retrieval-pilot-20261007).
