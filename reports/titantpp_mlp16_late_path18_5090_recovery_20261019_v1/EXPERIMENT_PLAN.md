# 재부팅으로 중단된 5090 백본 실험 복구

**현재 기준선과 원본을 보존한다 — 완료**

- 대상은 paper_research 저장소, `codex/contextual-retrieval-pilot-20261007` 브랜치와 5090 전용 실험 경로다. 사용자2026-10-19의 원인 확인·Intermittent4 복구·Validation 재검증·회수 승인을 적용한다.
- 재부팅17:24:09 KST를 확인했고 원본1126파일을 별도 archive/manifest로 보관했다. 재부팅을 촉발한 세부 원인은 관리자 system journal 접근이 없어 미확정이다. 기존 source·완료RAF6·checkpoint·실패·원래 중단 종료 코드 미확정은 보존한다.

**과학 조건을 유지한 새 복구 경로를 검증한다 — 진행 중**

- 새 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_mlp16_late_path18_5090_recovery_20261019_v1_5090`이다. 기존 root에서 학습을 재시작하지 않는다.
- 과학 계약·source revision·모델·입력·log-MSE·시간 가중치0·두 log1p 입력·AdamW·예산·선택 기준은 원형이다. 운영 코드와 출력 경로만 별도로 둔다. 상대 입력 경로를 유지하는 동일 SHA의 입력 사본을 새 source에 둔다.
- 중단3개는 모델/optimizer/RNG/loader 상태와 기존 history·best·시간 head 검증 횟수를 이어서 복원한다. 미시작62 용량대조1개는 같은 seed·공통 초기화로 처음 시작한다. 5090 최대3병렬이며 원래 조건36h 및 절대 host deadline을 연장하지 않는다.
- 모델·재개 adapter와 운영 runtime은 독립 파일로 병렬 구현하고 공통 계약·커밋·배포·native 검증은 순서대로 진행한다. 기존 source963과 신규 운영 코드를 각각 SHA로 결속한다. 기존 CNN과 다른 실험은 재학습·재추론하지 않는다.

**native 검증 뒤 남은 네 학습만 실행한다 — 다음 작업**

- 원본 파일 SHA·새 source/입력 SHA·실제 boot/GPU 유휴·같은 Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads를 확인한다.
- 합성 exact-resume 일치와 3개 동시 native qualification을 통과한 뒤 새 exclusive intent/claim에서만 실행한다. 새 학습 실패를 자동 재시도하지 않고 원문과 실제 child 종료 코드를 보존한다. 복구 child의 고정 제한시간 외에는 종료 신호를 보내지 않는다.

**선택 Validation을 재현하고 원본을 회수한다 — 외부 작업 대기**

- Intermittent4 학습이 모두 끝나면 기존 RAF6을 포함한 5090 전체10개의 선택 checkpoint를 native Validation으로 재검증한다. 최초 strict finite fullValidation 수량RMSE 최소 epoch를 고정하고 같은 checkpoint의 MAE·상한도달 지표·시간 진단값과 동일 평가 대상을 확인한다. Test는 실행하지 않는다.
- 실제 train4/Validation10/dispatcher 종료0, 소유 PID/GPU 부재, terminal binary·모든 archive/manifest/원본 SHA와 원래 root의 불변을 확인한 뒤 새 retrieval에 한 번 회수한다. wrapper 자신의 OS 종료와 직접 관측한 child 종료를 구분한다.
- 연결이 끊겨도 원격 복구는 tmux에서 이어지며, 유한 로컬 완료 처리에서 회수·검증까지 연결한다. 반복 자동화는 만들지 않는다. 최종 검증 전 진행을 완료로 표시하지 않는다.

**증적을 독립 커밋으로 정리한다 — 다음 작업**

- 신규 운영 코드·검사·원인 및 복구 기록만 paper_research의 위 브랜치에 커밋한다. 기존 사용자 변경이나 다른 실험의 산출물은 포함하지 않는다. Push·공용 Runtime 재배포·인증 변경은 하지 않는다.
