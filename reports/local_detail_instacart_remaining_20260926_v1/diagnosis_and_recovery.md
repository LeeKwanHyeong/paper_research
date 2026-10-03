# Instacart seed62 남은 네 모델 복구 — 5090

**멈춘 작업 진단과 기존 결과 보존 — 완료**
- 2026-09-26 사용자 지시: “원인확인하고 수정해서 남은 학습 진행하다”. 동일 범위 재승인 없이 이 네 모델에 대한 복구를 한 번 실행한다.
- 기존5090은 seed52 여섯 모델과 seed62 B·이력 보완 TitanTPP의 8조건·16replays까지 완료했다. RMTPP62는 03:03:29UTC 이후 첫 epoch/체크포인트/이력 저장 없이 3시간 이상 정지했다. CPU 프로세스는 살아 있고 GPU 사용률은0, 사용 메모리834MiB였다.
- SIGINT에 traceback/종료 응답이 없었다. 호스트 ptrace 권한상 내부 stack을 얻지 못했다. **내부 native 정지의 정확한 원인은 미확정**이다. 장기 실행 프로세스의 상태 누적은 가설이며 확정 원인으로 주장하지 않는다.
- 감독 프로세스3097647에 사용자 승인 범위의SIGTERM을 한 번 보내 소유 worker3097753만 정리했다. 둘 다 종료, 기존tmux 종료, GPU 비움 확인. 기존run은 stopped/dispatch failed로 보존한다.
- 사전 수집58소형파일과16checkpoint 파일SHA를 기록했다. 61개 완료 결과·입력·체크포인트 파일SHA를 원격 복구 전 검사했다. 기존source·결과·claim을 삭제하거나 덮어쓰지 않는다.
- 원본 증적: `../local_detail_replication_5090_20260924_v1/monitor/20260926T060901Z/pre_recovery/collection_manifest.json`, `../local_detail_replication_5090_20260924_v1/monitor/20260926T060901Z/authorized_stop.json`.

**격리 복구 구현·검증 — 완료**
- 원래 동결 source98개 중97개는 SHA가 완전히 같다. 원래 runner에는 새 계약의 검증·선행 결과 검사 경로만 연결했다. 모델·학습·head/loss·loader·평가 코드는 변경하지 않았다.
- 각 모델을 독립된 새Python 프로세스에서 순서대로 실행한다. 학습/재평가 budget 호출에서10초 간격 진행 증적을 쓰고,5분간 진척이 없으면 stack을 남기며30분간 진척이 없으면 소유worker만 중단한다. 실패하면 다음 모델이나 재시도를 시작하지 않는다.
- CPU91테스트 통과: 네 실제trainer40epoch/selected·last 재평가를 합성데이터로 검증, 계약/seed/deadline 변경 거부, 진척 없는 프로세스 감지, 독립PID, 실패 시 다음 실행·재시도 차단. 호스트 native stall 자체의 재현 테스트는 불가능했음을 구분한다.
- 5090 자체CUDA qualification60synthetic updates 통과. Runtime/라이브러리/GPU UUID는 기존 계약과 동일하다.

**남은 학습 진행 — 진행 중**
- RMTPP → THP → NHP → SAHP, Instacart seed62만4조건·8validation replays. 기존 B/local과seed52 완료분은 재실행하지 않는다.
- max300/min40/patience40, batch128, AdamW, 동일split/초기화/head/loss와 strict-earliest raw-RMSE 체크포인트 선택 유지. same selected checkpoint의 시간·수량·구간 지표를 사용한다.
- RMTPP의 정지한 첫 시도에는 저장된진척이 없어 새초기화로 시작한다. 저장되지 않은 부분update수는 알 수 없으며, 최종 보고에서 실패시도와 소요시간을 삭제하지 않는다.
- 준비/대기 포함 공통마감 **2026-10-02 21:28:48KST** 유지. 추가240시간·추가seed·held-out·Runtime 변경·커밋/Push·외부공개 없음.
- 새root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/local_detail_instacart_seed62_remaining_5090_20260926_v1`
- contract canonical SHA: `a4a4f8bb11a86e0866dc2b1bb35bb9dd736a492b13ab7c60b656238aff6ad6a5`
- source closure SHA: `1ac3dd261582631a03f473571af980cf4a09bcf6123b56fe92106b6563cebfa3`
- dispatch SHA: `1f07d1b935f0dbfaa772ace1f8e0c41d9b977c74b9c216d85d601a9da829dcc6`
- 계약: `../../paper/contracts/local_detail_instacart_remaining_recovery_v1.json`; 본폴더의approval/start_permit/deployment_receipt/launch_receipt를 함께 확인한다.

**자동 모니터링·최종 감사 — 다음 작업**
- 한 회차 한 번 `/usr/local/bin/python3 search_artifacts/local_detail_instacart_remaining_5090_20260926_v1/monitor_once.py --host 5090`으로 읽는다. 새실행·retry·resume·claim삭제 금지. 소형상태/로그/PID/tmux/GPU/실제진척만저장하고 내부반복polling은하지않는다. 일반상태는조용히,의미있는전환·완료·오류에만알린다.
- 5080은24조건48replays 완료 및 감사된 `../local_detail_replication_5080_20260924_v1/monitor/20260925T204245Z/terminal_5080`을 재사용한다.
- 최종 총54조건108replays =seed42재사용18/36 +5080신규24/48 +5090기존완료8/16 +이번복구4/8. seed62 paired/partition은B/local원본2조건을JSON으로합쳐6모델학습노출/배치prefix를감사한다. 복구파일에 새4개와재사용2개를구분한다.
- 모든조건의 실제stop/step/표본수/초기화/strict선택/selected·last replay/partition을감사하고 validation만해석한다. 실행완료와성능기준통과를구분하고 불리한seed/실패시도를누락하지않는다. 정상완료확인후TitanTPP의주장범위와반례를함께정리한다.
- 기존 `vnc-hard-lmm-5090-screening`은PAUSED유지,현재작업archive금지.

## 재개 검증 결과 — 2026-09-26 15:32KST

RMTPP seed62가1epoch·15,557step을완료했다. 첫epoch train1,991,192표본/validation503,733표본, 초기상태·입력증적·train/validation 배치순서가 원본seed62 B와같음을확인했다. 전체stop/선택/replay감사는각조건완료후수행한다. 현재오류없음. 기존heartbeat5080-5090은새복구실행읽기전용모니터링으로갱신했다.

**남은 네 모델 학습·검증 — 진행 중**
- RMTPP 완료후 THP→NHP→SAHP를순서대로진행한다. 자동retry없음,기존공통마감유지.

**전체54조건 통합 감사·최종 비교 — 다음 작업**
- 모든조건terminal후원본과복구의출처를구분하고 실제조기종료·step·표본수·같은selected checkpoint의validation지표를감사한다. 실패시도와불리한seed를포함해이력보완TitanTPP의근거와한계를보고한다.
