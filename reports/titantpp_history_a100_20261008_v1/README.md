> 현재 운영은 사용자 추가 승인에 따라 **A100 동시3개**로 변경됐다. [PARALLEL3.md](PARALLEL3.md)를 우선한다. 아래는 초기 단일 worker 시도의 보존 기록이다.

# CNN+GRU54 이력 보정27 — A100 SXM 운영 이관

사용자가 2026-10-08 이력 보정 실험을 Runpod A100 SXM에서 실행하고 총비용을 USD50 이내로 제한하도록 승인했다. 대상은 Taxi·RAF·Intermittent × seed42/52/62 × full/보정없음/현재상태전용의 27 fresh fit이다. Instacart와 검색27은 이번 A100 실행에 포함하지 않는다.

## 실행 및 소유권

- Pod: `qq2uk2dpfi2rum`, NVIDIA A100-SXM4-80GB 1개, GPU UUID `GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de`.
- root: `/workspace/titantpp_contribution_history_a100_20261008_v1_a100`.
- 원래 RTX history27은 시작되지 않았으며 `ownership_reservation.json`에 A100 소유권을 기록했다. 기존 시간별 자동화에서 RTX history/retrieval 활성화를 제외했다. Instacart는 계속 관측한다.
- 소스129·데이터7개·초기화·loss·optimizer·B128·최소40/최대300/patience40·최초 strict finite fullValidation 전체 raw 수량RMSE 선택은 유지한다. 운영 호스트와 총 비용 기한만 변경한다.
- 실제 Runtime: Python3.12, torch2.11.0+cu130, CUDA13.0, cuDNN91900, numpy2.4.4, polars1.39.3. 격리 venv: `/opt/titantpp-history27-venv`.
- 순서: Taxi→RAF→Intermittent, 각 seed에서 세 arm 연속 실행, worker1. 완료 원본 재학습·자동 retry/resume 없음.

## 비용과 종료

GPU $1.59/h, container30GB + persistent20GB 저장비 포함 추정 $1.596944/h. 실제 배포 직전 잔액 $57.1935, 자동충전 비활성. 계정 충전은 이 작업에서 수행하지 않았다.

준비·학습 최대 $45, 원본 회수·종료·정리 예비금 $5, 총상한 $50. 설치시간부터 비용을 센다. 절대 학습 기한은 봉인 start permit, provider stopAfter는 그 후 최대1시간으로 요청했다. stopAfter 실행 보장은 주장하지 않으며 독립 로컬 비용 감시(caffeinate)와 실제 provider 종료 상태 확인을 병행한다. Pod 종료 요청과 실제 종료 확인은 구분한다. 27조건의 완료 비용·전체 ETA는 실측 전 미확정이다.

A100 재고가 있는 DC의 network volume 미지원으로 생성 요청은 HTTP400으로 거절됐고 해당 저장공간은 생성되지 않았다. Pod persistent20GB와 로컬 주기 복사를 사용한다. live backup은 일관성이 검증된 최종 원본이 아니며, 최종 archive/manifest SHA 검증을 별도로 수행한다. 계정 잔액0 또는 Pod 삭제 시 persistent 원본이 소실될 수 있으므로 회수 후 정리한다. 미회수 실패 시 과금을 끝없이 연장하지 않고 종료와 미회수 사실을 보고한다.

## 과학적 판정

같은 선택 epoch의 전체/큰수량 RMSE·MAE·TimeNLL과 비용을 보고한다. full이 두 대조보다 같은2개 이상 dataset의 각3seed에서 전체RMSE를 strict 개선하면 개발 근거로 지지한다. 현재상태 대조의 hidden에도 encoder prefix 이력이 포함되며 activation/depth/cost 차이는 남는다. Test 및 예측행은 열지 않는다. 독립 평가·통계적 유의성·외부 우월성의 증거로 확대하지 않는다.

## 후속 주제

- Instacart9: RTX5080/5090 기존 실행 유지.
- 이력 보정27: A100로 이관, native 검증과 실제 학습 증적으로 진행 상태를 판단.
- 고정 검색27: 대기. A100 full9를 RTX 검색 대조의 같은GPU mean으로 재사용할 수 없으므로 후속 GPU 배치 계약을 먼저 정리해야 한다. 이 기록은 검색27의 추가 A100 과금 승인이 아니다.
