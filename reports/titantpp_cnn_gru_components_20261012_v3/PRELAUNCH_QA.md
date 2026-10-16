# CNN×GRU components v3 독립 사전 QA

## Review Result: PASS (로컬 코드 범위)

상태: `LOCAL_READY_5080_NATIVE_PENDING`. 봉인 bundle 및 5080 단독 승인은 확인했다. 5080 실제 native qualification 통과는 아직 확인하지 않았다. QA의 원격 호출과 GPU 실행은 0이다.

### CRITICAL
- 없음.

### HIGH
- 현재 v3 코드에서 미해결 사항 없음. v2의 필수 `design_sha256` 누락은 실제 qualification exit1을 일으켰으며 v2 실패 원본을 보존한다. v3는 원래 frozen design 값을 복원하고 로컬/원격 계약 검사에서 누락·변경을 거부한다.

### MEDIUM
- 없음.

### LOW
- 없음.

## 검토 범위와 증거

- `prepare_titantpp_cnn_gru_components_v3.py`: 부모 v2 canonical, 원래 history 계약 canonical과 design 값, v2 배포 receipt SHA 및 절대 lease를 고정한다. source131, 36개 ID, 24:12 배정, 데이터·모델·초기화·손실·optimizer·선택 조건은 불변 비교한다.
- `run_titantpp_cnn_gru_component_campaign_v3.py`: 필수 design guard 외 함수는 v2와 AST가 같다. 동일 dataset/seed 네 구조의 3+1 실행, 소유 프로세스만 정리, 실패 claim 보존, 독립 host 승인과 해당 host qualification만 사용한다.
- `run_titantpp_cnn_gru_component_phase_v3.py`: v2와 byte-identical이다. 실제 자식 종료와 소유 프로세스 정리를 보존한다.
- `launch_titantpp_cnn_gru_components_v3.py`: 검증된 행정 health bridge의 읽기 전용 `/proc/locks` 확인과 엄격한 shutdown 대기 프로세스 식별을 포함한다. 실제 updater·미확인 조건은 차단한다. payload는 stdin JSON이며 안전한 archive member와 SHA를 검사한다.
- 5080 lease는 시작 `1791794258.15949`, deadline `1792399058.15949`를 계승한다. 만료·미래 시작·비유한 값·168시간 초과는 거부한다.
- 5090에 대한 승인 없는 모든 remote mode는 SSH 전에 차단하며, 5080 동작에서 5090 호출·receipt를 요구하지 않는다.

## 자동 검증

독립 실행 명령:

```sh
MPLCONFIGDIR=/private/tmp/component_qa_mpl /usr/local/bin/python3 -m unittest paper.tests.test_titantpp_component_campaign_v3 paper.tests.test_titantpp_component_preparation_v3 paper.tests.test_titantpp_component_launcher_v3 -v
```

41 tests PASS, 14.721초, 실제 OS exit0. 실데이터 CPU 계약 소비자 검증은 봉인 source131의 Train/Validation loader를 거쳐 자료 3개 × 구조 4개 = `time_interface` 12개와 `training_args` 12개를 통과했다. Train/Validation 대상 수는 Intermittent 393824/86285, Taxi 38393/8268, RAF 25779/6690이다. 성과 수치는 계산하지 않았다.

로컬 Python 3.12.10/Torch 2.14.0에서 모델 예측 0, 과학 fit 0, Test materialization false, CUDA 초기화 false였다. 이는 5080 Python 3.12.13/Torch 2.11.0+cu130의 실제 native 통과 증거가 아니다.

## 완료 조건

- [x] 필수 설계 메타데이터 복원과 누락 시 조기 차단
- [x] 실제 Train/Validation 입력과 12개 구조 interface/학습 인수 검증
- [x] 과학 조건·원본 코드 보존, 독립 host 승인과 5090 호출 차단 회귀 검증
- [x] 기존 5080 절대 lease 계승과 새 실패 root 재시도 금지
- [x] 생성된 immutable bundle 전체 stdlib SHA 및 5080 승인·release 확인
- [ ] 5080 실제 native/동시성 자식 exit0와 종료 후 GPU idle 확인
- [ ] 학습 시작 후 소유 PID/argv/GPU와 실제 저장 상태 확인

## 주장 범위

네 구조 대조는 동일 조건에서 각 계산 경로의 개발 성능 기여를 비교한다. 활성 파라미터 수가 다르고 FULL은 3+1 중 단독 wave이며, 데이터셋 사이 GPU/Runtime도 다르다. 참신성·보편적 필요성·인과적 시너지·통계적 유의성·독립 일반화의 증명으로 확대하지 않는다. 기존 Test 및 혼합 성과 파일은 열지 않았다.

## 생성 bundle의 독립 봉인 확인

stdlib SHA로 source131, input7, operation4, local tool26, archive143 regular member 전체를 검증했다. 원본 hold와 5080 singleton 승인/release의 raw SHA 결합, current 경로 및 contract digest도 통과했다. 5090 승인은 없으며 호출 이전 거부를 재확인했다.

- canonical: `24739d7fda028e8f10926d9db601a14ae8a20f999a0f3bfbe057e98c9ca67a52`
- source131: `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`
- operation4: `173d444ee00015c0a9ec16760521f0ae1c4ae13cf2b6e52e6c0c69e4ac979e3b`
- payload: `95128839a913cffe2894e8d71ac2947a3337f30d976ff25a4918bbf982b55fc7`
- local tool seal raw: `534fb173994cdc91df70abe10f410e4b57c591f3ba0193fcfc9c91b309dd768d`
- 5080 승인 raw: `d00ee066fff326355e74ddde240e8e14d7ec459ac97a60f9863b11ecd895ab60`

승인 원문은 “우선 5080 먼저 진행하자. 5090은 내가 진행하자 명령하면 그때 하고”이다. 승인된 범위는 5080 Taxi+RAF 최대24 fresh fits이며, 실제 qualification이 실패하면 해당 root의 failure/claim을 보존하고 자동 재시도하지 않는다.
