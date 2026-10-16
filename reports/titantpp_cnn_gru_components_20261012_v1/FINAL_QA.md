# CNN×GRU36 준비 완료의 독립 QA

상태: **LOCAL_READY_REMOTE_PENDING**. 이는 코드·계약·로컬 봉인 준비의 완료다. 실험36개 완료나 두 서버 native 검증 통과가 아니다.

검토 시각: 2026-10-12T17:10:38+09:00 (KST).

## Review Result: PASS (로컬 준비 범위)

### CRITICAL

없음.

### HIGH

미해결 없음. 다른 projection shape의 과도한 bitwise gate와 별도 session 자식의 SIGTERM 정리 누락을 수정 후 독립 회귀로 확인했다.

### MEDIUM

미해결 없음. 완료 보고의 train launch/exit/claim 및 worker 실제 exit 원문 결합 누락을 독립 재현했고 수정 후 negative regression을 확인했다.

### LOW

없음.

## 검증 요약

- 독립 로컬 회귀56개 PASS: 모델14, campaign16, preparation5, launcher21. 마지막 launcher21은1.395초였다. root의 같은 전체 suite56개 결과도2.036초 PASS로 전달받았으며 독립 결과와 구분한다.
- 같은 seed의 초기 tensor·RNG, FULL/NONE 원래 경로 일치, 비활성 경로·optimizer 제외, gradient와 인과성, Validation-only 및 checkpoint 잘못된 표기 거부를 확인했다.
- 정확한36ID,5080 Taxi+RAF24/5090 Intermittent12, 같은 dataset의 모든 seed·arm 같은 host,3+1 barrier/max3, fresh fit·재시도0, 양 host native 자격 선행을 확인했다.
- 새 지시 없는 원격 모드 거절, hold-release 원문 결합, 안전한 archive 추출, shell quoting, 원본·claim·실패 보존, 실제 소유 자식만 정리, 조회 실패와 학습 실패 구분을 확인했다.
- CPU synthetic/fake remote 회귀이며 실제 두 GPU의 CUDA 호환성·VRAM 수용·데이터 fit 성능을 증명하지 않는다. 그 항목은 이후 native gate에서 확인해야 한다.

## 독립 stdlib SHA 확인

| 대상 | 확인 값 |
|---|---|
| canonical contract | `593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3` |
| contract raw | `39c731aa31160b0009a65528dd4c6fda7ba3f8db7c99318b1c20604792f6995e` |
| source131 closure | `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53` |
| operation4 closure | `402561677270a201142edccdc5c5fece655ed10b103a5e6e328cf6f1c9d85f6e` |
| deployment archive raw | `4b7358bbb52b0066aaf1e681bfebb17740cc8b4eef0507be885c31964c22ee06` |
| local tools integrity raw | `c13a8fd652834b73ea66fa6998e14a26583ae9fa7494821262a16f618557c82f` |
| original hold raw | `e2441d121f4337b0f347891adfa285bb64030319a51e726b3cf75783a8e99983` |

대상은 `search_artifacts/titantpp_cnn_gru_components_20261012_v1`이다. source131·opaque input7·operation4·local tools12 전 파일과 archive143개 regular member의 exact 집합·내용 SHA를 대조했다. current의 실제 bundle 경로와 hold=true도 확인했다. 승인·release·start/training permits·실행 control 디렉터리는 존재하지 않았다.

## 준비 완료 조건

- [x] 과학129 보존 및 신규2 source/운영4/입력7/로컬도구12 봉인
- [x] 네 구조·공통 초기화·원경로 동일성·비활성 optimizer 제외
- [x] Train/Validation 경계·기존 selector/예산·새 학습36ID 고정
- [x] CPU 회귀와 과학·운영 독립 검토 및 발견 사항 해결
- [x] 사전 여섯 지표·paired contrast·FULL 대 세 대조 지지 기준·불리한 결과 포함
- [x] hold 보존·원격0·native0·학습0·Test 평가0
- [x] 원고 보존: main.tex SHA `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354`

## QA가 변경한 파일

`REVIEW_NOTES.md`, `PRELAUNCH_QA.md`, `FINAL_QA.md`만 작성했다. 소스·bundle·원고·Runtime·기존 결과 및 `.agents`는 변경하지 않았다. 적용 프로토콜 경로 부재와 대체 읽기 기록은 REVIEW_NOTES.md에 있다.

**새 지시 뒤 양 서버 상태를 확인한다 — 외부 작업 대기**

5080/5090에 대한 실제 사용자 시작 지시를 계약과 원본 hold에 결합한 다음 건강·GPU 유휴·기존 Runtime을 확인한다. 한 host가 점유되면 양 host 모두 대기하며 기존 작업을 중지하지 않는다.

**격리 배포와 양 서버 native 검증을 끝낸다 — 다음 작업**

새 root에 봉인 payload를 한 번 배포하고 qualification/합성3worker admission의 실제 exit0 및 GPU 해제를 양쪽 모두 확인한다. 로컬 PASS를 이 단계의 증거로 대체하지 않는다.

**36개 fresh fit과 원본·분석을 확정한다 — 다음 작업 / 앞선 gate 뒤**

전역 permit 뒤24/12개의 독립 큐를 시작한다. 실제 종료·원본 회수·manifest와 binary SHA 검증 후 동일 선택 epoch의 여섯 지표를 분석한다. 전체 큐 ETA, 학습 성능 및 결합 지지 여부는 현재 미확정이다.

이 대조는 파라미터 수가 다른 경로 제거 실험이다. 반복 관측 Validation의 개발 근거와 host·동시 부하 차이를 공개하며 참신성·보편적 필수성·통계 유의성·독립 일반화로 확대하지 않는다.
