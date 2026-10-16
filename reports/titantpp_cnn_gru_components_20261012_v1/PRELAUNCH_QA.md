# CNN×GRU 두 서버 준비의 독립 사전 QA

상태: **LOCAL_READY_REMOTE_PENDING**. 로컬 준비 QA는 완료됐으며 원격 실행은 사용자 지시 대기다. 이 문서는 native 통과나 실행 허가가 아니다.

## 검토 범위

- 신규 모델·runner·campaign·qualification·phase wrapper·prepare·launcher와 관련 CPU 회귀 테스트, README·실행 순서·공통 계약을 검토했다.
- 검토자는 소스 코드, Runtime, 기존 원본 및 열린 원고를 변경하지 않았다. SSH·GPU·학습·Test 성과 열람0이다.
- 과학129파일 및7개 입력은 기존 봉인과 stdlib SHA로 대조했다. 입력 파일은 opaque bytes로 해시했으며 Test 예측·성과를 해석하지 않았다.

## Review Result: PASS (로컬 준비 범위)

### CRITICAL

없음.

### HIGH

미해결 없음. 다른 shape의 GRU projection에 대한 bitwise gate와 상위 SIGTERM 때 별도 session 자식 정리 누락을 독립 확인해 담당자에게 수정 요청했고, 수정 후 회귀 검증이 통과했다. 이력과 파일 위치는 REVIEW_NOTES.md에 보존했다.

### MEDIUM

미해결 없음. train 완료 분기의 종료 증적 결합 누락을 fake snapshot으로 재현했고, 담당자가 train launch/exit/claim과 모든 worker actual exit0·raw SHA를 결합했다. 독립 launcher21개 회귀 PASS로 해결을 확인했다.

### LOW

없음.

## 과학 조건 확인

- 네 구조는 보정 없음/CNN-only/GRU-only/CNN+GRU이며 과거 CNN+MLP16을 CNN-only로 재명명하지 않는다.
- 동일 seed의 공통 tensor·construction RNG 및 FULL/NONE 원래 forward·gradient·optimizer update 일치를 확인했다. 비활성 경로는 호출·gradient·optimizer에서 제외한다.
- 활성 파라미터89,859/90,435/114,591/115,167, 등록115,167을 확인했다. 동일 용량 실험은 아니며 current encoder의 prefix 정보도 유지된다.
- 5080 Taxi+RAF24,5090 Intermittent12이며 같은 데이터셋의 세 seed와 네 구조는 같은 GPU·Runtime이다. host 간 Runtime과3+1 동시 부하 차이는 시간 비교의 한계다.
- 기존 데이터·Train 정규화·손실·optimizer·selector·B128/min40/max300/patience40을 유지한다. 신규 fit36은 fresh이며 warm start·resume·자동 retry0이다.
- 전체 Validation raw 수량 RMSE로 같은 선택 epoch의 여섯 지표를 모두 보고한다. FULL이 NONE/CNN/GRU보다 같은 두 개 이상 dataset의 모든3seed에서 strict 개선할 때만 사전 개발 지지다. 불리한 조건도 유지한다.
- 상호작용 M11−M10−M01+M00는 지표 척도에 의존한다. 참신성·유의성·기계적 시너지·보편적 필수성·새 독립 일반화로 확대하지 않는다.

## 안전 및 실행 경계 확인

- 원본 launch_hold를 보존하고 새 사용자 지시·계약·hold rawSHA에 결합된 별도 authorization/release가 없으면 SSH 전에 거절한다.
- 양 host 건강·유휴 확인 후 새 격리 root만 한 번 배포한다. 모든 payload member를 먼저 SHA 검사하고 링크·경로 이탈·중복·extra를 거절한다. intent·claim·실패와 출력은 덮어쓰지 않는다.
- 양 host native qualification의 실제 subprocess exit0와 raw evidence·GPU·Runtime·source·3worker admission이 모두 연결돼야 fit permit이 유효하다.
- 같은 dataset/seed의 세 대조→FULL 한 개 barrier 및 최대3worker를 유지한다. timeout과 SIGTERM은 소유 process group만 정리한다. wrapper 자체 OS exit를 직접 본 것으로 추정하지 않는다.

## 독립 로컬 검증 증적

- 모델14개 CPU 회귀 PASS.
- Campaign16개와 preparation5개, 총21개 stdlib 회귀 PASS. 실제 작은 CPU process tree의 SIGTERM 정리도 포함한다.
- Launcher21개 stdlib/fake-remote 회귀 PASS(1.395s). 독립 로컬 검증 합계56개 PASS다. Immutable bundle의 source131·input7·operation4·local tools12·tar143개 member를 독립 stdlib SHA로 모두 대조했고 통과했다.
- Native GPU qualification0, 실제 학습0, 새 Test 평가0. 준비 문서나 명령 예시를 실행 승인으로 해석하지 않는다.

## 최종 봉인 및 범위

- 독립 봉인 확인 시각: 2026-10-12T17:10:38+09:00 (KST).
- canonical: `593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3`.
- 원본 hold=true, authorization/release/approval/start/training permit 및 control 실행 디렉터리 부재를 확인했다.
- 열린 main.tex SHA는 `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354`로 보존됐다.
- 전체 봉인 SHA와 남은 의존 순서는 FINAL_QA.md에 기록했다.
