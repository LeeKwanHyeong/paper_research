# v3 필수 interface 계약 복원의 독립 QA

상태: 로컬 독립 QA PASS, 실제 5080 native 검증 대기. 5080 승인만 유효하며 5090은 별도 명령 전 모든 원격 호출 0이다. QA는 로컬 증적 검토만 수행한다.

## 현재 기준선

v2 actual native qualification은 design_sha256 누락으로 exit1이었다. fit0, 이전 root/claim/lease/원본 보존. 원 과학source131 `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`과 모델·데이터·초기화·손실·optimizer·selector·36ID/24:12배정/3+1은 변경하지 않는다.

v3는 원 frozen design `d3a579750f60c037e45b881ee03ad27a211bdbcc61f2677a4fa6ca128b0336a5` metadata를 복원한다.5080 lease는 기존 start1791794258.15949/deadline1792399058.15949를 유지하며168시간을 다시 늘리지 않는다.

## 필수 확인

- [x] v2 실패·v1/v2 코드와 원본 보존
- [x] 필수 design key 검증 및 실제 Train/Validation 12interfaces/12args 통과
- [x] source131·과학·학습·구조 불변
- [x] 5080 단독 승인 및5090모든호출0
- [x] 기존5080 lease 상한 계승
- [x] v3 immutable SHA·payload·localtools 검증
- [ ] 실제5080 native 종료/qualification/admission 증적 검증

과거 v1/v2의 CPU synthetic/fake tests만으로 실제 interface 통합을 보증할 수 없음을 명시한다. v3에서는 실제 Train/Validation interface CPU 검증을 추가했고 독립 41개 회귀검증에 포함해 통과했다. 로컬 Python/Torch가 실제 5080과 다르므로 native 검증을 대체하지 않는다.

## 결과와 변경 파일

현재 미해결 CRITICAL/HIGH/MEDIUM/LOW 발견 사항은 없다. QA는 보고서 PRELAUNCH_QA.md와 REVIEW_NOTES.md만 작성했다. 코드·봉인·원고·.agents 파일은 변경하지 않았다. NATIVE_QA.md는 실제 종료 원문을 확인한 뒤 작성한다.
