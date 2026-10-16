## Review Result: PASS

상태: 로컬 후보 QA 완료. 실제 주기 자동화 생성·배포 및 실관측 결과는 root가 별도로 검증한다.

### CRITICAL
- 없음.
### HIGH
- 없음.
### MEDIUM
- 미해결 없음. 기존 후보의 `latest_snapshot` 선택은 같은 상태의 다른 회차와 결합될 수 있었으나, 현재 controller 66–81행은 이번 호출 전후 신규 snapshot 정확히 1개, 전체 결과 canonical, 원관측시각을 함께 대조한다. 시계 역행·동시 추가·결과 불일치 모의 검증 PASS.
### LOW
- 없음.

요약: 승인된 v5 관측만 사용하며, 원격 변경·학습·새 평가 없이 host별 status 1회와 검증된 terminal cache 0회로 제한한다. 원본 회수와 최종 Validation 집계는 미완료로 분리한다. ETA는 같은 fit의 저장 완료 epoch와 full Validation history에만 의존하며, 미시작/다른 GPU 속도를 대입하지 않는다.

검증:
- ETA의 독립 stdlib 단위 테스트 17개 PASS (0.001초).
- Controller 독립 모의 사례 5개 PASS: 시계 역행에서 이번 snapshot 선택, 동시 추가 snapshot 차단, report 불일치 차단, 조회 실패와 학습 실패 분리, terminal cache 원시시각 유지 및 SSH0.
- Controller 114–127/175–179행의 별도 monitor seal·3600초 승인·v5 기존 bundle/양 host 승인 사전 검증을 검토.
- Controller 93–104행: 저장 checkpoint와 history/timing의 best 동일성을 확인한 경우에만 조건부 ETA 계산.
- Controller 181–198행: 중복 local observer lock, 각 host 한 번, host별 실패 독립 기록, 두 host terminal 판정.
- ETA 모듈: 연속·고유 1-based 완료 epoch 수가 저장 epoch와 같아야 하며, 첫 strict finite 비음수 RMSE 최소를 선택한다. 유한 양수 elapsed_seconds의 마지막 최대10개 중앙값을 사용한다. best+40/min40/max300 및 300epoch 대안을 구분한다. 불일치·미저장·nonfinite·overflow에서는 미확정.

완료 조건 체크:
- [x] 원격 mutation 0의 코드 경로와 host별 최대 status 1회 검토
- [x] terminal cache rawSHA/host/계약/재검증/원관측시각 및 SSH0
- [x] 당회 snapshot와 실제 반환 결과의 완전한 결합
- [x] 조회 실패를 수치/학습 실패로 바꾸지 않음
- [x] fit별 조건부 ETA, whole queue 미확정, deadline과 구분
- [x] 36개 원격 종료와 local 원본 회수·분석 완료를 분리
- [x] 기존 science131·sealed40 수정하지 않음, QA 직접 SSH/GPU/데이터 평가 0
- [ ] 배포 후 monitor 파일 seal·승인/자동화 일치 및 첫 실관측 확인 (root 담당)

변경 파일: 본 `/private/tmp/component_hourly_qa_20261012.md`만 작성. 프로젝트/원고/코드/원본 수정0, `.agents/` 수정0.

검토 후보 rawSHA:
- `/private/tmp/observe_titantpp_components_hourly_20261012.py`: `4e4d51038413814742951699560f4fc703ee0dcf80af2dc3908f4d0bbec827e3`
- `/private/tmp/component_hourly_eta_20261012.py`: `2b7718bc68b142dc8edea769b59a57369763908a93dcf66e94bcac3daf53fdda`
- `/private/tmp/test_component_hourly_eta_20261012.py`: `61e4aa64d1fea2b73b20d6823d4005b32e6693036dc6f6b16be59c33490bd915`
