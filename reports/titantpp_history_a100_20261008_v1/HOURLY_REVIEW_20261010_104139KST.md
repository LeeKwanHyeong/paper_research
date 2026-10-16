# 정기관측 — 5080 GPU 조회 오류 지속

작성 2026-10-10T10:41:39.915828+09:00. 계약·source/input/op/control 봉인과 A100 원본393파일·archive·manifest SHA 검증 PASS.

| 주제 | 상태 | 완료 / 진행 / 대기 / 미확정 |
|---|---|---|
| Instacart9 | 마지막 조건 미확정 | 8 / 0 / 0 / 1 |
| 이력 보정27 | 차단됨·부분 종료 | 10 / 0 / 15 / 2 |
| 고정 검색27 | 외부 작업 대기 | 0 / 0 / 27 / 0 |

이번 10:41 KST 회차 5080 SSH1회 관측에서 nvidia-smi GPU UUID/name 조회가 exit18로 실패했다(연속5회). 학습 실패 증거가 아니다. 마지막 성공 관측은 2026-10-10 05:30:23 KST, Instacart CNN+GRU54 seed62 E37/best1. 현재 PID/GPU/저장epoch/완료 여부와 ETA는 미확정. 사용자 조치가 필요하며 추가 SSH·반복조회·retry·resume·kill·재시작0.

5090 seed52 세조건 완료. CNN+GRU54 E55/best15, 기존 실제 2026-10-09 06:05:11 KST terminal cache 보존·SSH0. 최종 원본 회수 pending·remote0, Instacart 3seed 최종 비교 미완료.

A100은 마지막 실제 관측 2026-10-09 12:23:25 KST·삭제 확인12:23:55 KST 유지·원격0. Taxi9+RAF seed42 disabled E49/best9 총10완료, RAF full E23/best9·current E46/best9 부분중단·15미시작. Foreign GPU process group 운영 오류/dispatcher exit1 보존. 마지막 비용 추정$30.12/$50, 확정 청구 미확정. 종료 후 heartbeat 정지는 삭제 상태에 부합한다.

새 수량·큰수량·시간 성능 판정 없음. Q1 최종 판정·CPU binary 재추론·Test·독립 평가 미완료. 자동화 유지·다른 자동화 변경0.

**5080 상태 확인 — 다음 작업 / 차단됨**
- GPU 조회 오류와 실제 실행·종료 상태 확인이 필요하다.

**Instacart 3seed 비교 — 외부 작업 대기**
- 마지막 조건 종료와 원본 SHA 검증 후 집계한다.

**A100 오류 원인·검색 자원 계약 — 다음 작업 / 차단됨**
- 보존 증거로 운영 원인을 확인하고 공통 GPU/Runtime 계약을 정리한다. 후속 학습 자동 시작0.
