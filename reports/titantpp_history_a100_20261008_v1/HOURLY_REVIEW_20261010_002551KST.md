# 정기관측

봉인 계약·source/input/op/control SHA와 A100 부분 원본 393파일·archive/manifest SHA 검증 PASS.

| 주제 | 상태 | 완료 / 진행 / 대기 / 미확정 |
|---|---|---|
| Instacart 9 | 진행 중 | 8 / 1 / 0 / 0 |
| CNN+GRU54 이력 보정 27 | 차단됨·부분 종료 | 10 / 0 / 15 / 2 |
| 고정 검색 27 | 외부 작업 대기 | 0 / 0 / 27 / 0 |

5080 실제 관측: 2026-10-10 00:25:36 KST. 마지막 Instacart CNN+GRU54 seed62는 저장 E14 / best1. checkpoint 안정성·worker52338/timeout52337/supervisor41573 소유권 검증 PASS. 자체 최근 10완료 epoch 중앙값 790.270초. best 미갱신·현 속도 유지 시 E41까지 약 5시간 56분, E300까지 약 62시간 47분. E300 시나리오는 조건 36시간 한도를 초과하므로 완료 예측으로 쓰지 않는다. 전체 큐 ETA 미확정.

5090은 seed52 세 조건 완료. CNN+GRU54 E55 / best15. 실제 관측 2026-10-09 06:05:11 KST의 terminal cache를 유지하며 이번 SSH는 0회.

이번 회차 새 완료 결과는 없다. Instacart 3seed 전체 비교는 마지막 조건 종료와 원본 보존 검증 후 작성한다. 수량 RMSE/MAE와 같은 선택 epoch의 TimeNLL 및 host별 비용을 함께 보고한다.

A100 마지막 실제 관측 2026-10-09 12:23:25 KST, Pod 삭제 확인 12:23:55 KST 유지. Taxi9와 RAF seed42 보정없음 E49/best9 등 10개 완료. RAF full E23/best9·현재전용 E46/best9는 부분 중단, 15개 미시작. Foreign GPU process group 운영 검사 실패와 dispatcher exit1을 보존하며 수치 학습 실패로 판정하지 않는다. 원인 확인·사용자 조치가 필요하다. 마지막 비용 추정 $30.12/$50, 최종 청구는 미확정. 삭제된 Pod 접속 0회.

Instacart wrapper 1회, 5080 SSH 1회, 5090 SSH 0회, A100 원격 0회. 재실행·retry·resume·새 Pod·충전·다른 자동화 변경 0. 자동화는 유지한다. Q1 최종 판정·CPU binary 재추론 감사·Test·새 독립 평가는 미완료다.

**Instacart 최종 학습·보존 — 진행 중**
- 마지막 조건 종료, 소유 PID 부재, 원본 SHA 검증 후 3seed 결과를 집계한다.

**A100 운영 오류 원인 확인 — 다음 작업 / 차단됨**
- 보존된 로컬 증거로 원인을 확인한다. 자동 재실행은 하지 않는다.

**고정 검색 공통 조건 정리 — 외부 작업 대기**
- 후속 자원 계약을 정리하기 전 시작하지 않는다. A100 결과를 RTX의 같은 GPU 기준선으로 혼합하지 않는다.
