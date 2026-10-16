# 2026-10-08 17:04:00 KST 진행 확인

양쪽 소유 프로세스·GPU UUID·원본 SHA·저장 checkpoint가 확인됐다. 실패0, 미확정0이다.

| 주제 | 상태 | 완료 / 진행 / 대기 |
|---|---|---|
| Instacart 폭8·12·CNN+GRU54 | 진행 중 | 2 / 2 / 5 |
| CNN+GRU54 보정 대조 | Instacart 종료·원본 회수 대기 | 0 / 0 / 27 |
| CNN+GRU54 검색 대조 | 보정 대조 종료·원본 회수 대기 | 0 / 0 / 27 |

| GPU | 현재 조건 | 저장 / best epoch | best 미갱신 ETA | E300 시나리오 |
|---|---|---|---|---|
| 5080 | Instacart MLP12 seed42 | 35 / 28 | 약2시간12분 | 약17시간42분 |
| 5090 | Instacart MLP12 seed52 | 50 / 15 | 약21분 | 약17시간39분 |

ETA는 각 fit의 최근10개 완료 epoch 중앙값과 best 미갱신·현 속도 유지 가정이다. 전체 큐 ETA, endpoint 확인·원본 회수·CPU 감사 시간은 포함하지 않는다. 16:30 관측보다 두 fit 모두9epoch 진행했고5080 best는6에서28로 갱신됐다.

완료된 것은 Instacart MLP8 seed42/52 두 조건이다. MLP12와 CNN+GRU54의 같은 seed 대조가 끝나지 않아 구조 간 수량 우위·시간 손해를 아직 판정할 수 없다. 새 대표 보정·검색 기여 역시 미판정이다. 승인된 순서는 Instacart9→보정27→검색27이며, 앞 stage 전체 종료·원본 검증 후 다음 단계로 이어진다.

봉인 사전검증은 세 stage 모두PASS. 승인된 advance 도우미1회, 읽기전용 관측SSH host별1회, 기존launch 확인remote0, finalizerpending remote0. 새학습·자동retry·재추론·Test·Runtime 변경은 없었다. CPU binary 재추론 감사와 새 독립 평가는 별도 미완료다.
