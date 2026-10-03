# 5090 동기화 및 신규 A100 시간별 모니터 설정

5090은 9/9조건 원본 동기화·CPU 감사가 완료됐다. 신규 원본 130파일과 checkpoint 2개를 회수했고 기존 8조건의 증거를 재사용해 전체 checkpoint18개와 validation replay18개를 연결했다.

신규 routing/placement A100 Pod c88rn6dvhdafj0를 대상으로 이 채팅에 1시간 간격 heartbeat titantpp-a100을 활성화하고 저장 설정을 재검증했다. 삭제된 titantpp-36 및 다른 PAUSED 모니터는 변경하지 않았다.

10/3 10:06:09 KST 실제 읽기 전용 관측 1회에서 완료0·학습3·대기3·실패0·미확정0을 확인했다. Taxi recent4_attention은 저장138/best104, Intermittent recent4_attention은13/11, post_block은13/2다. Taxi post_block과 RAF 두 후보는 대기 중이다. 자체 속도가 없는 대기 조건으로 인해 전체 큐 ETA는 미확정이다. 상세 조건부 ETA는 관측 analysis.json에 보존했다.

모니터 검증은 합성·저장 snapshot 기반 15개 항목과 실제 관측으로 완료했다. macOS Python 실행 파일 대소문자 인식은 로컬 PID 재확인으로 수정·검증했으며 추가 원격 접속은 없었다. 기존 비용 관리자 PID53515와 전용 관리 경로가 일치한다. 총 승인 상한$100 및 자동충전 없음이 유지된다.

**남은 작업 — 진행 중**: 신규 A100 6조건의 학습과 시간별 진행·종료 예상 보고. 전 조건 terminal과 원본 회수·비용 보고·Pod 삭제를 확인한 뒤 이 모니터를 종료한다.

**남은 작업 — 다음 작업**: 이전 RunPod의 CPU 감사3조건은 별도 미완료다. 이전42조건 전체의 감사는39완료·3미완료이며 전체 연구 감사 완료로 표현하지 않는다.
