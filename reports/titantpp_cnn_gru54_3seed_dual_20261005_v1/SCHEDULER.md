# CNN+GRU54 3seed 매시간 관측

기존 자동화 `titantpp-gru-3seed`를 새6조건 관측으로 갱신해 ACTIVE 상태로 확인했습니다. 이름은 “TitanTPP CNN+GRU54 3seed”이며 기존 매시간 주기를 유지합니다. 기록은 scheduler_receipt.json, 실제 학습 진입은 launch_confirmation.json을 참조합니다.

두 서버에서 승인된 monitor.py를 각각 한 번만 호출하고, 이후 finalize_once.py의 terminal gate를 확인합니다. 미완료이면 로컬 pending만 남기며 추가 평가·학습·원격 회수를 하지 않습니다. 종료된 이전33조건 및 삭제된 A100은 다시 관측하지 않습니다. 신규6조건과 재사용3조건의 Validation·원본SHA 회수·실제 서버 종료 확인 후 이 자동화만 종료합니다. Test3seed·새 독립 평가·CPU 재추론 감사는 별도 후속입니다.
