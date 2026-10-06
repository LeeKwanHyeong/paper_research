# 시간 출력부6조건과 원래3seed 통합 관측

기존 자동화 titantpp-gru-3seed에 새 head6조건을 별도 집계하도록 연결하고 ACTIVE/readback을 확인했습니다. 원래9조건만 종료됐다고 삭제하지 않고 두 캠페인의 원본SHA회수와 실제 종료를 모두 확인한 뒤 이 자동화만 삭제합니다.

새5080 head6조건은 이미09:57:54 KST terminal·SHA회수 검증을 마쳤습니다. monitor_once.py는 원관측 시각을 보존한 terminal_cache를 재사용하고 SSH를 끝냅니다. finalize_once.py는 completion_receipt를 검증해 새 실행이나 재회수를 하지 않습니다. 원래5090 관측과 원래9조건 회수 gate는 유지합니다. Test/130개 학습/새평가/자동retry는 실행하지 않습니다.

실제 저장 프롬프트와 readback 증적: search_artifacts/titantpp_time_head_bias_refit_5080_20261006_v1/scheduler_prompt.txt·scheduler_receipt.json.
