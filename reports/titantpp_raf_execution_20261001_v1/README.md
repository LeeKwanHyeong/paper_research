# RAF 완료 결과

8모델×seed42·52·62=24조건의 학습·selected/last 재평가 및 원본 CPU 감사를 완료했다. [보고서](report.md), [집계와 구간별 원본값](comparison.json), [선택 checkpoint CSV](selected_conditions.csv), [selected/last CSV](endpoint_roles.csv), [검증](verification.json)을 제공한다.

원격 읽기 전용 collect.py, CPU 전용 audit.py, 로컬 기록 집계 summarize.py를 보존했다. collect.py와 audit.py는 완료 회수/감사를 덮어쓰지 않으며 정기 실행용이 아니다. summarize.py는 기존 감사 기록을 재사용한다. 신규 GPU 실행·학습·재평가는 없다.
