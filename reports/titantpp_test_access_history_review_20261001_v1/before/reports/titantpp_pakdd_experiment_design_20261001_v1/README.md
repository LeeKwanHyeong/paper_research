# TitanTPP PAKDD 실험 보강 설계

[결과와 남은 작업](report.md)

- [평가 데이터 성격](evaluation_population.md): 과거 test 열람 기억이 불확실하다는 연구자 답변과 사용 경로.
- [두 구조 비교](architecture_controls.md): 6,144개 파라미터의 현재 상태 대안, 고정 /8의 전체 분기 대안.
- [Deep Renewal 직접 비교](direct_baseline.md): 원문·공식 코드·입출력·native 분포·관측 시간 처리.
- [비실행 설계 계약](design_contract.json), [36조건 제안 목록](proposed_run_matrix.csv), [검증](verification.json).

현재 단계는 설계와 로컬 CPU 검증이다. 이 폴더에는 GPU 학습을 시작하는 실행기가 없다. 기존84조건 평가 준비와 원고는 보존한다. 이전 광범위 실험 계획의 실행 승인·비용 상한을 가져오지 않는다.
