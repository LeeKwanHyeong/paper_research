# 재현 자료

- 대상: 원본 uncapped RAF·Taxi·Intermittent, seed42/52/62. 추가72/82는 아직 진행 중이다.
- 입력: 시간 간격·수량 log1p. log 수량 MSE와 recorded positive-integer lognormal duration NLL을 가중치1로 합산한다.
- batch128, hidden64, AdamW lr0.001/weight_decay0.01/gradient clip1. 최대300·최소40·patience40.
- 전체 Validation raw 수량RMSE 최초 strict finite 최소 checkpoint를 고정한다. 동률 첫 epoch이며 MAE와 Test도 같은 가중치를 쓴다.
- 공유형3seed 학습5090/cuDNN92000, 고정Test 평가5080/cuDNN91900. Validation 재현 허용오차1e-5. 과거 외부 저장 평가 중 CPU 재현 기록도 있어 동일Runtime 새평가로 설명하지 않는다.
- Intermittent 최종5/5/11과 임시Test 고정본은 canonical 모델tensor SHA가 같음이 확인됐다. 기존 임시 원본의 명칭·receipt를 바꾸지 않고 별도PAPER_START_GATE 증거를 붙였다.
- SOURCE_VERIFICATION.json은 raw9 227개·RAF/Taxi 평가205개·Inter 평가92개 원본(총524개)과 archive/manifest/결과 SHA를 재검증한다. 소유프로세스 종료와 실제exit는 기존 완료receipt에 결속된다. wrapper 자신의OSexit는 직접 관측했다고 주장하지 않는다.
- 모든 입력/SHA/관측수/시간head 초기통계/모델별 정확설정은 REPRODUCIBILITY.json과 기존 FROZEN_CONDITIONS.json을 따른다. 새 학습·추론·원격호출로 결과를 재생성하지 않았다.
- RESULT_TABLES.json의 source_bindings와 SOURCE_VERIFICATION.json이 표의 근거이며, 표 생성은 build_evidence_tables.py로 이루어졌다. 기존 결과 파일은 수정하지 않았다.
