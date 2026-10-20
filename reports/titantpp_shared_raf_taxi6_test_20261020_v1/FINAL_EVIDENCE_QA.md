# RAF·Taxi Test6 종료 증적 독립 검증

판정: **PASS**. 새 원격 호출이나 평가 helper 재실행 없이 회수된 원본을 검사했습니다.

- 평가 archive/manifest **205개**, pipeline 결속 **202개**, 선택 원본 archive **67개**의 모든 SHA와 archive 내부 바이트를 대조했습니다.
- 원본 source983와 참조 결과280 SHA를 확인했습니다. RAF epoch9/5/19, Taxi epoch112/148/127의 binary·tensor·history·원 종료/launch 결속이 유지됩니다.
- qualification6·Validation6·Test6, 총18개 worker와 pipeline1 실제 exit0가 일치합니다. 마지막 Validation exit보다 첫 Test launch가 늦으며 gate6 전체가 결속돼 있습니다.
- Validation **44874행**의 정확 대상·truth·history와 개별 예측/시간손실을 고정 허용오차로 대조했습니다. Validation/Test 여섯 지표 총 **72개**를 저장 행으로 독립 재계산했습니다.
- 종료 시 소유PID/GPU부재 assertion 통과와 기존 중단 외부모델 CUDA PID351048/351049/353905 보존을 확인했습니다. GPU 전체가 비었다는 뜻은 아닙니다.
- 기존 Cap 사용자중지 증적과 최종 결과 JSON/Markdown SHA를 재검증했습니다. 기존 과학 원본·checkpoint를 수정하지 않았습니다.

로컬 증적 검증은 현재 GPU 유휴 확인을 대체하지 않습니다. 종료 당시 중단 프로세스의 전체 신원 행은 pipeline receipt에 중복 저장되지 않아, 보존한 계약·검토 코드의 exact identity/state/descendant assertion과 성공 exit를 함께 검증했습니다. wrapper 자신의 OS exit는 직접 관측하지 않았습니다.

이번 결과는 원본 RAF·Taxi6개에 한정됩니다. 5090 Intermittent를 포함한 전체9개 완료를 주장하지 않습니다. 학습5090/cuDNN92000과 평가5080/cuDNN91900 차이는 남지만 고정 Validation 재현 기준을 통과했습니다.
