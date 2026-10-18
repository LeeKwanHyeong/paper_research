# 원본 MLP 저장 예측 범위 검증 — 완료

기존 입력 122개 SHA를 다시 확인했고 모두 일치했다. 기존 분석 원본과 모델 코드는 수정하지 않았다.

## 같은 native 환경의 4구조 비교 가능 여부

MLP16은 정확한 CUDA qualification 디렉터리에 세 데이터셋 모두 존재하지 않는다. MLP4·AllAvailable·CurrentOnly 3구조의 seed42 Validation 최초 512행은 저장돼 있으며, 3데이터셋 × 3쌍 = 9쌍에서 target_index·target_id·entity_id·seq·split·raw_quantity·recorded_gap·last_quantity·mean_quantity·history_length가 정확히 일치했다. 각 receipt와 prediction part SHA, 선택 epoch·checkpoint·state digest·source 결속을 확인했다. 체크포인트는 모델로 읽지 않고 바이너리 SHA만 검증했다.

이 512행은 최초 4배치이며 무작위표본이나 전수 결과로 해석할 수 없다. MLP16까지 같은 qualification 환경에서 비교했다고 표현하면 안 된다.

## 여전히 부족한 자료

- MLP16과 CurrentOnly의 저장 Train 개별 예측은 찾지 못했다.
- MLP4와 두 구조 대조군의 seed52·62 개별 예측도 찾지 못했다.
- MLP4·AllAvailable Train 비교는 seed42 가중 층화표본만 가능하다.
- 전수 Validation 행은 MLP4 seed42와 MLP16 seed42·52·62에 한정된다.

## Runtime 한계

이번 3구조 qualification은 Linux CUDA/Torch2.11.0/cuDNN91900이다. 기존 MLP16 Taxi·RAF seed42 전수 Validation은 macOS CPU/Torch2.14.0이므로 비교 시 출처 차이를 명시해야 한다.

새 학습·추론·원격 조회·Test 열람·CNN 변경은 수행하지 않았다. 세부 경로와 SHA는 COVERAGE_AUDIT.json에 기록했다.
