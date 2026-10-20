# RAF·Taxi 원본 Shared15 Test6 독립 구현 검토

판정: **PASS**. 최종 파일에서 열린 중대 실행 오류는 없습니다. 로컬 합성 검사 75개와 subtest 2개가 실제 exit0으로 통과했습니다. 검사 전후 검토 파일 SHA가 같았습니다.

**원본과 평가 범위를 유지한다 — 완료**

- 원본 학습 source983와 보존한 운영 의존성3 SHA를 확인했습니다. RAF epoch9/5/19, Taxi epoch112/148/127의 선택값이 고정됩니다.
- 완료한 RAF·Taxi6만 읽기 전용 회수합니다. 원본9개 전체 완료를 주장하지 않으며 5090 Intermittent 학습은 변경하지 않습니다. 원래 train_launch.json/train_exit.json의 API와 신원 결속을 코드로 확인했습니다.
- 기존 비교군42개의 저장 결과와 참조280 SHA를 확인했습니다. 신규 추론은 수행하지 않았습니다.

**Validation 전체 통과 뒤 Test를 허용한다 — 완료**

- 합성6 → Validation6 → 개별 예측·시간손실·여섯 지표 재현 gate → Test6 순서와 최대3 병렬을 검증했습니다. 실패·timeout·재현 불일치 때 Test가 시작되지 않습니다.
- 학습5090/cuDNN92000과 평가5080/cuDNN91900을 구분하며 5080 GPU UUID를 고정합니다. 중단 외부 프로세스4의 신원을 보존하고 CUDA 문맥3개만 허용합니다. 종료 개수는 합성6+실데이터12=18, pipeline1입니다.
- 중지 관측 POST_STOP와 최종 FINAL_RECEIPT를 별도로 검증하는 최종 수정을 확인했습니다. 부모가 지적한5080 root suffix와 inspect 재컴파일 단계6 카운터 수정도 합성 검사에서 확인했습니다.
- 회수 스트림은 tar 원문을 stdout에, 진단 메시지는 stderr에 분리합니다. 로컬 동일 import의 stdout은0byte였습니다.

**실제 평가로 이어간다 — 다음 작업**

- 부모 작업에서 검토된 신규 코드 커밋 후 별도 평가경로를 준비하고 fresh5080 조건을 확인합니다.
- 실제 native Validation6 재현을 모두 통과한 경우에만 고정 Test6를 실행합니다. GPU/cuDNN 차이가 고정 허용오차를 넘으면 중지해야 합니다.

이 보고서는 구현·합성 검토 결과입니다. 원격 호출과 실제 native Validation/Test는 수행하지 않았으며 현재 GPU 유휴나 실제 평가 완료를 뜻하지 않습니다. 상세 SHA·검사 명령·보존 범위는 INDEPENDENT_QA.json에 기록했습니다.
