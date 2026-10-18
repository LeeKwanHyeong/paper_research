# Cap 외부27 — 학습 시작 확인

관측 시각: 2026-10-18 09:42:00 KST(5080),09:42:02 KST(5090). 문서 작성 시각을 새 관측으로 사용하지 않는다.

**학습 진입 — 진행 중**

S2P2·RMTPP·THP × Taxi·RAF·Intermittent × seed42·52·62=27개. 완료0·진행6·대기21이며 Test0이다.

| GPU | 데이터 | seed | 모델 | 저장 epoch | 현재 best epoch |
|---|---|---:|---|---:|---:|
| 5080 | Taxi | 42 | rmtpp | 3 | 3 |
| 5080 | Taxi | 42 | thp | 2 | 2 |
| 5080 | Taxi | 42 | s2p2_matched_head | 2 | 1 |
| 5090 | RAF | 42 | rmtpp | 7 | 3 |
| 5090 | RAF | 42 | thp | 5 | 3 |
| 5090 | RAF | 42 | s2p2_matched_head | 5 | 3 |

각GPU 최대3개를 유지한다. 5080 총12개/5090 총15개이며 같은 데이터·seed는 저장CapMLP와 같은GPU다. nativeRuntime/GPU/전체학습크기합성 admission과실제worker/PID/GPU연결·첫epoch를 확인했다. 메모리admission은순차peak3배추정이며3프로세스동시peak실측은아니다.

**고정 모델 평가와 비교 — 다음 작업**

서버별학습완료 → 최초최소Validation RMSE checkpoint고정 → 해당서버전체Validation재현gate → 동일checkpointTest → 종료원본회수·SHA확인 → 저장CapMLP4/16/AllAvailable/CurrentOnly와3seed평균±표본SD·seed별차이비교 순서다. 유한서버큐는Test까지자동연결하며새반복개선자동화는생성하지않았다.

Cap은관측과정답수량이모두제한된별도과제다. 원본수량과의점수크기만으로우월성을주장하지않는다. 세외부모델은공통head를사용한다. 기존CNN·CapMLP·원본·checkpoint·실패기록은보존했다.

소스커밋: `9d9e09a346ec6417307bc5237cf1f2f2e2e447f0`. 로컬79tests+24subtests와독립검토및실제배포archiveimport/native검증을통과했다. 상세조건은EXPERIMENT_PLAN.md,실행증적은START_RECEIPT.json,후속확인은PIPELINE_GUIDE.md에있다.
