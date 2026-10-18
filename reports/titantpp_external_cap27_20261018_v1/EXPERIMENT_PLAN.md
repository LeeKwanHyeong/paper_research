# Cap 외부 백본 27개 비교

## 비교 범위 확정 — 완료

- 2026-10-18 사용자 요청에 따라 S2P2·RMTPP·THP를 Taxi·RAF·Intermittent의 seed42·52·62에서 새로 학습한다. 총27개다.
- 기존 Cap36의 네 MLP는 저장 결과만 재사용한다. 기존 CNN·원본 데이터·checkpoint·실패 기록·공용 Runtime은 변경하지 않는다.
- Cap은 관측 이력과 정답 수량 모두 Train Q3로 제한한 별도 과제다. 상한은 Taxi65·RAF10·Intermittent20이다. 원본 데이터와의 raw RMSE/MAE 차이를 개선율로 해석하지 않는다.
- 외부 모델은 기존 `rmtpp`, `thp`, `s2p2_matched_head` 구현이다. 공통 수량·시간 head를 사용하므로 각 논문의 원형 전체 모델 재현이라는 주장은 하지 않는다. 새로운 백본 탐색은 아니다.

## 공통 조건과 학습 배정 — 완료

- Cap36의 입력9개, split·평가 대상, Cap Train 초기 통계와 시간 통계, lookback·max sequence length를 동일하게 상속한다. log1p와 count-only log regression, recorded positive-integer lognormal time likelihood를 유지한다.
- hidden64, AdamW lr0.001/weight decay0.01, gradient clipping1, batch128, 최대300epoch·최소40epoch·patience40, seed42·52·62다. 기존 checkpoint warm start는 없다.
- 전체 Validation 수량RMSE의 최초 strict finite 최소epoch를 선택한다. 동률은 첫 epoch를 유지하며 Test에서 checkpoint를 다시 고르지 않는다.
- 5080: Taxi9개와 Intermittent seed42의3개, 총12개. 5090: RAF9개와 Intermittent seed52·62의6개, 총15개. 같은 데이터셋·seed의 세 외부 모델과 저장 Cap MLP는 같은 서버에 배치한다.
- 각 서버 최대3개 병렬이며 순서는 데이터셋→seed→RMTPP/THP/S2P2로 고정한다. fit36시간·서버 큐168시간·평가 stage2시간 상한을 둔다. 이 새 실행의 제한이며 이전 실험 기한을 변경하지 않는다.
- 5080/cuDNN91900과5090/cuDNN92000 차이를 기록한다. Intermittent42와52·62의 서버가 다르다.

## 격리 구현과 실행 검증 — 진행 중

- 새 관리·worker·큐·집계 파일만 추가한다. 기존 Cap36과 외부 모델 구현을 덮어쓰지 않는다.
- 합성 입력에서 세 모델의 학습·선택·재로드·Validation 재현·Test 연결, 미래 정답 차단·batch 독립성과 평가 상태 불변을 확인한다. 합성 검사를 실데이터 평가 완료로 표시하지 않는다.
- 커밋된 source와 Cap 입력을 새 전용 경로에 패키징한다. 기존 경로 탐색에 필요한 빈 `sample_data`도 포함한다.
- 실제 GPU UUID·Runtime·입력/source와 현재 GPU 사용 상태를 확인한다. 각 외부 모델의 전체 학습 크기 forward/backward/optimizer step peak를 측정하고, 최대 peak의3배가 GPU85% 미만인지 확인한다. 이는 동시3프로세스 실측 peak가 아닌 보수적 추정이다.
- qualification 뒤 전용 tmux의 유한 큐를 한 번 시작한다. 자동 retry/resume·기존 프로세스 종료·원본 덮어쓰기는 없다. 실패 원문을 보존한다.

## 선택 모델 평가와 비교 — 다음 작업

- 각 서버의 모든 학습이 끝나면 선택 checkpoint를 고정해 개별 수량 예측·시간 손실·여섯 지표와 평가 대상을 Validation에서 재현한다. 서버별 전체 gate를 통과한 뒤 그 서버의 Test를 진행한다.
- 종료코드와 소유 PID/GPU 부재, binary·원본 archive/manifest SHA를 확인해 새 retrieval 경로에 회수한다. supervisor 자신의 OS exit는 직접 관측하지 않았음을 구분한다.
- 세 외부 모델과 네 저장 MLP의 Validation/Test를 데이터셋별로 나란히 제시한다. seed별 외부−각 MLP 절대·상대차이와 악화,3seed 평균±표본SD(ddof1)를 보존한다. 모델 간 Test 사후 최저와 Validation 선택을 구분한다.
- 전체 RMSE·MAE·TimeNLL과 `q==cap` 집단을 보고한다. `q>cap`은 빈 집합/N/A이며 예측 출력은 clipping하지 않는다. 서로 다른 데이터셋의 raw 오류는 평균내지 않는다.
- 기존 반복개선 자동화를 재개하거나 새 반복 자동화를 만들지 않는다. 각 서버의 승인된27개 학습·평가 큐만 완료 후 종료한다.
