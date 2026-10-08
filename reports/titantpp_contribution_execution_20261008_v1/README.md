# Instacart와 대표 구조의 두 기여를 순서대로 검증한다

사용자는 2026-10-08 이 문서의 세 주제 모두 구현·검증·5080/5090 학습을 승인했다. Test 접근, 공용 Runtime 변경, 자동 재시도는 포함하지 않는다. 과거 결과와 신규 결과를 섞어 하나의 대표 모델 성능으로 표시하지 않는다.

**Instacart의 비교 목적과 조건을 고정한다 — 계약 완료 / 학습 준비 중**
- 대상은 paper_research와 전용 5080/5090 root다. 폭8·12와 CNN+GRU54 각각 seed42/52/62의 9개를 새로 학습한다.
- 기존 폭16 selected binary 3개는 SHA로 보존한 역사적 참고 원본이다. 이번에 다시 학습하거나 같은 GPU의 짝지어진 효율 대조로 취급하지 않는다.
- 기존 Instacart 데이터·split·Train 통계, batch128, 길이64, lookback52, day 단위, 큰 수량 >35를 유지한다. 이 단계는 데이터 확장과 기준선 비교이며 보정·검색의 추가 효과를 직접 증명하지 않는다.
- 5080은 seed42/62의 6개, 5090은 seed52의 3개를 맡는다. 같은 seed의 세 구조는 같은 GPU/Runtime에서 비교한다.

**대표 CNN+GRU54에서 명시적 보정의 추가 효과를 확인한다 — 구현·로컬 테스트 완료 / Instacart 종료 대기**
- Taxi·Intermittent·RAF × seed42/52/62 × full·보정 없음·현재 상태 전용의 27개를 새로 학습한다.
- full과 현재 상태 전용은 활성 보정 파라미터 25,308개를 맞춘다. 현재 상태 전용은 보정 경로에서 직전 행·이전 GRU 상태를 직접 받지 않는다. 현재 인코더 상태 자체에는 과거 요약이 남는다.
- 현재 상태 전용의 비선형 변환·계산 깊이는 full과 다르므로 같은 파라미터 수가 같은 함수 용량이나 같은 비용을 뜻하지 않는다. 전체·큰 수량 RMSE/MAE/TimeNLL과 같은 GPU 비용을 함께 보고한다.
- 개발 근거 기준은 같은 두 개 이상 데이터 각각의 모든 3seed에서 full이 두 대조군 모두보다 전체 수량 RMSE를 엄격히 개선하는 것이다. 불리한 결과도 보존한다.

**고정 비중 검색의 추가 효과를 분리한다 — 구현·로컬 테스트 완료 / 이력 보정 종료 대기**
- 앞 단계의 새 full CNN+GRU54 9개를 mean 기준선으로 재사용한다. weighted·fixed·generic512의 27개만 새로 학습해 mean을 중복 학습하지 않는다.
- fixed와 일반 학습 검색은 추가 활성 파라미터 512개, prototype64, top4, 같은 관측 과거32 요약과 고정 온도1을 맞춘다. 기존 generic521은 이번 예산 대조군으로 쓰지 않는다.
- 두 쿼리의 cosine 평균에는 연결 표현과의 대수적 동등성이 있다. 이 비교는 분리 정규화와 공동 쿼리의 귀납 편향을 검증하며, 이름만으로 참신성을 주장하지 않는다. prototype은 학습된 정적 원형이며 실제 관측 사건 저장소가 아니다.
- 같은 두 개 이상 데이터 각각의 모든 3seed에서 fixed가 mean·weighted·generic512 모두보다 전체 RMSE를 개선하고, 해당 GPU의 길이64/256 모두에서 mean 대비 추론 +20%·학습step +30%·peak메모리 +20% 이내일 때 개발 단계 지지로 판정한다. 큰 수량과 시간 개선은 별도 결과에 근거한다.

**실제 실행과 원본을 보존한다 — 진행 중**
- 신규 fit은 9+27+27=63개다. 각 stage min40/max300/patience40, 조건36시간·host168시간, worker1, fresh fit, 자동 retry/resume 없음이다. 마감은 완료 ETA가 아니다.
- Instacart → 이력 보정 → 검색 순서다. 앞 stage의 전체 조건·실제 프로세스 종료·manifest/archive와 selected/last binary rawSHA·최종 Validation 보고 검증 후 다음 stage를 시작한다.
- core 세 데이터는 5080 Taxi/RAF18개, 5090 Intermittent9개씩 맡는다. GPU 간 시간을 섞지 않는다.
- Validation 최초 strict finite 최소 전체 raw 수량 RMSE의 같은 epoch에서 여섯 지표를 집계한다. seed 평균과 표본 SD, 손해 사례, native 비용을 기록한다. 신규 Test·CPU binary 재추론 감사·새 독립 평가는 별도 미완료다.

**남은 작업을 의존관계에 따라 이어간다 — 다음 작업**
- 각 GPU native qualification·실제 종료를 확인해 Instacart 학습을 시작한다. 후속 두 stage는 승인된 완료 gate에 따라 연결한다.
- 과학 주장에 필요한 성능 판정은 학습·원본 회수 후 수행한다. 구현 완료를 기여 검증 완료로 표시하지 않는다.

실행 저장소: paper_research, 브랜치 `codex/contextual-retrieval-pilot-20261007`. 신규 파일만 별도 커밋하며 기존 사용자 변경은 보존한다. Push/MR은 이번 실행 범위에 포함하지 않는다.
