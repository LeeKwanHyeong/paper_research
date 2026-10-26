# 전체 과거 조건의 지원 범위와 PHEN 관측 기록 확보

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-26
- Verification Status: 전체 수집·독립 QA·저장 inventory 재대조 완료

## 원본과 분석 범위를 고정한다 — 완료
- 대상은 Intermittent Train·Validation이다. seed42 best epoch1/last41, seed52 best5/last45의 기존 checkpoint와 frozen source를 사용한다.
- 재학습·가중치 갱신·Test 열람·기존 Runtime 변경·원고 변경은 수행하지 않는다. 양 서버의 기존 작업과 모든 checkpoint·실패 기록을 보존한다.
- 직전 수량·간격·변화·이력 길이의 비교 조건과 최소 표본 기준은 support/support_spec.json에 결과 확인 전에 고정했다.

## 전체 원자료에서 비교 가능한 과거 조건을 확인한다 — 완료
- 전체 Train393,824건과 Validation86,285건을 집계하고 기존 2,048건 표본과 비교했다. 독립 QA를 통과했다.
- 정확한 조건에서 각 split에 6건·2개 시계열 이상인 조건은 532개이며, 해당 Validation은 29,720건(34.44%)이다. 같은 정확한 조건이 Train에 없는 Validation은 45,792건(53.07%)이다.
- 표본 수 기준을 충족해도 서로 다른 Prototype 검색 조합이 존재하는지는 아직 확인하지 않았다. 이 집계만으로 모듈 효과나 인과관계를 판단하지 않는다.

## 필요한 전체 관측을 기존 checkpoint에서 확보한다 — 완료
- 기존 best Validation172,570건과 v4에서 정상 종료한 seed42 best Train393,824건을 재사용했다. v4 전체 dispatcher 실패는 그대로 보존하며, 정상 종료한 자식의 결과만 독립 QA 근거로 사용한다.
- v6의 5080에서 seed42 last Train·Validation480,109건, v8의 5090에서 seed52 best Train과 last Train·Validation873,933건을 수집했다.
- 네 checkpoint 각각 Train393,824건·Validation86,285건으로 총1,920,436건이다. 신규 실행의 실제 collector/child exit0, 종료 후 보호 상태, 원본 및 checkpoint 상태 불변, 사건 신원·중복·누락·청크 SHA·NLL 산술·전체 집계를 독립 검증했다.
- Prototype top4 번호·cosine, 수량 예측, 시간 중심·폭·NLL을 함께 저장했다. best Train의 head 입력64차원만 별도 보존했고 head 학습은 하지 않았다.
- 재사용 Validation의 μ·σ는 기존 float32 export이고 NLL은 원래 float64다. 없는 native log-MSE·학습 손실·head preactivation·hidden은 채우지 않았다. 신규 시간 관측은 native float64다.

## 수치 기준 변경과 실패 이력을 구분해 보존한다 — 완료
- 기존 cosine 기준과 log-MSE 기준에서 각각 실패를 기록했다. 실패 원문과 실제 종료 판정은 수정하지 않는다.
- 사용자는 cosine 절대허용차1e-7→1e-6, 이어 log-MSE 절대허용차1e-7→1e-6을 각각 명시 승인했다. 상대허용차1e-5와 수량 예측·시간 NLL·Prototype 번호/순서·신원 기준은 유지했다.
- 승인 원문과 별도 계약을 SHA로 연결했다. v6 5080은 두 split 각2,048건, v8 5090은 세 cohort 각2,048건의 고정 참조를 먼저 통과한 뒤 전체 관측을 수집했다.
- v7 best-only 대안은 준비 스크립트만 작성했고 실행하지 않았다. 운영 오류와 수치 검증 실패는 과거 PHEN 학습 실패나 Validation 악화의 원인으로 해석하지 않는다.

## 최종 목록과 보고서를 원본에 연결한다 — 완료
- 최종 inventory를 원본에서 재계산한 뒤 별도 파일로 생성하고, 저장된 inventory와 요약표도 다시 대조했다. check-only→write→check-only는 모두 actual exit0이다. 이전 PARTIAL_COLLECTION과 모든 FAIL 기록은 보존했다.
- 고유 관측 Parquet434개와 best Train 은닉벡터 NPZ194개는 합계322,028,937bytes다. Archive·소스·로그·manifest의 중복 보관 용량은 제외했다.

## 남은 분석의 순서를 정한다 — 다음 작업
1. 고정한 정확한 과거 조건 532개 안에서 실제 Prototype 검색 조합별 사건·시계열 수를 확인한다. 비교가 가능한 조건에서 과거 수량·간격 이상의 설명력이 남는지 분석한다.
2. 같은 조건에서 Train·Validation의 시간 중심 오차와 폭을 best·last별로 비교한다. 시간 NLL의 악화가 중심 이동과 폭 중 어느 쪽에 연결되는지 확인한다.
- 두 분석은 같은 고정 자료를 읽는 작업이므로 병렬 진행할 수 있다. 수집 완료만으로 Prototype의 해석력이나 시간 손실 악화 원인을 확정하지 않는다.
